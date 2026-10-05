from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from synaploom.core.graph_state import GraphState
from synaploom.plasticity.policies import (
    ConsolidationPolicy,
    EdgeInitPolicy,
    NoConsolidation,
    NoEdges,
    NoRecruitment,
    RecruitmentPolicy,
    StructuralObservation,
)
from synaploom.events import JsonlEventLogger, StructuralEvent


@dataclass(frozen=True)
class StructuralPlan:
    step: int
    recruit_node_id: int | None = None
    routing_key: Tensor | None = None
    edge_targets: tuple[int, ...] = ()
    consolidate_node_ids: tuple[int, ...] = ()


class StructuralController:
    """Slow-timescale controller for controlled structural adaptation."""

    def __init__(
        self,
        *,
        recruitment_policy: RecruitmentPolicy | None = None,
        edge_init_policy: EdgeInitPolicy | None = None,
        consolidation_policy: ConsolidationPolicy | None = None,
        event_logger: JsonlEventLogger | None = None,
        imprint_noise_std: float = 0.0,
        edge_gate_logit: float = 0.0,
    ) -> None:
        if imprint_noise_std < 0:
            raise ValueError("imprint_noise_std must be non-negative")
        self.recruitment_policy = recruitment_policy or NoRecruitment()
        self.edge_init_policy = edge_init_policy or NoEdges()
        self.consolidation_policy = consolidation_policy or NoConsolidation()
        self.event_logger = event_logger
        self.imprint_noise_std = float(imprint_noise_std)
        self.edge_gate_logit = float(edge_gate_logit)
        self._observation: StructuralObservation | None = None
        self._last_applied_step = -1

    def observe(self, observation: StructuralObservation) -> None:
        self._observation = observation

    def should_update(self, graph_state: GraphState) -> bool:
        observation = self._require_observation()
        if observation.step <= self._last_applied_step:
            return False
        return (
            self.recruitment_policy.should_recruit(observation, graph_state)
            or bool(self.consolidation_policy.node_ids(observation, graph_state))
        )

    def propose(self, graph_state: GraphState) -> StructuralPlan:
        observation = self._require_observation()
        if observation.step <= self._last_applied_step:
            return StructuralPlan(step=observation.step)

        consolidate = self.consolidation_policy.node_ids(observation, graph_state)
        recruit_node_id: int | None = None
        routing_key: Tensor | None = None
        edge_targets: tuple[int, ...] = ()

        if self.recruitment_policy.should_recruit(observation, graph_state):
            free = torch.nonzero(graph_state.dormant_mask, as_tuple=False).flatten()
            if free.numel() > 0:
                recruit_node_id = int(free[0].item())
                routing_key = self._imprint_routing_key(observation.routing_context, graph_state)
                edge_targets = self.edge_init_policy.targets(
                    observation,
                    graph_state,
                    recruit_node_id,
                )

        return StructuralPlan(
            step=observation.step,
            recruit_node_id=recruit_node_id,
            routing_key=routing_key,
            edge_targets=edge_targets,
            consolidate_node_ids=consolidate,
        )

    @torch.no_grad()
    def apply(self, graph_state: GraphState, plan: StructuralPlan) -> list[StructuralEvent]:
        if plan.step <= self._last_applied_step:
            return []

        events: list[StructuralEvent] = []

        if plan.consolidate_node_ids:
            graph_state.consolidate(list(plan.consolidate_node_ids))
            for node_id in plan.consolidate_node_ids:
                events.append(
                    StructuralEvent(
                        step=plan.step,
                        kind="consolidate",
                        structural_version=int(graph_state.structural_version.item()),
                        node_id=node_id,
                    )
                )

        if plan.recruit_node_id is not None:
            graph_state.recruit(plan.recruit_node_id, plan.routing_key)
            events.append(
                StructuralEvent(
                    step=plan.step,
                    kind="recruit",
                    structural_version=int(graph_state.structural_version.item()),
                    node_id=plan.recruit_node_id,
                )
            )
            for dst in plan.edge_targets:
                graph_state.add_edge(
                    plan.recruit_node_id,
                    dst,
                    gate=self.edge_gate_logit,
                )
                events.append(
                    StructuralEvent(
                        step=plan.step,
                        kind="edge_add",
                        structural_version=int(graph_state.structural_version.item()),
                        src=plan.recruit_node_id,
                        dst=dst,
                        metadata={"gate_logit": self.edge_gate_logit},
                    )
                )

        self._last_applied_step = plan.step
        if self.event_logger is not None:
            for event in events:
                self.event_logger.append(event)
        return events

    def state_dict(self) -> dict[str, int]:
        return {"last_applied_step": self._last_applied_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._last_applied_step = int(state.get("last_applied_step", -1))

    def _require_observation(self) -> StructuralObservation:
        if self._observation is None:
            raise RuntimeError("observe() must be called before controller operations")
        return self._observation

    def _imprint_routing_key(self, context: Tensor, graph_state: GraphState) -> Tensor:
        device = graph_state.routing_keys.device
        dtype = graph_state.routing_keys.dtype
        context = context.to(device=device, dtype=dtype).flatten()
        if context.shape != (graph_state.config.router_dim,):
            raise ValueError("routing context has wrong shape")

        recruited = graph_state.routing_keys[graph_state.recruited_mask].detach()
        target_norm = recruited.norm(dim=1).median().clamp_min(1e-6)
        context_norm = context.norm().clamp_min(1e-6)
        key = context / context_norm * target_norm

        if self.imprint_noise_std > 0:
            key = key + torch.randn_like(key) * target_norm * self.imprint_noise_std
            key_norm = key.norm().clamp_min(1e-6)
            key = key / key_norm * target_norm
        return key.detach()
