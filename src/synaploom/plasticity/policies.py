from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from synaploom.core.graph_state import GraphState
from synaploom.core.outputs import ForwardTrace
from synaploom.core.states import NodeState


@dataclass(frozen=True)
class StructuralObservation:
    step: int
    task_boundary: bool
    selected_node_ids: tuple[int, ...]
    router_weights: tuple[float, ...]
    routing_context: torch.Tensor

    @classmethod
    def from_trace(
        cls,
        step: int,
        trace: ForwardTrace,
        *,
        task_boundary: bool = False,
    ) -> "StructuralObservation":
        detached = trace.detached()
        return cls(
            step=int(step),
            task_boundary=bool(task_boundary),
            selected_node_ids=tuple(int(v) for v in detached.selected_node_ids.tolist()),
            router_weights=tuple(float(v) for v in detached.router_weights.tolist()),
            routing_context=detached.routing_context.clone(),
        )


class RecruitmentPolicy(Protocol):
    def should_recruit(self, observation: StructuralObservation, graph_state: GraphState) -> bool:
        ...


@dataclass(frozen=True)
class NoRecruitment:
    def should_recruit(self, observation: StructuralObservation, graph_state: GraphState) -> bool:
        return False


@dataclass(frozen=True)
class FixedScheduleRecruitment:
    steps: frozenset[int]

    def __init__(self, steps: tuple[int, ...] | list[int] | set[int]) -> None:
        object.__setattr__(self, "steps", frozenset(int(step) for step in steps))

    def should_recruit(self, observation: StructuralObservation, graph_state: GraphState) -> bool:
        return observation.step in self.steps and bool(graph_state.dormant_mask.any())


@dataclass(frozen=True)
class TaskBoundaryRecruitment:
    def should_recruit(self, observation: StructuralObservation, graph_state: GraphState) -> bool:
        return observation.task_boundary and bool(graph_state.dormant_mask.any())


class EdgeInitPolicy(Protocol):
    def targets(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
        new_node_id: int,
    ) -> tuple[int, ...]:
        ...


@dataclass(frozen=True)
class NoEdges:
    def targets(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
        new_node_id: int,
    ) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True)
class ConnectToSelectedTopR:
    top_r: int = 1

    def __post_init__(self) -> None:
        if self.top_r < 0:
            raise ValueError("top_r must be non-negative")

    def targets(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
        new_node_id: int,
    ) -> tuple[int, ...]:
        pairs = sorted(
            zip(observation.selected_node_ids, observation.router_weights),
            key=lambda item: item[1],
            reverse=True,
        )
        limit = min(self.top_r, graph_state.config.q_max)
        return tuple(node_id for node_id, _ in pairs if node_id != new_node_id)[:limit]


class ConsolidationPolicy(Protocol):
    def node_ids(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
    ) -> tuple[int, ...]:
        ...


@dataclass(frozen=True)
class NoConsolidation:
    def node_ids(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
    ) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True)
class TaskBoundaryTopImportance:
    top_n: int = 1

    def __post_init__(self) -> None:
        if self.top_n < 0:
            raise ValueError("top_n must be non-negative")

    def node_ids(
        self,
        observation: StructuralObservation,
        graph_state: GraphState,
    ) -> tuple[int, ...]:
        if not observation.task_boundary or self.top_n == 0:
            return ()

        plastic = torch.nonzero(
            graph_state.node_state == int(NodeState.PLASTIC),
            as_tuple=False,
        ).flatten()
        if plastic.numel() == 0:
            return ()

        scores = graph_state.node_importance_ema.index_select(0, plastic)
        n = min(self.top_n, int(plastic.numel()))
        _, local_ids = torch.topk(scores, k=n)
        selected = plastic.index_select(0, local_ids)
        return tuple(int(v) for v in selected.tolist())
