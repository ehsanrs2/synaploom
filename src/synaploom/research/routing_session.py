"""Opt-in inference/diagnostic routing facade; never a TrainingEngine model.

The legacy path delegates to SynapLoomModel.forward without changing defaults.
Sample routing is H1a-only (no edges/rounds), with a versioned [B,K] trace.
The session is deliberately NOT an nn.Module: no new parameters/state_dict keys,
no optimizer interface, and no TrainingEngine duck-typed compatibility.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from synaploom.model import SynapLoomModel


@dataclass(frozen=True)
class RoutingSpec:
    granularity: str = "batch"
    execution: str = "legacy"
    tie_policy: str = "legacy_topk"

    def __post_init__(self) -> None:
        valid = {
            ("batch", "legacy", "legacy_topk"),
            ("sample", "row_reference", "node_id_ascending_v1"),
            ("sample", "grouped", "node_id_ascending_v1"),
        }
        if (self.granularity, self.execution, self.tie_policy) not in valid:
            raise ValueError("unsupported routing spec; only registered combinations are valid")


@dataclass(frozen=True)
class SampleRoutingTraceV1:
    schema_version: int
    granularity: str
    requested_execution: str
    actual_execution: str
    tie_policy: str
    recruited_node_ids: Tensor
    selected_node_ids: Tensor
    router_weights: Tensor
    routing_context: Tensor
    unique_executed_ids: Tensor
    effective_expert_rows: int
    actual_expert_calls: int
    executed_edge_count: int
    structural_version: int


@dataclass(frozen=True)
class SampleRoutingOutputV1:
    logits: Tensor
    representation: Tensor
    aux_loss: Tensor
    trace: SampleRoutingTraceV1


class RoutingSession:
    """Explicit opt-in; not a checkpoint root, trainer or optimizer wrapper.

    Inference creates no autograd graph. Diagnostic backward intentionally permits
    an autograd graph, but this class performs no optimizer step or statistics update.
    Modes, RNG, source parameters and structural configuration remain caller-owned.
    """

    def __init__(self, model: SynapLoomModel, spec: RoutingSpec | None = None):
        if not isinstance(model, SynapLoomModel):
            raise TypeError("RoutingSession requires SynapLoomModel")
        if spec is not None and not isinstance(spec, RoutingSpec):
            raise TypeError("spec must be RoutingSpec")
        self.model = model
        self.spec = spec if spec is not None else RoutingSpec()
        if self.spec.granularity == "sample":
            self._validate_sample_capability()

    def _validate_sample_capability(self) -> None:
        g = self.model.config.graph
        if g.q_max != 0 or g.message_rounds != 0:
            raise ValueError("sample routing requires q_max=0 and message_rounds=0")
        self.model.graph_state.validate()

    def forward(
        self,
        x: Tensor,
        *,
        purpose: str = "inference",
        update_statistics: bool = False,
    ):
        if purpose not in ("inference", "diagnostic_backward"):
            raise ValueError("routing session permits only inference or diagnostic_backward")
        if self.spec.granularity == "batch":
            # Preserve the original output type and exact legacy forwarding.
            return self.model(x, update_statistics=update_statistics)

        # Reject unsupported behavior BEFORE any model forward or mutation.
        if update_statistics:
            raise ValueError("sample routing cannot update production statistics")
        self._validate_sample_capability()
        if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("sample routing requires nonempty rank-2 input")
        if x.shape[1] != self.model.config.input_dim:
            raise ValueError("sample routing input feature dimension mismatch")
        if x.device != self.model.graph_state.routing_keys.device:
            raise ValueError("sample routing inputs and parameters must share a device")
        if x.dtype != self.model.graph_state.routing_keys.dtype:
            raise ValueError("sample routing input and model dtype must match")

        context = torch.no_grad() if purpose == "inference" else nullcontext()
        with context:
            return self._sample_forward(x)

    def _sample_forward(self, x: Tensor) -> SampleRoutingOutputV1:
        model = self.model
        structural_version = int(model.graph_state.structural_version.item())
        recruited = model.graph_state.recruited_ids()  # ascending IDs
        if recruited.numel() == 0:
            raise ValueError("sample routing requires recruited capacity")
        k = min(model.config.graph.top_k, int(recruited.numel()))

        if self.spec.execution == "row_reference":
            representations, queries_list, ids_list, weights_list = [], [], [], []
            for sample in x.split(1):
                encoded = model.encoder(sample)
                query = model.router.query(encoded)[0]
                scores = torch.mv(model.graph_state.routing_keys[recruited], query)
                scores = scores / math.sqrt(model.config.graph.router_dim)
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("nonfinite routing scores")
                local = torch.argsort(scores, descending=True, stable=True)[:k]
                ids = recruited[local]
                weights = scores[local].softmax(dim=0)
                representation, _ = model.executor(
                    encoded, ids, weights, model.graph_state
                )
                representations.append(representation)
                queries_list.append(query)
                ids_list.append(ids)
                weights_list.append(weights)
            representation = torch.cat(representations, dim=0)
            queries = torch.stack(queries_list)
            selected = torch.stack(ids_list)
            weights = torch.stack(weights_list)
            actual_expert_calls = int(x.shape[0]) * k
            actual_execution = "row_reference"
        else:
            encoded = model.encoder(x)
            queries = model.router.query(encoded)
            scores = queries @ model.graph_state.routing_keys[recruited].T
            scores = scores / math.sqrt(model.config.graph.router_dim)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("nonfinite routing scores")
            local = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            selected = recruited[local]
            weights = scores.gather(1, local).softmax(dim=1)

            flattened = selected.reshape(-1)
            order = torch.argsort(flattened, stable=True)
            unique, counts = torch.unique_consecutive(
                flattened[order], return_counts=True
            )
            plan = torch.stack((unique, counts), dim=1).detach().cpu().tolist()
            grouped_slots = order.split([count for _, count in plan])
            grouped = []
            for (node_id, _), slots in zip(plan, grouped_slots):
                rows = torch.div(slots, k, rounding_mode="floor")
                grouped.append(
                    model.executor.microcircuits[node_id](
                        encoded.index_select(0, rows)
                    )
                )
            # Every assignment slot is unique; restore its original row/K position.
            values = torch.cat(grouped).index_select(0, torch.argsort(order))
            expert_values = values.reshape(len(x), k, encoded.shape[1])
            representation = (weights.unsqueeze(-1) * expert_values).sum(dim=1)
            actual_expert_calls = len(plan)
            actual_execution = "grouped"

        logits = model.head(representation)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite sample routing logits")
        if int(model.graph_state.structural_version.item()) != structural_version:
            raise RuntimeError("structural version changed during sample routing")
        union = torch.unique(selected.flatten(), sorted=True)
        trace = SampleRoutingTraceV1(
            schema_version=1,
            granularity="sample",
            requested_execution=self.spec.execution,
            actual_execution=actual_execution,
            tie_policy="node_id_ascending_v1",
            recruited_node_ids=recruited,
            selected_node_ids=selected,
            router_weights=weights,
            routing_context=queries,
            unique_executed_ids=union,
            effective_expert_rows=len(x) * k,
            actual_expert_calls=actual_expert_calls,
            executed_edge_count=0,
            structural_version=structural_version,
        )
        return SampleRoutingOutputV1(
            logits=logits,
            representation=representation,
            aux_loss=logits.new_zeros(()),
            trace=trace,
        )
