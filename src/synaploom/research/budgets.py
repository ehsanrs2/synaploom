from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from synaploom.model import SynapLoomModel


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass(frozen=True)
class SynapLoomBudget:
    total_trainable_parameters: int
    shared_forward_parameters: int
    total_microcircuit_parameters: int
    per_microcircuit_parameters: int
    executed_microcircuit_parameters: int
    recruited_routing_key_parameters: int
    total_routing_key_parameters: int
    relation_parameters: int
    relation_parameters_executed: int
    top_k: int
    recruited_nodes: int
    total_nodes: int

    @property
    def active_parameter_proxy(self) -> int:
        """Approximate per-batch trainable parameter footprint, not FLOPs."""
        return (
            self.shared_forward_parameters
            + self.executed_microcircuit_parameters
            + self.recruited_routing_key_parameters
            + self.relation_parameters_executed
        )


def synaploom_budget(model: SynapLoomModel) -> SynapLoomBudget:
    graph = model.graph_state
    cfg = graph.config
    per_micro = [
        sum(p.numel() for p in module.parameters() if p.requires_grad)
        for module in model.executor.microcircuits
    ]
    if not per_micro or len(set(per_micro)) != 1:
        raise RuntimeError("M1.3 budget helper expects equal-size microcircuits")

    encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    router_query = sum(p.numel() for p in model.router.query.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    relation = sum(p.numel() for p in model.executor.relations.parameters() if p.requires_grad)
    shared_forward = encoder + router_query + head
    recruited = graph.recruited_count()
    routing_total = graph.routing_keys.numel()
    routing_recruited = recruited * cfg.router_dim

    return SynapLoomBudget(
        total_trainable_parameters=count_trainable_parameters(model),
        shared_forward_parameters=shared_forward,
        total_microcircuit_parameters=sum(per_micro),
        per_microcircuit_parameters=per_micro[0],
        executed_microcircuit_parameters=cfg.top_k * per_micro[0],
        recruited_routing_key_parameters=routing_recruited,
        total_routing_key_parameters=routing_total,
        relation_parameters=relation,
        relation_parameters_executed=relation if cfg.message_rounds > 0 else 0,
        top_k=cfg.top_k,
        recruited_nodes=recruited,
        total_nodes=cfg.num_nodes,
    )


def _dense_parameter_count(*, input_dim: int, d_model: int, d_hidden: int, num_classes: int) -> int:
    encoder = 0 if input_dim == d_model else input_dim * d_model + d_model
    norm = 2 * d_model
    body = d_model * d_hidden + d_hidden + d_hidden * d_model + d_model
    head = d_model * num_classes + num_classes
    return encoder + norm + body + head


def dense_hidden_width_for_target(
    target_parameters: int,
    *,
    input_dim: int,
    d_model: int,
    num_classes: int,
    minimum: int = 1,
    maximum: int = 100_000,
) -> tuple[int, int]:
    """Return dense hidden width and its exact parameter count nearest to target."""
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid hidden-width search interval")

    slope = 2 * d_model + 1
    fixed = _dense_parameter_count(
        input_dim=input_dim, d_model=d_model, d_hidden=0, num_classes=num_classes
    )
    estimate = round((target_parameters - fixed) / slope)
    candidates = {
        max(minimum, min(maximum, estimate + offset))
        for offset in (-2, -1, 0, 1, 2)
    }
    best_width = min(
        candidates,
        key=lambda width: abs(
            _dense_parameter_count(
                input_dim=input_dim,
                d_model=d_model,
                d_hidden=width,
                num_classes=num_classes,
            )
            - target_parameters
        ),
    )
    count = _dense_parameter_count(
        input_dim=input_dim,
        d_model=d_model,
        d_hidden=best_width,
        num_classes=num_classes,
    )
    return best_width, count
