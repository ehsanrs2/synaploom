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


def estimate_synaploom_linear_macs_per_batch(
    model: SynapLoomModel,
    *,
    batch_size: int,
) -> int:
    """Estimate linear-layer MACs for M1.3 (message_rounds=0).

    Excludes LayerNorm/GELU/softmax and data movement. Router key scoring is
    batch-level, so its candidate cost is paid once per batch.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cfg = model.graph_state.config
    if cfg.message_rounds != 0:
        raise ValueError("M1.3 MAC estimator currently assumes message_rounds=0")

    input_dim = model.config.input_dim
    d = cfg.d_model
    h = cfg.d_hidden
    r = cfg.router_dim
    classes = model.config.num_classes
    recruited = model.graph_state.recruited_count()

    encoder = 0 if input_dim == d else batch_size * input_dim * d
    query = batch_size * d * r
    router_scores = recruited * r
    experts = batch_size * cfg.top_k * (d * h + h * d)
    head = batch_size * d * classes
    return int(encoder + query + router_scores + experts + head)


def estimate_dense_linear_macs_per_batch(
    *,
    input_dim: int,
    d_model: int,
    d_hidden: int,
    num_classes: int,
    batch_size: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoder = 0 if input_dim == d_model else batch_size * input_dim * d_model
    body = batch_size * (d_model * d_hidden + d_hidden * d_model)
    head = batch_size * d_model * num_classes
    return int(encoder + body + head)
