from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphConfig:
    num_nodes: int = 16
    initial_recruited: int = 4
    d_model: int = 32
    d_hidden: int = 64
    router_dim: int = 32
    top_k: int = 2
    q_max: int = 4
    message_rounds: int = 0
    relation_types: int = 1
    consolidated_plasticity: float = 0.1
    utilization_decay: float = 0.99

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not 1 <= self.initial_recruited <= self.num_nodes:
            raise ValueError("initial_recruited must be in [1, num_nodes]")
        if not 1 <= self.top_k <= self.initial_recruited:
            raise ValueError("top_k must be in [1, initial_recruited] for M1.0")
        if self.d_model <= 0 or self.d_hidden <= 0 or self.router_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.q_max < 0:
            raise ValueError("q_max must be non-negative")
        if self.message_rounds < 0:
            raise ValueError("message_rounds must be non-negative")
        if self.relation_types <= 0:
            raise ValueError("relation_types must be positive")
        if not 0.0 < self.consolidated_plasticity <= 1.0:
            raise ValueError("consolidated_plasticity must be in (0, 1]")
        if not 0.0 <= self.utilization_decay < 1.0:
            raise ValueError("utilization_decay must be in [0, 1)")


@dataclass(frozen=True)
class ModelConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    input_dim: int = 32
    num_classes: int = 10

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
