from __future__ import annotations

import torch
from torch import Tensor, nn

from synaploom.config import GraphConfig
from synaploom.core.states import NodeState


class GraphState(nn.Module):
    """Persistent structural state for the fixed-capacity M1 supergraph."""

    def __init__(self, config: GraphConfig) -> None:
        super().__init__()
        self.config = config
        n, q = config.num_nodes, config.q_max

        node_state = torch.full((n,), int(NodeState.DORMANT), dtype=torch.int8)
        node_state[: config.initial_recruited] = int(NodeState.PLASTIC)
        self.register_buffer("node_state", node_state)
        self.register_buffer("node_age", torch.zeros(n, dtype=torch.long))
        self.register_buffer("node_utilization_ema", torch.zeros(n))
        self.register_buffer("node_importance_ema", torch.zeros(n))

        plasticity = torch.zeros(n)
        plasticity[: config.initial_recruited] = 1.0
        self.register_buffer("node_plasticity", plasticity)

        self.routing_keys = nn.Parameter(torch.randn(n, config.router_dim) * 0.02)

        self.register_buffer("edge_dst", torch.full((n, q), -1, dtype=torch.long))
        self.register_buffer("edge_enabled", torch.zeros((n, q), dtype=torch.bool))
        self.register_buffer("edge_relation", torch.zeros((n, q), dtype=torch.long))
        self.register_buffer("edge_utilization_ema", torch.zeros((n, q)))
        self.edge_gate = nn.Parameter(torch.zeros(n, q))

        self.register_buffer("structural_version", torch.zeros((), dtype=torch.long))

    @property
    def recruited_mask(self) -> Tensor:
        return self.node_state != int(NodeState.DORMANT)

    @property
    def dormant_mask(self) -> Tensor:
        return self.node_state == int(NodeState.DORMANT)

    def recruited_ids(self) -> Tensor:
        return torch.nonzero(self.recruited_mask, as_tuple=False).flatten()

    def recruited_count(self) -> int:
        return int(self.recruited_mask.sum().item())

    def active_edge_count(self) -> int:
        return int(self.edge_enabled.sum().item())

    @torch.no_grad()
    def recruit(self, node_id: int, routing_key: Tensor | None = None) -> None:
        self._check_node_id(node_id)
        if self.node_state[node_id].item() != int(NodeState.DORMANT):
            raise ValueError(f"node {node_id} is already recruited")
        self.node_state[node_id] = int(NodeState.PLASTIC)
        self.node_plasticity[node_id] = 1.0
        self.node_age[node_id] = 0
        if routing_key is not None:
            key = routing_key.to(device=self.routing_keys.device, dtype=self.routing_keys.dtype)
            if key.shape != (self.config.router_dim,):
                raise ValueError("routing_key has wrong shape")
            self.routing_keys[node_id].copy_(key)
        self.structural_version.add_(1)

    @torch.no_grad()
    def consolidate(self, node_ids: Tensor | list[int]) -> None:
        ids = torch.as_tensor(node_ids, device=self.node_state.device, dtype=torch.long).flatten()
        for node_id in ids.tolist():
            self._check_node_id(node_id)
            if self.node_state[node_id].item() == int(NodeState.DORMANT):
                raise ValueError("cannot consolidate a dormant node")
        self.node_state[ids] = int(NodeState.CONSOLIDATED)
        self.node_plasticity[ids] = self.config.consolidated_plasticity
        self.structural_version.add_(1)

    @torch.no_grad()
    def add_edge(self, src: int, dst: int, *, relation: int = 0, gate: float = 1.0) -> int:
        self._check_node_id(src)
        self._check_node_id(dst)
        if not 0 <= relation < self.config.relation_types:
            raise ValueError("relation id out of range")
        if src == dst:
            raise ValueError("self edges are disabled in M1.0")
        existing = self.edge_enabled[src] & (self.edge_dst[src] == dst)
        if bool(existing.any()):
            raise ValueError(f"edge {src}->{dst} already exists")
        free = torch.nonzero(~self.edge_enabled[src], as_tuple=False).flatten()
        if free.numel() == 0:
            raise RuntimeError(f"node {src} has no free edge slots")
        slot = int(free[0].item())
        self.edge_dst[src, slot] = dst
        self.edge_relation[src, slot] = relation
        self.edge_enabled[src, slot] = True
        self.edge_gate[src, slot] = float(gate)
        self.structural_version.add_(1)
        return slot

    @torch.no_grad()
    def remove_edge(self, src: int, slot: int) -> None:
        self._check_node_id(src)
        if not 0 <= slot < self.config.q_max:
            raise IndexError("edge slot out of range")
        if not bool(self.edge_enabled[src, slot]):
            raise ValueError("edge slot is already disabled")
        self.edge_enabled[src, slot] = False
        self.edge_dst[src, slot] = -1
        self.edge_relation[src, slot] = 0
        self.edge_utilization_ema[src, slot] = 0.0
        self.edge_gate[src, slot] = 0.0
        self.structural_version.add_(1)

    @torch.no_grad()
    def update_utilization(self, selected_node_ids: Tensor, routing_weights: Tensor) -> None:
        decay = self.config.utilization_decay
        self.node_utilization_ema.mul_(decay)
        ids = selected_node_ids.to(self.node_utilization_ema.device)
        weights = routing_weights.detach().to(self.node_utilization_ema.device)
        self.node_utilization_ema[ids] += (1.0 - decay) * weights
        self.node_age[self.recruited_mask] += 1

    def validate(self) -> None:
        if bool((self.edge_enabled & (self.edge_dst < 0)).any()):
            raise RuntimeError("enabled edge has invalid destination")
        if bool((self.edge_dst >= self.config.num_nodes).any()):
            raise RuntimeError("edge destination exceeds node capacity")
        if bool((self.node_plasticity[self.dormant_mask] != 0).any()):
            raise RuntimeError("dormant node has nonzero plasticity")

    def _check_node_id(self, node_id: int) -> None:
        if not 0 <= node_id < self.config.num_nodes:
            raise IndexError(f"node id {node_id} out of range")
