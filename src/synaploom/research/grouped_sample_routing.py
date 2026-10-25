"""Grouped H1a sample-routing experiment; never installed in production.

Shares all parameters with the row-wise reference. One expert call per selected
expert union; only selected rows enter each expert. Unique slot restoration
avoids atomic scatter-add reductions and retains top-k reduction order.
"""
import math

import torch

from synaploom.research.sample_routing import SampleOutput, SampleRoutingReference


class GroupedSampleRouting(SampleRoutingReference):
    def forward(self, x):
        self._validate()
        if x.ndim != 2 or len(x) == 0 or x.shape[1] != self.model.config.input_dim:
            raise ValueError("expected nonempty [batch,input_dim]")
        model = self.model
        recruited = model.graph_state.recruited_ids()  # ascending node IDs
        if not len(recruited):
            raise ValueError("no recruited experts")
        z = model.encoder(x)
        queries = model.router.query(z)
        scores = queries @ model.graph_state.routing_keys[recruited].T
        scores = scores / math.sqrt(model.config.graph.router_dim)
        if not torch.isfinite(scores).all():
            raise ValueError("nonfinite routing scores")
        k = min(model.config.graph.top_k, len(recruited))
        local = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
        selected = recruited[local]
        weights = scores.gather(1, local).softmax(dim=1)

        # Group B*k unique assignment slots by expert, on the device. Only the
        # small dispatch plan is copied to the host, once per forward. Dynamic
        # dispatch still has synchronization costs; no CUDA-graph claim is made.
        flat = selected.flatten()
        order = torch.argsort(flat, stable=True)
        experts, counts = torch.unique_consecutive(flat[order], return_counts=True)
        plan = torch.stack((experts, counts), dim=1).detach().cpu().tolist()
        groups = order.split([count for _, count in plan])
        values = []
        for (expert, _), slots in zip(plan, groups):
            rows = torch.div(slots, k, rounding_mode="floor")
            values.append(model.executor.microcircuits[expert](z.index_select(0, rows)))
        # Every slot occurs exactly once. Restore row/top-k order before summing.
        restored = torch.cat(values).index_select(0, torch.argsort(order))
        states = restored.reshape(len(x), k, z.shape[1])
        representation = (weights.unsqueeze(-1) * states).sum(dim=1)
        return SampleOutput(model.head(representation), selected, weights, queries)
