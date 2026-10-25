"""Bounded CPU qualification; no training or scientific accuracy experiment."""
import argparse
import json
import time
from pathlib import Path
import torch
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.training.state_digest import model_state_sha256


def qualify():
    torch.set_num_threads(1)
    rows = []
    with torch.inference_mode():
        for seed in (7, 19):
            for recruited in (2, 6):
                torch.manual_seed(seed)
                model = SynapLoomModel(ModelConfig(input_dim=8, num_classes=3, graph=GraphConfig(
                    q_max=0, num_nodes=6, initial_recruited=recruited, top_k=2,
                    d_model=8, d_hidden=16, router_dim=8))).eval()
                ref = SampleRoutingReference(model)
                x = torch.randn(24, 8)
                before = model_state_sha256(model)
                full = ref(x)
                oracle = torch.cat([model(row[None]).logits for row in x])
                torch.testing.assert_close(full.logits, oracle, atol=1e-6, rtol=1e-6)
                delta = 0.
                for i in range(24):
                    for companions in (x[:0], x, x.flip(0)*100):
                        out = ref(torch.cat([x[i:i+1], companions]))
                        delta = max(delta, float((out.logits[0]-full.logits[i]).abs().max()))
                        torch.testing.assert_close(out.logits[0], full.logits[i], atol=1e-6, rtol=1e-6)
                        torch.testing.assert_close(out.weights[0], full.weights[i], atol=1e-6, rtol=1e-6)
                        assert out.logits[0].argmax() == full.logits[i].argmax()
                costs = {}
                for name, fn in [('batch', model), ('sample', ref)]:
                    for _ in range(2): fn(x)
                    values = []
                    for _ in range(5):
                        start = time.perf_counter(); fn(x); values.append(time.perf_counter()-start)
                    costs[name] = values
                assert before == model_state_sha256(model)
                rows.append(dict(seed=seed, recruited=recruited, state_sha256=before,
                    parameters=sum(p.numel() for p in model.parameters()),
                    top_k=2, expert_row_evaluations=48, sample_expert_calls=48,
                    batch_expert_calls=2, comparisons=72, max_logit_delta=delta,
                    singleton_max_delta=float((full.logits-oracle).abs().max()), wall_seconds=costs))
    return dict(label='UNTRAINED_SYNTHETIC_CPU_ONLY', torch=torch.__version__, cells=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'results.json').write_text(json.dumps(qualify(), indent=2))
