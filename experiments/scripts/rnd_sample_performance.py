"""Manual-only representative-shape synthetic profiling. No scientific data."""
import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import time
import torch
from synaploom.research.sample_performance import measure
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.training.state_digest import model_state_sha256


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shape = {k: getattr(args, k) for k in ('batch', 'width', 'hidden', 'nodes', 'recruited', 'top_k', 'classes')}
    timings = []
    for repeat in range(args.repeats):
        ordinary = measure(device=args.device, threads=args.threads, steps=args.steps, **shape)
        profiled = measure(device=args.device, threads=args.threads, steps=args.steps,
                           instrumented=True, trace_path=args.output / f'trace-{repeat}.json', **shape)
        if ordinary['final_state_sha256'] != profiled['final_state_sha256']:
            raise RuntimeError('instrumentation state parity failed')
        (args.output / f'operators-{repeat}.txt').write_text(profiled.pop('table'))
        ordinary.pop('table')
        timings.append(dict(ordinary=ordinary, profiled=profiled,
            instrumentation_extra_wall_seconds=profiled['wall_seconds']-ordinary['wall_seconds']))
    torch.set_num_threads(args.threads)
    torch.manual_seed(7)
    model = SynapLoomModel(ModelConfig(input_dim=args.width, num_classes=args.classes,
        graph=GraphConfig(num_nodes=args.nodes, initial_recruited=args.recruited,
        d_model=args.width, d_hidden=args.hidden, router_dim=64, top_k=args.top_k, q_max=0))).to(args.device).eval()
    reference = SampleRoutingReference(model)
    x = torch.randn(args.batch, args.width, device=args.device)
    def sync():
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
    costs = {}
    with torch.inference_mode():
        for name, fn in [('batch', lambda: model(x)), ('sample', lambda: reference(x)),
                         ('transfer_to_cpu', lambda: x.cpu()), ('idle_sync', sync)]:
            for _ in range(2): fn()
            values, events = [], []
            for _ in range(args.repeats):
                sync()
                start = time.perf_counter()
                if args.device.startswith('cuda'):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                fn()
                if args.device.startswith('cuda'): b.record()
                sync()
                values.append(time.perf_counter()-start)
                if args.device.startswith('cuda'): events.append(a.elapsed_time(b))
            costs[name] = dict(wall_seconds=values, cuda_stream_elapsed_ms=events)
    metadata = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
        cuda=torch.version.cuda, device=args.device, affinity=sorted(os.sched_getaffinity(0)),
        threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
        env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')},
        gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None,
        parameters=sum(p.numel() for p in model.parameters()), shape=shape,
        per_example_expert_calls=args.top_k, batch_expert_row_evaluations=args.batch*args.top_k,
        state_sha256=model_state_sha256(model))
    (args.output/'results.json').write_text(json.dumps(dict(metadata=metadata, timings=timings, costs=costs), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    for name, default in dict(threads=1, steps=5, repeats=5, batch=128, width=256,
                              hidden=512, nodes=16, recruited=4, top_k=4, classes=100).items():
        p.add_argument('--'+name, type=int, default=default)
    args = p.parse_args()
    if min(args.steps, args.repeats, args.batch, args.threads) < 1: p.error('positive counts required')
    run(args)
