"""Isolated forward/guard/hash profiling; synthetic, explicit device, no training."""
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.research.grouped_sample_routing import GroupedSampleRouting
from synaploom.training.numerics import (
    check_model_gradients, check_model_parameters, ensure_finite_tensor,
)
from synaploom.training.state_digest import model_state_sha256


def trace_summary(path):
    events = json.loads(Path(path).read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    runtime = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver')]
    launches = [e for e in runtime if 'launch' in e.get('name', '').lower()]
    syncs = [e for e in runtime if 'synchroniz' in e.get('name', '').lower()]
    copies = [e for e in runtime if 'memcpy' in e.get('name', '').lower()]
    return dict(kernel_count=len(kernels), kernel_duration_us=sum(e.get('dur', 0) for e in kernels),
        launch_api_count=len(launches), launch_api_duration_us=sum(e.get('dur', 0) for e in launches),
        synchronization_api_count=len(syncs), synchronization_api_duration_us=sum(e.get('dur', 0) for e in syncs),
        copy_api_count=len(copies), copy_api_duration_us=sum(e.get('dur', 0) for e in copies))


def timed(fn, device, repeats, trace_path):
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()
    for _ in range(2):
        fn()
    sync()
    walls, cpus, events = [], [], []
    for _ in range(repeats):
        sync()
        start, cpu = time.perf_counter(), time.process_time()
        fn()
        sync()
        walls.append(time.perf_counter()-start)
        cpus.append(time.process_time()-cpu)
    # Events and profiler are separate from ordinary wall measurements.
    if device == 'cuda':
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        events.append(a.elapsed_time(b))
    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    start = time.perf_counter()
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function(trace_path.stem):
            fn()
        sync()
    prof.export_chrome_trace(str(trace_path))
    instrumented = time.perf_counter()-start
    table_path = trace_path.with_suffix('.txt')
    table_path.write_text('\n'.join(line.rstrip() for line in
        prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=20).splitlines())+'\n')
    return dict(wall_seconds=walls, process_cpu_seconds=cpus, median_wall_seconds=statistics.median(walls),
        cuda_stream_elapsed_ms=events, instrumented_wall_with_export_seconds=instrumented,
        trace=trace_summary(trace_path))


def work(model, fn, x):
    calls = []
    handles = [expert.register_forward_hook(lambda m, a, o, i=i: calls.append((i, len(a[0]))))
               for i, expert in enumerate(model.executor.microcircuits)]
    try:
        fn(x)
    finally:
        for handle in handles:
            handle.remove()
    return dict(executed_expert_count=len({i for i, _ in calls}), expert_calls=len(calls),
                effective_expert_rows=sum(n for _, n in calls), calls=calls)


def cell(output, *, device='cpu', batch=4, recruited=4, k=2, width=8, hidden=16,
         nodes=6, router_dim=8, classes=3, repeats=3, seed=7):
    torch.manual_seed(seed)
    model = SynapLoomModel(ModelConfig(input_dim=width, num_classes=classes,
        graph=GraphConfig(q_max=0, num_nodes=nodes, initial_recruited=recruited,
            top_k=k, d_model=width, d_hidden=hidden, router_dim=router_dim))).to(device).eval()
    x = torch.randn(batch, width, generator=torch.Generator().manual_seed(seed+1)).to(device)
    old, new = SampleRoutingReference(model), GroupedSampleRouting(model)
    before = model_state_sha256(model)
    toler = 1e-6 if device == 'cpu' else 1e-5
    with torch.inference_mode():
        a, b = old(x), new(x)
        assert torch.equal(a.selected_node_ids, b.selected_node_ids)
        torch.testing.assert_close(a.logits, b.logits, atol=toler, rtol=toler)
        torch.testing.assert_close(a.weights, b.weights, atol=toler, rtol=toler)
        assert torch.equal(a.logits.argmax(-1), b.logits.argmax(-1))
        for i in range(batch):
            single = new(x[i:i+1])
            torch.testing.assert_close(single.logits[0], b.logits[i], atol=toler, rtol=toler)
            torch.testing.assert_close(single.weights[0], b.weights[i], atol=toler, rtol=toler)
        results = {}
        for name, fn in [('production_batch', model), ('row_reference', old), ('grouped', new)]:
            accounting = work(model, fn, x)
            assert accounting['effective_expert_rows'] == batch*k
            results[name] = dict(work=accounting, **timed(lambda: fn(x), device, repeats, output/f'{name}.json'))
            if device == 'cuda' and not results[name]['trace']['kernel_count']:
                raise RuntimeError('CUDA activity missing; rerun in a fresh CUDA profiling process')
            assert model_state_sha256(model) == before
    # Synthetic gradients solely for profiling scans; no optimizer or updates.
    new(x).logits.square().mean().backward()
    guards = {}
    probes = dict(finite_tensor=lambda: ensure_finite_tensor(a.logits, phase='rnd', tensor_name='logits'),
        gradient_diagnostics=lambda: check_model_gradients(model, phase='rnd'),
        parameter_diagnostics=lambda: check_model_parameters(model, phase='rnd'),
        model_state_sha256=lambda: model_state_sha256(model))
    for name, fn in probes.items():
        guards[name] = timed(fn, device, repeats, output/f'{name}.json')
        assert model_state_sha256(model) == before
    return dict(seed=seed, shape=dict(batch=batch, recruited=recruited, top_k=k, nodes=nodes,
        width=width, hidden=hidden, router_dim=router_dim, classes=classes), parameters=sum(p.numel() for p in model.parameters()),
        state_sha256=before, input_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
        parity=dict(logit_linf=float((a.logits-b.logits).abs().max()),
            weight_linf=float((a.weights-b.weights).abs().max()), selected_ids_equal=True,
            predictions_equal=True, singleton_allclose=True, state_unchanged=True),
        forward=results, independent_checks=guards)


def run(output, *, device='cpu', batches=(1, 4, 8), ratios=((2, 1), (4, 2), (6, 4)),
        width=8, hidden=16, nodes=6, router_dim=8, classes=3, repeats=3, threads=1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old_threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        cells = []
        for batch in batches:
            for recruited, k in ratios:
                path = output/f'b{batch}-r{recruited}-k{k}'
                path.mkdir()
                cells.append(cell(path, device=device, batch=batch, recruited=recruited,
                    k=k, width=width, hidden=hidden, nodes=nodes, router_dim=router_dim,
                    classes=classes, repeats=repeats))
        metadata = dict(label='SYNTHETIC_RND_NO_TRAINING', device=device, torch=torch.__version__,
            python=platform.python_version(), platform=platform.platform(), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
            threads=torch.get_num_threads(), interop=torch.get_num_interop_threads(),
            affinity=sorted(os.sched_getaffinity(0)), tf32=False,
            env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS')},
            residency='resident forwards; H2D setup excluded; D2H dispatch/guards/hash included',
            warmups=2, repeats=repeats)
        (output/'results.json').write_text(json.dumps(dict(metadata=metadata, cells=cells), indent=2)+'\n')
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
