"""Opt-in profiling of the unchanged TrainingEngine on synthetic data."""
from contextlib import ExitStack, contextmanager
import time
import sys
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.training.engine import TrainingEngine
from synaploom.training.state_digest import model_state_sha256
from synaploom.training.evaluation import evaluate_task
from synaploom.data.streams import TensorTask


@contextmanager
def regions(model, engine):
    # Wrappers delegate exactly once to the original function. No checks disabled.
    import synaploom.training.engine as module
    def wrapped(fn, name):
        def call(*args, **kwargs):
            with record_function(name):
                return fn(*args, **kwargs)
        return call
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys.modules[__name__], "model_state_sha256",
            wrapped(model_state_sha256, "model_state_sha256")))
        for name in ("ensure_finite_tensor", "check_model_gradients", "check_model_parameters",
                     "update_node_importance", "scale_node_owned_gradients"):
            stack.enter_context(patch.object(module, name, wrapped(getattr(module, name), name)))
        for obj, names in ((model, ("forward",)), (model.router, ("forward",)),
                           (model.executor, ("forward",)),
                           (engine.optimizer, ("step", "zero_grad"))):
            for name in names:
                stack.enter_context(patch.object(obj, name, wrapped(getattr(obj, name),
                                    type(obj).__name__ + "." + name)))
        yield


def measure(*, device="cpu", threads=1, steps=5, instrumented=False, trace_path=None):
    if steps < 1 or threads < 1:
        raise ValueError("positive steps and threads required")
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        torch.manual_seed(7)
        model = SynapLoomModel(ModelConfig(graph=GraphConfig(num_nodes=6,
            initial_recruited=6, d_model=8, d_hidden=16, router_dim=8,
            top_k=2), input_dim=8, num_classes=3)).to(device)
        engine = TrainingEngine(model, torch.optim.SGD(model.parameters(), lr=.001))
        x = torch.randn(24, 8, device=device)
        y = torch.arange(24, device=device) % 3
        task = TensorTask(task_id=0, classes=(0, 1, 2), train_x=x, train_y=y, eval_x=x, eval_y=y)
        def synchronize():
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
        def iteration(i):
            engine.train_step(x, y, step=i)
            before = model_state_sha256(model)
            with record_function("evaluation_with_hash_guards") if instrumented else ExitStack():
                evaluate_task(model, task, batch_size=8, device=device)
            if model_state_sha256(model) != before:
                raise RuntimeError("evaluation mutated state")
        for i in range(2):
            iteration(i)
        synchronize()
        start, cpu = time.perf_counter(), time.process_time()
        table = None
        if instrumented:
            activities = [ProfilerActivity.CPU]
            if str(device).startswith("cuda"):
                activities.append(ProfilerActivity.CUDA)
            with regions(model, engine), profile(activities=activities, record_shapes=True,
                                                  profile_memory=True) as prof:
                for i in range(steps):
                    iteration(i + 2)
                with record_function("explicit_device_synchronization"):
                    synchronize()
            table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
            if trace_path:
                prof.export_chrome_trace(str(trace_path))
        else:
            for i in range(steps):
                iteration(i + 2)
            synchronize()
        return {"label": "EXPLORATORY_SYNTHETIC", "instrumented": instrumented,
                "wall_seconds": time.perf_counter()-start,
                "process_cpu_seconds": time.process_time()-cpu,
                "steps": steps, "threads": threads,
                "interop_threads": torch.get_num_interop_threads(),
                "device": str(device), "torch": torch.__version__,
                "final_state_sha256": model_state_sha256(model), "table": table}
    finally:
        torch.set_num_threads(old_threads)
