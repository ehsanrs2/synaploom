"""D2: bounded *actual* CPU Runner probe on generated two-task tensors only.

Research-only. Calls the existing run_continual_experiment, observes real
optimizer.step post-hooks and real evaluator forwards, hashes on-disk source
bytes and in-memory synthetic stream tensors, and refuses science data/seeds.
It is deliberately NOT an M1.3b science wrapper, full 8000-step run, or
cryptographic proof of checkout/training process integrity.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import torch
from torch import nn

from synaploom.data import streams as stream_module
from synaploom.data.streams import ContinualStream, TensorTask
from synaploom.research.fixed_sparse_baseline import (
    FixedSparseMLP, install_fixed_mask_guard,
)
from synaploom.training import continual as continual_module
from synaploom.training import evaluation as evaluation_module
from synaploom.training.continual import ContinualConfig, run_continual_experiment
from synaploom.training.state_digest import model_state_sha256


KIND = "M13B_D2_ACTUAL_TINY_CPU_RUNNER_INSTRUMENTATION_V1"
ENGINEERING_SEEDS = (13, 17)
TASK_COUNT = 2
EPOCHS_PER_TASK = 2
TRAIN_COUNT = 8
EVAL_COUNT = 7
TRAIN_BATCH_SIZE = 3
EVAL_BATCH_SIZE = 4
STEPS_PER_EPOCH = 3
EXPECTED_STEPS = 12


def cpu_source_file_sha256() -> dict[str, str]:
    """Hash runtime import locations, not a signed source revision.

    The import paths are fixed; no CLI or user-specified arbitrary file path
    exists. A caller-supplied independent expected map may be checked BEFORE
    running, but this test does not possess a trusted signing authority.
    """
    modules = {
        "continual": continual_module,
        "evaluation": evaluation_module,
        "batch_iterator": stream_module,
    }
    result = {}
    for label, mod in modules.items():
        location = Path(mod.__file__).resolve(strict=True)
        if location.suffix != ".py":
            raise ValueError("expected Python source, not compiled or unknown module")
        result[label] = sha256(location.read_bytes()).hexdigest()
    return result


def _stream_digest(stream: ContinualStream) -> str:
    h = sha256()
    for task in stream:
        h.update(json.dumps({
            "id":task.task_id,"classes":list(task.classes),
        },sort_keys=True,separators=(",",":")).encode("ascii"))
        for name in ("train_x","train_y","eval_x","eval_y"):
            t=getattr(task,name)
            if t.device.type != "cpu":
                raise ValueError("the tiny test permits CPU tensors only")
            material=t.detach().contiguous()
            h.update(json.dumps({
                "name":name,"dtype":str(material.dtype),
                "shape":list(material.shape),
            },sort_keys=True,separators=(",",":")).encode("ascii"))
            h.update(material.numpy().tobytes())
    return h.hexdigest()


def _stream() -> ContinualStream:
    tasks=[]
    for task_id in range(TASK_COUNT):
        classes=(task_id*2,task_id*2+1)
        train_y=torch.tensor([classes[i%2] for i in range(TRAIN_COUNT)],dtype=torch.long)
        eval_y=torch.tensor([classes[i%2] for i in range(EVAL_COUNT)],dtype=torch.long)
        train_x=torch.nn.functional.one_hot(train_y,num_classes=4).float()
        eval_x=torch.nn.functional.one_hot(eval_y,num_classes=4).float()
        # Synthetic stable ID channel; no protected sample/feature content.
        train_ids=torch.tensor(
            [[-(task_id*100+i+1)/1000] for i in range(TRAIN_COUNT)],dtype=torch.float32
        )
        eval_ids=torch.tensor(
            [[(task_id*100+i)/1000] for i in range(EVAL_COUNT)],dtype=torch.float32
        )
        tasks.append(TensorTask(
            task_id=task_id,classes=classes,
            train_x=torch.cat((train_x,train_ids),dim=1),train_y=train_y,
            eval_x=torch.cat((eval_x,eval_ids),dim=1),eval_y=eval_y,
        ))
    return ContinualStream(tasks)


def expected_evaluation_events() -> list[dict[str, Any]]:
    """Expected real forward groups and ACTUAL step-hook count at each eval."""
    events=[]
    for stage in range(TASK_COUNT):
        for epoch in range(EPOCHS_PER_TASK):
            completed=(stage*EPOCHS_PER_TASK+epoch+1)*STEPS_PER_EPOCH
            for ids in (
                list(range(stage*100,stage*100+4)),
                list(range(stage*100+4,stage*100+7)),
            ):
                events.append({"steps_observed":completed,"sample_ids":ids})
        completed=(stage+1)*EPOCHS_PER_TASK*STEPS_PER_EPOCH
        for evaluated_task in range(stage+1):
            for ids in (
                list(range(evaluated_task*100,evaluated_task*100+4)),
                list(range(evaluated_task*100+4,evaluated_task*100+7)),
            ):
                events.append({"steps_observed":completed,"sample_ids":ids})
    return events


def validate_events(events: list[dict[str, Any]]) -> None:
    """Reject missing/shuffled/duplicated/tail-dropped evaluation groups."""
    if events != expected_evaluation_events():
        raise ValueError("real evaluator forward membership or hook-step sequence differs")


def _check_small_result(payload: Mapping[str, Any]) -> None:
    """Require complete tiny runner output, without pretending to certify 10 tasks."""
    if type(payload.get("global_steps")) is not int or payload["global_steps"] != EXPECTED_STEPS:
        raise ValueError("runner global step count mismatch")
    config=payload.get("run_config")
    if not isinstance(config, Mapping) or config.get("device")!="cpu":
        raise ValueError("tiny runner must execute on CPU")
    matrix=payload.get("accuracy_matrix")
    losses=payload.get("eval_loss_matrix")
    if not isinstance(matrix,list) or not isinstance(losses,list) or len(matrix)!=2 or len(losses)!=2:
        raise ValueError("runner did not produce 2x2 evaluation matrices")
    for values in (matrix,losses):
        for stage,row in enumerate(values):
            if not isinstance(row,list) or len(row)!=2:
                raise ValueError("invalid result matrix")
            for task,v in enumerate(row):
                if task>stage:
                    if v is not None:
                        raise ValueError("future-task matrix cell not null")
                elif type(v) not in (int,float) or not math.isfinite(v) or v<0 or (
                    values is matrix and v>1
                ):
                    raise ValueError("nonfinite or out-of-range result")
    curves=payload.get("current_task_learning_curves")
    if not isinstance(curves,list) or len(curves)!=2:
        raise ValueError("incomplete learning curves")
    for stage,curve in enumerate(curves):
        if not isinstance(curve,list) or len(curve)!=2:
            raise ValueError("missing tiny epoch observation")
        for epoch,point in enumerate(curve):
            if type(point.get("epoch")) is not int or point["epoch"]!=epoch+1:
                raise ValueError("epoch identity mismatch")
            expected=(stage*EPOCHS_PER_TASK+epoch+1)*STEPS_PER_EPOCH
            if type(point.get("global_step")) is not int or point["global_step"]!=expected:
                raise ValueError("tiny epoch optimizer checkpoint mismatch")
            for key in ("accuracy","loss"):
                v=point.get(key)
                if type(v) not in (int,float) or not math.isfinite(v) or v<0 or (
                    key=="accuracy" and v>1
                ):
                    raise ValueError("nonfinite or invalid tiny curve point")
    if not isinstance(payload.get("summary"),Mapping):
        raise ValueError("missing result summary")
    summary=payload["summary"]
    final=sum(matrix[-1])/TASK_COUNT
    late=sum(sum(point["accuracy"] for point in curve)/2 for curve in curves)/2
    for key,computed in (
        ("final_average_accuracy",final),
        ("late_task_within_task_mean_accuracy",late),
    ):
        declared=summary.get(key)
        if type(declared) not in (float,int) or not math.isfinite(declared) or abs(declared-computed)>1e-5:
            raise ValueError("runner summary endpoint inconsistent")


@contextmanager
def _preserve_cpu_rng():
    """Keep toy investigation from contaminating the caller's CPU RNG state."""
    import numpy as np
    py_state=random.getstate()
    torch_state=torch.get_rng_state().clone()
    np_state=np.random.get_state()
    try:
        yield
    finally:
        random.setstate(py_state)
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)


def run_tiny_cpu_probe(
    *,
    seed: int = 13,
    masked: bool = False,
    expected_source_sha256: Mapping[str,str] | None = None,
    fault_after_step: int | None = None,
) -> dict[str, Any]:
    """Run 12 *real* optimizer steps on synthetic CPU tensors, never M1.3b.

    Fault injection is for negative tests and returns explicit failure evidence.
    An expected hash map is supplied by caller, not trusted authority; matching
    it does NOT prove a signed checkout or executable integrity.
    """
    if type(seed) is not int or seed not in ENGINEERING_SEEDS:
        raise ValueError("only non-science engineering seeds 13 or 17 permitted")
    if type(masked) is not bool:
        raise ValueError("masked must be a boolean")
    if fault_after_step is not None and (
        type(fault_after_step) is not int or not 1<=fault_after_step<=EXPECTED_STEPS
    ):
        raise ValueError("invalid synthetic fault-injection step")
    source_before=cpu_source_file_sha256()
    if expected_source_sha256 is not None:
        if (not isinstance(expected_source_sha256,Mapping)
                or dict(expected_source_sha256)!=source_before):
            raise ValueError("caller-pinned source content does not match on-disk imports")

    with _preserve_cpu_rng():
        torch.manual_seed(seed)
        stream=_stream()
        stream_digest_before=_stream_digest(stream)
        if masked:
            model:nn.Module=FixedSparseMLP(
                input_dim=5,d_model=4,d_hidden=8,num_classes=4,
                active_first=12,active_second=12,mask_seed=41,
            )
        else:
            model=nn.Linear(5,4)
        optimizer=torch.optim.SGD(
            model.parameters(),lr=.01,momentum=.9,weight_decay=.0005,
        )
        mask_guard=install_fixed_mask_guard(model,optimizer) if masked else None
        start_state=model_state_sha256(model)
        observed_steps=0
        mask_pass_count=0
        eval_events:list[dict[str,Any]]=[]
        def on_step(opt, _args, _kwargs):
            nonlocal observed_steps,mask_pass_count
            observed_steps+=1
            if masked:
                model.assert_masks(opt)
                mask_pass_count+=1
            if fault_after_step==observed_steps:
                raise RuntimeError("INJECTED_SYNTHETIC_CPU_FAULT")
        def on_forward(module, inputs):
            if not module.training:
                tensor=inputs[0]
                if tensor.device.type!="cpu":
                    raise RuntimeError("unexpected non-CPU evaluation forward")
                ids=[int(round(float(item)*1000)) for item in tensor[:,-1].tolist()]
                eval_events.append({"steps_observed":observed_steps,"sample_ids":ids})
        step_handle=optimizer.register_step_post_hook(on_step)
        forward_handle=model.register_forward_pre_hook(on_forward)
        try:
            config=ContinualConfig(
                seed=seed,device="cpu",epochs_per_task=EPOCHS_PER_TASK,
                batch_size=TRAIN_BATCH_SIZE,eval_batch_size=EVAL_BATCH_SIZE,
                evaluate_future_tasks=False,apply_final_task_boundary=False,
                restrict_logits_to_seen_classes=True,
                current_task_eval_every_epochs=1,
            )
            result=run_continual_experiment(
                model,stream,optimizer,config=config,engine=None,
                metadata={"kind":KIND,"synthetic":True,"scientific_claim":"NONE"},
            )
            data=result.to_dict()
            _check_small_result(data)
            validate_events(eval_events)
            if observed_steps!=EXPECTED_STEPS or result.global_steps!=observed_steps:
                raise ValueError("optimizer hook-count disagrees with runner")
            if masked and mask_pass_count!=EXPECTED_STEPS:
                raise ValueError("fixed mask guard failed during a real optimizer step")
            if masked:
                model.assert_masks(optimizer)
            if _stream_digest(stream)!=stream_digest_before:
                raise ValueError("input stream tensors changed during actual CPU training")
            if cpu_source_file_sha256()!=source_before:
                raise ValueError("source files modified during probe")
            final_state=model_state_sha256(model)
            if final_state==start_state:
                raise ValueError("model state stayed unchanged after real optimizer steps")
            return {
                "kind":KIND,"status":"ACTUAL_TINY_SYNTHETIC_CPU_RUNNER_PASS",
                "seed":seed,"synthetic_model":"masked_mlp" if masked else "linear",
                "real_cpu_optimizer_post_step_observations":observed_steps,
                "real_cpu_mask_post_step_checks":mask_pass_count,
                "expected_optimizer_steps":EXPECTED_STEPS,
                "observed_eval_forward_batches":len(eval_events),
                "observed_eval_membership_matches_synthetic_ids":True,
                "stream_bytes_stable":True,"stream_sha256":stream_digest_before,
                "source_bytes_stable":True,"runtime_source_file_sha256":source_before,
                "caller_pin_checked":expected_source_sha256 is not None,
                "source_attested_by_independent_signer":False,
                "model_state_changed":True,
                "full_10_task_8000_step_contract_qualified":False,
                "science_runner_integration":False,
                "actual_science_optimizer_steps":0,
                "protected_eval_accessed":False,"gpu_used":False,
                "science_executable":False,"scientific_claim":"NONE",
            }
        except Exception as exc:
            return {
                "kind":KIND,"status":"SYNTHETIC_CPU_RUNNER_FAILURE_RETAINED",
                "seed":seed,"synthetic_model":"masked_mlp" if masked else "linear",
                "error_type":type(exc).__name__,
                "error_detail":str(exc)[:200],
                "observed_cpu_post_step_calls_before_failure":observed_steps,
                "observed_eval_forward_batches_before_failure":len(eval_events),
                "source_attested_by_independent_signer":False,
                "full_10_task_8000_step_contract_qualified":False,
                "science_runner_integration":False,
                "actual_science_optimizer_steps":0,
                "protected_eval_accessed":False,"gpu_used":False,
                "science_executable":False,"scientific_claim":"NONE",
            }
        finally:
            forward_handle.remove()
            step_handle.remove()
            if mask_guard is not None:
                mask_guard.remove()
