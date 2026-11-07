"""M1.3b D4 manual full-SHAPE engineering cost probe, never science.

Reproduces the H1a2 10-task/20-epoch, 8,000-step TRAIN geometry and
evaluation *shape* using ONLY original TRAIN features. A 100/class
diagnostic evaluation subset is selected from the SAME 500/class training
examples. Overlap is intentional and MUST NEVER be interpreted as accuracy.

Only a cost record (time, optimizer steps, model/cost provenance, CUDA memory)
may leave the worker. The existing trusted-file loader deserializes the
entire pinned bundle (which contains eval tensors), but the worker receives
ONLY training tensors and never uses original protected evaluation examples
for calibration. A separately authorized owner CLI invocation is needed.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import time
from typing import Any

import torch

from synaploom.data.streams import ContinualStream, TensorTask
from synaploom.protocols.h1a import H1A_CLASS_ORDER, build_h1a_variant
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256, h1a2_protocol
from synaploom.research.fixed_sparse_baseline import (
    build_h1a_fixed_sparse_feasibility, install_fixed_mask_guard,
)
from synaploom.training.continual import run_continual_experiment
from synaploom.training.state_digest import model_state_sha256


KIND = "M13B_D4_TRAIN_ONLY_FULLSHAPE_COST_PROBE_V1"
ARMS = ("progressive_reserve", "fixed_sparse_mlp", "all_recruited")
ENGINEERING_SEED = 13
EVAL_SUBSET_SEED = 41
MASK_SEED = 17
EXCLUDED_PROPOSED_SCIENCE_SEEDS = frozenset(range(501, 511))
TASKS = 10
CLASSES_PER_TASK = 10
TRAIN_PER_CLASS = 500
DIAGNOSTIC_EVAL_PER_CLASS = 100
EPOCHS_PER_TASK = 20
TRAIN_BATCH = 128
EVAL_BATCH = 256
STEPS_PER_EPOCH = 40
EXPECTED_STEPS = TASKS * EPOCHS_PER_TASK * STEPS_PER_EPOCH
EXPECTED_EVAL_CALLS = TASKS * EPOCHS_PER_TASK + TASKS * (TASKS+1)//2
EXPECTED_EVAL_EXAMPLES_PROCESSED = EXPECTED_EVAL_CALLS * CLASSES_PER_TASK * DIAGNOSTIC_EVAL_PER_CLASS


def describe_fullshape_cost_probe() -> dict[str, Any]:
    """Read-only plan, cheap CPU computation: not run authorization."""
    proto = h1a2_protocol()
    if not (proto.feature_dim==512 and proto.num_classes==100
            and proto.classes_per_task==CLASSES_PER_TASK
            and proto.epochs_per_task==EPOCHS_PER_TASK
            and proto.batch_size==TRAIN_BATCH
            and proto.eval_batch_size==EVAL_BATCH
            and proto.current_task_eval_every_epochs==1
            and proto.learning_rate==0.01):
        raise RuntimeError("H1a2 full-workload protocol no longer matches this cost probe")
    return {
        "schema": 1,
        "kind": KIND,
        "status": "MANUAL_COST_PROBE_PLAN_ONLY__OWNER_AUTHORIZATION_MISSING",
        "science_executable": False,
        "scientific_claim": "NONE",
        "owner_gpu_authorization_recorded": False,
        "manual_probe_available_but_not_authorized_by_this_plan": True,
        "arm_options_one_per_invocation": list(ARMS),
        "engineering_seed": ENGINEERING_SEED,
        "excluded_proposed_science_seeds": sorted(EXCLUDED_PROPOSED_SCIENCE_SEEDS),
        "feature_sha256_required": H1A2_FEATURE_SHA256,
        "protected_eval_partition_used": False,
        "original_training_partition_only": True,
        "training_diagnostic_eval_overlap": True,
        "diagnostic_accuracy_is_unusable": True,
        "accuracy_values_reported": False,
        "tasks": TASKS,
        "epochs_per_task": EPOCHS_PER_TASK,
        "classes_per_task": CLASSES_PER_TASK,
        "train_examples_per_task": CLASSES_PER_TASK*TRAIN_PER_CLASS,
        "diagnostic_eval_examples_per_task": CLASSES_PER_TASK*DIAGNOSTIC_EVAL_PER_CLASS,
        "train_batch_size": TRAIN_BATCH,
        "eval_batch_size": EVAL_BATCH,
        "train_batches_per_epoch": STEPS_PER_EPOCH,
        "last_training_batch_size": 8,
        "last_diagnostic_eval_batch_size": 232,
        "optimizer_steps_per_complete_trajectory": EXPECTED_STEPS,
        "current_task_epoch_eval_calls": TASKS*EPOCHS_PER_TASK,
        "task_stage_eval_calls": TASKS*(TASKS+1)//2,
        "total_task_eval_calls": EXPECTED_EVAL_CALLS,
        "diagnostic_eval_batches": EXPECTED_EVAL_CALLS*4,
        "total_diagnostic_eval_examples_processed": EXPECTED_EVAL_EXAMPLES_PROCESSED,
        "real_science_equivalent_accuracy_estimate": False,
        "real_science_equivalent_runtime_guarantee": False,
        "strict_parent_subprocess_wall_timeout_required": True,
        "possible_early_timeout": True,
        "no_automatic_gpu_dispatch": True,
        "no_confirmatory_runner_or_seed_use": True,
        "full_science_cost_unknown_until_owner_manual_engineering_probe": True,
    }


def make_fullshape_trainonly_stream(
    features: torch.Tensor, labels: torch.Tensor,
) -> tuple[ContinualStream, str]:
    """Full train shape, diagnostic eval same shape but derived from training.

    Diagnostic subset 100/class is deliberately a SUBSET of training: never
    a validation/test holdout. The original protected evaluation partition
    must not be supplied to this function. Hash selected source indices.
    """
    if (not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor)
            or features.shape != (50_000,512) or labels.shape != (50_000,)
            or not features.is_floating_point() or labels.is_floating_point()
            or not bool(torch.isfinite(features).all())):
        raise ValueError("requires original 50000x512 finite train features and integer targets")
    labels = labels.long()
    if (int(labels.min()) != 0 or int(labels.max()) != 99
            or not torch.equal(
                torch.bincount(labels, minlength=100),
                torch.full((100,), 500, dtype=torch.long)
            )):
        raise ValueError("requires exactly 500 original training samples per class")
    track = sha256(b"synaploom/m13b/d4/training-only-in-sample-eval-subset/v1\0")
    per_class: dict[int,tuple[list[int],list[int]]] = {}
    for class_id in H1A_CLASS_ORDER:
        ids = torch.nonzero(labels==class_id,as_tuple=False).flatten()
        if ids.numel()!=TRAIN_PER_CLASS:
            raise RuntimeError("bad class inventory")
        generator=torch.Generator(device="cpu")
        generator.manual_seed(EVAL_SUBSET_SEED+int(class_id))
        selected = ids[torch.randperm(TRAIN_PER_CLASS,generator=generator)[:DIAGNOSTIC_EVAL_PER_CLASS]]
        all_ids = ids.tolist()
        diagnostic_ids = selected.tolist()
        if (len(set(all_ids))!=TRAIN_PER_CLASS
                or len(set(diagnostic_ids))!=DIAGNOSTIC_EVAL_PER_CLASS
                or not set(diagnostic_ids).issubset(all_ids)):
            raise RuntimeError("bad diagnostic sample identity")
        per_class[class_id]=(all_ids,diagnostic_ids)
        track.update(int(class_id).to_bytes(2,"big"))
        for prefix,items in ((b"train-all",all_ids),(b"diagnostic-in-sample",diagnostic_ids)):
            track.update(prefix+b"\0")
            for idx in items:
                track.update(int(idx).to_bytes(4,"big"))
    tasks: list[TensorTask]=[]
    for t in range(TASKS):
        classes=tuple(H1A_CLASS_ORDER[t*CLASSES_PER_TASK:(t+1)*CLASSES_PER_TASK])
        tr=torch.tensor([idx for cls in classes for idx in per_class[cls][0]],dtype=torch.long)
        ev=torch.tensor([idx for cls in classes for idx in per_class[cls][1]],dtype=torch.long)
        tasks.append(TensorTask(
            task_id=t,classes=classes,
            train_x=features.index_select(0,tr).float().contiguous(),
            train_y=labels.index_select(0,tr),
            eval_x=features.index_select(0,ev).float().contiguous(),
            eval_y=labels.index_select(0,ev),
        ))
    return ContinualStream(tasks),track.hexdigest()


def execute_one_manual_cost_probe(
    stream: ContinualStream, *, arm: str,
    verified_feature_sha256: str, train_subset_sha256: str,
) -> dict[str,Any]:
    """GPU worker, called ONLY after explicit external manual owner gate.

    Does not save accuracy, predictions, model weights or scientific metrics.
    No in-process hard wall limit; the CLI MUST supervise this in a killed-on-
    timeout child process and retain incomplete/failure evidence.
    """
    describe_fullshape_cost_probe()
    if arm not in ARMS:
        raise ValueError("unknown primary engineering arm")
    if (verified_feature_sha256!=H1A2_FEATURE_SHA256
            or type(train_subset_sha256) is not str
            or len(train_subset_sha256)!=64
            or any(c not in "0123456789abcdef" for c in train_subset_sha256)):
        raise ValueError("untrusted or incomplete feature/subset digest")
    if ENGINEERING_SEED in EXCLUDED_PROPOSED_SCIENCE_SEEDS:
        raise RuntimeError("reserved confirmatory seed in engineering probe")
    if not isinstance(stream,ContinualStream) or len(stream)!=TASKS:
        raise ValueError("full task stream required")
    for stage,task in enumerate(stream):
        if (task.task_id!=stage or task.classes!=tuple(H1A_CLASS_ORDER[stage*10:(stage+1)*10])
                or task.train_x.shape!=(5000,512) or task.eval_x.shape!=(1000,512)
                or task.train_y.shape!=(5000,) or task.eval_y.shape!=(1000,)):
            raise ValueError("cost-only stream does not match full benchmark shapes")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no cost probe")
    proto=h1a2_protocol()
    if arm=="fixed_sparse_mlp":
        model,budget=build_h1a_fixed_sparse_feasibility(proto,seed=ENGINEERING_SEED,mask_seed=MASK_SEED)
        model.to("cuda:0")
        optimizer=torch.optim.SGD(model.parameters(),lr=proto.learning_rate,
                                  momentum=proto.momentum,weight_decay=proto.weight_decay)
        mask_guard=install_fixed_mask_guard(model,optimizer)
        engine=None
    else:
        model,optimizer,engine,_=build_h1a_variant(
            arm,stream=stream,seed=ENGINEERING_SEED,protocol=proto
        )
        model.to("cuda:0")
        mask_guard=None
        budget=None
    try:
        device=torch.device("cuda:0")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        result=run_continual_experiment(
            model,stream,optimizer,config=proto.continual_config(
                seed=ENGINEERING_SEED,device="cuda:0"
            ),engine=engine,metadata={
                "kind":KIND,
                "scientific_claim":"NONE",
                "original_training_partition_only":True,
                "diagnostic_eval_in_sample":True,
            }
        )
        torch.cuda.synchronize(device)
        wall=time.perf_counter()-started
        if (result.global_steps!=EXPECTED_STEPS
                or len(result.train_seconds_by_task)!=TASKS
                or len(result.eval_seconds_by_stage)!=TASKS
                or len(result.current_task_learning_curves)!=TASKS
                or any(len(x)!=EPOCHS_PER_TASK for x in result.current_task_learning_curves)):
            raise RuntimeError("incomplete or unbounded engineering probe")
        durations=[wall,*result.train_seconds_by_task,*result.eval_seconds_by_stage]
        if wall<=0 or any(not math.isfinite(x) or x<0 for x in durations):
            raise RuntimeError("invalid GPU cost timing")
        if budget is not None:
            model.assert_masks(optimizer)
            if model.mask_digest()!=budget.mask_sha256:
                raise RuntimeError("fixed mask changed during cost probe")
        return {
            "kind":KIND,"schema":1,
            "status":"ENGINEERING_FULLSHAPE_COST_ONLY__NOT_SCIENTIFIC_RESULT",
            "scientific_claim":"NONE",
            "protected_eval_partition_used":False,
            "diagnostic_eval_in_sample":True,
            "diagnostic_accuracy_disclosed":False,
            "science_executable":False,
            "arm":arm,
            "engineering_seed":ENGINEERING_SEED,
            "original_train_feature_sha256":verified_feature_sha256,
            "in_sample_diagnostic_selection_sha256":train_subset_sha256,
            "gpu_name":torch.cuda.get_device_name(device),
            "torch_version":torch.__version__,
            "cuda_runtime":torch.version.cuda,
            "optimizer_steps":result.global_steps,
            "train_seconds_by_task":list(result.train_seconds_by_task),
            "diagnostic_eval_seconds_by_stage":list(result.eval_seconds_by_stage),
            "wall_seconds":wall,
            "peak_allocated_bytes":int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)),
            "trainable_parameter_elements":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "final_model_state_sha256":model_state_sha256(model),
            "fixed_sparse_mask_sha256":budget.mask_sha256 if budget is not None else None,
            "accuracy_and_forgetting_values":None,
            "full_original_protected_eval_measured":False,
            "full_science_gpu_hours_estimated_automatically":None,
        }
    finally:
        if mask_guard is not None:
            mask_guard.remove()
