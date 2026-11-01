"""M1.3b train-only cost calibration: opt-in engineering, never confirmation.

This is a nonconfirmatory *one arm per invocation* RTX integration measurement.
The only data used by the continual harness is selected from the ORIGINAL
FROZEN TRAIN partition. Model accuracy scores are neither emitted nor used
to select seeds, configurations, arms or protocol. It is not a full science
runner, not a statistical comparison and not an estimate of actual GPU hours
without additional host evidence and workload review.
"""
from __future__ import annotations

from dataclasses import replace
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
from synaploom.research.m13b_synthetic_pilot import ARMS
from synaploom.training.continual import run_continual_experiment
from synaploom.training.state_digest import model_state_sha256


KIND = "M13B_TRAIN_ONLY_COST_CALIBRATION_V1"
SEED = 13
SPLIT_SEED = 41
MASK_SEED = 17
TRAIN_PER_CLASS = 496
HOLDOUT_PER_CLASS = 4
CLASS_COUNT_PER_TASK = 10
TASK_OPTIONS = (2, 10)
BATCH_SIZE = 128
EPOCHS = 1
FEATURE_SHA256 = H1A2_FEATURE_SHA256
MAX_STEPS = 390
CONFIRMATION_SEEDS = frozenset(range(501, 511))


def make_training_partition_calibration_stream(
    features: torch.Tensor, labels: torch.Tensor, *, tasks: int,
) -> tuple[ContinualStream, str]:
    """Exclude 4/class of *training* examples as diagnostics, never eval data.

    For 496 train examples/class, B128 yields 39 optimizer steps per task.
    This is 97.5% of the true 40-step-per-epoch science workload, but the
    40 train-derived eval examples/task are NOT the science 1000 eval examples.
    """
    if type(tasks) is not int or tasks not in TASK_OPTIONS:
        raise ValueError("only two or ten engineering tasks allowed")
    if (features.ndim != 2 or features.shape != (50_000, 512)
            or labels.shape != (50_000,)
            or not torch.isfinite(features).all()):
        raise ValueError("requires full frozen train-only 50k×512 features")
    if labels.is_floating_point() or not bool(torch.equal(
        torch.bincount(labels.long(), minlength=100),
        torch.full((100,), 500, dtype=torch.long),
    )):
        raise ValueError("requires frozen 500 examples per class train partition")

    track = sha256(b"synaploom/m13b/training-only-cost-split/v1\0")
    task_records = []
    for task_index in range(tasks):
        ids_train = []
        ids_holdout = []
        classes = tuple(H1A_CLASS_ORDER[
            task_index * CLASS_COUNT_PER_TASK:(task_index + 1) * CLASS_COUNT_PER_TASK
        ])
        for class_id in classes:
            eligible = (labels == class_id).nonzero(as_tuple=False).flatten().cpu()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(SPLIT_SEED + int(class_id))
            permutation = eligible[
                torch.randperm(int(eligible.numel()), generator=generator)
            ]
            train = permutation[:TRAIN_PER_CLASS].tolist()
            hold = permutation[TRAIN_PER_CLASS:TRAIN_PER_CLASS + HOLDOUT_PER_CLASS].tolist()
            if len(train) != TRAIN_PER_CLASS or len(hold) != HOLDOUT_PER_CLASS:
                raise ValueError("malformed original training class partition")
            if set(train).intersection(hold):
                raise ValueError("calibration train and holdout overlap")
            ids_train.extend(train)
            ids_holdout.extend(hold)
            track.update(int(class_id).to_bytes(2, "big"))
            for label, indices in ((b"train", train), (b"heldout-train", hold)):
                track.update(label + b"\0")
                for index in indices:
                    track.update(int(index).to_bytes(4, "big"))
        ti = torch.tensor(ids_train, dtype=torch.long)
        hi = torch.tensor(ids_holdout, dtype=torch.long)
        task_records.append(TensorTask(
            task_id=task_index,
            classes=classes,
            train_x=features.index_select(0, ti).float().contiguous(),
            train_y=labels.index_select(0, ti).long(),
            eval_x=features.index_select(0, hi).float().contiguous(),
            eval_y=labels.index_select(0, hi).long(),
        ))
    return ContinualStream(task_records), track.hexdigest()


def describe_cost_calibration() -> dict[str, Any]:
    return {
        "schema": 1,
        "kind": KIND,
        "status": "MANUAL_ENGINEERING_CALIBRATION_NOT_SCIENCE",
        "scientific_claim": "NONE",
        "science_executable": False,
        "protected_eval_partition_used": False,
        "feature_sha256_expected": FEATURE_SHA256,
        "data_scope": "TRUSTED_FROZEN_CIFAR_TRAIN_ONLY",
        "device": "cuda:0",
        "engineering_seed": SEED,
        "split_seed": SPLIT_SEED,
        "mask_seed": MASK_SEED,
        "candidate_arms": list(ARMS),
        "one_arm_per_manual_invocation": True,
        "task_options": list(TASK_OPTIONS),
        "training_examples_per_class": TRAIN_PER_CLASS,
        "train_only_holdout_examples_per_class": HOLDOUT_PER_CLASS,
        "optimizer_steps_per_task": 39,
        "optimizer_steps_at_2_tasks": 78,
        "optimizer_steps_at_10_tasks": 390,
        "maximum_optimizer_steps_per_invocation": MAX_STEPS,
        "full_science_optimizer_steps_per_trajectory": 8000,
        "full_science_evaluation_workload_reproduced": False,
        "full_science_gpu_time_estimate_produced": False,
        "confirmation_seeds_used": False,
        "not_a_strict_walltime_timeout": True,
    }


def run_one_arm_trainonly_cost_calibration(
    stream: ContinualStream,
    *,
    arm: str, tasks: int, frozen_sha256: str, split_sha256: str,
) -> dict[str, Any]:
    """Manually run <=390 train steps for one engineering arm on CUDA.

    Uses the existing continual harness, including the existing per-task
    controller semantics for progressive reserve and protected numerical
    checks. Results report cost ONLY, never toy/actual accuracies.
    """
    if arm not in ARMS:
        raise ValueError("unregistered engineering arm")
    if type(tasks) is not int or tasks not in TASK_OPTIONS or len(stream) != tasks:
        raise ValueError("task count mismatch")
    if SEED in CONFIRMATION_SEEDS or SPLIT_SEED in CONFIRMATION_SEEDS:
        raise RuntimeError("protected confirmatory seed was selected")
    if frozen_sha256 != FEATURE_SHA256 or type(split_sha256) is not str or len(split_sha256) != 64:
        raise ValueError("trusted feature or train-only split digest invalid")
    if not torch.cuda.is_available():
        raise RuntimeError("manual engineering calibration requires CUDA")
    for task in stream:
        if (task.train_x.shape != (TRAIN_PER_CLASS * CLASS_COUNT_PER_TASK, 512)
                or task.eval_x.shape != (HOLDOUT_PER_CLASS * CLASS_COUNT_PER_TASK, 512)):
            raise ValueError("incorrect calibrated training/eval shape")

    proto = replace(h1a2_protocol(), epochs_per_task=EPOCHS, seeds=(SEED,))
    device = torch.device("cuda:0")
    guard = None
    if arm == "fixed_sparse_mlp":
        model, budget = build_h1a_fixed_sparse_feasibility(
            proto, seed=SEED, mask_seed=MASK_SEED
        )
        if not budget.matched_total_2pct or not budget.matched_ideal_macs_10pct:
            raise RuntimeError("fixed sparse candidate budget gates failed")
        model.to(device)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=proto.learning_rate,
            momentum=proto.momentum, weight_decay=proto.weight_decay
        )
        guard = install_fixed_mask_guard(model, optimizer)
        engine = None
    else:
        model, optimizer, engine, _ = build_h1a_variant(
            arm, stream=stream, seed=SEED, protocol=proto
        )
        model.to(device)

    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        result = run_continual_experiment(
            model, stream, optimizer,
            config=proto.continual_config(seed=SEED, device="cuda:0"),
            engine=engine,
            metadata={
                "kind": KIND,
                "scientific_claim": "NONE",
                "training_partition_only": True,
                "fixed_arm": arm,
            },
        )
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
        expected_steps = tasks * 39
        if result.global_steps != expected_steps or expected_steps > MAX_STEPS:
            raise RuntimeError("calibration executed unexpected number of optimizer steps")
        if not (len(result.train_seconds_by_task) == tasks
                and len(result.eval_seconds_by_stage) == tasks):
            raise RuntimeError("calibration failed to record per-task timing")
        times = [wall, *result.train_seconds_by_task, *result.eval_seconds_by_stage]
        if not all(math.isfinite(t) and t >= 0 for t in times) or wall <= 0:
            raise RuntimeError("nonfinite calibration runtime")
        if arm == "fixed_sparse_mlp":
            model.assert_masks(optimizer)
            if model.mask_digest() != budget.mask_sha256:
                raise RuntimeError("fixed sparse mask changed during calibration")
        return {
            "schema": 1,
            "kind": KIND,
            "status": "ENGINEERING_COST_OBSERVATION_ONLY",
            "scientific_claim": "NONE",
            "science_executable": False,
            "arm": arm,
            "task_count": tasks,
            "expected_and_observed_steps": expected_steps,
            "epochs_per_task": EPOCHS,
            "device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "frozen_feature_sha256": frozen_sha256,
            "train_only_split_sha256": split_sha256,
            "protected_eval_partition_used": False,
            "excluded_science_seeds": sorted(CONFIRMATION_SEEDS),
            "wall_seconds": wall,
            "training_seconds_by_task": list(result.train_seconds_by_task),
            "diagnostic_eval_seconds_by_stage": list(result.eval_seconds_by_stage),
            "full_science_eval_workload_reproduced": False,
            "full_science_gpu_hours_inferred": None,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "trainable_parameter_elements": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "final_recruited_count": (
                model.graph_state.recruited_count()
                if hasattr(model, "graph_state") else None
            ),
            "model_state_sha256": model_state_sha256(model),
            "mask_sha256": budget.mask_sha256 if arm == "fixed_sparse_mlp" else None,
            "numeric_accuracy_or_forgetting_results_disclosed": False,
        }
    finally:
        if guard is not None:
            guard.remove()
