"""Pure analytical M1.3b per-task resource trajectory audit, not a runner.

Accounts for real CIFAR-100 per-task batch shapes (including 8-example tail),
the first-step-delayed recruitment on new tasks, and candidate fixed-sparse
ideal vs dense-masked *linear* MAC proxies. No data, CUDA or training access.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

import torch

from synaploom.model import SynapLoomModel
from synaploom.protocols.h1a2 import h1a2_protocol
from synaploom.research.budgets import (
    count_trainable_parameters,
    estimate_synaploom_linear_macs_per_batch,
)
from synaploom.research.fixed_sparse_baseline import build_h1a_fixed_sparse_feasibility


TRAIN_PER_CLASS = 500
AUDIT_KIND = "M13B_STAGEWISE_IDEAL_MAC_AUDIT_V1"
MASK_SEED = 20261026
CANDIDATE_INIT_SEED = 501  # Used only to instantiate candidate weights; no training.
RUN_AUTHORIZED = False


def _batch_shapes(examples: int, batch_size: int) -> tuple[int, ...]:
    if examples < 1 or batch_size < 1:
        raise ValueError("invalid analytical batch dimensions")
    return tuple(min(batch_size, examples - start) for start in range(0, examples, batch_size))


def _syn_macs(batch: int, recruited: int, *, input_dim: int, d: int, h: int,
              routing: int, k: int, classes: int) -> int:
    if not k <= recruited:
        raise ValueError("recruited capacity below top_k")
    # Matches existing estimate_synaploom_linear_macs_per_batch and batch router:
    # encoder, router query, one recruited-candidate score pass per batch,
    # k expert MLPs per example, classifier.
    encoder = 0 if input_dim == d else batch * input_dim * d
    return (encoder + batch * d * routing + recruited * routing
            + batch * k * (2 * d * h) + batch * d * classes)


def _fixed_macs(batch: int, *, input_dim: int, d: int, classes: int,
                body_weights: int) -> int:
    encoder = 0 if input_dim == d else batch * input_dim * d
    return encoder + batch * body_weights + batch * d * classes


def build_m13b_stagewise_audit() -> dict[str, Any]:
    """Offline analysis of both theoretical resource tiers across all ten tasks.

    No science run, no benchmark, no claim about actual GPU FLOPs or speed.
    """
    protocol = h1a2_protocol()
    if not (protocol.num_classes == 100 and protocol.classes_per_task == 10
            and protocol.num_nodes == 16 and protocol.initial_recruited == 4
            and protocol.top_k == 4 and protocol.epochs_per_task == 20
            and protocol.batch_size == 128 and protocol.eval_batch_size == 256):
        raise ValueError("unexpected frozen H1a2 protocol settings")

    candidate, budget = build_h1a_fixed_sparse_feasibility(
        protocol, seed=CANDIDATE_INIT_SEED, mask_seed=MASK_SEED
    )
    candidate.assert_masks()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        initial_reference = SynapLoomModel(protocol.model_config())
        all_reference = SynapLoomModel(protocol.model_config(all_recruited=True))
    physical_reference = count_trainable_parameters(initial_reference)
    if physical_reference != count_trainable_parameters(all_reference):
        raise RuntimeError("internal H1a physical parameter capacities differ")
    if budget.target_synaploom_total_trainable != physical_reference:
        raise RuntimeError("resource candidate references another physical capacity")

    args = {
        "input_dim": protocol.feature_dim, "d": protocol.d_model,
        "h": protocol.d_hidden, "routing": protocol.d_model,
        "k": protocol.top_k, "classes": protocol.num_classes,
    }
    if _syn_macs(protocol.batch_size, protocol.initial_recruited, **args) != (
        estimate_synaploom_linear_macs_per_batch(
            initial_reference, batch_size=protocol.batch_size
        )
    ):
        raise RuntimeError("independent analytical MAC definition diverged from source")
    if _syn_macs(protocol.batch_size, protocol.num_nodes, **args) != (
        estimate_synaploom_linear_macs_per_batch(
            all_reference, batch_size=protocol.batch_size
        )
    ):
        raise RuntimeError("all-recruited MAC definition diverged from source")

    examples_per_task = protocol.classes_per_task * TRAIN_PER_CLASS
    batch_shapes = _batch_shapes(examples_per_task, protocol.batch_size)
    steps_per_task = protocol.epochs_per_task * len(batch_shapes)
    if batch_shapes[-1] != 8 or len(batch_shapes) != 40 or steps_per_task != 800:
        raise RuntimeError("unexpected H1a2 batch geometry")
    totals = {"progressive_reserve": 0, "all_recruited": 0,
              "fixed_sparse_ideal": 0, "fixed_sparse_dense_kernel_linear_proxy": 0}
    stage_rows = []
    for task_index in range(protocol.num_classes // protocol.classes_per_task):
        before = protocol.initial_recruited + max(0, task_index - 1)
        after = protocol.initial_recruited + task_index
        if task_index == 0:
            before = after
        if after > protocol.num_nodes:
            raise RuntimeError("recruitment would exceed capacity")
        sums = {k: 0 for k in totals}
        for epoch in range(protocol.epochs_per_task):
            for batch_index, actual_batch in enumerate(batch_shapes):
                # H1a controller acts *after* first optimizer update on task
                # boundary: only first batch of tasks 1..9 sees old capacity.
                recruited = before if (
                    task_index > 0 and epoch == 0 and batch_index == 0
                ) else after
                sums["progressive_reserve"] += _syn_macs(actual_batch, recruited, **args)
                sums["all_recruited"] += _syn_macs(
                    actual_batch, protocol.num_nodes, **args
                )
                sums["fixed_sparse_ideal"] += _fixed_macs(
                    actual_batch, input_dim=protocol.feature_dim,
                    d=protocol.d_model, classes=protocol.num_classes,
                    body_weights=budget.body_active_weights,
                )
                sums["fixed_sparse_dense_kernel_linear_proxy"] += _fixed_macs(
                    actual_batch, input_dim=protocol.feature_dim,
                    d=protocol.d_model, classes=protocol.num_classes,
                    body_weights=budget.body_allocated_weights,
                )
        for key, value in sums.items():
            totals[key] += value
        stage_rows.append({
            "task_index": task_index,
            "samples_per_epoch": examples_per_task,
            "epochs": protocol.epochs_per_task,
            "batches_per_epoch": len(batch_shapes),
            "last_batch_size": batch_shapes[-1],
            "optimizer_steps": steps_per_task,
            "progressive_recruited_first_batch": before,
            "progressive_recruited_after_first_step": after,
            "recruitment_after_first_step": task_index > 0,
            "linear_mac_estimates": sums,
            "ideal_progressive_vs_fixed_relative_error": abs(
                sums["progressive_reserve"] - sums["fixed_sparse_ideal"]
            ) / sums["progressive_reserve"],
        })

    progressive = totals["progressive_reserve"]
    report = {
        "schema": 1,
        "kind": AUDIT_KIND,
        "status": "ANALYTICAL_RESOURCE_PREVIEW_NOT_EXECUTION",
        "science_executable": RUN_AUTHORIZED,
        "scientific_claim": "NONE",
        "new_scientific_experiments_run": False,
        "resource_units": "LINEAR_MAC_ESTIMATE_NOT_GPU_FLOPS",
        "feature_file_read": False,
        "cuda_used": False,
        "source_protocol": "EXP-H1A-002",
        "feature_dim": protocol.feature_dim,
        "num_classes": protocol.num_classes,
        "total_tasks": len(stage_rows),
        "train_examples_per_task": examples_per_task,
        "total_training_examples_presented_per_arm": (
            examples_per_task * protocol.epochs_per_task * len(stage_rows)
        ),
        "total_optimizer_steps_per_arm": steps_per_task * len(stage_rows),
        "total_optimizer_steps_for_3_arms_per_order": (
            steps_per_task * len(stage_rows) * 3
        ),
        "reference_total_trainable": physical_reference,
        "fixed_sparse_total_trainable": budget.total_allocated_trainable,
        "physical_trainable_relative_error": budget.total_parameter_relative_error,
        "fixed_sparse_mask_sha256": budget.mask_sha256,
        "fixed_sparse_active_body_weights": budget.body_active_weights,
        "fixed_sparse_allocated_body_weights": budget.body_allocated_weights,
        "fixed_sparse_body_density": budget.body_density,
        "stage_rows": stage_rows,
        "all_task_linear_mac_estimates": totals,
        "whole_run_ideal_progressive_vs_fixed_relative_error": abs(
            progressive - totals["fixed_sparse_ideal"]
        ) / progressive,
        "per_task_ideal_relative_error_at_most_10pct": all(
            x["ideal_progressive_vs_fixed_relative_error"] <= 0.10
            for x in stage_rows
        ),
        "actual_gpu_compute_observed": None,
        "actual_gpu_memory_observed": None,
        "scientific_protocol_frozen": False,
        "owner_decision_pending": True,
        "warnings": [
            "Ideal linear MACs omit normalization, activations, routing overhead, "
            "masked weight multiplication, optimizer/backprop, kernel launches and transfers.",
            "The fixed-sparse PyTorch matmul remains physically dense despite a sparse mask.",
            "Historical H1a batch-level sample-coupled routing behavior is unchanged.",
            "No CUDA speedup, scientific fairness promotion or accuracy benefit follows.",
        ],
    }
    without_digest = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    report["record_sha256"] = sha256(without_digest).hexdigest()
    return report
