"""Bounded CPU synthetic *engineering* pilot for M1.3b; never CIFAR science.

One fixed 3-task Gaussian stream, one epoch/task, one engineering seed and
three predefined arms. No user-supplied feature paths, GPU choice, class order,
confirmation seeds, scientific variant overrides or checkpoint loading.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any

import torch

from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.protocols.h1a import H1AProtocol, build_h1a_variant
from synaploom.research.fixed_sparse_baseline import (
    build_h1a_fixed_sparse_feasibility,
    install_fixed_mask_guard,
)
from synaploom.training.continual import run_continual_experiment
from synaploom.training.state_digest import model_state_sha256


PILOT_KIND = "M13B_SYNTHETIC_ENGINEERING_ONLY"
PILOT_SEED = 13
DATA_SEED = 41
MASK_SEED = 17
ARMS = ("progressive_reserve", "all_recruited", "fixed_sparse_mlp")
RESERVED_CONFIRMATION = frozenset(range(501, 511))


def _pilot_protocol() -> H1AProtocol:
    return H1AProtocol(
        feature_dim=8, num_classes=6, classes_per_task=2,
        d_model=8, d_hidden=12, num_nodes=4, initial_recruited=2, top_k=2,
        epochs_per_task=1, batch_size=4, eval_batch_size=4,
        learning_rate=0.01, momentum=0.9, weight_decay=5e-4,
        seeds=(PILOT_SEED,), current_task_eval_every_epochs=1,
    )


def _finite_metrics(summary: dict[str, Any]) -> dict[str, float]:
    keys = (
        "final_average_accuracy",
        "late_task_within_task_mean_accuracy",
        "average_forgetting",
        "backward_transfer",
    )
    values = {}
    for key in keys:
        item = summary.get(key)
        if type(item) not in (float, int) or not math.isfinite(float(item)):
            raise RuntimeError("synthetic pilot metric missing or nonfinite: " + key)
        values[key] = float(item)
    return values


def _stream_identity(stream: Any) -> str:
    """Fingerprint exact synthetic input and labels, not scientific data."""
    digest = sha256(b"synaploom/m13b/synthetic-stream/v1\0")
    for task in stream:
        digest.update(int(task.task_id).to_bytes(4, "big", signed=False))
        digest.update(bytes(task.classes))
        for tensor in (task.train_x, task.train_y, task.eval_x, task.eval_y):
            cpu = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(cpu.shape)).encode("ascii") + b"\0")
            digest.update(str(cpu.dtype).encode("ascii") + b"\0")
            digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def run_m13b_synthetic_engineering_pilot() -> dict[str, Any]:
    """Execute exactly 3 tiny CPU synthetic trajectories; never scientific runs.

    No external inputs, checkpoints, feature tensors, GPU or execution variants.
    The fixed sparse optimizer guard is installed and removed explicitly.
    Results are diagnostic smoke outputs and **must not** be pooled with H1a
    or M1.3b confirmatory accuracy/forgetting estimates.
    """
    if PILOT_SEED in RESERVED_CONFIRMATION or DATA_SEED in RESERVED_CONFIRMATION:
        raise RuntimeError("engineering seed overlaps protected confirmation seeds")
    protocol = _pilot_protocol()
    stream = make_gaussian_cluster_stream(
        num_tasks=3, classes_per_task=2, input_dim=8,
        train_per_class=4, eval_per_class=2, seed=DATA_SEED,
    )
    identity = _stream_identity(stream)
    arms: dict[str, dict[str, Any]] = {}
    # CPU-only; avoid multi-core timing variance and restore CPU thread count.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for variant in ARMS:
            guard = None
            if variant == "fixed_sparse_mlp":
                model, budget = build_h1a_fixed_sparse_feasibility(
                    protocol, seed=PILOT_SEED, mask_seed=MASK_SEED
                )
                if not budget.matched_total_2pct or not budget.matched_ideal_macs_10pct:
                    raise RuntimeError("synthetic mask resource feasibility gate failed")
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=protocol.learning_rate,
                    momentum=protocol.momentum, weight_decay=protocol.weight_decay,
                )
                engine = None
                guard = install_fixed_mask_guard(model, optimizer)
                provenance = {
                    "mask_sha256": budget.mask_sha256,
                    "body_active_weights": budget.body_active_weights,
                    "body_allocated_weights": budget.body_allocated_weights,
                }
            else:
                model, optimizer, engine, _ = build_h1a_variant(
                    variant, stream=stream, seed=PILOT_SEED, protocol=protocol
                )
                provenance = {"mask_sha256": None}

            try:
                result = run_continual_experiment(
                    model, stream, optimizer,
                    config=protocol.continual_config(seed=PILOT_SEED, device="cpu"),
                    engine=engine,
                    metadata={"kind": PILOT_KIND, "arm": variant},
                )
                if result.global_steps != 6 or result.accuracy_matrix.shape != (3, 3):
                    raise RuntimeError("unexpected synthetic pilot steps or task shape")
                metrics = _finite_metrics(result.summary())
                if variant == "fixed_sparse_mlp":
                    model.assert_masks(optimizer)
                    if model.mask_digest() != provenance["mask_sha256"]:
                        raise RuntimeError("fixed-sparse mask identity changed")
                arms[variant] = {
                    "status": "ENGINEERING_SMOKE_ONLY",
                    "steps": result.global_steps,
                    "metrics_synthetic_only": metrics,
                    "model_state_sha256": model_state_sha256(model),
                    **provenance,
                }
            finally:
                if guard is not None:
                    guard.remove()
    finally:
        torch.set_num_threads(previous_threads)
    if tuple(arms) != ARMS:
        raise RuntimeError("synthetic pilot missing an arm")
    return {
        "schema": 1,
        "kind": PILOT_KIND,
        "authorization": "SYNTHETIC_CPU_ENGINEERING_ONLY",
        "science_executable": False,
        "scientific_claim": "NONE",
        "source_dataset": "synthetic_gaussian_3x2_NOT_CIFAR100",
        "data_sha256": identity,
        "device": "cpu",
        "seed": PILOT_SEED,
        "data_seed": DATA_SEED,
        "mask_seed": MASK_SEED,
        "epochs_per_task": 1,
        "train_per_class": 4,
        "eval_per_class": 2,
        "protected_confirmation_seeds_untouched": True,
        "historical_h1a_results_unchanged": True,
        "arms": arms,
    }


def pilot_canonical_json(report: dict[str, Any]) -> bytes:
    """Allow finite-only standard JSON; stable key ordering for audit artifacts."""
    return json.dumps(
        report, sort_keys=True, ensure_ascii=True,
        separators=(",", ":"), allow_nan=False
    ).encode("ascii")
