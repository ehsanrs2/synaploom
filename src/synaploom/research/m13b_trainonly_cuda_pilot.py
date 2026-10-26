"""Manually invoked M1.3b GPU *engineering feasibility* on train-only features.

NOT the 30/90-trajectory study, not a scientific evaluation of CIFAR-100,
not an automatic GPU job. Explicit trusted, historically SHA-pinned input and
physical CUDA device required. Confirmatory seeds and protected eval features
are not used for model selection, training, or diagnostic validation.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
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


KIND = "M13B_TRAIN_ONLY_CUDA_FEASIBILITY_V1"
ENGINEERING_SEED = 13
SPLIT_SEED = 41
MASK_SEED = 17
TASKS = 2
TRAIN_PER_CLASS = 16
HOLDOUT_PER_CLASS = 4
MAX_FILE_BYTES = 512 * 1024 * 1024


def load_trusted_train_only_features(path: Path, *, trusted_source: bool) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Verify the original pinned file *before* trusting pickle; use same bytes.

    The file's eval tensors must exist but are neither returned nor used as
    validation/training data. A SHA pin does not authenticate an unknown origin.
    """
    if trusted_source is not True:
        raise ValueError("explicit independent trust of feature file origin required")
    with path.open("rb") as f:
        raw = f.read(MAX_FILE_BYTES + 1)
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("feature file exceeds bounded snapshot size or is empty")
    digest = sha256(raw).hexdigest()
    if digest != H1A2_FEATURE_SHA256:
        raise ValueError("feature SHA-256 does not match frozen H1a artifact")
    # WARNING: torch.load(...weights_only=False) uses pickle. Only the user's
    # independently trusted file, with the separately frozen digest, is allowed.
    bundle = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or not {
        "train_features", "train_targets", "eval_features", "eval_targets", "metadata"
    }.issubset(bundle):
        raise ValueError("invalid trusted feature bundle schema")
    features = bundle["train_features"]
    labels = bundle["train_targets"]
    if not isinstance(features, torch.Tensor) or features.shape != (50_000, 512):
        raise ValueError("unexpected frozen training feature shape")
    if not isinstance(labels, torch.Tensor) or labels.shape != (50_000,) or labels.is_floating_point():
        raise ValueError("unexpected training target tensor")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("training features contain unsupported type or NaN/Inf")
    if not bool(torch.equal(torch.bincount(labels.long(), minlength=100),
                           torch.full((100,), 500, dtype=torch.long))):
        raise ValueError("frozen training class inventory mismatch")
    meta = bundle["metadata"]
    if not isinstance(meta, dict) or any(
        meta.get(k) != v for k, v in {
            "dataset": "CIFAR100", "feature_extractor": "torchvision.resnet18",
            "weights": "IMAGENET1K_V1", "feature_dim": 512,
            "train_examples": 50_000, "eval_examples": 10_000,
        }.items()
    ):
        raise ValueError("frozen feature metadata mismatch")
    # Do not construct a stream from bundle["eval_features"] / eval_targets.
    return features, labels.long(), digest


def train_only_two_task_stream(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_order: tuple[int, ...] = H1A_CLASS_ORDER,
) -> tuple[ContinualStream, str]:
    """Select disjoint train/diagnostic subsamples from TRAIN partition only."""
    if len(class_order) != 100 or set(class_order) != set(range(100)):
        raise ValueError("expected complete frozen 100-class order")
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[1] != 512:
        raise ValueError("features must have dimension 512")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("training feature/target dimensions disagree")

    all_indices: list[tuple[list[int], list[int]]] = []
    audit = sha256(b"synaploom/m13b/trainonly-split/v1\0")
    for class_id in class_order[:TASKS * 10]:
        ids = (labels == class_id).nonzero(as_tuple=False).flatten().cpu()
        if ids.numel() < TRAIN_PER_CLASS + HOLDOUT_PER_CLASS:
            raise ValueError("insufficient train-only examples for class " + str(class_id))
        g = torch.Generator(device="cpu")
        g.manual_seed(SPLIT_SEED + class_id)
        selection = ids[torch.randperm(ids.numel(), generator=g)[:TRAIN_PER_CLASS + HOLDOUT_PER_CLASS]]
        train_ids = selection[:TRAIN_PER_CLASS].tolist()
        val_ids = selection[TRAIN_PER_CLASS:].tolist()
        if set(train_ids).intersection(val_ids):
            raise ValueError("train-only holdout overlaps training")
        all_indices.append((train_ids, val_ids))
        audit.update(int(class_id).to_bytes(2, "big"))
        for name, group in ((b"train", train_ids), (b"holdout", val_ids)):
            audit.update(name)
            for idx in group:
                audit.update(int(idx).to_bytes(4, "big"))

    tasks: list[TensorTask] = []
    for task_id in range(TASKS):
        groups = all_indices[task_id * 10:(task_id + 1) * 10]
        train = torch.tensor([i for x, _ in groups for i in x], dtype=torch.long)
        val = torch.tensor([i for _, x in groups for i in x], dtype=torch.long)
        classes = tuple(class_order[task_id * 10:(task_id + 1) * 10])
        tasks.append(TensorTask(
            task_id=task_id, classes=classes,
            train_x=features.index_select(0, train).float().contiguous(),
            train_y=labels.index_select(0, train).long(),
            eval_x=features.index_select(0, val).float().contiguous(),
            eval_y=labels.index_select(0, val).long(),
        ))
    return ContinualStream(tasks), audit.hexdigest()


def _validate_result(result: Any) -> dict[str, float]:
    if result.global_steps != 4 or result.accuracy_matrix.shape != (2, 2):
        raise RuntimeError("bounded GPU step count / task dimensions changed")
    out = {}
    for field in ("final_average_accuracy", "late_task_within_task_mean_accuracy",
                  "average_forgetting", "backward_transfer"):
        x = result.summary().get(field)
        if type(x) not in (int, float) or not math.isfinite(float(x)):
            raise RuntimeError("non-finite train-only engineering metric: " + field)
        out[field] = float(x)
    return out


def run_m13b_trainonly_cuda_feasibility(
    stream: ContinualStream,
    *,
    frozen_data_sha256: str,
    train_only_split_sha256: str,
    on_arm_complete: Any = None,
) -> dict[str, Any]:
    """One explicitly requested 3-arm GPU feasibility, never science results."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for explicit manual GPU pilot")
    if len(stream) != TASKS or any(
        task.train_x.shape != (160, 512) or task.eval_x.shape != (40, 512)
        for task in stream
    ):
        raise ValueError("unexpected train-only pilot stream shape")
    if frozen_data_sha256 != H1A2_FEATURE_SHA256:
        raise ValueError("feature provenance not frozen")
    if len(train_only_split_sha256) != 64:
        raise ValueError("invalid train-only split digest")
    protocol = replace(h1a2_protocol(), epochs_per_task=1, seeds=(ENGINEERING_SEED,),
                       batch_size=128, eval_batch_size=256,
                       current_task_eval_every_epochs=1)
    device = torch.device("cuda:0")
    arms = {}
    for arm in ARMS:
        hook = None
        if arm == "fixed_sparse_mlp":
            model, budget = build_h1a_fixed_sparse_feasibility(
                protocol, seed=ENGINEERING_SEED, mask_seed=MASK_SEED
            )
            if not budget.matched_total_2pct or not budget.matched_ideal_macs_10pct:
                raise RuntimeError("fixed-sparse candidate arithmetic budget gate failed")
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=protocol.learning_rate,
                                        momentum=protocol.momentum, weight_decay=protocol.weight_decay)
            hook = install_fixed_mask_guard(model, optimizer)
            engine = None
        else:
            model, optimizer, engine, _ = build_h1a_variant(
                arm, stream=stream, seed=ENGINEERING_SEED, protocol=protocol
            )
            model.to(device)
        try:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = run_continual_experiment(
                model, stream, optimizer,
                config=protocol.continual_config(seed=ENGINEERING_SEED, device="cuda:0"),
                engine=engine,
                metadata={"kind": KIND, "arm": arm, "scientific_claim": "NONE"},
            )
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            diagnostic = _validate_result(result)
            if arm == "fixed_sparse_mlp":
                model.assert_masks(optimizer)
                if model.mask_digest() != budget.mask_sha256:
                    raise RuntimeError("frozen fixed-sparse mask changed")
            entry = {
                "status": "GPU_TRAIN_ONLY_ENGINEERING_ONLY",
                "global_steps": result.global_steps,
                "wall_seconds": seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "physically_allocated_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "diagnostic_train_only_holdout_metrics": diagnostic,
                "model_state_sha256": model_state_sha256(model),
                "mask_sha256": budget.mask_sha256 if arm == "fixed_sparse_mlp" else None,
            }
            arms[arm] = entry
            if on_arm_complete is not None:
                on_arm_complete(arm, entry)
        finally:
            if hook is not None:
                hook.remove()
            del model, optimizer
            if engine is not None:
                del engine
            torch.cuda.empty_cache()

    return {
        "schema": 1, "kind": KIND, "science_executable": False,
        "scientific_claim": "NONE", "dataset_scope": "FROZEN_TRAIN_PARTITION_ONLY",
        "protected_eval_partition_used": False, "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "feature_sha256": frozen_data_sha256,
        "train_only_split_sha256": train_only_split_sha256,
        "seed": ENGINEERING_SEED, "mask_seed": MASK_SEED,
        "tasks": TASKS, "epochs_per_task": 1,
        "train_per_class": TRAIN_PER_CLASS, "holdout_per_class": HOLDOUT_PER_CLASS,
        "protected_confirmation_seeds_untouched": True,
        "arms": arms,
    }
