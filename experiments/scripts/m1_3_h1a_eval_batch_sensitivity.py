from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from synaploom.data.features import load_feature_bundle, make_split_feature_stream
from synaploom.protocols.h1a import H1A_CLASS_ORDER, build_h1a_variant
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.protocols.h1a_eval_batch_sensitivity import (
    BOOTSTRAP_RESAMPLES,
    CO_PRIMARY_METRICS,
    H1A_EVAL_BATCH_SIZES,
    H1A_EVALBATCH_EXPERIMENT_ID,
    H1A_EVALBATCH_ORDER_FINGERPRINT,
    H1A_EVALBATCH_REFERENCE_SIZE,
    H1A_EVALBATCH_RUNS,
    H1A_EVALBATCH_SEEDS,
    H1A_EVALBATCH_SUMMARIES,
    H1A_EVALBATCH_VARIANTS,
    PERCENT_POINT_METRICS,
    SECONDARY_METRICS,
    class_order_fingerprint,
    dry_run_payload,
    h1a_eval_batch_sensitivity_protocol,
    paired_batch_size_comparisons,
    secondary_batch_size_descriptives,
    validate_protocol,
)
from synaploom.research.preflight import audit_h1a_feature_bundle
from synaploom.training.continual import run_continual_experiment
from synaploom.training.numerics import NumericalInstabilityError


EXPECTED_STATE_HASH_SWEEPS_PER_RUN = 10 * 20 + 10


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _environment(device: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "requested_device": device,
        "git_commit": _git_commit(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.version.cuda
        payload["gpu"] = torch.cuda.get_device_name(
            torch.device(device) if device.startswith("cuda") else 0
        )
    return payload


def _ensure_finite_json(value: Any, *, location: str = "$") -> None:
    """Reject NaN/Inf and non-JSON values before declaring artifacts valid."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {location}: {value}")
        return
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _ensure_finite_json(child, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_finite_json(child, location=f"{location}[{index}]")
        return
    raise TypeError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_finite_json(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_value(summary: dict[str, Any], metric: str) -> float:
    value = float(summary[metric])
    if metric in PERCENT_POINT_METRICS:
        value *= 100.0
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {metric}: {value}")
    return value


def _validate_feature_stream(stream: Any, protocol: Any) -> None:
    if tuple(stream.feature_shape) != (protocol.feature_dim,):
        raise ValueError(
            f"feature dimension mismatch: {tuple(stream.feature_shape)}"
        )
    if len(stream) != 10 or stream.num_classes != protocol.num_classes:
        raise ValueError("O0 stream must contain exactly 10 tasks and 100 classes")
    for task in stream:
        if len(task.classes) != protocol.classes_per_task:
            raise ValueError(f"task {task.task_id} has an invalid class count")
        if int(task.eval_y.numel()) != 1000:
            raise ValueError(
                f"task {task.task_id} must have 1000 evaluation examples, "
                f"got {task.eval_y.numel()}"
            )


def _validated_batch_summaries(result: Any) -> dict[str, dict[str, float]]:
    if set(result.evaluation_batch_size_results) != set(H1A_EVAL_BATCH_SIZES):
        raise ValueError(
            "result is missing one or more registered evaluation batch sizes"
        )
    expected_sweeps = EXPECTED_STATE_HASH_SWEEPS_PER_RUN
    if len(result.evaluation_state_hashes) != expected_sweeps:
        raise ValueError(
            f"expected {expected_sweeps} state-hash sweeps, "
            f"got {len(result.evaluation_state_hashes)}"
        )
    for row in result.evaluation_state_hashes:
        if row["batch_sizes"] != list(H1A_EVAL_BATCH_SIZES):
            raise ValueError("state-hash sweep has a mismatched batch-size list")
        if row["state_sha256_before"] != row["state_sha256_after"]:
            raise ValueError(
                f"model-state hash mismatch in {row['checkpoint_id']}"
            )
        if row["state_unchanged"] is not True:
            raise ValueError(
                f"model-state invariance was not proven in {row['checkpoint_id']}"
            )

    selected: dict[str, dict[str, float]] = {}
    for batch_size in H1A_EVAL_BATCH_SIZES:
        summary = result.summary_for_eval_batch_size(batch_size)
        required = (*CO_PRIMARY_METRICS, *SECONDARY_METRICS)
        values = {metric: _metric_value(summary, metric) for metric in required}
        selected[str(batch_size)] = values
    return selected


def _summary_artifact_digest(summary: dict[str, Any]) -> str:
    encoded = json.dumps(
        summary, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EXP-H1A-EVALBATCH-001: evaluate the same H1a model checkpoints "
            "at all preregistered evaluation batch sizes."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Frozen CIFAR-100 feature bundle (required unless --dry-run is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/m1_3_h1a_eval_batch_sensitivity"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen run matrix without reading features or training.",
    )
    args = parser.parse_args()

    if not args.dry_run and args.features is None:
        parser.error("--features is required unless --dry-run is used")

    protocol = h1a_eval_batch_sensitivity_protocol()
    validate_protocol()
    if class_order_fingerprint(H1A_CLASS_ORDER) != H1A_EVALBATCH_ORDER_FINGERPRINT:
        raise ValueError("O0 class-order fingerprint differs from the preregistration")

    if args.dry_run:
        print(json.dumps(dry_run_payload(), indent=2, sort_keys=True, allow_nan=False))
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested through --device {args.device}, "
            "but torch.cuda.is_available() is false"
        )

    feature_audit = audit_h1a_feature_bundle(args.features)
    if feature_audit.sha256 != H1A2_FEATURE_SHA256:
        raise ValueError(
            "EXP-H1A-EVALBATCH-001 requires the frozen feature artifact SHA-256 "
            f"{H1A2_FEATURE_SHA256}, got {feature_audit.sha256}"
        )
    bundle = load_feature_bundle(args.features)
    feature_metadata = dict(bundle.get("metadata", {}))
    environment = _environment(args.device)

    stream = make_split_feature_stream(
        args.features,
        class_order=H1A_CLASS_ORDER,
        classes_per_task=protocol.classes_per_task,
    )
    _validate_feature_stream(stream, protocol)

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(
            "output directory must be empty to prevent mixing partial/stale runs: "
            f"{args.output}"
        )

    preflight = {
        "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
        "status": "ready",
        "feature_bundle": str(args.features),
        "feature_bundle_sha256": feature_audit.sha256,
        "feature_bundle_audit": feature_audit.to_dict(),
        "feature_bundle_metadata": feature_metadata,
        "environment": environment,
        "protocol": protocol.as_dict(),
        "class_order": list(H1A_CLASS_ORDER),
        "class_order_fingerprint_sha256": H1A_EVALBATCH_ORDER_FINGERPRINT,
        "seeds": list(H1A_EVALBATCH_SEEDS),
        "variants": list(H1A_EVALBATCH_VARIANTS),
        "evaluation_batch_sizes": list(H1A_EVAL_BATCH_SIZES),
        "reference_batch_size": H1A_EVALBATCH_REFERENCE_SIZE,
        "expected_training_runs": H1A_EVALBATCH_RUNS,
        "expected_run_size_summaries": H1A_EVALBATCH_SUMMARIES,
        "expected_state_hash_sweeps_per_run": EXPECTED_STATE_HASH_SWEEPS_PER_RUN,
    }
    _dump_json(args.output / "preflight.json", preflight)

    # Values are stored in percentage units; paired deltas are percentage points.
    per_seed_variant_batch_metrics: dict[
        int, dict[str, dict[int, dict[str, float]]]
    ] = {
        seed: {
            variant: {batch_size: {} for batch_size in H1A_EVAL_BATCH_SIZES}
            for variant in H1A_EVALBATCH_VARIANTS
        }
        for seed in H1A_EVALBATCH_SEEDS
    }
    trajectory_manifest: list[dict[str, Any]] = []
    run_size_manifest: list[dict[str, Any]] = []

    for seed in H1A_EVALBATCH_SEEDS:
        for variant in H1A_EVALBATCH_VARIANTS:
            run_path = args.output / f"{variant}__seed-{seed}.json"
            try:
                model, optimizer, engine, build_meta = build_h1a_variant(
                    variant,
                    stream=stream,
                    seed=seed,
                    protocol=protocol,
                )
                result = run_continual_experiment(
                    model,
                    stream,
                    optimizer,
                    config=protocol.continual_config(seed=seed, device=args.device),
                    engine=engine,
                    evaluation_batch_sizes=H1A_EVAL_BATCH_SIZES,
                    metadata={
                        "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
                        "parent_primary_experiment": "EXP-H1A-002",
                        "related_class_order_sensitivity": "EXP-H1A-ORDER-001",
                        "class_order_id": "O0",
                        "class_order": list(H1A_CLASS_ORDER),
                        "class_order_fingerprint_sha256": H1A_EVALBATCH_ORDER_FINGERPRINT,
                        "variant": variant,
                        "seed": seed,
                        "protocol": protocol.as_dict(),
                        "feature_bundle": str(args.features),
                        "feature_bundle_metadata": feature_metadata,
                        "feature_bundle_sha256": feature_audit.sha256,
                        "feature_bundle_audit": feature_audit.to_dict(),
                        "environment": environment,
                        "evaluation_batch_sizes": list(H1A_EVAL_BATCH_SIZES),
                        "reference_batch_size": H1A_EVALBATCH_REFERENCE_SIZE,
                        "build": build_meta,
                    },
                )
                batch_summaries = _validated_batch_summaries(result)
                for batch_size in H1A_EVAL_BATCH_SIZES:
                    per_seed_variant_batch_metrics[seed][variant][batch_size] = (
                        batch_summaries[str(batch_size)]
                    )

                payload = result.to_dict()
                payload.update(
                    {
                        "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
                        "status": "COMPLETE_NUMERICALLY_VALID",
                        "seed": seed,
                        "variant": variant,
                        "protocol": protocol.as_dict(),
                        "feature_bundle_sha256": feature_audit.sha256,
                        "class_order": list(H1A_CLASS_ORDER),
                        "class_order_fingerprint_sha256": H1A_EVALBATCH_ORDER_FINGERPRINT,
                        "evaluation_batch_size_summaries_percent": batch_summaries,
                        "primary_result_guardrail": (
                            "EXP-H1A-002 remains INCONCLUSIVE; this sensitivity "
                            "study cannot change its verdict."
                        ),
                    }
                )
                _dump_json(run_path, payload)
                run_artifact_sha256 = hashlib.sha256(run_path.read_bytes()).hexdigest()
                trajectory_manifest.append(
                    {
                        "seed": seed,
                        "variant": variant,
                        "status": "complete",
                        "artifact": str(run_path),
                        "artifact_sha256": run_artifact_sha256,
                        "state_hash_sweeps": len(result.evaluation_state_hashes),
                    }
                )
                for batch_size in H1A_EVAL_BATCH_SIZES:
                    one_summary = batch_summaries[str(batch_size)]
                    run_size_manifest.append(
                        {
                            "seed": seed,
                            "variant": variant,
                            "eval_batch_size": batch_size,
                            "trajectory_artifact": str(run_path),
                            "trajectory_artifact_sha256": run_artifact_sha256,
                            "summary_sha256": _summary_artifact_digest(one_summary),
                            "status": "complete",
                        }
                    )
                print(
                    json.dumps(
                        {
                            "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
                            "seed": seed,
                            "variant": variant,
                            "artifact": str(run_path),
                            "status": "complete",
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            except Exception as exc:
                failure_payload = {
                    "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
                    "status": (
                        "INVALID_NUMERICAL_INSTABILITY"
                        if isinstance(exc, NumericalInstabilityError)
                        else "INVALID_EXECUTION_OR_ARTIFACT_FAILURE"
                    ),
                    "seed": seed,
                    "variant": variant,
                    "feature_bundle_sha256": feature_audit.sha256,
                    "environment": environment,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_training_trajectories": len(trajectory_manifest),
                    "completed_run_size_summaries": len(run_size_manifest),
                    "expected_training_runs": H1A_EVALBATCH_RUNS,
                    "expected_run_size_summaries": H1A_EVALBATCH_SUMMARIES,
                }
                failure_path = args.output / f"failure__{variant}__seed-{seed}.json"
                _dump_json(failure_path, failure_payload)
                raise RuntimeError(
                    f"{H1A_EVALBATCH_EXPERIMENT_ID} invalidated/aborted; "
                    f"failure details: {failure_path}"
                ) from exc

    expected_trajectories = {
        (seed, variant)
        for seed in H1A_EVALBATCH_SEEDS
        for variant in H1A_EVALBATCH_VARIANTS
    }
    actual_trajectories = {
        (int(row["seed"]), str(row["variant"])) for row in trajectory_manifest
    }
    if len(trajectory_manifest) != H1A_EVALBATCH_RUNS or actual_trajectories != expected_trajectories:
        raise RuntimeError(
            "completed training manifest does not exactly match the registered 20-run matrix"
        )

    expected_run_sizes = {
        (seed, variant, batch_size)
        for seed in H1A_EVALBATCH_SEEDS
        for variant in H1A_EVALBATCH_VARIANTS
        for batch_size in H1A_EVAL_BATCH_SIZES
    }
    actual_run_sizes = {
        (int(row["seed"]), str(row["variant"]), int(row["eval_batch_size"]))
        for row in run_size_manifest
    }
    if len(run_size_manifest) != H1A_EVALBATCH_SUMMARIES or actual_run_sizes != expected_run_sizes:
        raise RuntimeError(
            "completed run-size manifest does not exactly match the registered 120 summaries"
        )

    comparisons = paired_batch_size_comparisons(per_seed_variant_batch_metrics)
    aggregate = {
        "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
        "status": "COMPLETE_NUMERICALLY_VALID",
        "parent_primary_experiment": "EXP-H1A-002",
        "parent_primary_verdict": "INCONCLUSIVE",
        "related_class_order_sensitivity_experiment": "EXP-H1A-ORDER-001",
        "related_class_order_sensitivity_verdict": "ORDER_SENSITIVE",
        "protocol": protocol.as_dict(),
        "feature_bundle_metadata": feature_metadata,
        "feature_bundle_sha256": feature_audit.sha256,
        "feature_bundle_audit": feature_audit.to_dict(),
        "environment": environment,
        "class_order": list(H1A_CLASS_ORDER),
        "class_order_fingerprint_sha256": H1A_EVALBATCH_ORDER_FINGERPRINT,
        "seeds": list(H1A_EVALBATCH_SEEDS),
        "variants": list(H1A_EVALBATCH_VARIANTS),
        "evaluation_batch_sizes": list(H1A_EVAL_BATCH_SIZES),
        "reference_batch_size": H1A_EVALBATCH_REFERENCE_SIZE,
        "expected_training_runs": H1A_EVALBATCH_RUNS,
        "completed_training_runs": len(trajectory_manifest),
        "expected_run_size_summaries": H1A_EVALBATCH_SUMMARIES,
        "completed_run_size_summaries": len(run_size_manifest),
        "expected_state_hash_sweeps_per_run": EXPECTED_STATE_HASH_SWEEPS_PER_RUN,
        "trajectory_manifest": trajectory_manifest,
        "run_size_manifest": run_size_manifest,
        "paired_comparisons": comparisons,
        "secondary_descriptive_metrics": secondary_batch_size_descriptives(
            per_seed_variant_batch_metrics
        ),
        "interpretation_guardrail": (
            "This is a separate measurement-sensitivity study on fresh training "
            "trajectories. It cannot change EXP-H1A-002 = INCONCLUSIVE or "
            "EXP-H1A-ORDER-001 = ORDER_SENSITIVE."
        ),
    }
    _dump_json(args.output / "aggregate.json", aggregate)
    print(
        json.dumps(
            {
                "experiment_id": H1A_EVALBATCH_EXPERIMENT_ID,
                "status": aggregate["status"],
                "completed_training_runs": len(trajectory_manifest),
                "completed_run_size_summaries": len(run_size_manifest),
                "direction_classification": comparisons[
                    "registered_direction_classification"
                ],
                "primary_exp_h1a_002_verdict": "INCONCLUSIVE",
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
