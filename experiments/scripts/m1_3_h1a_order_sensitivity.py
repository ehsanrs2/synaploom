from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from synaploom.data.features import load_feature_bundle, make_split_feature_stream
from synaploom.protocols.h1a import build_h1a_variant
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.protocols.h1a_order_sensitivity import (
    BOOTSTRAP_RESAMPLES,
    CLASS_ORDERS,
    CO_PRIMARY_METRICS,
    EXPECTED_ORDER_FINGERPRINTS,
    H1A_ORDER_EXPERIMENT_ID,
    H1A_ORDER_IDS,
    H1A_ORDER_RUNS,
    H1A_ORDER_SEEDS,
    H1A_ORDER_VARIANTS,
    ORDER_BOOTSTRAP_SEEDS,
    PERCENT_POINT_METRICS,
    POOLED_BOOTSTRAP_SEED,
    class_order_fingerprint,
    classify_order_sensitivity,
    h1a_order_sensitivity_protocol,
    paired_bootstrap_ci,
    seed_cluster_bootstrap_ci,
    validate_class_orders,
)
from synaploom.research.budgets import (
    count_trainable_parameters,
    estimate_dense_linear_macs_per_batch,
    estimate_synaploom_linear_macs_per_batch,
    synaploom_budget,
)
from synaploom.research.preflight import audit_h1a_feature_bundle
from synaploom.training.continual import run_continual_experiment
from synaploom.training.numerics import NumericalInstabilityError


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
        "torch": torch.__version__,
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
    """Reject NaN/Inf before persisting an artifact or declaring a run valid."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {location}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _ensure_finite_json(child, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _ensure_finite_json(child, location=f"{location}[{index}]")


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_finite_json(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metric_value(summary: Mapping[str, float], metric: str) -> float:
    value = float(summary[metric])
    if metric in PERCENT_POINT_METRICS:
        value *= 100.0
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {metric}: {value}")
    return value


def _paired_comparison(
    left: Mapping[int, Mapping[str, float]],
    right: Mapping[int, Mapping[str, float]],
    *,
    metric: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    paired_deltas: dict[int, float] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for seed in H1A_ORDER_SEEDS:
        if seed not in left or seed not in right:
            raise ValueError(f"missing paired seed {seed} for metric {metric}")
        treatment = _metric_value(left[seed], metric)
        control = _metric_value(right[seed], metric)
        delta = treatment - control
        paired_deltas[seed] = delta
        per_seed[str(seed)] = {
            "progressive_reserve_pp": treatment,
            "all_recruited_pp": control,
            "delta_pp": delta,
        }

    return {
        "metric": metric,
        "units": "percentage_points",
        "n_paired_seeds": len(paired_deltas),
        "mean_delta_pp": statistics.fmean(paired_deltas.values()),
        "ci95_pp": paired_bootstrap_ci(
            paired_deltas,
            samples=BOOTSTRAP_RESAMPLES,
            seed=bootstrap_seed,
        ),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": bootstrap_seed,
        "per_seed": per_seed,
    }


def _cluster_pooled_comparison(
    per_order: Mapping[str, Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    deltas_by_order: dict[str, dict[int, float]] = {
        order_id: {
            int(seed): float(value["delta_pp"])
            for seed, value in per_order[order_id][metric]["per_seed"].items()
        }
        for order_id in H1A_ORDER_IDS
    }
    cluster_means = {
        seed: statistics.fmean(
            deltas_by_order[order_id][seed] for order_id in H1A_ORDER_IDS
        )
        for seed in H1A_ORDER_SEEDS
    }
    all_deltas = [
        deltas_by_order[order_id][seed]
        for order_id in H1A_ORDER_IDS
        for seed in H1A_ORDER_SEEDS
    ]
    ci95 = seed_cluster_bootstrap_ci(
        deltas_by_order,
        seeds=H1A_ORDER_SEEDS,
        samples=BOOTSTRAP_RESAMPLES,
        seed=POOLED_BOOTSTRAP_SEED,
    )
    return {
        "metric": metric,
        "units": "percentage_points",
        "weighting": "equal class-order weight; resample seed clusters",
        "n_order_seed_pairs": len(all_deltas),
        "n_seed_clusters": len(cluster_means),
        "mean_delta_pp": statistics.fmean(all_deltas),
        "ci95_pp": ci95,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": POOLED_BOOTSTRAP_SEED,
        "per_seed_mean_delta_across_orders_pp": {
            str(seed): cluster_means[seed] for seed in H1A_ORDER_SEEDS
        },
        "per_order_mean_delta_pp": {
            order_id: statistics.fmean(deltas_by_order[order_id].values())
            for order_id in H1A_ORDER_IDS
        },
    }


def _describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot describe an empty set of values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("descriptive values must be finite")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _secondary_descriptives(
    by_order_variant_seed: Mapping[
        str, Mapping[str, Mapping[int, Mapping[str, float]]]
    ],
    *,
    metric: str,
) -> dict[str, Any]:
    per_order: dict[str, Any] = {}
    pooled_values: dict[str, list[float]] = {
        variant: [] for variant in H1A_ORDER_VARIANTS
    }
    all_pair_deltas: list[float] = []

    for order_id in H1A_ORDER_IDS:
        variant_summary: dict[str, dict[str, float | int]] = {}
        variant_seed_values: dict[str, dict[int, float]] = {}
        for variant in H1A_ORDER_VARIANTS:
            values = {
                seed: _metric_value(summary, metric)
                for seed, summary in by_order_variant_seed[order_id][variant].items()
            }
            if set(values) != set(H1A_ORDER_SEEDS):
                raise ValueError(f"incomplete {metric} values for {order_id}/{variant}")
            variant_seed_values[variant] = values
            variant_summary[variant] = _describe(list(values.values()))
            pooled_values[variant].extend(values[seed] for seed in H1A_ORDER_SEEDS)

        deltas = {
            seed: (
                variant_seed_values["progressive_reserve"][seed]
                - variant_seed_values["all_recruited"][seed]
            )
            for seed in H1A_ORDER_SEEDS
        }
        all_pair_deltas.extend(deltas.values())
        per_order[order_id] = {
            "variant_summary_pp": variant_summary,
            "paired_mean_delta_pp": statistics.fmean(deltas.values()),
            "paired_deltas_by_seed_pp": {
                str(seed): deltas[seed] for seed in H1A_ORDER_SEEDS
            },
        }

    return {
        "units": "percentage_points",
        "per_order": per_order,
        "pooled_equal_order_weight": {
            "variant_summary_pp": {
                variant: _describe(pooled_values[variant])
                for variant in H1A_ORDER_VARIANTS
            },
            "paired_mean_delta_pp": statistics.fmean(all_pair_deltas),
            "paired_delta_count": len(all_pair_deltas),
            "paired_deltas_order_then_seed_pp": all_pair_deltas,
        },
    }


def _validate_protocol(protocol: Any) -> None:
    settings = protocol.as_dict()
    if tuple(settings["seeds"]) != H1A_ORDER_SEEDS:
        raise ValueError("protocol seed list differs from the preregistered seed block")
    if settings["learning_rate"] != 0.01:
        raise ValueError("protocol learning rate must remain 0.01")
    if settings["eval_batch_size"] != 256:
        raise ValueError("protocol eval_batch_size must remain 256")
    if (
        settings["batch_size"] != 128
        or settings["epochs_per_task"] != 20
        or settings["classes_per_task"] != 10
        or settings["current_task_eval_every_epochs"] != 1
    ):
        raise ValueError("protocol training/task settings differ from the preregistration")
    if (
        settings["feature_dim"] != 512
        or settings["num_classes"] != 100
        or settings["momentum"] != 0.9
        or settings["weight_decay"] != 0.0005
    ):
        raise ValueError("protocol optimizer/feature settings differ from the preregistration")
    if (
        settings["d_model"] != 128
        or settings["d_hidden"] != 256
        or settings["num_nodes"] != 16
        or settings["initial_recruited"] != 4
        or settings["top_k"] != 4
        or protocol.graph_config().q_max != 0
        or protocol.graph_config().message_rounds != 0
    ):
        raise ValueError("protocol model/graph settings differ from the preregistration")


def _dry_run_payload(protocol: Any) -> dict[str, Any]:
    fingerprints = validate_class_orders()
    _validate_protocol(protocol)
    expected_runs = len(H1A_ORDER_IDS) * len(H1A_ORDER_SEEDS) * len(H1A_ORDER_VARIANTS)
    if expected_runs != H1A_ORDER_RUNS:
        raise ValueError(f"registered run count must be {H1A_ORDER_RUNS}, got {expected_runs}")
    return {
        "experiment_id": H1A_ORDER_EXPERIMENT_ID,
        "status": "DRY_RUN_ONLY",
        "protocol": protocol.as_dict(),
        "feature_sha256": H1A2_FEATURE_SHA256,
        "class_orders": {
            order_id: list(CLASS_ORDERS[order_id]) for order_id in H1A_ORDER_IDS
        },
        "order_fingerprints": fingerprints,
        "seeds": list(H1A_ORDER_SEEDS),
        "variants": list(H1A_ORDER_VARIANTS),
        "expected_runs": expected_runs,
        "co_primary_metrics": list(CO_PRIMARY_METRICS),
        "analysis": {
            "per_order_bootstrap_seeds": ORDER_BOOTSTRAP_SEEDS,
            "pooled_seed_cluster_bootstrap_seed": POOLED_BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "classification": [
                "ORDER_ROBUST_FAVORABLE",
                "ORDER_ROBUST_UNFAVORABLE",
                "ORDER_SENSITIVE",
                "INCONCLUSIVE",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-H1A-ORDER-001: preregistered H1a class-order sensitivity study."
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Frozen feature bundle (required unless --dry-run is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/m1_3_h1a_order_sensitivity"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen protocol and fingerprints without reading data or training.",
    )
    args = parser.parse_args()
    if not args.dry_run and args.features is None:
        parser.error("--features is required unless --dry-run is used")

    protocol = h1a_order_sensitivity_protocol()
    if args.dry_run:
        print(json.dumps(_dry_run_payload(protocol), indent=2, sort_keys=True, allow_nan=False))
        return

    fingerprints = validate_class_orders()
    _validate_protocol(protocol)
    expected_runs = len(H1A_ORDER_IDS) * len(H1A_ORDER_SEEDS) * len(H1A_ORDER_VARIANTS)
    if expected_runs != H1A_ORDER_RUNS:
        raise ValueError(f"registered run count must be {H1A_ORDER_RUNS}, got {expected_runs}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested through --device {args.device}, but torch.cuda.is_available() is false"
        )

    feature_audit = audit_h1a_feature_bundle(args.features)
    if feature_audit.sha256 != H1A2_FEATURE_SHA256:
        raise ValueError(
            "EXP-H1A-ORDER-001 requires the frozen feature artifact SHA-256 "
            f"{H1A2_FEATURE_SHA256}, got {feature_audit.sha256}"
        )
    bundle = load_feature_bundle(args.features)
    feature_metadata = dict(bundle.get("metadata", {}))
    environment = _environment(args.device)

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(
            f"output directory must be empty to prevent mixing partial/stale runs: {args.output}"
        )

    preflight = {
        "experiment_id": H1A_ORDER_EXPERIMENT_ID,
        "status": "ready",
        "feature_bundle": str(args.features),
        "feature_bundle_sha256": feature_audit.sha256,
        "feature_bundle_audit": feature_audit.to_dict(),
        "feature_bundle_metadata": feature_metadata,
        "environment": environment,
        "protocol": protocol.as_dict(),
        "order_fingerprints": fingerprints,
        "orders": {
            order_id: list(CLASS_ORDERS[order_id]) for order_id in H1A_ORDER_IDS
        },
        "seeds": list(H1A_ORDER_SEEDS),
        "variants": list(H1A_ORDER_VARIANTS),
        "expected_runs": expected_runs,
    }
    _dump_json(args.output / "preflight.json", preflight)

    by_order_variant_seed: dict[
        str, dict[str, dict[int, dict[str, float]]]
    ] = {
        order_id: {variant: {} for variant in H1A_ORDER_VARIANTS}
        for order_id in H1A_ORDER_IDS
    }
    run_manifest: list[dict[str, Any]] = []

    for order_id in H1A_ORDER_IDS:
        order = CLASS_ORDERS[order_id]
        if class_order_fingerprint(order) != EXPECTED_ORDER_FINGERPRINTS[order_id]:
            raise ValueError(f"class order changed after preflight: {order_id}")
        stream = make_split_feature_stream(
            args.features,
            class_order=order,
            classes_per_task=protocol.classes_per_task,
        )
        if tuple(stream.feature_shape) != (protocol.feature_dim,):
            raise ValueError(
                f"feature dimension mismatch for {order_id}: "
                f"{tuple(stream.feature_shape)}"
            )
        if len(stream) != 10 or stream.num_classes != protocol.num_classes:
            raise ValueError(
                f"{order_id} must contain exactly 10 tasks and 100 classes"
            )

        for seed in H1A_ORDER_SEEDS:
            for variant in H1A_ORDER_VARIANTS:
                run_path = args.output / f"order-{order_id}__{variant}__seed-{seed}.json"
                try:
                    model, optimizer, engine, build_meta = build_h1a_variant(
                        variant,
                        stream=stream,
                        seed=seed,
                        protocol=protocol,
                    )
                    if hasattr(model, "graph_state"):
                        initial_budget = synaploom_budget(model)
                        initial_macs = estimate_synaploom_linear_macs_per_batch(
                            model,
                            batch_size=protocol.batch_size,
                        )
                    else:
                        initial_budget = None
                        dense_width = int(build_meta["dense_hidden_width"])
                        initial_macs = estimate_dense_linear_macs_per_batch(
                            input_dim=protocol.feature_dim,
                            d_model=protocol.d_model,
                            d_hidden=dense_width,
                            num_classes=protocol.num_classes,
                            batch_size=protocol.batch_size,
                        )

                    result = run_continual_experiment(
                        model,
                        stream,
                        optimizer,
                        config=protocol.continual_config(
                            seed=seed,
                            device=args.device,
                        ),
                        engine=engine,
                        metadata={
                            "experiment_id": H1A_ORDER_EXPERIMENT_ID,
                            "parent_primary_experiment": "EXP-H1A-002",
                            "milestone": "M1.3a",
                            "hypothesis": "H1a-class-order-sensitivity",
                            "order_id": order_id,
                            "order_fingerprint_sha256": fingerprints[order_id],
                            "class_order": list(order),
                            "variant": variant,
                            "seed": seed,
                            "protocol": protocol.as_dict(),
                            "feature_bundle": str(args.features),
                            "feature_bundle_metadata": feature_metadata,
                            "feature_bundle_sha256": feature_audit.sha256,
                            "feature_bundle_audit": feature_audit.to_dict(),
                            "environment": environment,
                            "build": build_meta,
                            "initial_total_parameters": count_trainable_parameters(model),
                            "initial_budget": (
                                asdict(initial_budget) if initial_budget is not None else None
                            ),
                            "initial_linear_macs_per_batch_estimate": initial_macs,
                            "seed_block_source": "EXP-H1A-ORDER-001 preregistration",
                        },
                    )
                    if hasattr(model, "graph_state"):
                        result.metadata["final_budget"] = asdict(synaploom_budget(model))
                        result.metadata["final_linear_macs_per_batch_estimate"] = (
                            estimate_synaploom_linear_macs_per_batch(
                                model,
                                batch_size=protocol.batch_size,
                            )
                        )

                    summary = result.summary()
                    scalar_summary = {
                        key: float(value)
                        for key, value in summary.items()
                        if isinstance(value, (int, float)) and value is not None
                    }
                    for metric in (
                        "final_average_accuracy",
                        "late_task_within_task_mean_accuracy",
                        "average_forgetting",
                        "backward_transfer",
                    ):
                        if metric not in scalar_summary or not math.isfinite(scalar_summary[metric]):
                            raise ValueError(
                                f"missing/non-finite required summary metric {metric} "
                                f"for {order_id}/{variant}/seed-{seed}"
                            )

                    result_payload = result.to_dict()
                    _ensure_finite_json(result_payload)
                    run_path.write_text(
                        json.dumps(
                            result_payload,
                            indent=2,
                            sort_keys=True,
                            allow_nan=False,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    artifact_sha256 = hashlib.sha256(run_path.read_bytes()).hexdigest()
                    by_order_variant_seed[order_id][variant][seed] = scalar_summary
                    run_manifest.append(
                        {
                            "order_id": order_id,
                            "order_fingerprint_sha256": fingerprints[order_id],
                            "seed": seed,
                            "variant": variant,
                            "status": "complete",
                            "artifact": str(run_path),
                            "artifact_sha256": artifact_sha256,
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "experiment_id": H1A_ORDER_EXPERIMENT_ID,
                                "order_id": order_id,
                                "variant": variant,
                                "seed": seed,
                                "result": str(run_path),
                                "status": "complete",
                            },
                            sort_keys=True,
                            allow_nan=False,
                        )
                    )
                except Exception as exc:
                    invalid_numerics = isinstance(exc, NumericalInstabilityError)
                    failure_payload = {
                        "experiment_id": H1A_ORDER_EXPERIMENT_ID,
                        "status": (
                            "INVALID_NUMERICAL_INSTABILITY"
                            if invalid_numerics
                            else "INVALID_EXECUTION_OR_ARTIFACT_FAILURE"
                        ),
                        "order_id": order_id,
                        "order_fingerprint_sha256": fingerprints[order_id],
                        "class_order": list(order),
                        "variant": variant,
                        "seed": seed,
                        "feature_bundle_sha256": feature_audit.sha256,
                        "environment": environment,
                        "error_type": type(exc).__name__,
                        "error": (
                            exc.to_dict()
                            if isinstance(exc, NumericalInstabilityError)
                            else str(exc)
                        ),
                        "completed_runs_before_failure": len(run_manifest),
                        "expected_runs": expected_runs,
                    }
                    failure_path = args.output / (
                        f"failure__order-{order_id}__{variant}__seed-{seed}.json"
                    )
                    _dump_json(failure_path, failure_payload)
                    raise RuntimeError(
                        f"{H1A_ORDER_EXPERIMENT_ID} invalidated/aborted; "
                        f"failure details: {failure_path}"
                    ) from exc

    if len(run_manifest) != expected_runs:
        raise RuntimeError(
            f"incomplete run matrix: expected {expected_runs}, got {len(run_manifest)}"
        )
    run_keys = {
        (row["order_id"], int(row["seed"]), row["variant"])
        for row in run_manifest
    }
    expected_keys = {
        (order_id, seed, variant)
        for order_id in H1A_ORDER_IDS
        for seed in H1A_ORDER_SEEDS
        for variant in H1A_ORDER_VARIANTS
    }
    if run_keys != expected_keys:
        raise RuntimeError("completed run manifest does not exactly match the registered matrix")

    per_order_comparisons: dict[str, dict[str, Any]] = {}
    order_mean_deltas: dict[str, dict[str, float]] = {
        metric: {} for metric in CO_PRIMARY_METRICS
    }
    for order_index, order_id in enumerate(H1A_ORDER_IDS):
        left = by_order_variant_seed[order_id]["progressive_reserve"]
        right = by_order_variant_seed[order_id]["all_recruited"]
        per_order_comparisons[order_id] = {}
        for metric in CO_PRIMARY_METRICS:
            comparison = _paired_comparison(
                left,
                right,
                metric=metric,
                bootstrap_seed=ORDER_BOOTSTRAP_SEEDS[order_id],
            )
            per_order_comparisons[order_id][metric] = comparison
            order_mean_deltas[metric][order_id] = float(comparison["mean_delta_pp"])

    pooled_comparisons = {
        metric: _cluster_pooled_comparison(
            per_order_comparisons,
            metric=metric,
        )
        for metric in CO_PRIMARY_METRICS
    }
    pooled_cis = {
        metric: pooled_comparisons[metric]["ci95_pp"]
        for metric in CO_PRIMARY_METRICS
    }
    sensitivity_verdict = classify_order_sensitivity(
        pooled_cis,
        order_mean_deltas,
    )

    variant_descriptives: dict[str, Any] = {}
    for metric in CO_PRIMARY_METRICS:
        variant_descriptives[metric] = {
            order_id: {
                variant: _describe(
                    [
                        _metric_value(
                            by_order_variant_seed[order_id][variant][seed],
                            metric,
                        )
                        for seed in H1A_ORDER_SEEDS
                    ]
                )
                for variant in H1A_ORDER_VARIANTS
            }
            for order_id in H1A_ORDER_IDS
        }

    aggregate = {
        "experiment_id": H1A_ORDER_EXPERIMENT_ID,
        "parent_primary_experiment": "EXP-H1A-002",
        "milestone": "M1.3a",
        "status": "COMPLETE_NUMERICALLY_VALID",
        "registered_order_sensitivity_verdict": sensitivity_verdict,
        "protocol": protocol.as_dict(),
        "feature_bundle_metadata": feature_metadata,
        "feature_bundle_sha256": feature_audit.sha256,
        "feature_bundle_audit": feature_audit.to_dict(),
        "environment": environment,
        "class_orders": {
            order_id: {
                "classes": list(CLASS_ORDERS[order_id]),
                "fingerprint_sha256": fingerprints[order_id],
            }
            for order_id in H1A_ORDER_IDS
        },
        "seeds": list(H1A_ORDER_SEEDS),
        "variants": list(H1A_ORDER_VARIANTS),
        "expected_runs": expected_runs,
        "completed_runs": len(run_manifest),
        "run_manifest": run_manifest,
        "per_order_variant_descriptives_pp": variant_descriptives,
        "per_order_paired_comparisons": per_order_comparisons,
        "pooled_seed_cluster_comparisons": pooled_comparisons,
        "descriptive_secondary_metrics": {
            metric: _secondary_descriptives(
                by_order_variant_seed,
                metric=metric,
            )
            for metric in ("average_forgetting", "backward_transfer")
        },
        "registered_classification_inputs": {
            "pooled_cis_pp": pooled_cis,
            "order_mean_deltas_pp": order_mean_deltas,
        },
        "protocol_note": (
            "This sensitivity result does not alter the frozen EXP-H1A-002 "
            "primary verdict, which remains INCONCLUSIVE."
        ),
    }
    _dump_json(args.output / "aggregate.json", aggregate)
    print(
        json.dumps(
            {
                "aggregate": str(args.output / "aggregate.json"),
                "experiment_id": H1A_ORDER_EXPERIMENT_ID,
                "completed_runs": len(run_manifest),
                "order_sensitivity_verdict": sensitivity_verdict,
                "primary_exp_h1a_002_verdict": "INCONCLUSIVE",
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
