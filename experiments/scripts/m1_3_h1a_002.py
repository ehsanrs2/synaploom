from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch

from synaploom.data.features import load_feature_bundle, make_split_feature_stream
from synaploom.protocols.h1a import H1A_CLASS_ORDER, H1A_VARIANTS, build_h1a_variant
from synaploom.protocols.h1a2 import (
    H1A2_EXPERIMENT_ID,
    H1A2_FEATURE_SHA256,
    h1a2_protocol,
)
from synaploom.research.budgets import (
    count_trainable_parameters,
    estimate_dense_linear_macs_per_batch,
    estimate_synaploom_linear_macs_per_batch,
    synaploom_budget,
)
from synaploom.research.preflight import audit_h1a_feature_bundle
from synaploom.training.continual import run_continual_experiment, save_continual_result
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


def _aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None
        ]
        if not values:
            continue
        out[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    return out


def _paired_bootstrap(
    left: dict[int, dict[str, float]],
    right: dict[int, dict[str, float]],
    *,
    metric: str,
    samples: int = 10_000,
    seed: int = 1337,
) -> dict[str, object]:
    shared = sorted(set(left).intersection(right))
    valid = [
        s for s in shared
        if metric in left[s] and metric in right[s]
    ]
    deltas = [float(left[s][metric] - right[s][metric]) for s in valid]
    if not deltas:
        return {
            "metric": metric,
            "n": 0,
            "mean_delta": None,
            "ci95": None,
            "per_seed": {},
        }

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(statistics.fmean(draw))
    means.sort()

    lo = means[int(0.025 * (samples - 1))]
    hi = means[int(0.975 * (samples - 1))]
    return {
        "metric": metric,
        "n": len(deltas),
        "mean_delta": statistics.fmean(deltas),
        "ci95": [lo, hi],
        "per_seed": {
            str(s): float(left[s][metric] - right[s][metric])
            for s in valid
        },
    }


def _primary_metric_verdict(result: dict[str, object]) -> str:
    ci = result.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return "invalid"
    lo, hi = float(ci[0]), float(ci[1])
    if lo > 0.0:
        return "favorable"
    if hi < 0.0:
        return "unfavorable"
    return "inconclusive"


def _overall_verdict(primary: dict[str, dict[str, object]]) -> str:
    verdicts = [
        _primary_metric_verdict(primary["final_average_accuracy"]),
        _primary_metric_verdict(primary["late_task_within_task_mean_accuracy"]),
    ]
    if verdicts == ["favorable", "favorable"]:
        return "FAVORABLE"
    if verdicts == ["unfavorable", "unfavorable"]:
        return "UNFAVORABLE"
    if "invalid" in verdicts:
        return "INVALID"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-H1A-002: preregistered H1a rerun after stability calibration."
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Frozen feature bundle (required unless --dry-run is used).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/m1_3_h1a_002"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen protocol without reading data or training.",
    )
    args = parser.parse_args()
    if not args.dry_run and args.features is None:
        parser.error("--features is required unless --dry-run is used")

    protocol = h1a2_protocol()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "experiment_id": H1A2_EXPERIMENT_ID,
                    "protocol": protocol.as_dict(),
                    "feature_sha256": H1A2_FEATURE_SHA256,
                    "variants": list(H1A_VARIANTS),
                    "expected_runs": len(protocol.seeds) * len(H1A_VARIANTS),
                    "overall_rule": (
                        "FAVORABLE iff both co-primary CIs > 0; "
                        "UNFAVORABLE iff both < 0; otherwise INCONCLUSIVE; "
                        "INVALID on numerical/artifact failure"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    feature_audit = audit_h1a_feature_bundle(args.features)
    if feature_audit.sha256 != H1A2_FEATURE_SHA256:
        raise ValueError(
            "EXP-H1A-002 requires the frozen feature artifact SHA-256 "
            f"{H1A2_FEATURE_SHA256}, got {feature_audit.sha256}"
        )
    bundle = load_feature_bundle(args.features)
    feature_metadata = dict(bundle.get("metadata", {}))
    stream = make_split_feature_stream(
        args.features,
        class_order=H1A_CLASS_ORDER,
        classes_per_task=protocol.classes_per_task,
    )

    if tuple(stream.feature_shape) != (protocol.feature_dim,):
        raise ValueError(
            f"feature dimension mismatch: stream={tuple(stream.feature_shape)} "
            f"protocol={(protocol.feature_dim,)}"
        )
    if len(stream) != 10 or stream.num_classes != protocol.num_classes:
        raise ValueError("EXP-H1A-002 stream must contain 10 tasks and 100 classes")

    environment = _environment(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    aggregate_rows: dict[str, list[dict[str, float]]] = {
        variant: [] for variant in H1A_VARIANTS
    }
    by_seed: dict[str, dict[int, dict[str, float]]] = {
        variant: {} for variant in H1A_VARIANTS
    }

    for seed in protocol.seeds:
        for variant in H1A_VARIANTS:
            model, optimizer, engine, build_meta = build_h1a_variant(
                variant,
                stream=stream,
                seed=seed,
                protocol=protocol,
            )
            run_config = protocol.continual_config(
                seed=seed,
                device=args.device,
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

            try:
                result = run_continual_experiment(
                    model,
                    stream,
                    optimizer,
                    config=run_config,
                    engine=engine,
                    metadata={
                        "experiment_id": H1A2_EXPERIMENT_ID,
                        "milestone": "M1.3",
                        "hypothesis": "H1a-controlled-reserve-recruitment",
                        "variant": variant,
                        "seed": seed,
                        "protocol": protocol.as_dict(),
                        "class_order": list(H1A_CLASS_ORDER),
                        "feature_bundle": str(args.features),
                        "feature_bundle_metadata": feature_metadata,
                        "feature_bundle_sha256": feature_audit.sha256,
                        "feature_bundle_audit": feature_audit.to_dict(),
                        "environment": environment,
                        "build": build_meta,
                        "initial_total_parameters": count_trainable_parameters(model),
                        "initial_budget": (
                            asdict(initial_budget)
                            if initial_budget is not None
                            else None
                        ),
                        "initial_linear_macs_per_batch_estimate": initial_macs,
                        "optimizer_calibration_source": "EXP-H1A-STABILITY-001",
                    },
                )
            except NumericalInstabilityError as exc:
                failure_path = (
                    args.output
                    / f"{variant}__seed-{seed}__failure.json"
                )
                failure_path.write_text(
                    json.dumps(
                        {
                            "experiment_id": H1A2_EXPERIMENT_ID,
                            "status": "INVALID_NUMERICAL_INSTABILITY",
                            "variant": variant,
                            "seed": seed,
                            "feature_bundle_sha256": feature_audit.sha256,
                            "environment": environment,
                            "error": exc.to_dict(),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    "EXP-H1A-002 aborted on numerical instability; "
                    f"details: {failure_path}"
                ) from exc

            if hasattr(model, "graph_state"):
                result.metadata["final_budget"] = asdict(synaploom_budget(model))
                result.metadata["final_linear_macs_per_batch_estimate"] = (
                    estimate_synaploom_linear_macs_per_batch(
                        model,
                        batch_size=protocol.batch_size,
                    )
                )

            path = args.output / f"{variant}__seed-{seed}.json"
            save_continual_result(result, path)

            summary = result.summary()
            scalar_summary = {
                key: float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float)) and value is not None
            }
            aggregate_rows[variant].append(scalar_summary)
            by_seed[variant][seed] = scalar_summary

            print(
                json.dumps(
                    {
                        "experiment_id": H1A2_EXPERIMENT_ID,
                        "variant": variant,
                        "seed": seed,
                        "result": str(path),
                        "status": "complete",
                    },
                    sort_keys=True,
                )
            )

    primary_metrics = (
        "final_average_accuracy",
        "late_task_within_task_mean_accuracy",
        "average_forgetting",
        "backward_transfer",
    )
    primary_comparison = {
        metric: _paired_bootstrap(
            by_seed["progressive_reserve"],
            by_seed["all_recruited"],
            metric=metric,
        )
        for metric in primary_metrics
    }

    primary_verdicts = {
        "final_average_accuracy": _primary_metric_verdict(
            primary_comparison["final_average_accuracy"]
        ),
        "late_task_within_task_mean_accuracy": _primary_metric_verdict(
            primary_comparison["late_task_within_task_mean_accuracy"]
        ),
    }
    overall = _overall_verdict(primary_comparison)

    aggregate = {
        "experiment_id": H1A2_EXPERIMENT_ID,
        "milestone": "M1.3",
        "hypothesis": "H1a-controlled-reserve-recruitment",
        "protocol": protocol.as_dict(),
        "class_order": list(H1A_CLASS_ORDER),
        "feature_bundle_metadata": feature_metadata,
        "feature_bundle_sha256": feature_audit.sha256,
        "feature_bundle_audit": feature_audit.to_dict(),
        "environment": environment,
        "variants": {
            variant: _aggregate(rows)
            for variant, rows in aggregate_rows.items()
        },
        "paired_comparisons": {
            "progressive_minus_all_recruited": primary_comparison
        },
        "preregistered_primary_verdicts": primary_verdicts,
        "preregistered_overall_h1a_verdict": overall,
    }

    aggregate_path = args.output / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "aggregate": str(aggregate_path),
                "experiment_id": H1A2_EXPERIMENT_ID,
                "overall_h1a_verdict": overall,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
