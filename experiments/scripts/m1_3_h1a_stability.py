from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch

from synaploom.data.features import make_split_feature_stream
from synaploom.protocols.h1a import H1A_CLASS_ORDER, H1AProtocol, build_h1a_variant
from synaploom.protocols.h1a_stability import H1AStabilityProtocol
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Numerical-stability calibration for the invalid EXP-H1A-001 run."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/m1_3_h1a_stability"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    calibration = H1AStabilityProtocol()
    base_protocol = H1AProtocol()
    feature_audit = audit_h1a_feature_bundle(args.features)
    stream = make_split_feature_stream(
        args.features,
        class_order=H1A_CLASS_ORDER,
        classes_per_task=base_protocol.classes_per_task,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    environment = _environment(args.device)
    candidate_summaries: list[dict[str, object]] = []
    selected_learning_rate: float | None = None

    for learning_rate in calibration.learning_rates:
        protocol = replace(
            base_protocol,
            learning_rate=float(learning_rate),
            seeds=calibration.seeds,
        )
        records: list[dict[str, object]] = []
        candidate_stable = True

        for seed in calibration.seeds:
            for variant in calibration.variants:
                record: dict[str, object] = {
                    "learning_rate": float(learning_rate),
                    "seed": int(seed),
                    "variant": variant,
                    "status": "pending",
                }
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
                        config=protocol.continual_config(
                            seed=seed,
                            device=args.device,
                        ),
                        engine=engine,
                        metadata={
                            "experiment_id": calibration.experiment_id,
                            "purpose": "numerical-stability-only",
                            "learning_rate": float(learning_rate),
                            "seed": int(seed),
                            "variant": variant,
                            "feature_bundle_sha256": feature_audit.sha256,
                            "environment": environment,
                            "build": build_meta,
                        },
                    )
                    result_path = (
                        args.output
                        / f"lr-{learning_rate:g}__{variant}__seed-{seed}.json"
                    )
                    save_continual_result(result, result_path)
                    record.update(
                        {
                            "status": "stable",
                            "result": str(result_path),
                            "max_gradient_norm_by_task": result.max_gradient_norm_by_task,
                            "max_parameter_abs_by_task": result.max_parameter_abs_by_task,
                        }
                    )
                except NumericalInstabilityError as exc:
                    candidate_stable = False
                    failure_path = (
                        args.output
                        / f"lr-{learning_rate:g}__{variant}__seed-{seed}__failure.json"
                    )
                    failure = {
                        **record,
                        "status": "numerical_failure",
                        "error": exc.to_dict(),
                        "feature_bundle_sha256": feature_audit.sha256,
                        "environment": environment,
                    }
                    failure_path.write_text(
                        json.dumps(failure, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    record.update(
                        {
                            "status": "numerical_failure",
                            "failure": str(failure_path),
                            "error": exc.to_dict(),
                        }
                    )
                records.append(record)
                print(json.dumps(record, sort_keys=True))

        summary = {
            "learning_rate": float(learning_rate),
            "stable": candidate_stable,
            "stable_runs": sum(record["status"] == "stable" for record in records),
            "failed_runs": sum(
                record["status"] == "numerical_failure" for record in records
            ),
            "expected_runs": calibration.expected_runs_per_learning_rate,
            "records": records,
        }
        candidate_summaries.append(summary)

        if candidate_stable:
            selected_learning_rate = float(learning_rate)
            break

    aggregate = {
        "experiment_id": calibration.experiment_id,
        "purpose": "numerical-stability-calibration-only",
        "feature_bundle_sha256": feature_audit.sha256,
        "environment": environment,
        "calibration": asdict(calibration),
        "selection_rule": calibration.selection_rule(),
        "candidate_summaries": candidate_summaries,
        "selected_learning_rate": selected_learning_rate,
        "scientific_metrics_used_for_selection": False,
        "h1a_scientific_conclusion": "NOT_EVALUATED",
    }
    path = args.output / "aggregate.json"
    path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "aggregate": str(path),
        "selected_learning_rate": selected_learning_rate,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
