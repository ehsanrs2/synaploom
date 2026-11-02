"""M1.3b D2 pre-outcome evaluation policy and fail-closed endpoint guard.

Research-only validation for a FUTURE scientific runner, not a replacement of
the production evaluator, and not evidence of learning gains. All inputs are
JSON-like observations. Future runner integration remains explicitly NO-GO.
"""
from __future__ import annotations

from hashlib import sha256
import math
from typing import Any, Mapping

from synaploom.protocols.h1a2 import h1a2_protocol
from synaploom.research.m13b_manifest import canonical_json

KIND = "M13B_D2_EVAL_POLICY_AND_FAIL_CLOSED_ENDPOINTS_V1"
SOURCE_REFERENCE_COMMIT = "8eefa0a143ba02708fc4e1185003237df861e4de"
METRICS = ("final_average_accuracy", "late_task_within_task_mean_accuracy")
EXPECTED_TASKS = 10
EXPECTED_EPOCHS = 20
PRIMARY_REFERENCE_EVAL_BATCH_SIZE = 256


def evaluation_policy_candidate() -> dict[str, Any]:
    """Read-only manifest of existing behavior; no permission or new science."""
    protocol = h1a2_protocol()
    if not (
        protocol.num_classes == 100
        and protocol.classes_per_task == 10
        and protocol.epochs_per_task == EXPECTED_EPOCHS
        and protocol.batch_size == 128
        and protocol.eval_batch_size == PRIMARY_REFERENCE_EVAL_BATCH_SIZE
        and protocol.current_task_eval_every_epochs == 1
    ):
        raise RuntimeError("base H1a2 evaluation protocol changed")
    p: dict[str, Any] = {
        "schema": 1,
        "kind": KIND,
        "status": "CPU_POLICY_GUARD_CANDIDATE__NOT_FROZEN_OR_EXECUTABLE",
        "source_reference_not_execution_attestation": SOURCE_REFERENCE_COMMIT,
        "primary_order": "O0",
        "candidate_primary_arms": ["progressive_reserve", "fixed_sparse_mlp"],
        "co_primary_metrics": list(METRICS),
        "task_count": EXPECTED_TASKS,
        "epochs_per_task": EXPECTED_EPOCHS,
        "train_batch_size": protocol.batch_size,
        "eval_batch_size": protocol.eval_batch_size,
        "evaluation_shuffle": False,
        "evaluation_drop_last": False,
        "evaluation_partial_final_batch": "INCLUDED",
        "evaluate_future_tasks": False,
        "restrict_logits_to_seen_classes": True,
        "current_task_eval_every_epochs": 1,
        "routing": "LEGACY_BATCH_MEAN_TOP_K__BATCH_COMPOSITION_DEPENDENT",
        "accuracy_matrix_required": "10x10_lower_triangle_finite_[0,1]_upper_triangle_null",
        "learning_curves_required": "10_tasks_x_20_epochs_complete_finite_[0,1]",
        "missing_or_invalid_observation_policy": "FAIL_CLOSED__NO_VALID_CO_PRIMARY_VERDICT",
        "inference_or_scientific_outcome_claim": "NONE",
        "gpu_executable": False,
        "protected_eval_accessed": False,
        "owner_science_execution_approved": False,
        "runner_integrated": False,
        "mandatory_preoutcome_checks": [
            "Confirm the actual final runner uses the same fixed eval policy.",
            "Declare and bound batch-composition sensitivity on synthetic/nonprotected inputs.",
            "Do not choose evaluation settings from protected accuracy outcomes.",
            "Fail closed on any missing/nonfinite/out-of-range required accuracy point.",
            "Pin source, class order, feature bundle, fixed mask, complete seed registry and retry policy separately.",
        ],
    }
    p["record_sha256"] = sha256(canonical_json(p)).hexdigest()
    return p


def _accuracy_fraction(value: Any, where: str) -> float:
    """Require finite JSON numeric accuracy in [0,1], never None or booleans."""
    if type(value) not in (int, float):
        raise ValueError(f"{where} must be a numeric accuracy fraction")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{where} must be finite and in [0,1]")
    return out


def _nonnegative_float(value: Any, where: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{where} must be numeric")
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{where} must be finite nonnegative")
    return out


def validate_co_primary_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Check a COMPLETE candidate 10-task/20-epoch result and compute endpoints.

    Standalone, read-only pre-science guard NOT yet connected to science runner.
    Reject missing lower triangle values that legacy torch.nanmean would omit.
    Reject absent epochs in learning curves; no silent denominator changes.

    Input uses JSON-ready ContinualRunResult.to_dict() structure. This validates
    completeness, not provenance or inferential validity, which are separate.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("observation must be a mapping")

    config = payload.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError("missing run_config")
    expected_config: dict[str, Any] = {
        "epochs_per_task": EXPECTED_EPOCHS,
        "batch_size": 128,
        "eval_batch_size": PRIMARY_REFERENCE_EVAL_BATCH_SIZE,
        "evaluate_future_tasks": False,
        "restrict_logits_to_seen_classes": True,
        "current_task_eval_every_epochs": 1,
    }
    for field, expected in expected_config.items():
        value = config.get(field)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"unapproved evaluation/training config: {field}")

    matrix = payload.get("accuracy_matrix")
    if not isinstance(matrix, list) or len(matrix) != EXPECTED_TASKS:
        raise ValueError("required accuracy matrix must have exactly ten rows")
    for stage, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != EXPECTED_TASKS:
            raise ValueError("accuracy matrix must be 10x10")
        for task_index, value in enumerate(row):
            if task_index <= stage:
                _accuracy_fraction(value, f"accuracy_matrix[{stage}][{task_index}]")
            elif value is not None:
                raise ValueError("future-task accuracy must be explicit null")

    curves = payload.get("current_task_learning_curves")
    if not isinstance(curves, list) or len(curves) != EXPECTED_TASKS:
        raise ValueError("expected one complete learning curve per task")
    task_means: list[float] = []
    previous_global_step = -1
    for task_index, curve in enumerate(curves):
        if not isinstance(curve, list) or len(curve) != EXPECTED_EPOCHS:
            raise ValueError(f"task {task_index} must have 20 evaluated epochs")
        points: list[float] = []
        for idx, point in enumerate(curve):
            if not isinstance(point, Mapping):
                raise ValueError("learning curve point must be a mapping")
            epoch = point.get("epoch")
            if type(epoch) is not int or epoch != idx + 1:
                raise ValueError("missing or misordered epoch")
            step = point.get("global_step")
            if type(step) is not int or step <= previous_global_step:
                raise ValueError("global steps must increase for all epochs/tasks")
            previous_global_step = step
            points.append(_accuracy_fraction(point.get("accuracy"), "epoch accuracy"))
            _nonnegative_float(point.get("loss"), "epoch evaluation loss")
        task_means.append(sum(points) / EXPECTED_EPOCHS)

    final_mean = sum(_accuracy_fraction(x, "final row") for x in matrix[-1]) / EXPECTED_TASKS
    late_mean = (task_means[-2] + task_means[-1]) / 2
    endpoints = {
        "final_average_accuracy": final_mean,
        "late_task_within_task_mean_accuracy": late_mean,
    }

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("missing summary for endpoint cross-check")
    for name in METRICS:
        declared = _accuracy_fraction(summary.get(name), f"summary.{name}")
        if not math.isclose(declared, endpoints[name], rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(f"reported {name} does not match complete raw observations")

    return {
        "kind": "M13B_D2_EVAL_ENDPOINT_OBSERVATION_VALIDATION_V1",
        "status": "ENDPOINT_COMPLETENESS_PASS__NO_SCIENTIFIC_INFERENCE",
        "reference_eval_batch_size": PRIMARY_REFERENCE_EVAL_BATCH_SIZE,
        "task_count": EXPECTED_TASKS,
        "epochs_per_task": EXPECTED_EPOCHS,
        "co_primary_metric_values": endpoints,
        "science_executable": False,
        "scientific_claim": "NONE",
    }
