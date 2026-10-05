from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable

import torch
from torch import nn

from synaploom.benchmarks.active_budget import synaploom_capacity
from synaploom.benchmarks.budget import budget_report, nearest_dense_hidden_width
from synaploom.benchmarks.h1 import H1Protocol
from synaploom.benchmarks.models import (
    H1ModelShape,
    make_all_recruited_routed_model,
    make_dense_model,
    make_growth_model,
    make_routed_no_growth_model,
)
from synaploom.data.cifar100 import make_split_cifar100_stream
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import NoConsolidation, NoEdges, TaskBoundaryRecruitment
from synaploom.training.continual import ContinualConfig, ContinualRunResult, run_continual_experiment
from synaploom.training.engine import TrainingEngine


@dataclass(frozen=True)
class RuntimeProfile:
    wall_seconds: float
    peak_cuda_bytes: int | None


@dataclass(frozen=True)
class ModelRun:
    name: str
    seed: int
    result: ContinualRunResult
    runtime: RuntimeProfile
    budget: dict[str, Any]


def _optimizer(model: nn.Module, protocol: H1Protocol) -> torch.optim.Optimizer:
    if protocol.optimizer.lower() != "adamw":
        raise ValueError("M1.3 currently freezes optimizer to AdamW")
    return torch.optim.AdamW(
        model.parameters(),
        lr=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
    )


def _run_once(
    name: str,
    model: nn.Module,
    stream,
    protocol: H1Protocol,
    *,
    controller: StructuralController | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelRun:
    optimizer = _optimizer(model, protocol)
    engine = TrainingEngine(model, optimizer, controller=controller) if controller else None

    if torch.cuda.is_available() and str(protocol_device(protocol)).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    result = run_continual_experiment(
        model,
        stream,
        optimizer,
        config=ContinualConfig(
            epochs_per_task=protocol.epochs_per_task,
            batch_size=protocol.batch_size,
            eval_batch_size=protocol.eval_batch_size,
            seed=metadata["seed"] if metadata else 0,
            device=str(protocol_device(protocol)),
        ),
        engine=engine,
        metadata=metadata,
    )
    wall = time.perf_counter() - started
    peak = None
    if torch.cuda.is_available() and str(protocol_device(protocol)).startswith("cuda"):
        peak = int(torch.cuda.max_memory_allocated())

    return ModelRun(
        name=name,
        seed=int(metadata["seed"] if metadata else 0),
        result=result,
        runtime=RuntimeProfile(wall_seconds=wall, peak_cuda_bytes=peak),
        budget=asdict(budget_report(model)),
    )


def protocol_device(protocol: H1Protocol) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dense_matches(shape: H1ModelShape) -> dict[str, tuple[int, int]]:
    reference = make_growth_model(shape)
    capacity = synaploom_capacity(
        reference,
        executed_nodes=shape.top_k,
        recruited_nodes=shape.initial_recruited,
    )
    total_width, total_count = nearest_dense_hidden_width(
        lambda width: make_dense_model(shape, d_hidden=width),
        target_parameters=capacity.total_parameters,
        minimum=8,
        maximum=8192,
    )
    executed_width, executed_count = nearest_dense_hidden_width(
        lambda width: make_dense_model(shape, d_hidden=width),
        target_parameters=capacity.estimated_executed_parameters,
        minimum=8,
        maximum=8192,
    )
    return {
        "dense_total_match": (total_width, total_count),
        "dense_executed_match": (executed_width, executed_count),
    }


def run_h1_suite(
    root: str | Path,
    *,
    protocol: H1Protocol,
    shape: H1ModelShape | None = None,
    download: bool = False,
) -> list[ModelRun]:
    shape = shape or H1ModelShape()
    stream = make_split_cifar100_stream(
        root,
        num_tasks=protocol.num_tasks,
        class_order_seed=protocol.class_order_seed,
        download=download,
        normalize=protocol.normalize,
    )
    dense_matches = build_dense_matches(shape)

    runs: list[ModelRun] = []
    for seed in protocol.model_seeds:
        base_metadata = {
            "seed": int(seed),
            "protocol": protocol.to_dict(),
            "shape": asdict(shape),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(protocol_device(protocol)),
        }

        torch.manual_seed(seed)
        growth = make_growth_model(shape)
        growth_controller = StructuralController(
            recruitment_policy=TaskBoundaryRecruitment(),
            edge_init_policy=NoEdges(),
            consolidation_policy=NoConsolidation(),
        )
        runs.append(
            _run_once(
                "synaploom_growth",
                growth,
                stream,
                protocol,
                controller=growth_controller,
                metadata={**base_metadata, "condition": "synaploom_growth"},
            )
        )

        torch.manual_seed(seed)
        routed = make_routed_no_growth_model(shape)
        runs.append(
            _run_once(
                "routed_no_growth",
                routed,
                stream,
                protocol,
                metadata={**base_metadata, "condition": "routed_no_growth"},
            )
        )

        torch.manual_seed(seed)
        all_recruited = make_all_recruited_routed_model(shape)
        runs.append(
            _run_once(
                "routed_all_recruited",
                all_recruited,
                stream,
                protocol,
                metadata={**base_metadata, "condition": "routed_all_recruited"},
            )
        )

        for condition, (width, _) in dense_matches.items():
            torch.manual_seed(seed)
            dense = make_dense_model(shape, d_hidden=width)
            runs.append(
                _run_once(
                    condition,
                    dense,
                    stream,
                    protocol,
                    metadata={
                        **base_metadata,
                        "condition": condition,
                        "dense_hidden_width": width,
                    },
                )
            )
    return runs


def aggregate_runs(runs: list[ModelRun]) -> dict[str, Any]:
    grouped: dict[str, list[ModelRun]] = {}
    for run in runs:
        grouped.setdefault(run.name, []).append(run)

    output: dict[str, Any] = {}
    for name, group in grouped.items():
        keys = [
            "final_average_accuracy",
            "average_forgetting",
            "backward_transfer",
            "late_task_learning_accuracy",
            "late_task_learning_gain",
        ]
        metrics: dict[str, Any] = {}
        for key in keys:
            values = [float(item.result.summary()[key]) for item in group]
            metrics[key] = {
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        wall = [item.runtime.wall_seconds for item in group]
        output[name] = {
            "n_seeds": len(group),
            "metrics": metrics,
            "wall_seconds": {
                "mean": mean(wall),
                "std": pstdev(wall) if len(wall) > 1 else 0.0,
                "values": wall,
            },
            "budget": group[0].budget,
        }
    return output


def save_suite(runs: list[ModelRun], path: str | Path) -> None:
    payload = {
        "runs": [
            {
                "name": run.name,
                "seed": run.seed,
                "result": run.result.to_dict(),
                "runtime": asdict(run.runtime),
                "budget": run.budget,
            }
            for run in runs
        ],
        "aggregate": aggregate_runs(runs),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
