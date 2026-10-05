from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    import numpy as np
except ImportError:
    np = None
from torch import Tensor, nn

from synaploom.data.streams import ContinualStream, iter_tensor_batches
from synaploom.training.class_incremental import class_incremental_cross_entropy
from synaploom.training.continual_metrics import (
    average_accuracy_by_stage,
    average_forgetting,
    backward_transfer,
    diagonal_learning_accuracy,
    final_average_accuracy,
    late_task_learning_accuracy,
    matrix_to_jsonable,
)
from synaploom.training.engine import TrainingEngine
from synaploom.training.evaluation import evaluate_task, logits_from_output


@dataclass(frozen=True)
class ContinualConfig:
    epochs_per_task: int = 1
    batch_size: int = 32
    eval_batch_size: int = 128
    seed: int = 0
    device: str = "cpu"
    evaluate_future_tasks: bool = False
    apply_final_task_boundary: bool = False
    restrict_logits_to_seen_classes: bool = False

    def __post_init__(self) -> None:
        if self.epochs_per_task <= 0:
            raise ValueError("epochs_per_task must be positive")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")


@dataclass(frozen=True)
class StructuralSnapshot:
    stage: int
    task_id: int
    structural_version: int | None
    recruited_nodes: int | None
    active_edges: int | None
    mean_executed_nodes: float | None
    selection_entropy: float | None
    max_selection_share: float | None
    node_selection_counts: dict[int, int]
    node_utilization_ema: dict[int, float] = field(default_factory=dict)


@dataclass
class ContinualRunResult:
    accuracy_matrix: Tensor
    eval_loss_matrix: Tensor
    train_loss_by_task: list[float]
    structural_history: list[StructuralSnapshot]
    global_steps: int
    run_config: dict[str, Any] = field(default_factory=dict)
    stream_manifest: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "final_average_accuracy": final_average_accuracy(self.accuracy_matrix),
            "average_forgetting": average_forgetting(self.accuracy_matrix),
            "backward_transfer": backward_transfer(self.accuracy_matrix),
            "late_task_learning_accuracy": late_task_learning_accuracy(
                self.accuracy_matrix, last_n=min(2, self.accuracy_matrix.shape[0])
            ),
            "average_accuracy_by_stage": average_accuracy_by_stage(
                self.accuracy_matrix
            ).detach().cpu().tolist(),
            "diagonal_learning_accuracy": diagonal_learning_accuracy(
                self.accuracy_matrix
            ).detach().cpu().tolist(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy_matrix": matrix_to_jsonable(self.accuracy_matrix),
            "eval_loss_matrix": matrix_to_jsonable(self.eval_loss_matrix),
            "train_loss_by_task": [float(v) for v in self.train_loss_by_task],
            "structural_history": [asdict(item) for item in self.structural_history],
            "global_steps": int(self.global_steps),
            "run_config": self.run_config,
            "stream_manifest": self.stream_manifest,
            "metadata": self.metadata,
            "summary": self.summary(),
        }


def save_continual_result(result: ContinualRunResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _selection_stats(
    counts: Counter[int], executed_sizes: list[int]
) -> tuple[float | None, float | None, float | None]:
    if not counts:
        return None, None, None
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probabilities)
    max_share = max(probabilities)
    mean_executed = (
        sum(executed_sizes) / len(executed_sizes) if executed_sizes else None
    )
    return mean_executed, entropy, max_share


def _snapshot(
    model: nn.Module,
    *,
    stage: int,
    task_id: int,
    counts: Counter[int],
    executed_sizes: list[int],
) -> StructuralSnapshot:
    graph_state = getattr(model, "graph_state", None)
    mean_executed, entropy, max_share = _selection_stats(counts, executed_sizes)
    if graph_state is None:
        return StructuralSnapshot(
            stage=stage,
            task_id=task_id,
            structural_version=None,
            recruited_nodes=None,
            active_edges=None,
            mean_executed_nodes=mean_executed,
            selection_entropy=entropy,
            max_selection_share=max_share,
            node_selection_counts=dict(sorted(counts.items())),
            node_utilization_ema={},
        )
    recruited_ids = graph_state.recruited_ids().detach().cpu().tolist()
    utilization = graph_state.node_utilization_ema.detach().cpu()
    return StructuralSnapshot(
        stage=stage,
        task_id=task_id,
        structural_version=int(graph_state.structural_version.item()),
        recruited_nodes=graph_state.recruited_count(),
        active_edges=graph_state.active_edge_count(),
        mean_executed_nodes=mean_executed,
        selection_entropy=entropy,
        max_selection_share=max_share,
        node_selection_counts=dict(sorted(counts.items())),
        node_utilization_ema={
            int(node_id): float(utilization[node_id].item())
            for node_id in recruited_ids
        },
    )


def run_continual_experiment(
    model: nn.Module,
    stream: ContinualStream,
    optimizer: torch.optim.Optimizer,
    *,
    config: ContinualConfig | None = None,
    engine: TrainingEngine | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContinualRunResult:
    """Train sequentially and evaluate the standard stage-by-task matrix."""
    config = config or ContinualConfig()
    _set_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    if engine is not None:
        if engine.model is not model:
            raise ValueError("engine.model must be the model passed to run_continual_experiment")
        if engine.optimizer is not optimizer:
            raise ValueError("engine.optimizer must be the optimizer passed to run_continual_experiment")

    num_tasks = len(stream)
    accuracy = torch.full((num_tasks, num_tasks), float("nan"))
    eval_loss = torch.full((num_tasks, num_tasks), float("nan"))
    train_loss_by_task: list[float] = []
    structural_history: list[StructuralSnapshot] = []
    global_step = 0

    seen_classes: list[int] = []
    for stage, task in enumerate(stream):
        seen_classes.extend(int(v) for v in task.classes)
        allowed_classes = tuple(sorted(set(seen_classes))) if config.restrict_logits_to_seen_classes else None
        losses: list[float] = []
        selection_counts: Counter[int] = Counter()
        executed_sizes: list[int] = []

        for epoch in range(config.epochs_per_task):
            num_batches = (int(task.train_x.shape[0]) + config.batch_size - 1) // config.batch_size
            batches = iter_tensor_batches(
                task.train_x,
                task.train_y,
                batch_size=config.batch_size,
                shuffle=True,
                seed=config.seed + stage * 10007 + epoch,
            )
            for batch_index, (x, y) in enumerate(batches):
                x, y = x.to(device), y.to(device)
                is_last_batch = (
                    epoch == config.epochs_per_task - 1
                    and batch_index == num_batches - 1
                )
                boundary = is_last_batch and (
                    stage < num_tasks - 1 or config.apply_final_task_boundary
                )

                if engine is not None:
                    result = engine.train_step(
                        x,
                        y,
                        step=global_step,
                        task_boundary=boundary,
                        allowed_classes=allowed_classes,
                    )
                    losses.append(result.loss)
                    trace = result.trace
                    ids = [int(v) for v in trace.selected_node_ids.tolist()]
                else:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    logits = logits_from_output(output)
                    aux_loss = getattr(output, "aux_loss", None)
                    loss = class_incremental_cross_entropy(
                        logits,
                        y,
                        allowed_classes=allowed_classes,
                    )
                    if isinstance(aux_loss, Tensor):
                        loss = loss + aux_loss
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach().item()))
                    trace = getattr(output, "trace", None)
                    if trace is not None and hasattr(model, "graph_state"):
                        model.graph_state.update_utilization(
                            trace.selected_node_ids,
                            trace.router_weights,
                        )
                    ids = (
                        [int(v) for v in trace.selected_node_ids.detach().cpu().tolist()]
                        if trace is not None
                        else []
                    )

                if ids:
                    selection_counts.update(ids)
                    executed_sizes.append(len(ids))
                global_step += 1

        train_loss_by_task.append(sum(losses) / max(1, len(losses)))
        structural_history.append(
            _snapshot(
                model,
                stage=stage,
                task_id=task.task_id,
                counts=selection_counts,
                executed_sizes=executed_sizes,
            )
        )

        eval_limit = num_tasks if config.evaluate_future_tasks else stage + 1
        for eval_task_index in range(eval_limit):
            result = evaluate_task(
                model,
                stream[eval_task_index],
                batch_size=config.eval_batch_size,
                device=device,
                allowed_classes=allowed_classes,
            )
            accuracy[stage, eval_task_index] = result.accuracy
            eval_loss[stage, eval_task_index] = result.loss

    stream_manifest = [
        {
            "task_id": int(task.task_id),
            "classes": [int(v) for v in task.classes],
            "train_examples": int(task.train_y.numel()),
            "eval_examples": int(task.eval_y.numel()),
        }
        for task in stream
    ]
    return ContinualRunResult(
        accuracy_matrix=accuracy,
        eval_loss_matrix=eval_loss,
        train_loss_by_task=train_loss_by_task,
        structural_history=structural_history,
        global_steps=global_step,
        run_config=asdict(config),
        stream_manifest=stream_manifest,
        metadata=dict(metadata or {}),
    )
