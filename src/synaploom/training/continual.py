from __future__ import annotations

import json
import math
import random
import time
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
from synaploom.training.state_digest import model_state_sha256
from synaploom.training.numerics import (
    NumericalContext,
    check_model_gradients,
    check_model_parameters,
    ensure_finite_tensor,
)


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
    current_task_eval_every_epochs: int = 0

    def __post_init__(self) -> None:
        if self.epochs_per_task <= 0:
            raise ValueError("epochs_per_task must be positive")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.current_task_eval_every_epochs < 0:
            raise ValueError("current_task_eval_every_epochs must be non-negative")


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
    current_task_learning_curves: list[list[dict[str, float | int]]] = field(default_factory=list)
    train_seconds_by_task: list[float] = field(default_factory=list)
    eval_seconds_by_stage: list[float] = field(default_factory=list)
    peak_cuda_memory_bytes: int | None = None
    max_gradient_norm_by_task: list[float] = field(default_factory=list)
    max_parameter_abs_by_task: list[float] = field(default_factory=list)
    evaluation_batch_size_results: dict[int, dict[str, Any]] = field(default_factory=dict)
    evaluation_state_hashes: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _summary_for(
        accuracy_matrix: Tensor,
        learning_curves: list[list[dict[str, float | int]]],
    ) -> dict[str, Any]:
        curve_means = [
            sum(float(point["accuracy"]) for point in curve) / len(curve)
            for curve in learning_curves
            if curve
        ]
        late_curve_mean = (
            sum(curve_means[-min(2, len(curve_means)):]) / min(2, len(curve_means))
            if curve_means
            else None
        )
        return {
            "final_average_accuracy": final_average_accuracy(accuracy_matrix),
            "average_forgetting": average_forgetting(accuracy_matrix),
            "backward_transfer": backward_transfer(accuracy_matrix),
            "late_task_learning_accuracy": late_task_learning_accuracy(
                accuracy_matrix, last_n=min(2, accuracy_matrix.shape[0])
            ),
            "average_accuracy_by_stage": average_accuracy_by_stage(
                accuracy_matrix
            ).detach().cpu().tolist(),
            "diagonal_learning_accuracy": diagonal_learning_accuracy(
                accuracy_matrix
            ).detach().cpu().tolist(),
            "within_task_mean_accuracy_by_task": curve_means,
            "late_task_within_task_mean_accuracy": late_curve_mean,
        }

    def summary(self) -> dict[str, Any]:
        return self._summary_for(self.accuracy_matrix, self.current_task_learning_curves)

    def summary_for_eval_batch_size(self, batch_size: int) -> dict[str, Any]:
        """Return standard metrics for one registered evaluation batch size."""
        if batch_size not in self.evaluation_batch_size_results:
            if batch_size == self.run_config.get("eval_batch_size"):
                return self.summary()
            raise KeyError(f"no evaluation results for batch size {batch_size}")
        item = self.evaluation_batch_size_results[batch_size]
        return self._summary_for(
            item["accuracy_matrix"],
            item["current_task_learning_curves"],
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "accuracy_matrix": matrix_to_jsonable(self.accuracy_matrix),
            "eval_loss_matrix": matrix_to_jsonable(self.eval_loss_matrix),
            "train_loss_by_task": [float(v) for v in self.train_loss_by_task],
            "structural_history": [asdict(item) for item in self.structural_history],
            "global_steps": int(self.global_steps),
            "run_config": self.run_config,
            "stream_manifest": self.stream_manifest,
            "metadata": self.metadata,
            "current_task_learning_curves": self.current_task_learning_curves,
            "train_seconds_by_task": self.train_seconds_by_task,
            "eval_seconds_by_stage": self.eval_seconds_by_stage,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "max_gradient_norm_by_task": self.max_gradient_norm_by_task,
            "max_parameter_abs_by_task": self.max_parameter_abs_by_task,
            "summary": self.summary(),
        }
        if self.evaluation_batch_size_results:
            payload["evaluation_batch_size_results"] = {
                str(batch_size): {
                    "eval_batch_size": int(batch_size),
                    "accuracy_matrix": matrix_to_jsonable(item["accuracy_matrix"]),
                    "eval_loss_matrix": matrix_to_jsonable(item["eval_loss_matrix"]),
                    "current_task_learning_curves": item["current_task_learning_curves"],
                    "summary": self._summary_for(
                        item["accuracy_matrix"],
                        item["current_task_learning_curves"],
                    ),
                }
                for batch_size, item in sorted(self.evaluation_batch_size_results.items())
            }
            payload["evaluation_state_hashes"] = self.evaluation_state_hashes
        return payload


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
    current_task_learning_curves: list[list[dict[str, float | int]]] = []
    train_seconds_by_task: list[float] = []
    eval_seconds_by_stage: list[float] = []
    max_gradient_norm_by_task: list[float] = []
    max_parameter_abs_by_task: list[float] = []
    global_step = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    seen_classes: list[int] = []
    for stage, task in enumerate(stream):
        seen_classes.extend(int(v) for v in task.classes)
        allowed_classes = tuple(sorted(set(seen_classes))) if config.restrict_logits_to_seen_classes else None
        losses: list[float] = []
        task_curve: list[dict[str, float | int]] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_started = time.perf_counter()
        selection_counts: Counter[int] = Counter()
        executed_sizes: list[int] = []
        task_max_gradient_norm = 0.0
        task_max_parameter_abs = 0.0

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
                        stage=stage,
                        task_id=task.task_id,
                        epoch=epoch,
                        batch_index=batch_index,
                    )
                    losses.append(result.loss)
                    task_max_gradient_norm = max(
                        task_max_gradient_norm,
                        result.gradient_norm_l2,
                    )
                    task_max_parameter_abs = max(
                        task_max_parameter_abs,
                        result.max_abs_parameter,
                    )
                    trace = result.trace
                    ids = [int(v) for v in trace.selected_node_ids.tolist()]
                else:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    output = model(x)
                    logits = logits_from_output(output)
                    context = NumericalContext(
                        stage=stage,
                        task_id=task.task_id,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                    )
                    ensure_finite_tensor(
                        logits,
                        phase="train_forward",
                        tensor_name="logits",
                        context=context,
                    )
                    aux_loss = getattr(output, "aux_loss", None)
                    if isinstance(aux_loss, Tensor):
                        ensure_finite_tensor(
                            aux_loss,
                            phase="train_forward",
                            tensor_name="aux_loss",
                            context=context,
                        )
                    loss = class_incremental_cross_entropy(
                        logits,
                        y,
                        allowed_classes=allowed_classes,
                    )
                    if isinstance(aux_loss, Tensor):
                        loss = loss + aux_loss
                    ensure_finite_tensor(
                        loss,
                        phase="train_loss",
                        tensor_name="loss",
                        context=context,
                    )
                    loss.backward()
                    gradient_diagnostics = check_model_gradients(
                        model,
                        phase="backward_gradients",
                        context=context,
                    )
                    optimizer.step()
                    parameter_diagnostics = check_model_parameters(
                        model,
                        phase="post_optimizer_parameters",
                        context=context,
                    )
                    task_max_gradient_norm = max(
                        task_max_gradient_norm,
                        gradient_diagnostics.total_l2_norm,
                    )
                    task_max_parameter_abs = max(
                        task_max_parameter_abs,
                        parameter_diagnostics.max_abs,
                    )
                    losses.append(float(loss.detach().item()))
                    trace = getattr(output, "trace", None)
                    if trace is not None and hasattr(model, "graph_state"):
                        model.graph_state.update_utilization(
                            trace.selected_node_ids,
                            trace.router_weights,
                        )
                        ensure_finite_tensor(
                            model.graph_state.node_utilization_ema,
                            phase="structural_statistics",
                            tensor_name="node_utilization_ema",
                            context=context,
                        )
                        ensure_finite_tensor(
                            model.graph_state.edge_utilization_ema,
                            phase="structural_statistics",
                            tensor_name="edge_utilization_ema",
                            context=context,
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

            if (
                config.current_task_eval_every_epochs > 0
                and (epoch + 1) % config.current_task_eval_every_epochs == 0
            ):
                curve_eval = evaluate_task(
                    model,
                    task,
                    batch_size=config.eval_batch_size,
                    device=device,
                    allowed_classes=allowed_classes,
                    numerical_context=NumericalContext(
                        stage=stage,
                        task_id=task.task_id,
                        epoch=epoch,
                        global_step=global_step,
                    ),
                )
                task_curve.append(
                    {
                        "epoch": int(epoch + 1),
                        "global_step": int(global_step),
                        "accuracy": float(curve_eval.accuracy),
                        "loss": float(curve_eval.loss),
                    }
                )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds_by_task.append(time.perf_counter() - train_started)
        train_loss_by_task.append(sum(losses) / max(1, len(losses)))
        current_task_learning_curves.append(task_curve)
        max_gradient_norm_by_task.append(task_max_gradient_norm)
        max_parameter_abs_by_task.append(task_max_parameter_abs)
        structural_history.append(
            _snapshot(
                model,
                stage=stage,
                task_id=task.task_id,
                counts=selection_counts,
                executed_sizes=executed_sizes,
            )
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_started = time.perf_counter()
        eval_limit = num_tasks if config.evaluate_future_tasks else stage + 1
        for eval_task_index in range(eval_limit):
            result = evaluate_task(
                model,
                stream[eval_task_index],
                batch_size=config.eval_batch_size,
                device=device,
                allowed_classes=allowed_classes,
                numerical_context=NumericalContext(
                    stage=stage,
                    task_id=stream[eval_task_index].task_id,
                    global_step=global_step,
                ),
            )
            accuracy[stage, eval_task_index] = result.accuracy
            eval_loss[stage, eval_task_index] = result.loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        eval_seconds_by_stage.append(time.perf_counter() - eval_started)

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
        current_task_learning_curves=current_task_learning_curves,
        train_seconds_by_task=train_seconds_by_task,
        eval_seconds_by_stage=eval_seconds_by_stage,
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        max_gradient_norm_by_task=max_gradient_norm_by_task,
        max_parameter_abs_by_task=max_parameter_abs_by_task,
    )
