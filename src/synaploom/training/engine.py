from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from synaploom.core.outputs import ForwardTrace
from synaploom.events import StructuralEvent
from synaploom.model import SynapLoomModel
from synaploom.plasticity.consolidation import scale_node_owned_gradients
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import StructuralObservation
from synaploom.plasticity.statistics import update_node_importance
from synaploom.training.class_incremental import class_incremental_cross_entropy
from synaploom.training.numerics import (
    NumericalContext,
    check_model_gradients,
    check_model_parameters,
    ensure_finite_tensor,
)


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    trace: ForwardTrace
    events: tuple[StructuralEvent, ...]
    gradient_norm_l2: float
    max_abs_gradient: float
    max_abs_parameter: float


class TrainingEngine:
    """Minimal M1 trainer that enforces the structural mutation boundary."""

    def __init__(
        self,
        model: SynapLoomModel,
        optimizer: torch.optim.Optimizer,
        *,
        controller: StructuralController | None = None,
        criterion: Callable[[Tensor, Tensor], Tensor] | None = None,
        importance_decay: float = 0.99,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.controller = controller
        self.criterion = criterion or F.cross_entropy
        self.importance_decay = float(importance_decay)

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        *,
        step: int,
        task_boundary: bool = False,
        allowed_classes: tuple[int, ...] | list[int] | None = None,
        stage: int | None = None,
        task_id: int | None = None,
        epoch: int | None = None,
        batch_index: int | None = None,
    ) -> TrainStepResult:
        context = NumericalContext(
            stage=stage,
            task_id=task_id,
            epoch=epoch,
            batch_index=batch_index,
            global_step=step,
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        output = self.model(x, update_statistics=False)
        ensure_finite_tensor(
            output.logits,
            phase="train_forward",
            tensor_name="logits",
            context=context,
        )
        ensure_finite_tensor(
            output.aux_loss,
            phase="train_forward",
            tensor_name="aux_loss",
            context=context,
        )
        ensure_finite_tensor(
            output.trace.router_weights,
            phase="train_forward",
            tensor_name="router_weights",
            context=context,
        )
        ensure_finite_tensor(
            output.trace.routing_context,
            phase="train_forward",
            tensor_name="routing_context",
            context=context,
        )

        if allowed_classes is None:
            loss = self.criterion(output.logits, y)
        else:
            loss = class_incremental_cross_entropy(
                output.logits,
                y,
                allowed_classes=allowed_classes,
            )
        loss = loss + output.aux_loss
        ensure_finite_tensor(
            loss,
            phase="train_loss",
            tensor_name="loss",
            context=context,
        )

        loss.backward()
        check_model_gradients(
            self.model,
            phase="backward_gradients",
            context=context,
        )

        update_node_importance(
            self.model.executor,
            self.model.graph_state,
            decay=self.importance_decay,
        )
        ensure_finite_tensor(
            self.model.graph_state.node_importance_ema,
            phase="structural_statistics",
            tensor_name="node_importance_ema",
            context=context,
        )
        scale_node_owned_gradients(self.model.executor, self.model.graph_state)

        gradient_diagnostics = check_model_gradients(
            self.model,
            phase="protected_gradients",
            context=context,
        )
        self.optimizer.step()

        parameter_diagnostics = check_model_parameters(
            self.model,
            phase="post_optimizer_parameters",
            context=context,
        )

        self.model.graph_state.update_utilization(
            output.trace.selected_node_ids,
            output.trace.router_weights,
        )
        ensure_finite_tensor(
            self.model.graph_state.node_utilization_ema,
            phase="structural_statistics",
            tensor_name="node_utilization_ema",
            context=context,
        )
        ensure_finite_tensor(
            self.model.graph_state.edge_utilization_ema,
            phase="structural_statistics",
            tensor_name="edge_utilization_ema",
            context=context,
        )

        events: list[StructuralEvent] = []
        if self.controller is not None:
            observation = StructuralObservation.from_trace(
                step,
                output.trace,
                task_boundary=task_boundary,
            )
            self.controller.observe(observation)
            if self.controller.should_update(self.model.graph_state):
                plan = self.controller.propose(self.model.graph_state)
                events = self.controller.apply(self.model.graph_state, plan)

        return TrainStepResult(
            loss=float(loss.detach().item()),
            trace=output.trace.detached(),
            events=tuple(events),
            gradient_norm_l2=gradient_diagnostics.total_l2_norm,
            max_abs_gradient=gradient_diagnostics.max_abs,
            max_abs_parameter=parameter_diagnostics.max_abs,
        )
