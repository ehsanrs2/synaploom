from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from synaploom.core.outputs import ForwardTrace
from synaploom.model import SynapLoomModel
from synaploom.plasticity.consolidation import scale_node_owned_gradients
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import StructuralObservation
from synaploom.plasticity.statistics import update_node_importance
from synaploom.events import StructuralEvent
from synaploom.training.class_incremental import class_incremental_cross_entropy


@dataclass(frozen=True)
class TrainStepResult:
    loss: float
    trace: ForwardTrace
    events: tuple[StructuralEvent, ...]


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
    ) -> TrainStepResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        output = self.model(x, update_statistics=False)
        if allowed_classes is None:
            loss = self.criterion(output.logits, y)
        else:
            loss = class_incremental_cross_entropy(
                output.logits,
                y,
                allowed_classes=allowed_classes,
            )
        loss = loss + output.aux_loss
        loss.backward()

        update_node_importance(
            self.model.executor,
            self.model.graph_state,
            decay=self.importance_decay,
        )
        scale_node_owned_gradients(self.model.executor, self.model.graph_state)
        self.optimizer.step()

        self.model.graph_state.update_utilization(
            output.trace.selected_node_ids,
            output.trace.router_weights,
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
        )
