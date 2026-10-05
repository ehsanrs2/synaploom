from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch

from synaploom.baselines import DenseControlModel, make_no_growth_routed_control
from synaploom.config import GraphConfig, ModelConfig
from synaploom.data.streams import ContinualStream
from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import FixedScheduleRecruitment, NoConsolidation, NoEdges
from synaploom.research.budgets import (
    count_trainable_parameters,
    dense_hidden_width_for_target,
    synaploom_budget,
)
from synaploom.training.continual import ContinualConfig
from synaploom.training.engine import TrainingEngine


H1A_CLASS_ORDER: tuple[int, ...] = (
    75, 59, 1, 39, 93, 87, 37, 51, 7, 34,
    21, 41, 19, 86, 27, 89, 44, 0, 20, 14,
    49, 91, 30, 69, 23, 70, 62, 9, 29, 76,
    6, 15, 71, 32, 96, 95, 24, 99, 50, 66,
    80, 88, 53, 98, 68, 10, 97, 18, 73, 61,
    4, 45, 40, 72, 74, 63, 31, 28, 5, 92,
    90, 25, 3, 38, 48, 79, 94, 35, 65, 58,
    13, 60, 47, 54, 57, 85, 82, 42, 83, 43,
    56, 8, 17, 2, 84, 78, 16, 36, 67, 64,
    46, 81, 52, 55, 77, 12, 22, 33, 26, 11,
)

H1A_VARIANTS = (
    "progressive_reserve",
    "all_recruited",
    "fixed_small",
    "dense_active_proxy",
    "dense_total",
)


@dataclass(frozen=True)
class H1AProtocol:
    feature_dim: int = 512
    num_classes: int = 100
    classes_per_task: int = 10
    d_model: int = 128
    d_hidden: int = 256
    num_nodes: int = 16
    initial_recruited: int = 4
    top_k: int = 4
    epochs_per_task: int = 20
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    current_task_eval_every_epochs: int = 1

    def graph_config(self, *, all_recruited: bool = False) -> GraphConfig:
        initial = self.num_nodes if all_recruited else self.initial_recruited
        return GraphConfig(
            num_nodes=self.num_nodes,
            initial_recruited=initial,
            d_model=self.d_model,
            d_hidden=self.d_hidden,
            router_dim=self.d_model,
            top_k=self.top_k,
            q_max=0,
            message_rounds=0,
            relation_types=1,
        )

    def model_config(self, *, all_recruited: bool = False) -> ModelConfig:
        return ModelConfig(
            graph=self.graph_config(all_recruited=all_recruited),
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
        )

    def continual_config(self, *, seed: int, device: str) -> ContinualConfig:
        return ContinualConfig(
            epochs_per_task=self.epochs_per_task,
            batch_size=self.batch_size,
            eval_batch_size=self.eval_batch_size,
            seed=seed,
            device=device,
            restrict_logits_to_seen_classes=True,
            current_task_eval_every_epochs=self.current_task_eval_every_epochs,
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def task_start_recruitment_steps(stream: ContinualStream, protocol: H1AProtocol) -> tuple[int, ...]:
    """Steps whose *first batch* belongs to a new task after task 0."""
    starts: list[int] = []
    global_step = 0
    for task_index, task in enumerate(stream):
        if task_index > 0:
            starts.append(global_step)
        batches = math.ceil(int(task.train_y.numel()) / protocol.batch_size)
        global_step += batches * protocol.epochs_per_task
    return tuple(starts)


def _optimizer(model: torch.nn.Module, protocol: H1AProtocol) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=protocol.learning_rate,
        momentum=protocol.momentum,
        weight_decay=protocol.weight_decay,
    )


def build_h1a_variant(
    variant: str,
    *,
    stream: ContinualStream,
    seed: int,
    protocol: H1AProtocol,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, TrainingEngine | None, dict[str, object]]:
    if variant not in H1A_VARIANTS:
        raise ValueError(f"unknown H1a variant: {variant}")
    torch.manual_seed(seed)

    reference = make_no_growth_routed_control(protocol.model_config())
    reference_budget = synaploom_budget(reference)

    if variant in {"progressive_reserve", "fixed_small"}:
        model = reference
    elif variant == "all_recruited":
        torch.manual_seed(seed)
        model = make_no_growth_routed_control(protocol.model_config(all_recruited=True))
    elif variant in {"dense_active_proxy", "dense_total"}:
        target = (
            reference_budget.active_parameter_proxy
            if variant == "dense_active_proxy"
            else reference_budget.total_trainable_parameters
        )
        width, matched = dense_hidden_width_for_target(
            target,
            input_dim=protocol.feature_dim,
            d_model=protocol.d_model,
            num_classes=protocol.num_classes,
        )
        torch.manual_seed(seed)
        model = DenseControlModel(
            input_dim=protocol.feature_dim,
            d_model=protocol.d_model,
            d_hidden=width,
            num_classes=protocol.num_classes,
        )
        optimizer = _optimizer(model, protocol)
        return model, optimizer, None, {
            "variant": variant,
            "target_parameters": target,
            "actual_parameters": matched,
            "dense_hidden_width": width,
            "parameter_error": matched - target,
        }
    else:
        raise AssertionError("unreachable")

    optimizer = _optimizer(model, protocol)
    if variant == "progressive_reserve":
        controller = StructuralController(
            recruitment_policy=FixedScheduleRecruitment(
                task_start_recruitment_steps(stream, protocol)
            ),
            edge_init_policy=NoEdges(),
            consolidation_policy=NoConsolidation(),
        )
        engine: TrainingEngine | None = TrainingEngine(
            model,
            optimizer,
            controller=controller,
        )
    else:
        # Routed controls deliberately use the exact same trainer path.
        engine = TrainingEngine(model, optimizer, controller=None)

    budget = synaploom_budget(model)
    return model, optimizer, engine, {
        "variant": variant,
        "budget": asdict(budget),
        "task_start_recruitment_steps": (
            list(task_start_recruitment_steps(stream, protocol))
            if variant == "progressive_reserve"
            else []
        ),
        "total_parameters": count_trainable_parameters(model),
    }
