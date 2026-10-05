from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class H1Protocol:
    dataset: str = "split-cifar100"
    num_tasks: int = 10
    classes_per_task: int = 10
    class_order_seed: int = 1701
    model_seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    epochs_per_task: int = 10
    batch_size: int = 128
    eval_batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    use_augmentation: bool = False
    replay_examples: int = 0
    normalize: bool = True
    message_rounds: int = 0
    primary_metric: str = "final_average_accuracy"

    def __post_init__(self) -> None:
        if self.dataset != "split-cifar100":
            raise ValueError("M1.3 protocol currently supports split-cifar100 only")
        if self.num_tasks * self.classes_per_task != 100:
            raise ValueError("Split CIFAR-100 must cover exactly 100 classes")
        if not self.model_seeds:
            raise ValueError("at least one model seed is required")
        if self.epochs_per_task <= 0:
            raise ValueError("epochs_per_task must be positive")
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.replay_examples != 0:
            raise ValueError("primary H1 protocol is no-replay; replay is a separate reference track")
        if self.message_rounds != 0:
            raise ValueError("primary H1 protocol isolates recruitment before graph message passing")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_seeds"] = list(self.model_seeds)
        return data


def default_h1_protocol() -> H1Protocol:
    return H1Protocol()
