import pytest
import torch
from torch import nn

from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.training.continual import ContinualConfig, run_continual_experiment
from synaploom.training.numerics import (
    NumericalContext,
    NumericalInstabilityError,
    ensure_finite_tensor,
)


def test_nonfinite_tensor_error_preserves_context() -> None:
    context = NumericalContext(
        stage=2,
        task_id=2,
        epoch=4,
        batch_index=7,
        global_step=123,
    )
    with pytest.raises(NumericalInstabilityError) as captured:
        ensure_finite_tensor(
            torch.tensor([1.0, float("nan")]),
            phase="train_loss",
            tensor_name="loss",
            context=context,
        )

    payload = captured.value.to_dict()
    assert payload["phase"] == "train_loss"
    assert payload["tensor_name"] == "loss"
    assert payload["context"]["global_step"] == 123
    assert payload["details"]["nonfinite_count"] == 1


class _NonFiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * torch.tensor(float("nan"), device=x.device)


def test_continual_runner_fails_on_first_nonfinite_forward() -> None:
    stream = make_gaussian_cluster_stream(
        num_tasks=1,
        classes_per_task=2,
        input_dim=4,
        train_per_class=4,
        eval_per_class=2,
        seed=3,
    )
    model = _NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(NumericalInstabilityError) as captured:
        run_continual_experiment(
            model,
            stream,
            optimizer,
            config=ContinualConfig(
                epochs_per_task=1,
                batch_size=4,
                eval_batch_size=4,
                seed=3,
            ),
        )

    payload = captured.value.to_dict()
    assert payload["phase"] == "train_forward"
    assert payload["context"] == {
        "stage": 0,
        "task_id": 0,
        "epoch": 0,
        "batch_index": 0,
        "global_step": 0,
    }
