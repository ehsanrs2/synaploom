import pytest
import torch

from synaploom.training.continual_metrics import (
    average_accuracy_by_stage,
    average_forgetting,
    backward_transfer,
    diagonal_learning_accuracy,
    final_average_accuracy,
    late_task_learning_accuracy,
)


def test_continual_metrics_from_accuracy_matrix() -> None:
    matrix = torch.tensor(
        [
            [0.80, float("nan"), float("nan")],
            [0.70, 0.90, float("nan")],
            [0.60, 0.85, 0.75],
        ]
    )

    assert final_average_accuracy(matrix) == pytest.approx((0.60 + 0.85 + 0.75) / 3)
    assert average_forgetting(matrix) == pytest.approx((0.20 + 0.05) / 2)
    assert backward_transfer(matrix) == pytest.approx((-0.20 - 0.05) / 2)
    assert late_task_learning_accuracy(matrix, last_n=2) == pytest.approx((0.90 + 0.75) / 2)
    assert torch.allclose(diagonal_learning_accuracy(matrix), torch.tensor([0.80, 0.90, 0.75]))
    assert torch.allclose(
        average_accuracy_by_stage(matrix),
        torch.tensor([0.80, 0.80, (0.60 + 0.85 + 0.75) / 3]),
    )
