from __future__ import annotations

import pytest
import torch

from synaploom.baselines import DenseControlModel
from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.training.continual import ContinualConfig, run_continual_experiment


def _stream():
    return make_gaussian_cluster_stream(
        num_tasks=2,
        classes_per_task=2,
        input_dim=6,
        train_per_class=10,
        eval_per_class=8,
        separation=5.0,
        noise_std=0.6,
        seed=5,
    )


def test_multi_eval_records_each_size_without_retraining_and_hashes_state() -> None:
    torch.manual_seed(17)
    model = DenseControlModel(input_dim=6, d_model=8, d_hidden=12, num_classes=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    sizes = (2, 8, 16)
    config = ContinualConfig(
        epochs_per_task=2,
        batch_size=8,
        eval_batch_size=8,
        seed=17,
        current_task_eval_every_epochs=1,
        restrict_logits_to_seen_classes=True,
    )

    result = run_continual_experiment(
        model,
        _stream(),
        optimizer,
        config=config,
        evaluation_batch_sizes=sizes,
    )

    assert set(result.evaluation_batch_size_results) == set(sizes)
    assert torch.allclose(
        result.accuracy_matrix,
        result.evaluation_batch_size_results[8]["accuracy_matrix"],
        equal_nan=True,
    )
    assert len(result.evaluation_state_hashes) == 6
    for sweep in result.evaluation_state_hashes:
        assert sweep["batch_sizes"] == list(sizes)
        assert sweep["state_sha256_before"] == sweep["state_sha256_after"]
        assert sweep["state_unchanged"] is True

    for size in sizes:
        curves = result.evaluation_batch_size_results[size][
            "current_task_learning_curves"
        ]
        assert len(curves) == 2
        assert all(len(task_curve) == 2 for task_curve in curves)
        assert "final_average_accuracy" in result.summary_for_eval_batch_size(size)

    payload = result.to_dict()
    assert set(payload["evaluation_batch_size_results"]) == {str(size) for size in sizes}
    assert len(payload["evaluation_state_hashes"]) == 6


def test_multi_eval_requires_unique_sizes_and_the_reference_size() -> None:
    model = DenseControlModel(input_dim=6, d_model=8, d_hidden=12, num_classes=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    config = ContinualConfig(epochs_per_task=1, batch_size=8, eval_batch_size=8)

    with pytest.raises(ValueError, match="must be unique"):
        run_continual_experiment(
            model,
            _stream(),
            optimizer,
            config=config,
            evaluation_batch_sizes=(4, 8, 8),
        )

    with pytest.raises(ValueError, match="reference evaluation size"):
        run_continual_experiment(
            model,
            _stream(),
            optimizer,
            config=config,
            evaluation_batch_sizes=(4, 16),
        )

class _MutatingEvaluationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)
        self.register_buffer("evaluation_calls", torch.tensor(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            self.evaluation_calls.add_(1)
        return self.linear(x)


def test_multi_eval_fails_closed_if_evaluation_mutates_model_state() -> None:
    model = _MutatingEvaluationModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    config = ContinualConfig(
        epochs_per_task=1,
        batch_size=8,
        eval_batch_size=8,
        seed=23,
        current_task_eval_every_epochs=1,
        restrict_logits_to_seen_classes=True,
    )

    with pytest.raises(RuntimeError, match="model state changed during evaluation sweep"):
        run_continual_experiment(
            model,
            _stream(),
            optimizer,
            config=config,
            evaluation_batch_sizes=(4, 8),
        )
