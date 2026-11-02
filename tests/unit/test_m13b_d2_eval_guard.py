"""CPU-only evidence for proposed M1.3b D2 evaluation gate.

Synthetic only. No CIFAR feature bundle, model training or actual scientific
data is loaded. The future confirmatory runner is not implemented here.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
import torch
from torch import nn

from synaploom.config import GraphConfig, ModelConfig
from synaploom.core.router import SparseRouter
from synaploom.core.graph_state import GraphState
from synaploom.model import SynapLoomModel
from synaploom.research.fixed_sparse_baseline import FixedSparseMLP
from synaploom.research.m13b_d2_eval_guard import (
    KIND, METRICS, evaluation_policy_candidate,
    validate_co_primary_observation,
)
from synaploom.research.m13b_manifest import canonical_json
from synaploom.training.evaluation import evaluate_task
from synaploom.data.streams import TensorTask
from synaploom.training.state_digest import model_state_sha256


def complete_synthetic_observation():
    matrix = [
        [None if j > i else 0.3 + j / 100 for j in range(10)]
        for i in range(10)
    ]
    curves = [
        [
            {"epoch": epoch, "global_step": 800 * task + 40 * epoch,
             "accuracy": 0.4 + task / 100 + epoch / 1000, "loss": 1.0}
            for epoch in range(1, 21)
        ]
        for task in range(10)
    ]
    final = sum(matrix[-1]) / 10
    means = [sum(p["accuracy"] for p in curve) / 20 for curve in curves]
    late = (means[-2] + means[-1]) / 2
    return {
        "accuracy_matrix": matrix,
        "current_task_learning_curves": curves,
        "run_config": {
            "epochs_per_task": 20, "batch_size": 128, "eval_batch_size": 256,
            "evaluate_future_tasks": False,
            "restrict_logits_to_seen_classes": True,
            "current_task_eval_every_epochs": 1,
        },
        "summary": {
            "final_average_accuracy": final,
            "late_task_within_task_mean_accuracy": late,
        },
    }


def test_policy_manifest_is_deterministic_prescience_only():
    p = evaluation_policy_candidate()
    assert p["kind"] == KIND
    assert p["status"] == "CPU_POLICY_GUARD_CANDIDATE__NOT_FROZEN_OR_EXECUTABLE"
    assert p["eval_batch_size"] == 256
    assert p["train_batch_size"] == 128
    assert p["evaluation_shuffle"] is False
    assert p["evaluation_drop_last"] is False
    assert p["evaluation_partial_final_batch"] == "INCLUDED"
    assert p["current_task_eval_every_epochs"] == 1
    assert p["evaluate_future_tasks"] is False
    assert p["restrict_logits_to_seen_classes"] is True
    assert p["co_primary_metrics"] == list(METRICS)
    assert "BATCH_COMPOSITION_DEPENDENT" in p["routing"]
    assert p["gpu_executable"] is False
    assert p["protected_eval_accessed"] is False
    assert p["owner_science_execution_approved"] is False
    assert p["runner_integrated"] is False
    assert p["record_sha256"] == sha256(canonical_json({
        key: value for key, value in p.items() if key != "record_sha256"
    })).hexdigest()
    assert p == evaluation_policy_candidate()


def test_complete_synthetic_observation_passes_without_statistical_claim():
    payload = complete_synthetic_observation()
    original = deepcopy(payload)
    result = validate_co_primary_observation(payload)
    assert payload == original
    assert result["status"] == "ENDPOINT_COMPLETENESS_PASS__NO_SCIENTIFIC_INFERENCE"
    assert result["science_executable"] is False
    assert result["scientific_claim"] == "NONE"
    assert result["reference_eval_batch_size"] == 256
    assert result["co_primary_metric_values"]["final_average_accuracy"] == pytest.approx(.345)
    assert result["co_primary_metric_values"]["late_task_within_task_mean_accuracy"] == pytest.approx(
        payload["summary"]["late_task_within_task_mean_accuracy"]
    )


@pytest.mark.parametrize("bad", [
    None, float("nan"), float("inf"), -0.01, 1.01, True, "0.4",
])
def test_missing_or_bad_required_accuracy_fails_closed(bad):
    data = complete_synthetic_observation()
    data["accuracy_matrix"][9][3] = bad
    with pytest.raises(ValueError):
        validate_co_primary_observation(data)


def test_future_task_must_be_explicit_null_and_matrix_full():
    a = complete_synthetic_observation()
    a["accuracy_matrix"][0][2] = 0.5
    with pytest.raises(ValueError):
        validate_co_primary_observation(a)
    a = complete_synthetic_observation()
    a["accuracy_matrix"][9] = a["accuracy_matrix"][9][:-1]
    with pytest.raises(ValueError):
        validate_co_primary_observation(a)


@pytest.mark.parametrize("wrong", [
    None, float("nan"), float("inf"), -0.2, 1.4, True, "0.4",
])
def test_missing_nonfinite_or_invalid_epoch_accuracy_fails_closed(wrong):
    x = complete_synthetic_observation()
    x["current_task_learning_curves"][9][19]["accuracy"] = wrong
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)


def test_missing_epoch_or_inconsistent_step_fails_closed():
    x = complete_synthetic_observation()
    x["current_task_learning_curves"][9].pop()
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)
    x = complete_synthetic_observation()
    x["current_task_learning_curves"][3][4]["epoch"] = 4
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)
    x = complete_synthetic_observation()
    x["current_task_learning_curves"][2][0]["global_step"] = 1
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)


@pytest.mark.parametrize("field,bad", [
    ("eval_batch_size", 128), ("eval_batch_size", "256"),
    ("epochs_per_task", 1), ("batch_size", 64),
    ("restrict_logits_to_seen_classes", False),
    ("evaluate_future_tasks", True),
    ("current_task_eval_every_epochs", 0),
])
def test_unapproved_eval_config_fails_closed(field,bad):
    x = complete_synthetic_observation()
    x["run_config"][field] = bad
    with pytest.raises(ValueError, match="unapproved"):
        validate_co_primary_observation(x)


def test_missing_wrong_summary_and_nonfinite_eval_loss_rejected():
    x = complete_synthetic_observation()
    x["summary"]["final_average_accuracy"] += 0.05
    with pytest.raises(ValueError, match="does not match"):
        validate_co_primary_observation(x)
    x = complete_synthetic_observation()
    x["summary"]["late_task_within_task_mean_accuracy"] = None
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)
    x = complete_synthetic_observation()
    x["current_task_learning_curves"][0][0]["loss"] = float("nan")
    with pytest.raises(ValueError):
        validate_co_primary_observation(x)


def test_batch_mean_router_selection_changes_with_companion_only():
    cfg = GraphConfig(num_nodes=3, initial_recruited=3, top_k=1,
                      d_model=2, d_hidden=4, router_dim=2)
    state = GraphState(cfg)
    router = SparseRouter(cfg)
    with torch.no_grad():
        router.query.weight.copy_(torch.eye(2))
        state.routing_keys.copy_(torch.tensor([
            [1., 0.], [0., 1.], [-1., -1.],
        ]))
    anchor = torch.tensor([[2., 0.]])
    companion = torch.tensor([[-2., 6.]])
    reference = router(anchor, state)
    composed = router(torch.cat((anchor, companion)), state)
    assert reference.selected_node_ids.tolist() == [0]
    assert composed.selected_node_ids.tolist() == [1]
    # Both routings act on exactly the SAME anchor. Companion changes route.
    assert not torch.allclose(reference.batch_query, composed.batch_query)
    assert state.recruited_count() == 3


def test_fixed_masked_mlp_anchor_logits_are_companion_invariant():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = FixedSparseMLP(
            input_dim=2, d_model=2, d_hidden=4, num_classes=2,
            active_first=4, active_second=4, mask_seed=13,
        ).eval()
    anchor = torch.tensor([[2., 0.]])
    composed = torch.tensor([[2., 0.], [-2., 6.]])
    before = model_state_sha256(model)
    with torch.inference_mode():
        a = model(anchor).logits
        b = model(composed).logits[:1]
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert model_state_sha256(model) == before
    model.assert_masks()


def test_actual_eval_task_iterator_keeps_order_and_final_partial_batch():
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 1])
    features = torch.nn.functional.one_hot(labels, num_classes=2).float()
    task = TensorTask(
        task_id=0, classes=(0,1), train_x=features, train_y=labels,
        eval_x=features, eval_y=labels,
    )

    class Seen(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunks = []
        def forward(self, x):
            self.chunks.append((len(x), x.argmax(dim=1).tolist()))
            return x * 10

    model = Seen()
    result = evaluate_task(model, task, batch_size=4)
    assert model.chunks == [(4, [1,0,1,0]), (3, [1,0,1])]
    assert result.examples == 7
    assert result.accuracy == 1
