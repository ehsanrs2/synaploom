"""Actual CPU count witness on artificial data; no scientific run."""
from copy import deepcopy
from fractions import Fraction

import pytest
import torch

from synaploom.baselines import DenseControlModel
from synaploom.data.streams import TensorTask
from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.research.m13b_d1a_runner_control_cpu import (
    exact_endpoints, synthetic_observation_from_events, synthetic_registry,
)
from synaploom.training.continual import ContinualConfig, run_continual_experiment
from synaploom.training.evaluation import TaskEvaluation, evaluate_task


def test_actual_integer_count_with_partial_232_tail():
    y = torch.arange(1000, dtype=torch.long) % 2
    x = torch.nn.functional.one_hot(y, 2).float()
    x[-1] = x[-1].flip(0)
    task = TensorTask(task_id=0, classes=(0, 1), train_x=x[:2],
                      train_y=y[:2], eval_x=x, eval_y=y)
    result = evaluate_task(torch.nn.Identity(), task, batch_size=256)
    assert result.examples == 1000
    assert result.correct == 999
    assert result.accuracy == 999/1000
    assert result == TaskEvaluation(result.loss, result.accuracy, result.examples)


def test_real_continual_cpu_observer_counts_and_default_parity():
    def run(observer=None):
        torch.manual_seed(5)
        model = DenseControlModel(input_dim=6, d_model=8, d_hidden=12, num_classes=4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        stream = make_gaussian_cluster_stream(
            num_tasks=2, classes_per_task=2, input_dim=6,
            train_per_class=6, eval_per_class=5, separation=5.0,
            noise_std=0.6, seed=5,
        )
        kwargs = {} if observer is None else {"evaluation_count_observer": observer}
        result = run_continual_experiment(
            model, stream, optimizer,
            config=ContinualConfig(epochs_per_task=2, batch_size=8,
                                   eval_batch_size=4, seed=5,
                                   current_task_eval_every_epochs=1,
                                   restrict_logits_to_seen_classes=True),
            **kwargs,
        )
        return result, model.state_dict()
    observed = []
    before, state_before = run()
    after, state_after = run(observed.append)
    assert torch.allclose(before.accuracy_matrix, after.accuracy_matrix,
                          atol=0, rtol=0, equal_nan=True)
    assert before.current_task_learning_curves == after.current_task_learning_curves
    assert before.global_steps == after.global_steps == 8
    assert state_before.keys() == state_after.keys()
    assert all(torch.equal(v, state_after[k]) for k, v in state_before.items())
    assert len(observed) == 7
    assert [e["phase"] for e in observed] == [
        "epoch", "epoch", "stage", "epoch", "epoch", "stage", "stage"]
    for e in observed:
        assert e["examples"] == 10 and type(e["correct"]) is int
        assert 0 <= e["correct"] <= e["examples"]
        if e["phase"] == "epoch":
            point = after.current_task_learning_curves[e["stage"]][e["epoch"]-1]
            assert e["global_step"] == point["global_step"]
            assert e["correct"]/10 == point["accuracy"]
        else:
            assert float(after.accuracy_matrix[e["stage"], e["task_index"]]) == pytest.approx(e["correct"]/10)


def invented_full_events():
    events = []
    for s in range(10):
        for epoch in range(1,21):
            events.append(dict(phase="epoch",stage=s,task_index=s,task_id=s,
                               epoch=epoch,global_step=(s*20+epoch)*40,
                               batch_size=256,correct=700,examples=1000))
        for t in range(s+1):
            events.append(dict(phase="stage",stage=s,task_index=t,task_id=t,
                               epoch=0,global_step=(s+1)*800,
                               batch_size=256,correct=500,examples=1000))
    return events


def test_full_event_adapter_exact_endpoints():
    x = invented_full_events()
    slot = synthetic_registry()[0]["slot_id"]
    assert len(x) == 255
    assert exact_endpoints(synthetic_observation_from_events(slot,x)) == (
        Fraction(1,2), Fraction(7,10))


@pytest.mark.parametrize("mutate", [
    lambda x: x.pop(),
    lambda x: x[20].update(phase="epoch"),
    lambda x: x[-1].update(examples=999),
    lambda x: x[0].update(correct=True),
    lambda x: x[0].update(batch_size=128),
    lambda x: x[-1].update(global_step=7999),
    lambda x: x[0].update(task_id=1),
    lambda x: x[0].update(stage=False),
    lambda x: x[0].update(extra="not registered"),
])
def test_full_event_adapter_rejects_drift(mutate):
    events = deepcopy(invented_full_events())
    mutate(events)
    with pytest.raises(ValueError):
        synthetic_observation_from_events(synthetic_registry()[0]["slot_id"],events)
