import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from synaploom.training.state_digest import model_state_sha256
import synaploom.research.sample_performance as benchmark
import synaploom.training.evaluation as evaluation

from synaploom.research.sample_performance import measure


def test_bounded_profile_parity(tmp_path):
    kwargs = dict(steps=1, batch=4, width=8, hidden=16, nodes=4, recruited=2, top_k=2, classes=3)
    a = measure(**kwargs)
    b = measure(**kwargs, instrumented=True, trace_path=tmp_path/'trace.json')
    assert a['final_state_sha256'] == b['final_state_sha256']
    assert 'SparseRouter.forward' in b['table']
    assert (tmp_path/'trace.json').exists()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_host_evaluation_residency_state_and_schema(device, tmp_path):
    original_batches = evaluation.iter_tensor_batches
    original_evaluate = benchmark.evaluate_task
    original_select = torch.Tensor.index_select
    evaluations, selections = [], []

    def checked_select(tensor, dim, ids):
        # Check actual index operations, not merely intended residency.
        assert tensor.device == ids.device
        selections.append((tensor.device.type, ids.device.type))
        return original_select(tensor, dim, ids)

    def checked_batches(x, y, **kwargs):
        assert x.device.type == y.device.type == 'cpu'
        with patch.object(torch.Tensor, 'index_select', checked_select):
            yield from original_batches(x, y, **kwargs)

    def checked_evaluate(model, task, **kwargs):
        assert next(model.parameters()).device.type == device
        assert all(t.device.type == 'cpu' for t in (
            task.train_x, task.train_y, task.eval_x, task.eval_y))
        before = model_state_sha256(model)
        result = original_evaluate(model, task, **kwargs)
        assert before == model_state_sha256(model)
        assert result.examples == 4
        evaluations.append(result)
        return result

    kwargs = dict(device=device, steps=1, batch=4, width=8, hidden=16,
                  nodes=4, recruited=2, top_k=2, classes=3)
    with patch.object(evaluation, 'iter_tensor_batches', checked_batches), \
         patch.object(benchmark, 'evaluate_task', checked_evaluate):
        ordinary = measure(**kwargs)
        profiled = measure(**kwargs, instrumented=True, trace_path=tmp_path/'trace.json')
    assert len(evaluations) == 6  # two warmups + one measured, twice
    assert len(selections) >= 12
    assert evaluations[:3] == evaluations[3:]
    assert ordinary['final_state_sha256'] == profiled['final_state_sha256']
    frozen = json.loads((Path(__file__).parents[2]/
        'docs/rnd/sample_routing_v1/profile_smoke/results.json').read_text())
    baseline = frozen['timings'][0]['ordinary']
    assert ordinary.keys() == (baseline.keys() | {'table'})
    assert ordinary['shape'] == baseline['shape']
    # The frozen archive was produced with a CUDA-enabled PyTorch build. Even
    # CPU-only model initialization/optimizer arithmetic can differ between
    # builds/hardware, so an absolute runtime-state hash is NOT portable.
    # Preserve the frozen record and verify its internal hash consistency,
    # while requiring exact hashes for ordinary/profiled runs in this process.
    assert len(ordinary['final_state_sha256']) == 64
    assert baseline['final_state_sha256'] == frozen['timings'][0]['profiled']['final_state_sha256']
