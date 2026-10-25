import pytest
import torch
from synaploom.baselines import DenseControlModel
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.batch_composition import diagnose, grouping, compare, tensor_sha256
from synaploom.training.state_digest import model_state_sha256


def fixture():
    torch.manual_seed(7)
    return torch.randn(24, 8), torch.arange(24) % 3


def test_grouping():
    x, y = fixture()
    for kind in ('same', 'other', 'mixed'):
        ids = grouping(y, 0, 8, kind, 17)
        assert ids == grouping(y, 0, 8, kind, 17)
        assert ids[0] == 0 and len(set(ids)) == 8
        if kind != 'mixed':
            assert all((y[i] == y[0]) == (kind == 'same') for i in ids[1:])
    with pytest.raises(ValueError):
        grouping(y, 0, 9, 'same', 7)


def test_dense_and_shared_accounting():
    x, y = fixture()
    model = DenseControlModel(input_dim=8, d_model=8, d_hidden=16, num_classes=3)
    model.train()
    before = model_state_sha256(model)
    result = diagnose(model, x, y, anchors=[0, 1], sizes=(2, 4))
    assert model.training and result['state_sha256'] == before == model_state_sha256(model)
    assert result['unique_anchors'] == 2 and len(result['rows']) == 12
    for row in result['rows']:
        assert row['member_ids'].count(row['anchor_id']) == 1
        assert row['anchor_sha256'] == tensor_sha256(x[row['anchor_id']])
        assert row['logit_linf'] < 1e-6
        assert row['permutation']['logit_linf'] < 1e-6
        assert row['weight_l1'] == 0
    for summary in result['summary'].values():
        assert summary['correct'] == summary['baseline_correct']


def test_fixed_set_weight_dependence_and_permutation():
    x, y = fixture()
    model = SynapLoomModel(ModelConfig(graph=GraphConfig(num_nodes=6, initial_recruited=2,
        top_k=2, d_model=8, d_hidden=16, router_dim=8), input_dim=8, num_classes=3))
    result = diagnose(model, x, y, anchors=[0, 1], sizes=(4,))
    assert all(r['jaccard'] == 1 for r in result['rows'])
    assert any(r['weight_l1'] > 1e-6 for r in result['rows'])
    assert any(r['logit_linf'] > 1e-6 for r in result['rows'])
    assert all(r['permutation']['logit_linf'] < 1e-6 for r in result['rows'])
    assert result == diagnose(model, x, y, anchors=[0, 1], sizes=(4,))


def test_weight_alignment():
    a = (torch.tensor([1., 2.]), {1: .3, 2: .7}, [0.])
    b = (a[0], {2: .7, 1: .3}, [0.])
    assert compare(a, b)['weight_l1'] == 0
    b = (a[0], {2: .7, 3: .3}, [0.])
    assert compare(a, b)['weight_l1'] == pytest.approx(.6)


def test_mutation_rejected_even_on_exception():
    class Mutator(torch.nn.Module):
        def __init__(self, fail):
            super().__init__()
            self.register_buffer('counter', torch.zeros(1))
            self.fail = fail
        def forward(self, x):
            self.counter.add_(1)
            if self.fail:
                raise ValueError('failure')
            return x[:, :3]
    x, y = fixture()
    for fail in (False, True):
        model = Mutator(fail)
        with pytest.raises(RuntimeError, match='mutation'):
            diagnose(model, x, y, anchors=[0], sizes=(2,))
        assert model.training


def test_duplicate_ids_anchors_rejected():
    x, y = fixture()
    model = torch.nn.Linear(8, 3)
    with pytest.raises(ValueError):
        diagnose(model, x, y, sample_ids=[0]*24)
    with pytest.raises(ValueError):
        diagnose(model, x, y, anchors=[0, 0])


def test_grouping_rng_isolation_and_mixed():
    _, y = fixture()
    state = torch.random.get_rng_state().clone()
    for size in (2, 4, 8):
        ids = grouping(y, 0, size, 'mixed', 7)
        assert len(set(y[ids].tolist())) > 1
    assert torch.equal(state, torch.random.get_rng_state())
