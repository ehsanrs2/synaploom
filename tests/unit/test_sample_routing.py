import pytest
import torch
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.baselines import DenseControlModel
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.training.state_digest import model_state_sha256


def fixture(seed=7, recruited=2):
    torch.manual_seed(seed)
    model = SynapLoomModel(ModelConfig(input_dim=8, num_classes=3, graph=GraphConfig(
        num_nodes=6, initial_recruited=recruited, top_k=2, d_model=8,
        d_hidden=16, router_dim=8, q_max=0)))
    return model, torch.randn(24, 8)


@pytest.mark.parametrize('seed', [7, 19])
@pytest.mark.parametrize('recruited', [2, 6])
def test_invariance_singleton_dense_and_state(seed, recruited):
    model, x = fixture(seed, recruited)
    ref = SampleRoutingReference(model).eval()
    assert ref.model is model
    before = model_state_sha256(model)
    with torch.inference_mode():
        full = ref(x)
        for i in range(len(x)):
            singleton = ref(x[i:i+1])
            ordinary = model(x[i:i+1])
            torch.testing.assert_close(singleton.logits, ordinary.logits, atol=1e-6, rtol=1e-6)
            assert set(singleton.selected_node_ids[0].tolist()) == set(ordinary.trace.selected_node_ids.tolist())
            for node, weight in zip(singleton.selected_node_ids[0], singleton.weights[0]):
                j = (ordinary.trace.selected_node_ids == node).nonzero()[0, 0]
                torch.testing.assert_close(weight, ordinary.trace.router_weights[j], atol=1e-6, rtol=1e-6)
            for companions in (x[:0], x.flip(0), x[:3] * 100, x[3:11] * -20):
                out = ref(torch.cat([x[i:i+1], companions]))
                torch.testing.assert_close(out.logits[:1], singleton.logits, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(out.weights[:1], singleton.weights, atol=1e-6, rtol=1e-6)
                assert out.logits[0].argmax() == singleton.logits[0].argmax()
            torch.testing.assert_close(full.logits[i:i+1], singleton.logits, atol=1e-6, rtol=1e-6)
        permutation = torch.randperm(len(x))
        out = ref(x[permutation])
        torch.testing.assert_close(out.logits, full.logits[permutation], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(out.weights, full.weights[permutation], atol=1e-6, rtol=1e-6)
        assert torch.equal(out.selected_node_ids, full.selected_node_ids[permutation])
        torch.testing.assert_close(ref(x).logits, full.logits, atol=0, rtol=0)
        dense = DenseControlModel(input_dim=8, d_model=8, d_hidden=16, num_classes=3).eval()
        dense_before = model_state_sha256(dense)
        # Independent control has its own parameters; never a routed parity claim.
        from synaploom.training.evaluation import logits_from_output
        expected = logits_from_output(dense(x))
        actual = torch.cat([logits_from_output(dense(row[None])) for row in x])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert dense_before == model_state_sha256(dense)
    assert before == model_state_sha256(model)


def test_fixed_set_variable_weights_negative_control():
    model, x = fixture()
    a, b = model(x[:1]), model(x)
    assert set(a.trace.selected_node_ids.tolist()) == set(b.trace.selected_node_ids.tolist())
    assert not torch.allclose(a.trace.router_weights, b.trace.router_weights, atol=1e-6, rtol=1e-6)
    assert (a.logits[0] - b.logits[0]).abs().max() > 1e-6


def test_execution_gradient_isolation_and_ties():
    model, x = fixture(recruited=4)
    with torch.no_grad():
        model.graph_state.routing_keys.zero_()
    calls = []
    handles = [expert.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
               for i, expert in enumerate(model.executor.microcircuits)]
    before = model_state_sha256(model)
    output = SampleRoutingReference(model)(x[:3])
    assert output.selected_node_ids.tolist() == [[0, 1]] * 3
    output.logits.square().sum().backward()
    for h in handles:
        h.remove()
    assert calls == [0, 1] * 3
    for i, expert in enumerate(model.executor.microcircuits):
        assert all((p.grad is not None) == (i < 2) for p in expert.parameters())
    assert model.graph_state.routing_keys.grad[2:].count_nonzero() == 0
    assert before == model_state_sha256(model)


@pytest.mark.parametrize('kwargs', [{'q_max': 1}, {'q_max': 0, 'message_rounds': 1}])
def test_unsupported(kwargs):
    model = SynapLoomModel(ModelConfig(graph=GraphConfig(**kwargs)))
    with pytest.raises(ValueError, match='requires'):
        SampleRoutingReference(model)
