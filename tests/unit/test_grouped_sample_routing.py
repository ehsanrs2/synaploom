import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.research.grouped_sample_routing import GroupedSampleRouting
from synaploom.training.state_digest import model_state_sha256

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))]


@pytest.fixture(autouse=True)
def bounded_threads_and_precision():
    threads = torch.get_num_threads()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32


def fixture(seed, recruited, k, device):
    torch.manual_seed(seed)
    model = SynapLoomModel(ModelConfig(input_dim=5, num_classes=3,
        graph=GraphConfig(num_nodes=6, initial_recruited=recruited, top_k=k,
                          q_max=0, d_model=8, d_hidden=16, router_dim=8))).to(device)
    x = torch.randn(8, 5, device=device)
    return model, x


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('seed', [7, 19])
@pytest.mark.parametrize('recruited,k', [(2, 1), (4, 2), (6, 4)])
def test_parity_invariance_and_gradients(device, seed, recruited, k):
    model, x = fixture(seed, recruited, k, device)
    x.requires_grad_(True)
    old, new = SampleRoutingReference(model), GroupedSampleRouting(model)
    before = model_state_sha256(model)
    atol = 1e-6 if device == 'cpu' else 1e-5
    old_out, out = old(x), new(x)
    assert torch.equal(old_out.selected_node_ids, out.selected_node_ids)
    for name in ('logits', 'weights', 'queries'):
        torch.testing.assert_close(getattr(old_out, name), getattr(out, name), atol=atol, rtol=atol)
    assert torch.equal(old_out.logits.argmax(-1), out.logits.argmax(-1))
    with torch.inference_mode():
        new.eval()
        for i in range(len(x)):
            single = new(x[i:i+1])
            production = model(x[i:i+1])
            torch.testing.assert_close(single.logits, production.logits, atol=atol, rtol=atol)
            assert torch.equal(single.selected_node_ids[0], production.trace.selected_node_ids)
            torch.testing.assert_close(single.weights[0], production.trace.router_weights,
                                       atol=atol, rtol=atol)
            for companions in (x[:0], x, x.flip(0)*20):
                other = new(torch.cat((x[i:i+1], companions)))
                torch.testing.assert_close(other.logits[:1], single.logits, atol=atol, rtol=atol)
                torch.testing.assert_close(other.weights[:1], single.weights, atol=atol, rtol=atol)
                assert torch.equal(other.selected_node_ids[:1], single.selected_node_ids)
                assert other.logits[0].argmax() == single.logits[0].argmax()
        perm = torch.tensor([4, 2, 7, 0, 3, 1, 6, 5], device=device)
        reordered = new(x[perm])
        torch.testing.assert_close(reordered.logits, out.logits[perm], atol=atol, rtol=atol)
        torch.testing.assert_close(reordered.weights, out.weights[perm], atol=atol, rtol=atol)
        assert torch.equal(reordered.selected_node_ids, out.selected_node_ids[perm])
        torch.testing.assert_close(new(x).logits, out.logits, atol=0, rtol=0)
    old_out.logits.square().sum().backward()
    input_gradient = x.grad.clone()
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    x.grad = None
    new(x).logits.square().sum().backward()
    torch.testing.assert_close(x.grad, input_gradient, atol=2e-5, rtol=2e-5)
    for name, p in model.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, gradients[name], atol=2e-5, rtol=2e-5)
    assert before == model_state_sha256(model)


@pytest.mark.parametrize('device', DEVICES)
def test_exact_ties_dispatch_and_dormant_gradient_isolation(device):
    model, x = fixture(7, 4, 2, device)
    with torch.no_grad():
        model.graph_state.routing_keys.zero_()
    calls = []
    handles = [m.register_forward_hook(lambda m, a, o, i=i: calls.append((i, len(a[0]))))
               for i, m in enumerate(model.executor.microcircuits)]
    out = GroupedSampleRouting(model)(x)
    for handle in handles:
        handle.remove()
    assert out.selected_node_ids.tolist() == [[0, 1]]*len(x)
    assert calls == [(0, len(x)), (1, len(x))]
    out.logits.sum().backward()
    for i, expert in enumerate(model.executor.microcircuits):
        assert all((p.grad is not None) == (i < 2) for p in expert.parameters())
    assert model.graph_state.routing_keys.grad[2:].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cross_device_parameters_identical():
    cpu, x = fixture(19, 4, 2, 'cpu')
    gpu, _ = fixture(7, 4, 2, 'cuda')
    gpu.load_state_dict(cpu.state_dict())
    a, b = GroupedSampleRouting(cpu)(x), GroupedSampleRouting(gpu)(x.cuda())
    assert torch.equal(a.selected_node_ids, b.selected_node_ids.cpu())
    torch.testing.assert_close(a.logits, b.logits.cpu(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a.weights, b.weights.cpu(), atol=1e-5, rtol=1e-5)


def test_unsupported_input_and_nonfinite():
    with pytest.raises(ValueError, match='requires'):
        GroupedSampleRouting(SynapLoomModel(ModelConfig()))
    model, x = fixture(7, 4, 2, 'cpu')
    ref = GroupedSampleRouting(model)
    for bad in (x[:0], x[0], x[:, :2]):
        with pytest.raises(ValueError, match='nonempty'):
            ref(bad)
    before = model_state_sha256(model)
    with pytest.raises(ValueError, match='nonfinite'):
        ref(torch.full_like(x, float('nan')))
    assert before == model_state_sha256(model)
