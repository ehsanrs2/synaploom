import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.research.grouped_sample_routing import GroupedSampleRouting
from synaploom.research.routing_design_probes import SingletonProbe, SlotRestoreProbe, PrecomputedRowsProbe
from synaploom.training.state_digest import model_state_sha256


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('probe', [SingletonProbe, SlotRestoreProbe, PrecomputedRowsProbe])
@pytest.mark.parametrize('batch,recruited,k', [(1, 4, 4), (8, 4, 1), (8, 16, 4)])
def test_probe_parity_mutations_and_isolation(device, probe, batch, recruited, k):
    threads = torch.get_num_threads()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(19)
        model = SynapLoomModel(ModelConfig(input_dim=5, num_classes=3,
            graph=GraphConfig(q_max=0, num_nodes=18, initial_recruited=recruited,
                top_k=k, d_model=8, d_hidden=16, router_dim=8))).to(device)
        x = torch.randn(batch, 5, device=device, requires_grad=True)
        candidate, baseline = probe(model), GroupedSampleRouting(model)
        atol = 1e-6 if device == 'cpu' else 1e-5
        for mutation in (False, True):
            if mutation:
                model.graph_state.recruit(recruited)
                model.graph_state.consolidate([0])
            before = model_state_sha256(model)
            model.zero_grad(set_to_none=True)
            x.grad = None
            a = baseline(x)
            a.logits.square().sum().backward()
            expected = {n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}
            expected_x = x.grad.clone()
            model.zero_grad(set_to_none=True)
            x.grad = None
            b = candidate(x)
            assert torch.equal(a.selected_node_ids, b.selected_node_ids)
            torch.testing.assert_close(a.logits, b.logits, atol=atol, rtol=atol)
            torch.testing.assert_close(a.weights, b.weights, atol=atol, rtol=atol)
            if probe is SingletonProbe and batch == 1:
                exact = SampleRoutingReference(model)(x)
                for field in ('logits', 'weights', 'queries', 'selected_node_ids'):
                    assert torch.equal(getattr(b, field), getattr(exact, field))
            elif probe in (SlotRestoreProbe, PrecomputedRowsProbe):
                # These ablations change only integer dispatch/restoration.
                assert torch.equal(a.logits, b.logits)
                assert torch.equal(a.weights, b.weights)
            b.logits.square().sum().backward()
            torch.testing.assert_close(x.grad, expected_x, atol=2e-5, rtol=2e-5)
            for name,p in model.named_parameters():
                if expected[name] is None:
                    assert p.grad is None
                else:
                    torch.testing.assert_close(p.grad, expected[name], atol=2e-5, rtol=2e-5)
            calls=[]
            handles=[m.register_forward_hook(lambda m,a,o,i=i:calls.append((i,len(a[0]))))
                     for i,m in enumerate(model.executor.microcircuits)]
            with torch.inference_mode():
                out = candidate(x)
                for handle in handles: handle.remove()
                assert sum(n for _,n in calls) == batch*k
                assert {i for i,_ in calls} == set(out.selected_node_ids.flatten().tolist())
                assert len(calls) == len(set(out.selected_node_ids.flatten().tolist()))
                single = candidate(x[:1])
                composed = candidate(torch.cat((x[:1], x.flip(0)*20)))
                torch.testing.assert_close(single.logits, composed.logits[:1], atol=atol, rtol=atol)
                torch.testing.assert_close(single.weights, composed.weights[:1], atol=atol, rtol=atol)
                assert torch.equal(single.selected_node_ids, composed.selected_node_ids[:1])
                perm = torch.arange(batch-1,-1,-1,device=device)
                p = candidate(x[perm])
                torch.testing.assert_close(p.logits, out.logits[perm], atol=atol, rtol=atol)
                torch.testing.assert_close(p.weights, out.weights[perm], atol=atol, rtol=atol)
                assert torch.equal(p.selected_node_ids, out.selected_node_ids[perm])
            assert model_state_sha256(model) == before
        with torch.no_grad(): model.graph_state.routing_keys.zero_()
        tied = candidate(x)
        assert tied.selected_node_ids.tolist() == [list(range(k))]*batch
        for bad in (x[:0], x[:, :2]):
            with pytest.raises(ValueError): candidate(bad)
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
