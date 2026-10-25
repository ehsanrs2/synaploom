"""Stage-1 opt-in routing contract tests (not a training integration test)."""
from __future__ import annotations

import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.core.outputs import ModelOutput
from synaploom.model import SynapLoomModel
from synaploom.research.routing_session import (
    RoutingSession,
    RoutingSpec,
    SampleRoutingOutputV1,
)
from synaploom.training.state_digest import model_state_sha256


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA unavailable"
    )),
]


@pytest.fixture(autouse=True)
def qualified_precision():
    threads = torch.get_num_threads()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_num_threads(threads)


def make_model(device="cpu", *, recruited=4, k=2, q_max=0, rounds=0):
    torch.manual_seed(19)
    model = SynapLoomModel(ModelConfig(
        input_dim=5, num_classes=3,
        graph=GraphConfig(
            num_nodes=6, initial_recruited=recruited, top_k=k,
            q_max=q_max, message_rounds=rounds,
            d_model=8, d_hidden=16, router_dim=8,
        ),
    )).to(device)
    return model


def sample_session(model, execution):
    return RoutingSession(model, RoutingSpec(
        granularity="sample", execution=execution,
        tie_policy="node_id_ascending_v1",
    ))


@pytest.mark.parametrize("device", DEVICES)
def test_default_delegates_unmodified_production(device):
    model = make_model(device, q_max=2)
    x = torch.randn(4, 5, device=device)
    saved_keys = set(model.state_dict())
    before = model_state_sha256(model)
    with torch.no_grad():
        original = model(x)
        observed = RoutingSession(model).forward(x)
    assert type(observed) is ModelOutput
    torch.testing.assert_close(original.logits, observed.logits, atol=0, rtol=0)
    torch.testing.assert_close(original.trace.selected_node_ids,
                               observed.trace.selected_node_ids, atol=0, rtol=0)
    assert saved_keys == set(model.state_dict())
    assert before == model_state_sha256(model)
    assert not isinstance(RoutingSession(model), torch.nn.Module)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("recruited,k", [(2, 1), (4, 2), (4, 4)])
def test_sample_row_grouped_parity_accounting_and_invariance(device, recruited, k):
    model = make_model(device, recruited=recruited, k=k)
    x = torch.randn(8, 5, device=device)
    row, grouped = sample_session(model, "row_reference"), sample_session(model, "grouped")
    before = model_state_sha256(model)
    rng_before = torch.get_rng_state().clone()
    model_mode_before = model.training
    atol = 1e-6 if device == "cpu" else 1e-5
    a, b = row.forward(x), grouped.forward(x)
    assert isinstance(b, SampleRoutingOutputV1)
    assert not b.logits.requires_grad
    assert b.trace.schema_version == 1
    assert b.trace.selected_node_ids.shape == (8, k)
    assert b.trace.router_weights.shape == (8, k)
    assert b.trace.routing_context.shape == (8, 8)
    assert b.trace.effective_expert_rows == len(x) * k
    assert b.trace.actual_expert_calls == len(b.trace.unique_executed_ids)
    assert a.trace.actual_expert_calls == len(x) * k
    assert torch.equal(a.trace.selected_node_ids, b.trace.selected_node_ids)
    for field in ("logits", "representation"):
        torch.testing.assert_close(getattr(a, field), getattr(b, field),
                                   atol=atol, rtol=atol)
    for field in ("router_weights", "routing_context"):
        torch.testing.assert_close(getattr(a.trace, field), getattr(b.trace, field),
                                   atol=atol, rtol=atol)

    for anchor in (0, 3):
        single = grouped.forward(x[anchor:anchor+1])
        for companions in (x, x.flip(0) * 20):
            c = grouped.forward(torch.cat((x[anchor:anchor+1], companions)))
            torch.testing.assert_close(single.logits, c.logits[:1], atol=atol, rtol=atol)
            torch.testing.assert_close(single.trace.router_weights,
                                       c.trace.router_weights[:1], atol=atol, rtol=atol)
            assert torch.equal(single.trace.selected_node_ids,
                               c.trace.selected_node_ids[:1])
    permutation = torch.tensor([5, 4, 7, 2, 0, 3, 1, 6], device=device)
    reordered = grouped.forward(x[permutation])
    torch.testing.assert_close(b.logits[permutation], reordered.logits,
                               atol=atol, rtol=atol)
    assert torch.equal(b.trace.selected_node_ids[permutation],
                       reordered.trace.selected_node_ids)
    assert model_mode_before == model.training
    assert torch.equal(rng_before, torch.get_rng_state())
    assert before == model_state_sha256(model)


@pytest.mark.parametrize("device", DEVICES)
def test_explicit_diagnostic_backward_preserves_parameters_and_gradient_isolation(device):
    model = make_model(device, recruited=4, k=2)
    with torch.no_grad():
        model.graph_state.routing_keys.zero_()
    x = torch.randn(8, 5, device=device, requires_grad=True)
    start = model_state_sha256(model)
    reference = sample_session(model, "row_reference")
    candidate = sample_session(model, "grouped")
    a = reference.forward(x, purpose="diagnostic_backward")
    a.logits.square().sum().backward()
    old_input_grad = x.grad.clone()
    old_grads = {
        name: (None if p.grad is None else p.grad.clone())
        for name, p in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    x.grad = None
    calls = []
    handles = [
        m.register_forward_hook(
            lambda _m, inputs, _out, i=i: calls.append((i, len(inputs[0])))
        )
        for i, m in enumerate(model.executor.microcircuits)
    ]
    try:
        b = candidate.forward(x, purpose="diagnostic_backward")
    finally:
        for handle in handles:
            handle.remove()
    assert b.trace.selected_node_ids.tolist() == [[0, 1]] * len(x)
    assert calls == [(0, 8), (1, 8)]
    b.logits.square().sum().backward()
    torch.testing.assert_close(x.grad, old_input_grad, atol=2e-5, rtol=2e-5)
    for name, parameter in model.named_parameters():
        if old_grads[name] is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, old_grads[name],
                                       atol=2e-5, rtol=2e-5)
    for node_id in range(2, 6):
        assert all(p.grad is None for p in model.executor.microcircuits[node_id].parameters())
    assert torch.count_nonzero(model.graph_state.routing_keys.grad[2:]) == 0
    assert start == model_state_sha256(model)


@pytest.mark.parametrize("device", DEVICES)
def test_exact_tie_policy_and_strict_checkpoint_compatibility(device):
    model = make_model(device)
    x = torch.randn(3, 5, device=device)
    baseline_keys = set(model.state_dict())
    original_state = model.state_dict()
    with torch.no_grad():
        model.graph_state.routing_keys.zero_()
    sample = sample_session(model, "grouped")
    out = sample.forward(x)
    assert out.trace.selected_node_ids.tolist() == [[0, 1]] * 3
    assert out.trace.tie_policy == "node_id_ascending_v1"
    assert out.trace.unique_executed_ids.tolist() == [0, 1]
    assert set(model.state_dict()) == baseline_keys
    clone = make_model(device)
    clone.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        sample.forward(x).logits,
        sample_session(clone, "grouped").forward(x).logits,
        atol=0, rtol=0,
    )
    assert set(original_state) == set(clone.state_dict())


@pytest.mark.parametrize("device", DEVICES)
def test_recruitment_and_consolidation_rechecked_between_calls(device):
    model = make_model(device, recruited=2, k=2)
    session = sample_session(model, "grouped")
    x = torch.randn(3, 5, device=device)
    first = session.forward(x)
    model.graph_state.recruit(2)
    model.graph_state.consolidate([0])
    digest = model_state_sha256(model)
    second = session.forward(x)
    assert first.trace.structural_version != second.trace.structural_version
    assert len(second.trace.recruited_node_ids) == 3
    assert second.trace.effective_expert_rows == 6
    assert digest == model_state_sha256(model)


def test_fail_closed_spec_topology_purpose_and_statistics():
    model = make_model()
    for invalid in (
        ("batch", "grouped", "legacy_topk"),
        ("sample", "legacy", "node_id_ascending_v1"),
        ("sample", "grouped", "legacy_topk"),
    ):
        with pytest.raises(ValueError, match="unsupported routing spec"):
            RoutingSpec(*invalid)
    for q_max, rounds in ((1, 0), (0, 1)):
        with pytest.raises(ValueError, match="requires q_max=0"):
            sample_session(make_model(q_max=q_max, rounds=rounds), "grouped")
    session = sample_session(model, "grouped")
    x = torch.randn(3, 5)
    before = model_state_sha256(model)
    for kwargs in (
        {"purpose": "training"},
        {"purpose": "unknown"},
        {"update_statistics": True},
    ):
        with pytest.raises(ValueError):
            session.forward(x, **kwargs)
        assert before == model_state_sha256(model)
    for bad in (x[:0], x[0], x[:, :2], torch.full_like(x, float("nan"))):
        with pytest.raises(ValueError):
            session.forward(bad)
    assert before == model_state_sha256(model)
    with pytest.raises(TypeError, match="RoutingSpec"):
        RoutingSession(model, spec="sample")


def test_sample_transaction_detects_structural_change_during_forward():
    model = make_model()
    session = sample_session(model, "grouped")
    x = torch.randn(3, 5)
    handle = model.head.register_forward_hook(
        lambda _m, _inputs, _output: model.graph_state.consolidate([0])
    )
    try:
        with pytest.raises(RuntimeError, match="structural version changed"):
            session.forward(x)
    finally:
        handle.remove()
