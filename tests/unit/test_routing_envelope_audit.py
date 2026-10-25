"""Bounded trusted full-envelope shadow audit tests; no scientific training."""
from __future__ import annotations

import copy
from hashlib import sha256
import random

import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.plasticity.controller import StructuralController
from synaploom.research.routing_envelope_audit import audit_trusted_routing_envelope
from synaploom.research.routing_provenance import make_routing_metadata
from synaploom.research.routing_session import RoutingSpec
from synaploom.training.checkpoint import save_checkpoint
from synaploom.training.state_digest import model_state_sha256


def make_bundle(seed=19, *, opt_type="adam"):
    torch.manual_seed(seed)
    model = SynapLoomModel(ModelConfig(input_dim=5, num_classes=3,
        graph=GraphConfig(num_nodes=6, initial_recruited=4, top_k=2,
            q_max=0, message_rounds=0, d_model=8, d_hidden=16, router_dim=8)))
    if opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    ctrl = StructuralController()
    return model, optimizer, ctrl


def write_checkpoint(tmp_path, bundle, *, malformed=None):
    model, optimizer, ctrl = bundle
    extra = {"routing": make_routing_metadata(
        model, RoutingSpec(),
        implementation_revision="envelope-audit-test",
        precision_policy="fp32-tf32-off")}
    path = tmp_path / "trusted.pt"
    save_checkpoint(path, model, optimizer=optimizer, controller=ctrl,
        global_step=17, extra=extra)
    if malformed is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        malformed(payload)
        torch.save(payload, path)
    return path, sha256(path.read_bytes()).hexdigest()


def audit(path, digest, bundle, **kwargs):
    return audit_trusted_routing_envelope(
        path, *bundle, requested=RoutingSpec(),
        expected_sha256=digest, trusted_source=True, **kwargs)


def seeded_optimizer_state(bundle):
    _, optimizer, _ = bundle
    # Synthetic gradient assignment only. NOT model training or an optimizer
    # transaction in the checkpoint audit under test.
    for group in optimizer.param_groups:
        for p in group["params"]:
            p.grad = torch.full_like(p, 0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_adam_full_envelope_shadow_and_live_unchanged(tmp_path):
    source = make_bundle()
    seeded_optimizer_state(source)
    source[0].graph_state.recruit(4)
    source[2].load_state_dict({"last_applied_step": 5})
    path, digest = write_checkpoint(tmp_path, source)
    target = make_bundle(103)
    seeded_optimizer_state(target)
    target[2].load_state_dict({"last_applied_step": 2})
    target[0].eval()
    before = model_state_sha256(target[0])
    before_opt = copy.deepcopy(target[1].state_dict())
    before_ctrl = copy.deepcopy(target[2].state_dict())
    before_rng = torch.get_rng_state().clone()
    before_python = random.getstate()
    result = audit(path, digest, target)
    assert result.checkpoint_sha256 == digest
    assert result.global_step == 17
    assert result.routing.replay_semantics_match
    assert result.shadow_model_sha256 == model_state_sha256(source[0])
    assert result.optimizer_groups == 1
    assert result.optimizer_state_entries == len(source[1].state_dict()["state"])
    assert result.controller_last_applied_step == 5
    assert result.cpu_rng_checked and not result.live_state_restored
    assert not result.resume_authorized
    assert model_state_sha256(target[0]) == before
    assert not target[0].training
    assert target[2].state_dict() == before_ctrl
    assert target[1].state_dict()["param_groups"] == before_opt["param_groups"]
    for k, v in before_opt["state"].items():
        for name, value in v.items():
            actual = target[1].state_dict()["state"][k][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, actual)
            else:
                assert value == actual
    assert torch.equal(before_rng, torch.get_rng_state())
    assert random.getstate() == before_python


def test_sgd_momentum_shadow_preserves_optimizer_param_links(tmp_path):
    source = make_bundle(opt_type="sgd")
    seeded_optimizer_state(source)
    path, digest = write_checkpoint(tmp_path, source)
    target = make_bundle(77, opt_type="sgd")
    got = audit(path, digest, target)
    assert got.optimizer_state_entries == len(list(source[0].parameters()))
    assert got.optimizer_groups == 1
    assert not got.resume_authorized


@pytest.mark.parametrize("mutator,pattern", [
    (lambda p: p.pop("optimizer"), "requires optimizer"),
    (lambda p: p.pop("controller"), "requires optimizer"),
    (lambda p: p["controller"].update(last_applied_step="bad"), "controller state schema"),
    (lambda p: p["controller"].update(last_applied_step=30), "controller step"),
    (lambda p: p["optimizer"]["param_groups"][0].update(lr=0.2), "optimizer state scalar"),
    (lambda p: p["optimizer"]["param_groups"][0]["params"].pop(), "parameter group"),
    (lambda p: p["rng_state"].update(torch_cpu="bad"), "CPU torch RNG"),
    (lambda p: p["rng_state"].pop("python"), "RNG state keys"),
    (lambda p: p["extra"]["routing"].update(tie_policy="node_id_ascending_v1"), "unsupported routing spec"),
    (lambda p: p["model"].pop("graph_state.node_state"), "keys mismatch"),
])
def test_corrupt_envelopes_rejected_without_live_mutation(tmp_path, mutator, pattern):
    source = make_bundle()
    seeded_optimizer_state(source)
    path, _ = write_checkpoint(tmp_path, source, malformed=mutator)
    target = make_bundle(112)
    before = model_state_sha256(target[0])
    rng = torch.get_rng_state().clone()
    original_controller = target[2].state_dict().copy()
    with pytest.raises((ValueError, RuntimeError), match=pattern):
        audit(path, sha256(path.read_bytes()).hexdigest(), target)
    assert before == model_state_sha256(target[0])
    assert torch.equal(rng, torch.get_rng_state())
    assert target[2].state_dict() == original_controller


def test_trust_pin_bounds_and_sample_train_not_authorized(tmp_path):
    source = make_bundle()
    path, digest = write_checkpoint(tmp_path, source)
    target = make_bundle(22)
    original = model_state_sha256(target[0])
    with pytest.raises(ValueError, match="trusted_source"):
        audit_trusted_routing_envelope(
            path, *target, requested=RoutingSpec(), expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        audit(path, "0" * 64, target)
    with pytest.raises(ValueError, match="byte limit"):
        audit(path, digest, target, max_bytes=10)
    with pytest.raises(ValueError, match="legacy batch"):
        audit_trusted_routing_envelope(
            path, *target, requested=RoutingSpec(
                "sample", "grouped", "node_id_ascending_v1"),
            expected_sha256=digest, trusted_source=True)
    assert model_state_sha256(target[0]) == original


def test_optimizer_link_mismatch_rejected_before_unpickling(tmp_path):
    source = make_bundle()
    path, digest = write_checkpoint(tmp_path, source)
    target = make_bundle(11)
    # Optimizer intentionally bound to a DIFFERENT set of parameters.
    other = make_bundle(17)
    bad = (target[0], other[1], target[2])
    with pytest.raises(ValueError, match="exactly the model"):
        audit(path, digest, bad)
