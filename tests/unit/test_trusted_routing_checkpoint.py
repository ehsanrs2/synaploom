"""Trusted model-only routing loader: CPU gates, no optimizer/training replay."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import random

import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.routing_provenance import make_routing_metadata
from synaploom.research.routing_session import RoutingSpec
from synaploom.research.trusted_routing_checkpoint import (
    load_trusted_routing_model_checkpoint,
)
from synaploom.training.checkpoint import save_checkpoint
from synaploom.training.state_digest import model_state_sha256


BATCH = RoutingSpec()
SAMPLE = RoutingSpec("sample", "grouped", "node_id_ascending_v1")


def make_model(seed=19):
    torch.manual_seed(seed)
    model = SynapLoomModel(ModelConfig(
        input_dim=5, num_classes=3,
        graph=GraphConfig(
            num_nodes=6, initial_recruited=4, top_k=2, q_max=0,
            message_rounds=0, d_model=8, d_hidden=16, router_dim=8,
        ),
    ))
    return model


def save(tmp_path, source, *, spec=SAMPLE, with_routing=True):
    path = tmp_path / "checkpoint.pt"
    extra = {}
    if with_routing:
        extra["routing"] = make_routing_metadata(
            source, spec,
            implementation_revision="test-commit",
            precision_policy="fp32-tf32-off",
        )
    source_optim = torch.optim.Adam(source.parameters(), lr=0.001)
    save_checkpoint(
        path, source, optimizer=source_optim,
        global_step=17, extra=extra,
    )
    return path, sha256(path.read_bytes()).hexdigest()


def call(path, digest, target, *, requested=SAMPLE, **kwargs):
    return load_trusted_routing_model_checkpoint(
        path, target, requested=requested, expected_sha256=digest,
        trusted_source=True, **kwargs,
    )


def test_trusted_exact_sha_model_only_load_and_optimizer_rng_mode_preservation(tmp_path):
    source = make_model()
    source.graph_state.recruit(4)
    path, digest = save(tmp_path, source)
    target = make_model(67)
    target.eval()
    optimizer = torch.optim.Adam(target.parameters(), lr=0.04)
    optimizer_before = deepcopy(optimizer.state_dict())
    rng_before = torch.get_rng_state().clone()
    py_before = random.getstate()
    result = call(path, digest, target)
    assert result.checkpoint_sha256 == digest
    assert result.checkpoint_bytes == path.stat().st_size
    assert result.global_step == 17
    assert result.routing.replay_semantics_match
    assert result.routing.source_granularity == "sample"
    assert not result.optimizer_restored
    assert not result.controller_restored
    assert not result.rng_restored
    assert model_state_sha256(target) == model_state_sha256(source)
    assert result.loaded_model_sha256 == model_state_sha256(source)
    assert not target.training
    assert optimizer.state_dict() == optimizer_before
    assert torch.equal(rng_before, torch.get_rng_state())
    assert random.getstate() == py_before


def test_reject_untrusted_missing_digest_wrong_digest_and_oversize_before_mutation(tmp_path):
    source = make_model()
    path, digest = save(tmp_path, source)
    target = make_model(91)
    old = model_state_sha256(target)
    with pytest.raises(ValueError, match="trusted_source"):
        load_trusted_routing_model_checkpoint(
            path, target, requested=SAMPLE, expected_sha256=digest,
        )
    with pytest.raises(ValueError, match="expected_sha256"):
        call(path, "not-digest", target)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        call(path, "0" * 64, target)
    with pytest.raises(ValueError, match="byte limit"):
        call(path, digest, target, max_bytes=10)
    assert old == model_state_sha256(target)


def test_legacy_requires_explicit_counterfactual_and_never_calls_resume(tmp_path):
    source = make_model()
    path, digest = save(tmp_path, source, with_routing=False)
    target = make_model(60)
    before = model_state_sha256(target)
    with pytest.raises(ValueError, match="replay rejected"):
        call(path, digest, target)
    assert before == model_state_sha256(target)
    with pytest.raises(ValueError, match="authorization"):
        call(path, digest, target, intent="counterfactual")
    assert before == model_state_sha256(target)
    result = call(
        path, digest, target, intent="counterfactual",
        allow_counterfactual=True,
    )
    assert result.routing.explicit_counterfactual
    assert result.routing.legacy_missing_metadata
    assert not result.routing.replay_semantics_match
    assert model_state_sha256(target) == model_state_sha256(source)


@pytest.mark.parametrize("corrupt,match", [
    (lambda p: p["model"].pop("graph_state.node_state"), "keys mismatch"),
    (lambda p: p["model"].update(
        {"graph_state.node_state": p["model"]["graph_state.node_state"].float()}),
        "shape/dtype/layout"),
    (lambda p: p["model"].update({"unknown.state": torch.zeros(1)}), "keys mismatch"),
    (lambda p: p["extra"]["routing"]["topology"].update(input_dim=999),
        "topology differs"),
    (lambda p: p["extra"]["routing"].update(structural_version=999),
        "structural version"),
    (lambda p: p["extra"]["routing"].update(training_granularity="sample"),
        "sample-trained"),
    (lambda p: p["model"]["graph_state.node_plasticity"].__setitem__(5, 1.0),
        "dormant node"),
    (lambda p: p.update(global_step=-1), "global_step"),
    (lambda p: p.pop("rng_state"), "missing required"),
])
def test_invalid_payload_rejected_before_live_state_changes(tmp_path, corrupt, match):
    source = make_model()
    path, _ = save(tmp_path, source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    corrupt(payload)
    torch.save(payload, path)
    digest = sha256(path.read_bytes()).hexdigest()
    target = make_model(50)
    state_before = model_state_sha256(target)
    rng_before = torch.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError), match=match):
        call(path, digest, target)
    assert model_state_sha256(target) == state_before
    assert torch.equal(rng_before, torch.get_rng_state())


def test_explicit_batch_metadata_and_counterfactual_semantics(tmp_path):
    source = make_model()
    path, digest = save(tmp_path, source, spec=BATCH)
    target = make_model(72)
    with pytest.raises(ValueError, match="replay rejected"):
        call(path, digest, target)
    assert call(path, digest, target, requested=BATCH).routing.replay_semantics_match


def test_commit_error_rolls_back_live_model_state(tmp_path, monkeypatch):
    source = make_model()
    path, digest = save(tmp_path, source)
    target = make_model(50)
    before = model_state_sha256(target)
    original = SynapLoomModel.load_state_dict
    throw_once = [True]

    def fail_after_copy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is target and throw_once[0]:
            throw_once[0] = False
            raise RuntimeError("injected commit failure")
        return result

    monkeypatch.setattr(SynapLoomModel, "load_state_dict", fail_after_copy)
    with pytest.raises(RuntimeError, match="injected commit failure"):
        call(path, digest, target)
    assert model_state_sha256(target) == before


def test_tampered_checkpoint_bytes_with_matching_digest_do_not_change_model(tmp_path):
    path = tmp_path / "tampered.pt"
    path.write_bytes(b"not a valid PyTorch checkpoint")
    target = make_model()
    before = model_state_sha256(target)
    digest = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(Exception):
        call(path, digest, target)
    assert model_state_sha256(target) == before
