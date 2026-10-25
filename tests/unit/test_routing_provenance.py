"""Routing provenance preflight tests; no checkpoint loading or optimizer steps."""
from copy import deepcopy
import random

import pytest
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.routing_session import RoutingSpec
from synaploom.research.routing_provenance import (
    make_routing_metadata,
    preflight_routing_extra,
)
from synaploom.training.state_digest import model_state_sha256


BATCH = RoutingSpec()
SAMPLE = RoutingSpec("sample", "grouped", "node_id_ascending_v1")
ROW = RoutingSpec("sample", "row_reference", "node_id_ascending_v1")


def model(*, edges=0):
    torch.manual_seed(19)
    return SynapLoomModel(ModelConfig(
        input_dim=5, num_classes=3,
        graph=GraphConfig(num_nodes=6, initial_recruited=4, top_k=2,
                          q_max=edges, message_rounds=0,
                          d_model=8, d_hidden=16, router_dim=8),
    ))


def meta(m, spec=SAMPLE):
    return make_routing_metadata(
        m, spec, implementation_revision="research-routing-session-v1",
        precision_policy="fp32-tf32-off",
    )


def test_explicit_sample_metadata_round_trip_and_read_only():
    m = model()
    before = model_state_sha256(m)
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    rec = meta(m)
    assert rec["schema"] == 1
    assert rec["training_granularity"] == "batch"
    assert rec["granularity"] == "sample"
    assert rec["topology"]["top_k"] == 2
    assert rec["structural_version"] == 0
    out = preflight_routing_extra(
        {"routing": rec}, model=m, requested=SAMPLE, intent="replay"
    )
    assert out.replay_semantics_match and not out.legacy_missing_metadata
    assert out.source_granularity == "sample"
    assert not out.explicit_counterfactual
    assert model_state_sha256(m) == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng


def test_missing_metadata_legacy_default_and_explicit_counterfactual():
    m = model()
    legacy = preflight_routing_extra({}, model=m, requested=BATCH)
    assert legacy.legacy_missing_metadata and legacy.replay_semantics_match
    with pytest.raises(ValueError, match="replay rejected"):
        preflight_routing_extra({}, model=m, requested=SAMPLE)
    with pytest.raises(ValueError, match="authorization"):
        preflight_routing_extra({}, model=m, requested=SAMPLE,
                                intent="counterfactual")
    sample = preflight_routing_extra(
        {}, model=m, requested=SAMPLE,
        intent="counterfactual", allow_counterfactual=True
    )
    assert sample.legacy_missing_metadata and sample.explicit_counterfactual
    assert not sample.replay_semantics_match
    assert sample.training_granularity == "batch"


def test_sample_execution_change_requires_counterfactual():
    m = model()
    extra = {"routing": meta(m, ROW)}
    with pytest.raises(ValueError, match="replay rejected"):
        preflight_routing_extra(extra, model=m, requested=SAMPLE)
    out = preflight_routing_extra(
        extra, model=m, requested=SAMPLE,
        intent="counterfactual", allow_counterfactual=True
    )
    assert out.explicit_counterfactual
    assert out.source_execution == "row_reference"
    assert out.requested_execution == "grouped"


@pytest.mark.parametrize("mutator,pattern", [
    (lambda x: x.update(schema=2), "schema"),
    (lambda x: x.update(training_granularity="sample"), "sample-trained"),
    (lambda x: x.update(tie_policy="legacy_topk"), "unsupported routing spec"),
    (lambda x: x.update(structural_version=-1), "structural version"),
    (lambda x: x.update(implementation_revision=""), "provenance"),
    (lambda x: x.update(precision_policy=None), "provenance"),
    (lambda x: x.update(topology={"num_nodes": 6}), "topology schema"),
    (lambda x: x["topology"].update(input_dim=9), "topology differs"),
    (lambda x: x.update(unregistered_field=True), "field set"),
])
def test_corrupted_or_incompatible_metadata_rejects_without_mutation(mutator, pattern):
    m = model()
    before = model_state_sha256(m)
    rng = torch.get_rng_state().clone()
    corrupted = deepcopy(meta(m))
    mutator(corrupted)
    with pytest.raises(ValueError, match=pattern):
        preflight_routing_extra(
            {"routing": corrupted}, model=m, requested=SAMPLE
        )
    assert before == model_state_sha256(m)
    assert torch.equal(rng, torch.get_rng_state())


def test_graph_topology_and_input_guards():
    m = model(edges=2)
    batch = meta(m, BATCH)
    assert preflight_routing_extra({"routing": batch}, model=m, requested=BATCH).replay_semantics_match
    with pytest.raises(ValueError, match="requires q_max"):
        meta(m, SAMPLE)
    with pytest.raises(ValueError, match="requires q_max"):
        preflight_routing_extra({}, model=m, requested=SAMPLE,
                                intent="counterfactual", allow_counterfactual=True)
    with pytest.raises(TypeError):
        preflight_routing_extra({}, model=m, requested="sample")
    with pytest.raises(TypeError):
        preflight_routing_extra([], model=m, requested=BATCH)
    with pytest.raises(ValueError, match="intent"):
        preflight_routing_extra({}, model=m, requested=BATCH, intent="train")


def test_checkpoint_metadata_tracks_structural_version_without_loading():
    m = model()
    old = meta(m)
    m.graph_state.recruit(4)
    new = meta(m)
    assert new["structural_version"] == old["structural_version"] + 1
    assert new["topology"] == old["topology"]
    assert preflight_routing_extra({"routing": new}, model=m, requested=SAMPLE).replay_semantics_match
