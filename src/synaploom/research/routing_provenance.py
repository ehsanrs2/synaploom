"""Research-only routing metadata and read-only checkpoint semantic preflight.

This does NOT load pickle files or mutate a model, optimizer or RNG.
The caller supplies the extra mapping from a trusted, separately inspected
checkpoint. This is not a production load/save adapter or an authentication gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from synaploom.model import SynapLoomModel
from synaploom.research.routing_session import RoutingSpec


_TOPOLOGY_KEYS = (
    "input_dim", "num_classes", "num_nodes", "d_model", "d_hidden",
    "router_dim", "top_k", "q_max", "message_rounds",
)
_META_KEYS = frozenset((
    "schema", "granularity", "execution", "tie_policy",
    "training_granularity", "topology", "structural_version",
    "implementation_revision", "precision_policy",
))


@dataclass(frozen=True)
class RoutingPreflightV1:
    schema: int
    intent: str
    source_granularity: str
    requested_granularity: str
    source_execution: str
    requested_execution: str
    training_granularity: str
    legacy_missing_metadata: bool
    explicit_counterfactual: bool
    replay_semantics_match: bool


def _topology(model: SynapLoomModel) -> dict[str, int]:
    if not isinstance(model, SynapLoomModel):
        raise TypeError("routing provenance requires SynapLoomModel")
    g = model.config.graph
    return {
        "input_dim": model.config.input_dim,
        "num_classes": model.config.num_classes,
        "num_nodes": g.num_nodes,
        "d_model": g.d_model,
        "d_hidden": g.d_hidden,
        "router_dim": g.router_dim,
        "top_k": g.top_k,
        "q_max": g.q_max,
        "message_rounds": g.message_rounds,
    }


def make_routing_metadata(
    model: SynapLoomModel,
    spec: RoutingSpec,
    *,
    implementation_revision: str,
    precision_policy: str,
) -> dict[str, Any]:
    """Create the extra['routing'] schema; Stage 1 is always batch-trained."""
    topology = _topology(model)
    if not isinstance(spec, RoutingSpec):
        raise TypeError("spec must be RoutingSpec")
    for name, value in (
        ("implementation_revision", implementation_revision),
        ("precision_policy", precision_policy),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(name + " must be a nonempty string")
    if spec.granularity == "sample" and (
        topology["q_max"] or topology["message_rounds"]
    ):
        raise ValueError("sample routing requires q_max=0 and message_rounds=0")
    return {
        "schema": 1,
        "granularity": spec.granularity,
        "execution": spec.execution,
        "tie_policy": spec.tie_policy,
        "training_granularity": "batch",
        "topology": topology,
        "structural_version": int(model.graph_state.structural_version.item()),
        "implementation_revision": implementation_revision,
        "precision_policy": precision_policy,
    }


def preflight_routing_extra(
    extra: Mapping[str, Any] | None,
    *,
    model: SynapLoomModel,
    requested: RoutingSpec,
    intent: str = "replay",
    allow_counterfactual: bool = False,
) -> RoutingPreflightV1:
    """Pure semantic preflight; refuses mismatched replay before model loading.

    The caller must separately verify checkpoint bytes and state_dict shape,
    and use trusted checkpoint sources. This function does not load, save,
    authenticate or apply optimizer/controller/RNG state.
    """
    topology = _topology(model)
    if not isinstance(requested, RoutingSpec):
        raise TypeError("requested must be RoutingSpec")
    if intent not in ("replay", "counterfactual"):
        raise ValueError("intent must be replay or counterfactual")
    if intent == "counterfactual" and not allow_counterfactual:
        raise ValueError("explicit counterfactual authorization required")
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise TypeError("checkpoint extra must be a mapping")

    legacy = "routing" not in extra
    if legacy:
        source = RoutingSpec()
        training_granularity = "batch"
    else:
        raw = extra["routing"]
        if not isinstance(raw, Mapping) or set(raw) != _META_KEYS:
            raise ValueError("invalid routing metadata field set")
        if type(raw["schema"]) is not int or raw["schema"] != 1:
            raise ValueError("unsupported routing metadata schema")
        source = RoutingSpec(
            granularity=raw["granularity"],
            execution=raw["execution"],
            tie_policy=raw["tie_policy"],
        )
        training_granularity = raw["training_granularity"]
        if training_granularity != "batch":
            raise ValueError("unsupported sample-trained checkpoint")
        saved_topology = raw["topology"]
        if not isinstance(saved_topology, Mapping) or set(saved_topology) != set(_TOPOLOGY_KEYS):
            raise ValueError("invalid checkpoint topology schema")
        if any(type(saved_topology[key]) is not int for key in _TOPOLOGY_KEYS):
            raise ValueError("invalid checkpoint topology value")
        if dict(saved_topology) != topology:
            raise ValueError("checkpoint routing topology differs from model")
        if type(raw["structural_version"]) is not int or raw["structural_version"] < 0:
            raise ValueError("invalid checkpoint structural version")
        for key in ("implementation_revision", "precision_policy"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise ValueError("invalid routing provenance " + key)

    if requested.granularity == "sample" and (
        topology["q_max"] or topology["message_rounds"]
    ):
        raise ValueError("sample routing requires q_max=0 and message_rounds=0")
    matches = source == requested
    if intent == "replay" and not matches:
        raise ValueError("checkpoint routing semantics mismatch: replay rejected")

    return RoutingPreflightV1(
        schema=1,
        intent=intent,
        source_granularity=source.granularity,
        requested_granularity=requested.granularity,
        source_execution=source.execution,
        requested_execution=requested.execution,
        training_granularity=training_granularity,
        legacy_missing_metadata=legacy,
        explicit_counterfactual=(intent == "counterfactual"),
        replay_semantics_match=(matches and intent == "replay"),
    )
