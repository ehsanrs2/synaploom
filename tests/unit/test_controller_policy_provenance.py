"""Controller policy identity: deterministic allowlist, immutable provenance."""
from __future__ import annotations

from copy import deepcopy
import random

import pytest
import torch

from synaploom.plasticity.controller import StructuralController
from synaploom.plasticity.policies import (
    ConnectToSelectedTopR,
    FixedScheduleRecruitment,
    NoConsolidation,
    NoEdges,
    NoRecruitment,
    SeededRandomR,
    TaskBoundaryRecruitment,
    TaskBoundaryTopImportance,
)
from synaploom.research.controller_policy_provenance import (
    controller_policy_sha256,
    snapshot_controller_policy,
    verify_controller_policy,
)


def configured(**overrides):
    opts = dict(
        recruitment_policy=FixedScheduleRecruitment([7, 3, 7]),
        edge_init_policy=SeededRandomR(top_r=2, seed=41),
        consolidation_policy=TaskBoundaryTopImportance(top_n=1),
        imprint_noise_std=0.03,
        edge_gate_logit=0.2,
    )
    opts.update(overrides)
    return StructuralController(**opts)


def test_policy_identity_deterministic_readonly_and_not_resume_approval():
    ctrl = configured()
    ctrl.load_state_dict({"last_applied_step": 4})
    before = ctrl.state_dict().copy()
    rng = torch.get_rng_state().clone()
    py = random.getstate()
    record = snapshot_controller_policy(ctrl)
    assert record["recruitment"]["steps"] == [3, 7]
    assert record["edge_init"] == {"type": "SeededRandomR", "top_r": 2, "seed": 41}
    assert record["consolidation"] == {"type": "TaskBoundaryTopImportance", "top_n": 1}
    pin = controller_policy_sha256(record)
    assert len(pin) == 64
    result = verify_controller_policy(deepcopy(record), ctrl, expected_sha256=pin)
    assert result.policy_matches and result.policy_sha256 == pin
    assert result.resume_authorized is False
    assert ctrl.state_dict() == before
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py
    assert controller_policy_sha256(dict(reversed(list(record.items())))) == pin


def test_identical_controller_state_can_hide_distinct_policies():
    old = configured(recruitment_policy=FixedScheduleRecruitment([3]))
    new = configured(recruitment_policy=FixedScheduleRecruitment([999]))
    old.load_state_dict({"last_applied_step": 8})
    new.load_state_dict({"last_applied_step": 8})
    assert old.state_dict() == new.state_dict()
    record = snapshot_controller_policy(old)
    with pytest.raises(ValueError, match="differs"):
        verify_controller_policy(
            record, new, expected_sha256=controller_policy_sha256(record),
        )


@pytest.mark.parametrize("override", [
    {"recruitment_policy": FixedScheduleRecruitment([3, 9])},
    {"recruitment_policy": TaskBoundaryRecruitment()},
    {"edge_init_policy": SeededRandomR(top_r=2, seed=42)},
    {"edge_init_policy": ConnectToSelectedTopR(top_r=2)},
    {"consolidation_policy": NoConsolidation()},
    {"consolidation_policy": TaskBoundaryTopImportance(top_n=2)},
    {"imprint_noise_std": 0.07},
    {"edge_gate_logit": 0.21},
])
def test_all_identity_fields_affect_compatibility(override):
    source = configured()
    record = snapshot_controller_policy(source)
    target = configured(**override)
    with pytest.raises(ValueError, match="differs"):
        verify_controller_policy(
            record, target, expected_sha256=controller_policy_sha256(record)
        )


def test_noop_policy_variants_supported_without_side_effects():
    for ctrl in (StructuralController(), StructuralController(
        recruitment_policy=NoRecruitment(),
        edge_init_policy=NoEdges(),
        consolidation_policy=NoConsolidation(),
    ), configured(
        recruitment_policy=TaskBoundaryRecruitment(),
        edge_init_policy=ConnectToSelectedTopR(top_r=3),
        consolidation_policy=NoConsolidation(),
    )):
        record = snapshot_controller_policy(ctrl)
        assert verify_controller_policy(
            record, ctrl, expected_sha256=controller_policy_sha256(record)
        ).policy_matches


@pytest.mark.parametrize("corrupt,pattern", [
    (lambda r: r.update(schema=2), "schema"),
    (lambda r: r.update(new_field=True), "schema"),
    (lambda r: r.update(kind="bad"), "kind"),
    (lambda r: r["recruitment"].update(steps=[1, 7]), "SHA-256"),
    (lambda r: r["edge_init"].update(seed=1234), "SHA-256"),
    (lambda r: r.update(edge_gate_logit=float("nan")), "noncanonical"),
])
def test_corrupt_records_fail_closed(corrupt, pattern):
    ctrl = configured()
    record = snapshot_controller_policy(ctrl)
    pin = controller_policy_sha256(record)
    corrupt(record)
    with pytest.raises(ValueError, match=pattern):
        verify_controller_policy(record, ctrl, expected_sha256=pin)


def test_rehashed_tampering_still_rejected_against_target():
    ctrl = configured()
    record = snapshot_controller_policy(ctrl)
    record["recruitment"]["steps"] = [3, 700]
    new_pin = controller_policy_sha256(record)
    with pytest.raises(ValueError, match="differs"):
        verify_controller_policy(record, ctrl, expected_sha256=new_pin)


def test_unknown_policy_subclass_external_logger_nonfinite_are_unsupported():
    class CustomNoRecruitment(NoRecruitment):
        pass

    with pytest.raises(ValueError, match="unsupported recruitment"):
        snapshot_controller_policy(configured(recruitment_policy=CustomNoRecruitment()))
    with pytest.raises(ValueError, match="external event logger"):
        snapshot_controller_policy(configured(event_logger=object()))
    for bad in (float("nan"), float("inf"), float("-inf")):
        ctrl = configured()
        ctrl.edge_gate_logit = bad
        with pytest.raises(ValueError, match="finite"):
            snapshot_controller_policy(ctrl)


def test_invalid_expected_digest_and_structural_step_not_policy_identity():
    ctrl = configured()
    rec = snapshot_controller_policy(ctrl)
    with pytest.raises(ValueError, match="expected_sha256"):
        verify_controller_policy(rec, ctrl, expected_sha256="0")
    before = controller_policy_sha256(rec)
    ctrl.load_state_dict({"last_applied_step": 987})
    assert controller_policy_sha256(snapshot_controller_policy(ctrl)) == before
