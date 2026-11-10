"""D2 CPU-only strict output-boundary qualification, no science or GPU."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

import pytest
import torch

from synaploom.data.streams import iter_tensor_batches
from synaploom.protocols.h1a_order_sensitivity import CLASS_ORDERS, EXPECTED_ORDER_FINGERPRINTS
from synaploom.protocols.h1a2 import H1A2_FEATURE_SHA256
from synaploom.research.fixed_sparse_baseline import (
    FixedSparseMLP, install_fixed_mask_guard,
)
from synaploom.research.m13b_d2_runner_output_contract_cpu import (
    ARMS, REFERENCE_SOURCE_MAIN, STEPS_PER_TRAJECTORY,
    SYNTHETIC_ENGINEERING_SEEDS, reference_batch_ranges,
    reference_position_digest, validate_shape_and_policy,
    validate_synthetic_registry,
)
from synaploom.training.continual import (
    ContinualConfig, ContinualRunResult, StructuralSnapshot,
)


def _mask(seed: int) -> str:
    # Explicitly a fabricated digest, never model or actual data attestation.
    return sha256(f"synthetic-mask-fixture-{seed}".encode()).hexdigest()


def _fixture(seed: int = 13, arm: str = "progressive_reserve") -> dict:
    accuracy = torch.full((10, 10), float("nan"))
    losses = torch.full((10, 10), float("nan"))
    for stage in range(10):
        for task in range(stage + 1):
            accuracy[stage, task] = 0.3 + 0.005 * task + 0.001 * stage
            losses[stage, task] = 1.0 + task / 100
    curves = [
        [
            {
                "epoch": epoch + 1,
                "global_step": (stage * 20 + epoch + 1) * 40,
                "accuracy": .40 + .002 * stage + .0005 * (epoch + 1),
                "loss": 1.0,
            }
            for epoch in range(20)
        ]
        for stage in range(10)
    ]
    config = ContinualConfig(
        seed=seed, device="cpu", epochs_per_task=20,
        batch_size=128, eval_batch_size=256,
        evaluate_future_tasks=False, apply_final_task_boundary=False,
        restrict_logits_to_seen_classes=True,
        current_task_eval_every_epochs=1,
    )
    manifest = [
        {
            "task_id": stage,
            "classes": list(CLASS_ORDERS["O0"][stage * 10:(stage + 1) * 10]),
            "train_examples": 5000,
            "eval_examples": 1000,
        }
        for stage in range(10)
    ]
    structural = [
        StructuralSnapshot(
            stage=stage, task_id=stage, structural_version=None,
            recruited_nodes=None, active_edges=None,
            mean_executed_nodes=None, selection_entropy=None,
            max_selection_share=None, node_selection_counts={},
        )
        for stage in range(10)
    ]
    metadata = {
        "kind": "SYNTHETIC_CPU_D2_RUNNER_FIXTURE_V1",
        "synthetic": True, "scientific_claim": "NONE",
        "protected_eval_accessed": False, "actual_optimizer_steps": 0,
        "attempt_index": 0, "terminal_state": "SYNTHETIC_COMPLETE_NOT_TRAINED",
        "arm": arm, "seed": seed, "order_id": "O0",
        "source_commit_declared": REFERENCE_SOURCE_MAIN,
        "feature_sha256_declared": H1A2_FEATURE_SHA256,
        "order_sha256_declared": EXPECTED_ORDER_FINGERPRINTS["O0"],
        "position_only_batch_digest_declared": reference_position_digest(),
        "mask_sha256_declared": _mask(seed) if arm == "fixed_sparse_mlp" else None,
    }
    result = ContinualRunResult(
        accuracy_matrix=accuracy, eval_loss_matrix=losses,
        train_loss_by_task=[1.0] * 10,
        structural_history=structural, global_steps=8000,
        run_config=asdict(config), stream_manifest=manifest, metadata=metadata,
        current_task_learning_curves=curves,
        train_seconds_by_task=[.1] * 10,
        eval_seconds_by_stage=[.2] * 10,
        max_gradient_norm_by_task=[1.0] * 10,
        max_parameter_abs_by_task=[2.0] * 10,
    )
    return result.to_dict()  # Production serialization boundary, but fabricated contents.


def test_reference_batch_positions_match_real_iterator_with_tail_232():
    assert reference_batch_ranges() == ((0, 256), (256, 512), (512, 768), (768, 1000))
    assert len(reference_position_digest()) == 64
    ids = torch.arange(1000, dtype=torch.int64)
    batches = list(iter_tensor_batches(
        ids[:, None].float(), ids, batch_size=256, shuffle=False, seed=999
    ))
    assert [len(y) for _, y in batches] == [256, 256, 256, 232]
    assert torch.equal(torch.cat([y for _, y in batches]), ids)


@pytest.mark.parametrize("arm", ARMS)
def test_exact_continual_run_result_to_dict_boundary_passes_synthetic_only(arm):
    data = _fixture(13, arm)
    before = deepcopy(data)
    report = validate_shape_and_policy(
        data, seed=13, arm=arm,
        expected_mask_sha256=_mask(13) if arm == "fixed_sparse_mlp" else None,
    )
    assert data == before
    assert report["status"].startswith("SYNTHETIC_CPU_SHAPE_GUARD_PASS")
    assert report["independent_source_feature_mask_attestation"] is False
    assert report["science_executable"] is False
    assert report["runner_integrated_with_science"] is False
    assert report["position_only_batch_layout"][-1] == [768, 1000]


@pytest.mark.parametrize("mutator", [
    lambda x: x.update(global_steps=7999),
    lambda x: x["current_task_learning_curves"][0][0].update(global_step=41),
    lambda x: x["current_task_learning_curves"][2][19].update(global_step=9999),
    lambda x: x["current_task_learning_curves"][9].pop(),
    lambda x: x["accuracy_matrix"][5].__setitem__(4, None),
    lambda x: x["eval_loss_matrix"][3].__setitem__(2, float("nan")),
    lambda x: x["eval_loss_matrix"][0].__setitem__(9, 1.0),
    lambda x: x["stream_manifest"][1].update(train_examples=4999),
    lambda x: x["stream_manifest"][5]["classes"].reverse(),
    lambda x: x["train_seconds_by_task"].pop(),
    lambda x: x["structural_history"][7].update(stage=8),
    lambda x: x["metadata"].update(actual_optimizer_steps=8000),
    lambda x: x["metadata"].update(scientific_claim="BETTER"),
    lambda x: x["metadata"].update(protected_eval_accessed=True),
    lambda x: x["metadata"].update(feature_sha256_declared="0"*64),
    lambda x: x["metadata"].update(order_sha256_declared="1"*64),
    lambda x: x["metadata"].update(position_only_batch_digest_declared="2"*64),
    lambda x: x["metadata"].update(attempt_index=1),
    lambda x: x["run_config"].update(device="cuda:0"),
    lambda x: x["run_config"].update(seed=501),
    lambda x: x.update(peak_cuda_memory_bytes=5000),
    lambda x: x.update(evaluation_batch_size_results={"128": {}}),
])
def test_missing_inconsistent_or_false_provenance_rejected(mutator):
    data = _fixture()
    mutator(data)
    with pytest.raises(ValueError):
        validate_shape_and_policy(data, seed=13, arm="progressive_reserve")


def test_wrong_mask_digest_and_reserved_scientific_seed_fail_closed():
    fixed = _fixture(17, "fixed_sparse_mlp")
    with pytest.raises(ValueError, match="mask digest"):
        validate_shape_and_policy(fixed, seed=17, arm="fixed_sparse_mlp",
                                  expected_mask_sha256=sha256(b"other").hexdigest())
    with pytest.raises(ValueError):
        validate_shape_and_policy(_fixture(), seed=501, arm="progressive_reserve")


def test_complete_two_arm_pair_registry_and_duplicate_or_missing_slots():
    records = [
        {"seed": seed, "arm": arm, "observation": _fixture(seed, arm)}
        for seed in SYNTHETIC_ENGINEERING_SEEDS for arm in ARMS
    ]
    expected = {seed: _mask(seed) for seed in SYNTHETIC_ENGINEERING_SEEDS}
    report = validate_synthetic_registry(records, expected_mask_sha256_by_seed=expected)
    assert report["record_count"] == 4
    assert report["paired_seed_count"] == 2
    assert report["actual_optimizer_steps"] == 0
    assert report["source_feature_mask_crypto_attested"] is False
    assert not report["science_executable"]
    with pytest.raises(ValueError):
        validate_synthetic_registry(records[:-1], expected_mask_sha256_by_seed=expected)
    duplicated = deepcopy(records)
    duplicated[2] = duplicated[0]
    with pytest.raises(ValueError):
        validate_synthetic_registry(duplicated, expected_mask_sha256_by_seed=expected)
    tampered = deepcopy(records)
    tampered[1]["observation"]["metadata"]["terminal_state"] = "FAILED"
    with pytest.raises(ValueError):
        validate_synthetic_registry(tampered, expected_mask_sha256_by_seed=expected)


def test_real_fixed_mask_post_optimizer_hook_cpu_and_state_leakage_guard():
    # Small CPU model with actual SGD momentum, not the 8k science path.
    torch.manual_seed(71)
    model = FixedSparseMLP(
        input_dim=2, d_model=2, d_hidden=4, num_classes=2,
        active_first=3, active_second=3, mask_seed=41,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=.0005)
    original = model.mask_digest()
    handle = install_fixed_mask_guard(model, optimizer)
    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                model(torch.randn(4, 2)).logits, torch.tensor([0, 1, 0, 1])
            )
            loss.backward()
            optimizer.step()
            model.assert_masks(optimizer)
            assert model.mask_digest() == original
        layer = model.first
        with torch.no_grad():
            optimizer.state[layer.weight]["momentum_buffer"][~layer.mask] = 1.0
        with pytest.raises(ValueError, match="inactive optimizer state"):
            model.assert_masks(optimizer)
        model.project_masks_(optimizer)
        model.assert_masks(optimizer)
    finally:
        handle.remove()
