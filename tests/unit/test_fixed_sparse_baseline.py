"""Fixed-sparse candidate: engineering-only CPU tests; no science experiments."""
from __future__ import annotations

from io import BytesIO

import pytest
import torch

from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.protocols.h1a import H1AProtocol
from synaploom.protocols.h1a2 import h1a2_protocol
from synaploom.research.fixed_sparse_baseline import (
    FixedSparseMLP,
    build_h1a_fixed_sparse_feasibility,
    install_fixed_mask_guard,
)
from synaploom.training.class_incremental import class_incremental_cross_entropy
from synaploom.training.continual import ContinualConfig, run_continual_experiment


def tiny_protocol() -> H1AProtocol:
    return H1AProtocol(
        feature_dim=8, num_classes=6, classes_per_task=2,
        d_model=8, d_hidden=12, num_nodes=4,
        initial_recruited=2, top_k=2,
        epochs_per_task=1, batch_size=4, eval_batch_size=4,
        learning_rate=0.01, seeds=(13,),
    )


def make_tiny(seed=13, mask_seed=17):
    model, report = build_h1a_fixed_sparse_feasibility(
        tiny_protocol(), seed=seed, mask_seed=mask_seed
    )
    assert report.matched_total_2pct
    assert report.matched_ideal_macs_10pct
    return model, report


def test_h1a_scale_budget_and_nonzero_vs_implemented_macs():
    model, report = build_h1a_fixed_sparse_feasibility(
        h1a2_protocol(), seed=503, mask_seed=17
    )
    assert report.schema == 1
    assert report.total_allocated_trainable == sum(p.numel() for p in model.parameters())
    assert report.total_parameter_relative_error <= 0.02
    assert report.ideal_mac_relative_error <= 0.10
    assert report.matched_total_2pct and report.matched_ideal_macs_10pct
    assert report.body_active_weights < report.body_allocated_weights
    assert report.implementation_dense_linear_macs_per_batch > report.ideal_fixed_sparse_linear_macs_per_batch
    assert report.optimizer_state_bytes is None
    assert report.observed_wall_seconds is None
    assert report.observed_cuda_peak_bytes is None
    assert report.fidelity_status == "CANDIDATE_NOT_SCIENCE_APPROVED"
    assert model.first.mask.sum().item() == report.first_mask_active
    assert model.second.mask.sum().item() == report.second_mask_active
    assert model.mask_digest() == report.mask_sha256
    model.assert_masks()


def test_build_does_not_change_global_torch_rng_and_mask_is_seeded():
    before = torch.get_rng_state().clone()
    a, ar = make_tiny(seed=42, mask_seed=19)
    assert torch.equal(torch.get_rng_state(), before)
    b, br = make_tiny(seed=42, mask_seed=19)
    assert ar.mask_sha256 == br.mask_sha256
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    c, cr = make_tiny(seed=42, mask_seed=20)
    assert cr.mask_sha256 != ar.mask_sha256
    assert ar.total_allocated_trainable == cr.total_allocated_trainable
    assert ar.body_active_weights == cr.body_active_weights


@pytest.mark.parametrize("optim_name", ["SGD", "AdamW"])
def test_masked_gradients_weights_and_optimizer_moments_never_revive(optim_name):
    model, record = make_tiny()
    if optim_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.2
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.003, weight_decay=0.2
        )
    hook = install_fixed_mask_guard(model, optimizer)
    original_digest = model.mask_digest()
    initial_masks = [l.mask.clone() for l in model.masked_layers()]
    try:
        for i in range(3):
            x = torch.randn(5, 8)
            y = torch.tensor([0, 1, 2, 1, 0])
            optimizer.zero_grad(set_to_none=True)
            loss = class_incremental_cross_entropy(model(x).logits, y, (0, 1, 2))
            loss.backward()
            for layer in model.masked_layers():
                assert layer.weight.grad is not None
                assert torch.count_nonzero(layer.weight.grad[~layer.mask]) == 0
            optimizer.step()  # post-hook projects weights and all matching moment tensors
            model.assert_masks(optimizer)
            assert model.mask_digest() == original_digest
            for layer, mask in zip(model.masked_layers(), initial_masks):
                assert torch.equal(layer.mask, mask)
        # Even a corrupted inactive state is zeroed at the next optimizer step.
        layer = model.first
        state = optimizer.state[layer.weight]
        key = "momentum_buffer" if optim_name == "SGD" else "exp_avg"
        with torch.no_grad():
            state[key][~layer.mask] = 1.0
        with pytest.raises(ValueError, match="optimizer state"):
            model.assert_masks(optimizer)
        optimizer.step()
        model.assert_masks(optimizer)
    finally:
        hook.remove()


def test_masked_linear_no_execution_of_inactive_weights_and_output_shape():
    model, _ = make_tiny()
    x = torch.randn(3, 8)
    output = model(x)
    assert output.logits.shape == (3, 6)
    assert output.representation.shape == (3, 8)
    with torch.no_grad():
        first = model.first
        previous = output.logits.detach().clone()
        first.weight[~first.mask] = 1_000_000.0
        # Mask multiplication in the forward path excludes inactive weights
        # even if a caller has forgotten the post-step projection.
        torch.testing.assert_close(model(x).logits, previous, rtol=0, atol=0)
        first.project_()
    model.assert_masks()


def test_optimizer_binding_and_frozen_mask_validation():
    model, _ = make_tiny()
    other, _ = make_tiny(seed=14)
    wrong_optim = torch.optim.SGD(other.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="exactly once"):
        install_fixed_mask_guard(model, wrong_optim)
    with pytest.raises(TypeError):
        install_fixed_mask_guard(other.first, wrong_optim)
    with torch.no_grad():
        model.first.weight[~model.first.mask] = 10.0
    with pytest.raises(ValueError, match="weights"):
        model.assert_masks()
    model.project_masks_()
    model.assert_masks()


def test_mask_buffer_persists_exactly_and_strict_state_load():
    model, record = make_tiny()
    buf = BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    state = torch.load(buf, weights_only=True)
    clone, clone_report = make_tiny(seed=14, mask_seed=17)
    clone.load_state_dict(state, strict=True)
    assert clone.mask_digest() == record.mask_sha256
    assert clone_report.mask_sha256 == record.mask_sha256
    x = torch.randn(2, 8)
    torch.testing.assert_close(clone(x).logits, model(x).logits, atol=0, rtol=0)


def test_engineering_only_tiny_continual_harness_compatibility():
    protocol = tiny_protocol()
    model, _ = build_h1a_fixed_sparse_feasibility(protocol, seed=13, mask_seed=17)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    hook = install_fixed_mask_guard(model, optimizer)
    try:
        stream = make_gaussian_cluster_stream(
            num_tasks=3, classes_per_task=2, input_dim=8,
            train_per_class=4, eval_per_class=2, seed=41,
        )
        result = run_continual_experiment(
            model, stream, optimizer,
            config=ContinualConfig(
                epochs_per_task=1, batch_size=4, eval_batch_size=4,
                seed=13, restrict_logits_to_seen_classes=True,
                current_task_eval_every_epochs=1,
            ),
        )
        assert result.global_steps == 6
        assert result.accuracy_matrix.shape == (3, 3)
        assert torch.isfinite(result.accuracy_matrix[torch.isfinite(result.accuracy_matrix)]).all()
        assert len(result.current_task_learning_curves) == 3
        model.assert_masks(optimizer)
    finally:
        hook.remove()


def test_tiny_mask_invalid_settings_rejected():
    with pytest.raises(ValueError, match="active_weights"):
        FixedSparseMLP(
            input_dim=8, d_model=8, d_hidden=12, num_classes=6,
            active_first=0, active_second=4, mask_seed=1
        )
    with pytest.raises(ValueError, match="mask_seed"):
        FixedSparseMLP(
            input_dim=8, d_model=8, d_hidden=12, num_classes=6,
            active_first=3, active_second=4, mask_seed=-10
        )


def test_checkpoint_cannot_silently_replace_predeclared_seeded_mask():
    original, report = make_tiny(seed=13, mask_seed=17)
    incompatible, other_report = make_tiny(seed=13, mask_seed=18)
    assert other_report.mask_sha256 != report.mask_sha256
    incompatible.load_state_dict(original.state_dict(), strict=True)
    assert incompatible.mask_digest() == report.mask_sha256
    with pytest.raises(ValueError, match="frozen mask identity"):
        incompatible.assert_masks()
    optimizer = torch.optim.SGD(incompatible.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="frozen mask identity"):
        install_fixed_mask_guard(incompatible, optimizer)


def test_modified_mask_buffer_fails_identity_check_without_rewiring():
    model, report = make_tiny()
    with torch.no_grad():
        model.second.mask[0, 0] = ~model.second.mask[0, 0]
    assert model.mask_digest() != report.mask_sha256
    with pytest.raises(ValueError, match="frozen mask identity"):
        model.assert_masks()
