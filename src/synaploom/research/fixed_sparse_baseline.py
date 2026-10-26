"""M1.3b fixed-sparse *candidate* — standalone research-only feasibility adapter.

Not a preregistered scientific baseline. Masks never rewire. Physical trainable
weights are counted INCLUDING zeros, while ideal nonzero MACs are distinguished
from the dense masked-kernel implementation. No training engine is modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from synaploom.baselines import DenseOutput
from synaploom.protocols.h1a import H1AProtocol
from synaploom.research.budgets import (
    count_trainable_parameters,
    dense_hidden_width_for_target,
    estimate_synaploom_linear_macs_per_batch,
)
from synaploom.model import SynapLoomModel


class FixedMaskedLinear(nn.Linear):
    """An unstructured constant mask, not a sparse CPU/CUDA kernel."""

    def __init__(self, in_features: int, out_features: int, *, active_weights: int, mask_seed: int):
        """Initialize dense weights and a persistent, seeded boolean weight mask.

        ``active_weights`` is the exact number of enabled matrix entries; bias
        stays dense. Mask selection uses a local CPU RNG, independently of weight
        initialization, and inactive weights are initially zeroed.
        Raise ValueError unless the active count is an integer in [1, matrix
        size] and mask_seed is a nonnegative integer; booleans are rejected.
        """
        super().__init__(in_features, out_features)
        size = self.weight.numel()
        if type(active_weights) is not int or not 0 < active_weights <= size:
            raise ValueError("active_weights must be in [1, matrix weight count]")
        if type(mask_seed) is not int or mask_seed < 0:
            raise ValueError("mask_seed must be a nonnegative integer")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mask_seed)
        selected = torch.randperm(size, generator=generator)[:active_weights]
        mask = torch.zeros(size, dtype=torch.bool)
        mask[selected] = True
        self.register_buffer("mask", mask.reshape_as(self.weight), persistent=True)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the masked weight and dense bias using dense linear computation."""
        return F.linear(x, self.weight * self.mask, self.bias)

    @torch.no_grad()
    def project_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Multiply weights and optional same-shaped optimizer state by the mask in place.

        Only state associated with this layer's weight is projected; bias and
        differently shaped state, such as scalar step counters, are untouched.
        """
        self.weight.mul_(self.mask)
        if optimizer is not None:
            for key, item in optimizer.state.get(self.weight, {}).items():
                # Enforce inactive momentum/Adam moments, not just weights.
                if isinstance(item, Tensor) and item.shape == self.weight.shape:
                    item.mul_(self.mask)

    def verify_masked_zeros(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Raise ValueError if an inactive weight or matching optimizer entry is nonzero."""
        if bool(torch.any(self.weight.detach()[~self.mask] != 0)):
            raise ValueError("inactive weights became nonzero")
        if optimizer is not None:
            for item in optimizer.state.get(self.weight, {}).values():
                if isinstance(item, Tensor) and item.shape == self.weight.shape:
                    if bool(torch.any(item.detach()[~self.mask] != 0)):
                        raise ValueError("inactive optimizer state became nonzero")


class FixedSparseMLP(nn.Module):
    """Same high-level residual MLP pattern as DenseControlModel.

    Encoder, head, biases and norm remain dense. Only two body matrices use
    fixed masks; neither contains dynamic pruning nor regrowth.
    """

    def __init__(
        self, *, input_dim: int, d_model: int, d_hidden: int, num_classes: int,
        active_first: int, active_second: int, mask_seed: int,
    ):
        """Initialize the residual MLP and record its construction mask fingerprint.

        active_first and active_second specify exact enabled weight counts for
        the two body matrices, using mask_seed and mask_seed + 1 respectively.
        An encoder is allocated only when input_dim differs from d_model.
        Invalid active counts or a negative mask_seed propagate ValueError from
        FixedMaskedLinear.
        """
        super().__init__()
        self.encoder = nn.Identity() if input_dim == d_model else nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.first = FixedMaskedLinear(d_model, d_hidden, active_weights=active_first, mask_seed=mask_seed)
        self.activation = nn.GELU()
        self.second = FixedMaskedLinear(d_hidden, d_model, active_weights=active_second, mask_seed=mask_seed + 1)
        self.head = nn.Linear(d_model, num_classes)
        # A persistent mask buffer loaded from an unrelated checkpoint is NOT
        # automatically the frozen mask associated with this declared seed.
        self._expected_mask_sha256 = self.mask_digest()

    def forward(self, x: Tensor) -> DenseOutput:
        """Return class logits and the residual representation for input features.

        The final input dimension is input_dim; output final dimensions are
        num_classes and d_model respectively, preserving leading dimensions.
        """
        z = self.encoder(x)
        representation = z + self.second(self.activation(self.first(self.norm(z))))
        return DenseOutput(logits=self.head(representation), representation=representation)

    def masked_layers(self) -> tuple[FixedMaskedLinear, FixedMaskedLinear]:
        """Return the two body layers in forward order."""
        return self.first, self.second

    @torch.no_grad()
    def project_masks_(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Project both body weights and optional same-shaped optimizer state in place."""
        for layer in self.masked_layers():
            layer.project_(optimizer)

    def assert_masks(self, optimizer: torch.optim.Optimizer | None = None) -> None:
        """Raise ValueError for changed masks or nonzero inactive weights or state.

        Check masks against their construction fingerprint, including after a
        state_dict load. With optimizer, also check same-shaped body-weight state.
        """
        if self.mask_digest() != self._expected_mask_sha256:
            raise ValueError("frozen mask identity differs from construction seed")
        for layer in self.masked_layers():
            layer.verify_masked_zeros(optimizer)

    def mask_digest(self) -> str:
        """Return a SHA-256 hex digest of current mask names, shapes, and bits.

        Mask data is hashed on CPU; learned weights are excluded.
        """
        payload = sha256()
        for name, layer in zip(("first", "second"), self.masked_layers()):
            mask = layer.mask.detach().to(device="cpu").contiguous()
            payload.update(name.encode("ascii") + b"\0")
            payload.update(json.dumps(list(mask.shape), separators=(",", ":")).encode("ascii") + b"\0")
            payload.update(mask.to(torch.uint8).numpy().tobytes())
        return payload.hexdigest()


@dataclass(frozen=True)
class FixedSparseFeasibilityV1:
    schema: int
    input_dim: int
    d_model: int
    d_hidden: int
    num_classes: int
    mask_seed: int
    mask_sha256: str
    total_allocated_trainable: int
    target_synaploom_total_trainable: int
    total_parameter_relative_error: float
    body_allocated_weights: int
    body_active_weights: int
    first_mask_active: int
    second_mask_active: int
    body_density: float
    target_synaploom_linear_macs_per_batch: int
    ideal_fixed_sparse_linear_macs_per_batch: int
    implementation_dense_linear_macs_per_batch: int
    ideal_mac_relative_error: float
    mask_buffer_bytes: int
    optimizer_state_bytes: int | None
    observed_wall_seconds: float | None
    observed_cuda_peak_bytes: int | None
    matched_total_2pct: bool
    matched_ideal_macs_10pct: bool
    fidelity_status: str = "CANDIDATE_NOT_SCIENCE_APPROVED"

    def as_dict(self) -> dict[str, Any]:
        """Return the feasibility fields as a new dictionary."""
        return asdict(self)


def _check_optimizer_binding(model: FixedSparseMLP, optimizer: torch.optim.Optimizer) -> None:
    """Raise ValueError unless optimizer binds every model parameter exactly once."""
    params = [id(p) for p in model.parameters()]
    bound = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if len(bound) != len(params) or len(set(bound)) != len(bound) or set(bound) != set(params):
        raise ValueError("optimizer must own every model parameter exactly once")


def install_fixed_mask_guard(
    model: FixedSparseMLP, optimizer: torch.optim.Optimizer,
) -> Any:
    """Install post-step projection AFTER optimizer creation and model.to(device).

    Immediately project body weights and same-shaped optimizer state, then
    validate mask identity and inactive zeros. Repeat after each optimizer step.
    Return a removable hook handle; keep the hook installed throughout training
    and reinstall it when reconstructing an optimizer.

    Raise TypeError for a model other than FixedSparseMLP and ValueError unless
    optimizer owns every model parameter exactly once. Mask validation errors
    propagate at installation and after steps; projection precedes validation
    and is not rolled back on failure.
    """
    if not isinstance(model, FixedSparseMLP):
        raise TypeError("requires FixedSparseMLP")
    _check_optimizer_binding(model, optimizer)
    model.project_masks_(optimizer)
    model.assert_masks(optimizer)

    def _post_step(opt: torch.optim.Optimizer, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Project body weights and optimizer state, then propagate mask validation errors."""
        model.project_masks_(opt)
        model.assert_masks(opt)

    return optimizer.register_step_post_hook(_post_step)


def build_h1a_fixed_sparse_feasibility(
    protocol: H1AProtocol,
    *,
    seed: int,
    mask_seed: int = 20261026,
) -> tuple[FixedSparseMLP, FixedSparseFeasibilityV1]:
    """Construct a budget-sized *candidate*, not a frozen final comparator.

    Choose a hidden width near the protocol reference's full physical trainable
    allocation, including inactive nodes, and body mask counts near its initial
    routing linear MAC budget at protocol.batch_size. seed controls weight
    initialization; mask_seed independently controls fixed mask selection.
    The caller's CPU RNG state is restored; optimizer guards are not installed.

    Return the model and a resource ledger separating allocated parameters,
    ideal sparse and dense implementation MACs per batch, and mask bytes.
    Optimizer memory, wall time, and CUDA peak memory remain unmeasured (None).
    The 2% parameter and 10% ideal MAC gates are reported, not enforced, and
    do not establish measured GPU compute parity or scientific approval.

    Raise TypeError for a non-H1AProtocol and ValueError for invalid seed types,
    a negative mask_seed, invalid capacity, or an infeasible body active count.
    Configuration and budget helper validation errors also propagate, including
    ValueError for invalid model dimensions, recruitment counts, or batch size.
    """
    if not isinstance(protocol, H1AProtocol):
        raise TypeError("requires H1AProtocol")
    if type(seed) is not int or type(mask_seed) is not int or mask_seed < 0:
        raise ValueError("invalid seed or mask_seed")
    if protocol.num_nodes < protocol.top_k:
        raise ValueError("invalid H1a capacity")
    # fork_rng preserves global CPU RNG while generating a deterministic
    # resource-reference model. No training examples or labels are accessed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        reference = SynapLoomModel(protocol.model_config())
        ref_total = count_trainable_parameters(reference)
        ref_macs = estimate_synaploom_linear_macs_per_batch(
            reference, batch_size=protocol.batch_size
        )
        width, _ = dense_hidden_width_for_target(
            ref_total, input_dim=protocol.feature_dim, d_model=protocol.d_model,
            num_classes=protocol.num_classes,
        )
        body_each = protocol.d_model * width
        body_total = body_each * 2
        encoder_macs = (0 if protocol.feature_dim == protocol.d_model
                        else protocol.feature_dim * protocol.d_model)
        head_macs = protocol.d_model * protocol.num_classes
        dense_shared_macs = protocol.batch_size * (encoder_macs + head_macs)
        desired_body = round((ref_macs - dense_shared_macs) / protocol.batch_size)
        if not 2 <= desired_body <= body_total:
            raise ValueError("no feasible fixed-sparse density for declared H1a budgets")
        active_first = desired_body // 2
        active_second = desired_body - active_first
        # Hidden layer weights account for half of the body matrix pool each.
        # Masks are deterministic with independent fixed integer seeds.
        torch.manual_seed(seed)
        model = FixedSparseMLP(
            input_dim=protocol.feature_dim, d_model=protocol.d_model,
            d_hidden=width, num_classes=protocol.num_classes,
            active_first=active_first, active_second=active_second,
            mask_seed=mask_seed,
        )

    model.assert_masks()
    allocated = count_trainable_parameters(model)
    ideal_macs = dense_shared_macs + protocol.batch_size * desired_body
    dense_implementation_macs = dense_shared_macs + protocol.batch_size * body_total
    perr = abs(allocated - ref_total) / ref_total
    merr = abs(ideal_macs - ref_macs) / ref_macs
    info = FixedSparseFeasibilityV1(
        schema=1,
        input_dim=protocol.feature_dim, d_model=protocol.d_model,
        d_hidden=width, num_classes=protocol.num_classes,
        mask_seed=mask_seed, mask_sha256=model.mask_digest(),
        total_allocated_trainable=allocated,
        target_synaploom_total_trainable=ref_total,
        total_parameter_relative_error=perr,
        body_allocated_weights=body_total, body_active_weights=desired_body,
        first_mask_active=active_first, second_mask_active=active_second,
        body_density=desired_body / body_total,
        target_synaploom_linear_macs_per_batch=ref_macs,
        ideal_fixed_sparse_linear_macs_per_batch=ideal_macs,
        implementation_dense_linear_macs_per_batch=dense_implementation_macs,
        ideal_mac_relative_error=merr,
        mask_buffer_bytes=body_total,  # bool masks, one byte per element
        optimizer_state_bytes=None, observed_wall_seconds=None,
        observed_cuda_peak_bytes=None,
        matched_total_2pct=perr <= 0.02,
        matched_ideal_macs_10pct=merr <= 0.10,
    )
    return model, info
