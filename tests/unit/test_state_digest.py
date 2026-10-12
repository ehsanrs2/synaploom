from __future__ import annotations

import torch
from torch import nn

from synaploom.training.state_digest import model_state_sha256


class _ForwardRegistrationOrder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0, 2.0]))
        self.beta = nn.Parameter(torch.tensor([3.0]))


class _ReverseRegistrationOrder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([3.0]))
        self.alpha = nn.Parameter(torch.tensor([1.0, 2.0]))


def test_state_digest_is_deterministic_and_independent_of_registration_order() -> None:
    forward = _ForwardRegistrationOrder()
    reverse = _ReverseRegistrationOrder()

    assert tuple(forward.state_dict()) != tuple(reverse.state_dict())
    assert model_state_sha256(forward) == model_state_sha256(reverse)
    assert model_state_sha256(forward) == model_state_sha256(forward)


def test_state_digest_changes_when_a_state_tensor_changes() -> None:
    model = _ForwardRegistrationOrder()
    initial = model_state_sha256(model)
    with torch.no_grad():
        model.alpha[0] += 0.25

    assert model_state_sha256(model) != initial


def test_state_digest_rejects_unsupported_sparse_state() -> None:
    model = nn.Module()
    model.register_buffer(
        "sparse",
        torch.sparse_coo_tensor(
            indices=torch.tensor([[0]]),
            values=torch.tensor([1.0]),
            size=(1,),
        ),
    )

    try:
        model_state_sha256(model)
    except TypeError as exc:
        assert "unsupported state_dict tensor layout" in str(exc)
    else:
        raise AssertionError("sparse state must fail closed")
