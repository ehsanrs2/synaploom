import pytest
import torch

from synaploom.training.class_incremental import class_incremental_cross_entropy, mask_logits_to_classes


def test_seen_class_mask_prevents_future_class_prediction() -> None:
    logits = torch.tensor([[0.0, 1.0, 100.0, 99.0]])
    masked = mask_logits_to_classes(logits, (0, 1))
    assert masked.argmax(dim=-1).item() == 1
    assert masked[0, 2] < -1e20


def test_masked_loss_rejects_unseen_target() -> None:
    logits = torch.randn(2, 4)
    with pytest.raises(ValueError):
        class_incremental_cross_entropy(logits, torch.tensor([0, 3]), allowed_classes=(0, 1))
