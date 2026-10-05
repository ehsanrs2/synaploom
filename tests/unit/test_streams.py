import torch

from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.data.streams import iter_tensor_batches


def test_gaussian_stream_is_deterministic_and_class_incremental() -> None:
    a = make_gaussian_cluster_stream(
        num_tasks=3, classes_per_task=2, input_dim=5, train_per_class=4, eval_per_class=3, seed=9
    )
    b = make_gaussian_cluster_stream(
        num_tasks=3, classes_per_task=2, input_dim=5, train_per_class=4, eval_per_class=3, seed=9
    )

    assert len(a) == 3
    assert a.num_classes == 6
    assert a[0].classes == (0, 1)
    assert a[1].classes == (2, 3)
    assert torch.equal(a[2].train_x, b[2].train_x)
    assert torch.equal(a[2].train_y, b[2].train_y)


def test_tensor_batch_order_is_seeded() -> None:
    x = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    y = torch.arange(12)
    first = list(iter_tensor_batches(x, y, batch_size=5, shuffle=True, seed=3))
    second = list(iter_tensor_batches(x, y, batch_size=5, shuffle=True, seed=3))
    assert torch.equal(torch.cat([v for _, v in first]), torch.cat([v for _, v in second]))
