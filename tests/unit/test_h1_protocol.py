from synaploom.benchmarks.h1 import default_h1_protocol
from synaploom.data.cifar100 import cifar100_class_order


def test_h1_protocol_is_frozen_recruitment_only() -> None:
    protocol = default_h1_protocol()
    assert protocol.num_tasks == 10
    assert protocol.classes_per_task == 10
    assert len(protocol.model_seeds) == 5
    assert protocol.replay_examples == 0
    assert protocol.message_rounds == 0
    assert protocol.use_augmentation is False


def test_cifar_class_order_is_seeded_permutation() -> None:
    a = cifar100_class_order(1701)
    b = cifar100_class_order(1701)
    c = cifar100_class_order(1702)
    assert a == b
    assert a != c
    assert sorted(a) == list(range(100))
