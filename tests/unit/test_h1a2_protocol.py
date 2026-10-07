from synaploom.protocols.h1a2 import (
    H1A2_EXPERIMENT_ID,
    H1A2_FEATURE_SHA256,
    H1A2_LEARNING_RATE,
    H1A2_SEEDS,
    h1a2_protocol,
)
from synaploom.protocols.h1a_stability import H1A_STABILITY_SEEDS


def test_h1a2_protocol_uses_calibrated_learning_rate_and_fresh_seeds() -> None:
    protocol = h1a2_protocol()
    assert H1A2_EXPERIMENT_ID == "EXP-H1A-002"
    assert H1A2_LEARNING_RATE == 0.01
    assert H1A2_FEATURE_SHA256 == "c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47"
    assert protocol.learning_rate == 0.01
    assert protocol.seeds == H1A2_SEEDS
    assert len(protocol.seeds) == 10
    assert set(protocol.seeds).isdisjoint(set(range(10)))
    assert set(protocol.seeds).isdisjoint(set(H1A_STABILITY_SEEDS))


def test_h1a2_changes_only_lr_and_seeds_from_original_protocol() -> None:
    from synaploom.protocols.h1a import H1AProtocol

    original = H1AProtocol().as_dict()
    rerun = h1a2_protocol().as_dict()

    changed = {
        key
        for key in original
        if original[key] != rerun[key]
    }
    assert changed == {"learning_rate", "seeds"}
