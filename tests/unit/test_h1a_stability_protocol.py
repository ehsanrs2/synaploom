from synaploom.protocols.h1a import H1A_VARIANTS
from synaploom.protocols.h1a_stability import H1AStabilityProtocol


def test_stability_protocol_is_independent_of_primary_seeds() -> None:
    protocol = H1AStabilityProtocol()
    assert protocol.seeds == (101, 102, 103)
    assert set(protocol.seeds).isdisjoint(set(range(10)))
    assert protocol.learning_rates == (0.03, 0.01, 0.003)
    assert protocol.variants == H1A_VARIANTS
    assert protocol.expected_runs_per_learning_rate == 15


def test_stability_selection_rule_ignores_accuracy() -> None:
    rule = H1AStabilityProtocol().selection_rule().lower()
    assert "highest" in rule
    assert "nan/inf" in rule
    assert "accuracy" in rule
    assert "ignored" in rule
