from __future__ import annotations

from dataclasses import dataclass

from synaploom.protocols.h1a import H1A_VARIANTS


H1A_STABILITY_SEEDS: tuple[int, ...] = (101, 102, 103)
H1A_STABILITY_LEARNING_RATES: tuple[float, ...] = (0.03, 0.01, 0.003)


@dataclass(frozen=True)
class H1AStabilityProtocol:
    experiment_id: str = "EXP-H1A-STABILITY-001"
    seeds: tuple[int, ...] = H1A_STABILITY_SEEDS
    learning_rates: tuple[float, ...] = H1A_STABILITY_LEARNING_RATES
    variants: tuple[str, ...] = H1A_VARIANTS

    @property
    def expected_runs_per_learning_rate(self) -> int:
        return len(self.seeds) * len(self.variants)

    def selection_rule(self) -> str:
        return (
            "Evaluate learning rates in descending order and select the highest "
            "candidate that completes every seed x variant run with no detected "
            "NaN/Inf. Accuracy and continual-learning metrics are ignored for "
            "selection. If no candidate passes, select none and open a separate "
            "stability study for additional controls such as gradient clipping."
        )
