from __future__ import annotations

from dataclasses import replace

from synaploom.protocols.h1a import H1AProtocol


H1A2_EXPERIMENT_ID = "EXP-H1A-002"
H1A2_SEEDS: tuple[int, ...] = (201, 202, 203, 204, 205, 206, 207, 208, 209, 210)
H1A2_LEARNING_RATE = 0.01
H1A2_FEATURE_SHA256 = "c635bebdc5c67b231b4ff97ab09e01572e0cd5bf843a427091e518113a2b8a47"


def h1a2_protocol() -> H1AProtocol:
    """Return the frozen scientific rerun protocol after stability calibration."""
    return replace(
        H1AProtocol(),
        learning_rate=H1A2_LEARNING_RATE,
        seeds=H1A2_SEEDS,
    )
