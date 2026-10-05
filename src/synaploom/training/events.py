"""Backward-compatible event imports.

Structural events live at synaploom.events to keep the control plane independent
from the training package import graph.
"""

from synaploom.events import EventKind, JsonlEventLogger, StructuralEvent

__all__ = ["EventKind", "JsonlEventLogger", "StructuralEvent"]
