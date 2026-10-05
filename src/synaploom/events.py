from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


EventKind = Literal["recruit", "edge_add", "edge_remove", "consolidate", "note"]


@dataclass(frozen=True)
class StructuralEvent:
    step: int
    kind: EventKind
    structural_version: int
    node_id: int | None = None
    src: int | None = None
    dst: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class JsonlEventLogger:
    """Append-only structural event log suitable for experiment auditing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: StructuralEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
