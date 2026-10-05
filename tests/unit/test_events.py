import json
from pathlib import Path

from synaploom.training.events import JsonlEventLogger, StructuralEvent


def test_event_log_is_append_only_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)
    logger.append(StructuralEvent(step=10, kind="recruit", structural_version=1, node_id=4))
    logger.append(StructuralEvent(step=20, kind="consolidate", structural_version=2, node_id=0))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["kind"] for row in rows] == ["recruit", "consolidate"]
    assert rows[0]["node_id"] == 4
