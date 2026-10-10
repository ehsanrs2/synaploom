from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_h1a2_runner_dry_run_does_not_require_features() -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "experiments" / "scripts" / "m1_3_h1a_002.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["experiment_id"] == "EXP-H1A-002"
    assert payload["protocol"]["learning_rate"] == 0.01
    assert payload["protocol"]["seeds"] == list(range(201, 211))
    assert payload["expected_runs"] == 50
