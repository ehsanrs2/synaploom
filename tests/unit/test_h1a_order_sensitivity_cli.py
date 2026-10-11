from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_order_sensitivity_dry_run_emits_frozen_protocol_without_features(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[2]
    script = root / "experiments" / "scripts" / "m1_3_h1a_order_sensitivity.py"
    output = tmp_path / "must_not_be_created_by_dry_run"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dry-run",
            "--output",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["experiment_id"] == "EXP-H1A-ORDER-001"
    assert payload["status"] == "DRY_RUN_ONLY"
    assert payload["expected_runs"] == 60
    assert payload["seeds"] == list(range(301, 311))
    assert payload["variants"] == ["progressive_reserve", "all_recruited"]
    assert list(payload["class_orders"]) == ["O0", "O1", "O2"]
    assert payload["order_fingerprints"] == {
        "O0": "81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633",
        "O1": "c0dc7d8b234ec06670526cbc8d917f0e7e14ed9bbaee640272eab95573f1f8ed",
        "O2": "b610995c70c2269fb28e7492f1c1eb2870703fa134f967e185ca02dbd44e2e63",
    }
    assert payload["protocol"]["learning_rate"] == 0.01
    assert payload["protocol"]["eval_batch_size"] == 256
    assert not output.exists()
