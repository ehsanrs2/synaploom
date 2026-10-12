from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_eval_batch_sensitivity_dry_run_needs_no_feature_bundle(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    script = (
        root
        / "experiments"
        / "scripts"
        / "m1_3_h1a_eval_batch_sensitivity.py"
    )
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

    assert payload["experiment_id"] == "EXP-H1A-EVALBATCH-001"
    assert payload["status"] == "DRY_RUN_ONLY"
    assert payload["seeds"] == list(range(401, 411))
    assert payload["variants"] == ["progressive_reserve", "all_recruited"]
    assert payload["evaluation_batch_sizes"] == [32, 64, 128, 256, 512, 1024]
    assert payload["reference_batch_size"] == 256
    assert payload["expected_training_runs"] == 20
    assert payload["expected_run_size_summaries"] == 120
    assert payload["expected_state_hash_sweeps_per_run"] == 210
    assert payload["class_order_fingerprint_sha256"] == (
        "81d7f0637b8e61477c1ef80b7eef33270aae8d81dfcaffad6e58252fd004b633"
    )
    assert not output.exists()
