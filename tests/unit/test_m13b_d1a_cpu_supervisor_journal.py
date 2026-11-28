"""Real Linux CPU process tests; no training, datasets, GPU or science seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from synaploom.research import m13b_d1a_cpu_supervisor_journal as s

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux fixture only")


def test_actual_success_and_durable_chain(tmp_path):
    root = tmp_path / "session"
    result = s.run_fixture(root)
    assert result["status"] == "COMPLETE"
    assert result["started"] == 2
    assert result["terminal_states"] == ["COMPLETE", "COMPLETE"]
    assert result["not_started"] == []
    assert not result["science_executable"]
    assert not result["gpu_release_verified"]
    assert not result["external_custody_verified"]
    assert s.inspect_fixture(root) == result
    assert len(result["local_chain_head_sha256"]) == 64
    with pytest.raises(FileExistsError):
        s.run_fixture(root)


def test_nonzero_exit_stops_without_second_attempt(tmp_path):
    root = tmp_path / "session"
    r = s.run_fixture(root, ("fail", "ok"))
    assert r["status"] == "FAILED"
    assert r["terminal_states"] == ["FAILED"]
    assert r["not_started"] == ["toy-1"]
    assert not (root / "1.start.json").exists()
    end = json.loads((root / "0.terminal.json").read_text())
    assert end["returncode"] == 7
    assert end["direct_child_reaped_or_not_spawned"] is True


def test_actual_timeout_kills_and_reaps_direct_child(tmp_path):
    root = tmp_path / "session"
    r = s.run_fixture(root, ("timeout", "ok"))
    assert r["status"] == "TIMEOUT"
    assert r["terminal_states"] == ["TIMEOUT"]
    assert not r["direct_child_release_unknown"]
    end = json.loads((root / "0.terminal.json").read_text())
    assert end["returncode"] == -9
    assert end["direct_child_reaped_or_not_spawned"] is True
    assert r["not_started"] == ["toy-1"]


def test_real_supervisor_sigkill_after_durable_start(tmp_path):
    # Kill a separate supervisor after durable START but before it creates
    # a worker. No orphan child is created by this fault-injection test.
    root = tmp_path / "crashed"
    source = str(Path(s.__file__).resolve().parents[2])
    script = (
        "import sys,time\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "from pathlib import Path\n"
        "from synaploom.research import m13b_d1a_cpu_supervisor_journal as s\n"
        "def pause_before_spawn(case):\n"
        "    time.sleep(60)\n"
        "s._child=pause_before_spawn\n"
        "s.run_fixture(Path(sys.argv[2]))\n"
    )
    proc = subprocess.Popen([sys.executable, "-I", "-c", script, source, str(root)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 8
        while not (root / "0.start.json").exists():
            if proc.poll() is not None or time.monotonic() > deadline:
                pytest.fail("fault-injection supervisor did not reach START")
            time.sleep(.01)
        # Test waits until contents are visible; actual fsync is done in
        # production before _child. Lock is still held by the live parent.
        assert s.inspect_fixture(root)["status"] == "BUSY_BLOCKED"
        proc.kill()
        proc.wait(timeout=5)
        r = s.inspect_fixture(root)
        assert r["status"] in ("UNCLOSED_BLOCKED", "CORRUPT_BLOCKED")
        if r["status"] == "UNCLOSED_BLOCKED":
            assert r["dangling_start"]
            assert r["direct_child_release_unknown"]
            assert r["not_started"] == ["toy-1"]
        assert not r["retry_allowed"]
        with pytest.raises(FileExistsError):
            s.run_fixture(root)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_missing_terminal_and_close_blocks_without_repair(tmp_path):
    root = tmp_path / "session"
    s.run_fixture(root, ("fail", "ok"))
    (root / "0.terminal.json").unlink()
    (root / "close.json").unlink()
    before = sorted(p.name for p in root.iterdir())
    r = s.inspect_fixture(root)
    assert r["status"] == "UNCLOSED_BLOCKED"
    assert r["dangling_start"]
    assert r["direct_child_release_unknown"]
    assert sorted(p.name for p in root.iterdir()) == before


@pytest.mark.parametrize("damage", ["alter", "partial", "extra", "missing", "symlink"])
def test_corruption_fails_closed(tmp_path, damage):
    root = tmp_path / "session"
    s.run_fixture(root)
    terminal = root / "0.terminal.json"
    if damage == "alter":
        row = json.loads(terminal.read_text())
        row["end_ms"] += 1
        terminal.write_bytes(s._canonical(row))
    elif damage == "partial":
        terminal.write_text("{")
    elif damage == "extra":
        (root / "2.start.json").write_text("{}")
    elif damage == "missing":
        terminal.unlink()
    else:
        terminal.unlink()
        terminal.symlink_to(root / "manifest.json")
    assert s.inspect_fixture(root)["status"] == "CORRUPT_BLOCKED"


def test_budget_reservation_rejects_start_after_elapsed_setup(tmp_path, monkeypatch):
    root = tmp_path / "session"
    # Reduce the fixture budget; no actual wait or child is needed.
    monkeypatch.setattr(s, "BUDGET_MS", 1)
    monkeypatch.setattr(s, "_child", lambda case: pytest.fail("unexpected spawn"))
    r = s.run_fixture(root)
    assert r["status"] == "BUDGET_EXHAUSTED"
    assert r["started"] == 0
    assert r["not_started"] == list(s.PLAN)


def test_invalid_cases_do_not_create_session(tmp_path):
    for cases in (("ok",), ("ok", "unknown"), ("ok", 1), ["ok", "ok"]):
        with pytest.raises(ValueError):
            s.run_fixture(tmp_path / "bad", cases)
    assert not (tmp_path / "bad").exists()
