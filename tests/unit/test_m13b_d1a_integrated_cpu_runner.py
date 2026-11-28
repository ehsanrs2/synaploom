"""Integrated CPU-only qualification, including real live-worker parent death."""
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from synaploom.research import m13b_d1a_integrated_cpu_runner as runner
from synaploom.research import m13b_d1a_runner_control_cpu as control

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux fixture")


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrated") / "session"
    result = runner.run_fixture(root)
    return root, result


def test_full_720_actual_cpu_processes_and_closed_exact_analysis(full):
    root, result = full
    assert result["status"] == "COMPLETE"
    assert result["started"] == result["completed"] == 720
    assert result["not_started"] == 0
    assert result["analysis"] is None
    assert not result["science_executable"]
    assert len(list(root.iterdir())) == 2163
    audited = runner.inspect_fixture(root, expected_head=result["local_head_sha256"])
    assert audited["status"] == "COMPLETE"
    assert audited["caller_detached_head_matched"]
    assert not audited["external_custody_verified"]
    assert not audited["gpu_release_verified"]
    assert audited["analysis"]["toy_both_reject"]
    assert audited["analysis"]["actual_optimizer_steps"] == 0
    for name in control.METRICS:
        assert audited["analysis"]["toy_endpoints"][name]["mean_difference"] == "1/50"
    with pytest.raises(FileExistsError):
        runner.run_fixture(root)


def test_wrong_detached_head_blocks_analysis(full):
    root, _ = full
    result = runner.inspect_fixture(root, expected_head="0"*64)
    assert result["status"] == "CORRUPT_BLOCKED"
    assert result["analysis"] is None


def test_actual_failure_stops_full_registry_prefix(tmp_path):
    root = tmp_path / "session"
    r = runner.run_fixture(root, failure_index=2, failure_case="fail")
    assert r["status"] == "FAILED"
    assert r["started"] == 3 and r["completed"] == 2 and r["not_started"] == 717
    assert not (root/"3.start.json").exists()
    assert not (root/"2.observation.json").exists()
    r = runner.inspect_fixture(root, expected_head=r["local_head_sha256"])
    assert r["analysis"] is None


def test_actual_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "WORKER_MS", 500)
    root = tmp_path / "session"
    r = runner.run_fixture(root, failure_index=0, failure_case="sleep")
    assert r["status"] == "TIMEOUT"
    assert r["started"] == 1 and r["completed"] == 0
    terminal = json.loads((root/"0.terminal.json").read_text())
    assert terminal["direct_child_reaped_or_not_spawned"] is True


def test_budget_blocks_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "TOTAL_MS", 1)
    monkeypatch.setattr(runner, "_spawn", lambda *a: pytest.fail("unexpected spawn"))
    r = runner.run_fixture(tmp_path/"session")
    assert r["status"] == "BUDGET_EXHAUSTED"
    assert r["started"] == 0 and r["analysis"] is None


@pytest.mark.parametrize("damage", ["observation", "start", "extra", "close"])
def test_damage_blocks(tmp_path, damage):
    root = tmp_path/"session"
    r = runner.run_fixture(root, failure_index=1, failure_case="fail")
    if damage == "observation":
        obs = json.loads((root/"0.observation.json").read_text())
        obs["stage_correct"][0][0] += 1
        (root/"0.observation.json").write_bytes(runner.journal._canonical(obs))
    elif damage == "start":
        (root/"0.start.json").unlink()
    elif damage == "extra":
        (root/"2.start.json").write_text("{}")
    else:
        (root/"close.json").write_text("{")
    assert runner.inspect_fixture(root, expected_head=r["local_head_sha256"])["status"] == "CORRUPT_BLOCKED"


def test_live_worker_parent_death_and_no_retry(tmp_path):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        pytest.skip("pidfd support required for safe test process identity")
    root, marker = tmp_path/"session", tmp_path/"worker-pid"
    source = str(Path(runner.__file__).resolve().parents[2])
    script = (
        "import sys,time\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "from pathlib import Path\n"
        "from synaploom.research import m13b_d1a_integrated_cpu_runner as r\n"
        "original=r._spawn\n"
        "def spawn(i,case):\n"
        "    p=original(i,'sleep')\n"
        "    if p.stdout.readline()!=b'READY\\n': raise RuntimeError('not ready')\n"
        "    Path(sys.argv[3]).write_text(str(p.pid))\n"
        "    time.sleep(60)\n"
        "    return p\n"
        "r._spawn=spawn\n"
        "r.run_fixture(Path(sys.argv[2]))\n"
    )
    parent = subprocess.Popen([sys.executable,"-I","-c",script,source,str(root),str(marker)],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    fd = None
    try:
        until = time.monotonic()+10
        while not marker.exists() or not marker.read_text().strip():
            if parent.poll() is not None or time.monotonic()>until:
                pytest.fail("live child not witnessed")
            time.sleep(.01)
        fd = os.pidfd_open(int(marker.read_text()))
        poll = select.poll()
        poll.register(fd, select.POLLIN)
        assert not poll.poll(0), "worker already exited"
        assert runner.inspect_fixture(root)["status"] == "BUSY_BLOCKED"
        parent.kill()
        parent.wait(timeout=5)
        assert poll.poll(5000), "direct child did not exit after parent death"
        result = runner.inspect_fixture(root)
        assert result["status"] == "UNCLOSED_BLOCKED"
        assert result["dangling_start"] and result["started"] == 1
        assert result["direct_child_release_unknown"] and not result["retry_allowed"]
        assert result["analysis"] is None
        with pytest.raises(FileExistsError):
            runner.run_fixture(root)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if fd is not None:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.close(fd)


def test_invalid_fault_does_not_create_directory(tmp_path):
    root = tmp_path/"invalid"
    for kw in (dict(failure_index=True,failure_case="fail"),
               dict(failure_index=720,failure_case="fail"),
               dict(failure_case="sleep")):
        with pytest.raises(ValueError):
            runner.run_fixture(root, **kw)
        assert not root.exists()
