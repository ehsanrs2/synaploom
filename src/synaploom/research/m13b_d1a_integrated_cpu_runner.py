"""Integrated 720-slot CPU-only artificial runner. No real worker entry point.

Fixed subprocess source emits invented integer observations. Parent-death
signal covers only this direct, non-forking Linux child. Local journal and
caller-detached hashes are not authenticated scientific evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from synaploom.research import m13b_d1a_cpu_supervisor_journal as journal
from synaploom.research import m13b_d1a_runner_control_cpu as control

KIND = "M13B_D1A_INTEGRATED_CPU_FIXTURE_V1"
# Deliberately separate CPU fixture timing policy; not science authorization.
WORKER_MS = 5000
CLEANUP_MS = 1000
TOTAL_MS = 180_000
CHILD = r"""
import ctypes,json,os,signal,sys,time
parent=int(sys.argv[1])
index=int(sys.argv[2])
case=sys.argv[3]
# Linux PR_SET_PDEATHSIG. Check parent identity after installation, covering
# parent loss between fork/exec and installing the death signal.
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=parent:
    os._exit(91)
if case=="fail":
    sys.exit(7)
if case=="sleep":
    print("READY",flush=True)
    time.sleep(60)
    sys.exit(8)
if case!="ok" or not 0<=index<720:
    sys.exit(9)
draw=index//2
arms=("progressive_reserve","fixed_sparse_mlp")
arm=arms[(index%2)^(draw%2)]
slot=f"toy-draw-{draw:03d}-{arm}"
count=520 if arm==arms[0] else 500
print(json.dumps({"kind":"M13B_D1A_SYNTHETIC_RUNNER_CONTROL_V1",
                  "slot_id":slot,
                  "stage_correct":[[count]*(i+1) for i in range(10)],
                  "epoch_correct":[[count]*20 for _ in range(10)],
                  "epoch_steps":list(range(40,8001,40))},
                 sort_keys=True,separators=(",",":")))
"""


def _spawn(index: int, case: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", CHILD,
         str(os.getpid()), str(index), case],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env={"CUDA_VISIBLE_DEVICES": ""},
        start_new_session=True, close_fds=True,
    )


def _expected_observation(slot: dict[str, Any]) -> dict[str, Any]:
    # Separate parent-side reference, never reconstructed from worker output.
    count = 520 if slot["arm"] == control.ARMS[0] else 500
    return {"kind": control.KIND, "slot_id": slot["slot_id"],
            "stage_correct": [[count]*(i+1) for i in range(10)],
            "epoch_correct": [[count]*20 for _ in range(10)],
            "epoch_steps": list(range(40, 8001, 40))}


def _manifest(failure_index: int | None, failure_case: str | None) -> dict[str, Any]:
    if ((failure_index is None) != (failure_case is None)
            or (failure_index is not None and
                (type(failure_index) is not int or not 0 <= failure_index < 720
                 or failure_case not in ("fail", "sleep")))):
        raise ValueError("invalid fixed fixture fault")
    return {"kind": KIND, "registry_sha256": control.digest(control.synthetic_registry()),
            "count": 720, "worker_ms": WORKER_MS, "cleanup_ms": CLEANUP_MS,
            "total_ms": TOTAL_MS, "fault_index": failure_index,
            "fault_case": failure_case, "science_executable": False}


def run_fixture(root: Path, *, failure_index: int | None = None,
                failure_case: str | None = None) -> dict[str, Any]:
    """Run all 720 fixed artificial subprocesses, or stop on first failure."""
    manifest = _manifest(failure_index, failure_case)
    registry = control.synthetic_registry()
    root = Path(root)
    origin = time.monotonic_ns()
    root.mkdir(mode=0o700, exist_ok=False)
    journal._sync_dir(root.parent)
    with journal._lock(root):
        head = journal._write_new(root, "manifest.json", manifest)
        reason = "COMPLETE"
        started = 0
        for i, slot in enumerate(registry):
            start = journal._elapsed(origin)
            if start + WORKER_MS + CLEANUP_MS > TOTAL_MS:
                reason = "BUDGET_EXHAUSTED"
                break
            head = journal._write_new(root, f"{i}.start.json",
                {"slot_id": slot["slot_id"], "start_ms": start, "previous_sha256": head})
            started += 1
            deadline = origin + (start + WORKER_MS)*1_000_000
            cleanup = min(origin + TOTAL_MS*1_000_000, deadline + CLEANUP_MS*1_000_000)
            proc = None
            state, released, observation = "FAILED", True, None
            try:
                if time.monotonic_ns() >= deadline:
                    state = "TIMEOUT"
                else:
                    proc = _spawn(i, failure_case if i == failure_index else "ok")
                    raw, _ = proc.communicate(timeout=max(0, (deadline-time.monotonic_ns())/1e9))
                    if time.monotonic_ns() > deadline:
                        state = "TIMEOUT"
                    elif proc.returncode != 0:
                        state = "FAILED"
                    else:
                        value = json.loads(raw)
                        if (control.digest(value) != control.digest(_expected_observation(slot))):
                            state = "INVALID"
                        else:
                            control.exact_endpoints(value)
                            observation, state = value, "COMPLETE"
            except subprocess.TimeoutExpired:
                state = "TIMEOUT"
            except (ValueError, TypeError):
                state = "INVALID"
            except (OSError, KeyboardInterrupt):
                state = "INTERRUPTED"
                if proc is None:
                    released = False
            finally:
                worker_end = journal._elapsed(origin)
                if proc is not None:
                    released = journal._reap(proc, cleanup)
                    if proc.stdout is not None:
                        proc.stdout.close()
            release = journal._elapsed(origin)
            if not released:
                state = "UNKNOWN_RELEASE"
            elif (worker_end-start > WORKER_MS or release-worker_end > CLEANUP_MS
                    or release > TOTAL_MS):
                state = "TIMEOUT"
            obs_hash = None
            if state == "COMPLETE":
                obs_hash = control.digest(observation)
                journal._write_new(root, f"{i}.observation.json", observation)
            head = journal._write_new(root, f"{i}.terminal.json", {
                "slot_id": slot["slot_id"], "state": state,
                "worker_end_ms": worker_end, "release_ms": release,
                "direct_child_reaped_or_not_spawned": released,
                "observation_sha256": obs_hash, "previous_sha256": head})
            if state != "COMPLETE":
                reason = state
                break
        close = journal._elapsed(origin)
        if reason == "COMPLETE" and close > TOTAL_MS:
            reason = "BUDGET_EXHAUSTED"
        journal._write_new(root, "close.json",
            {"reason": reason, "started": started, "close_ms": close,
             "previous_sha256": head})
    return inspect_fixture(root)


def _scan(root: Path, expected_head: str | None) -> dict[str, Any]:
    names = {p.name for p in root.iterdir()}
    manifest, head = journal._read(root, "manifest.json")
    if journal._canonical(manifest) != journal._canonical(
            _manifest(manifest.get("fault_index"), manifest.get("fault_case"))):
        raise ValueError("manifest mismatch")
    registry = control.synthetic_registry()
    seen = {"session.lock", "manifest.json"}
    records, observations = [], []
    last = 0
    dangling = False
    started = 0
    failed = None
    for i, slot in enumerate(registry):
        sn, tn, on = f"{i}.start.json", f"{i}.terminal.json", f"{i}.observation.json"
        if sn not in names:
            break
        if failed:
            raise ValueError("attempt after failure")
        begin, h = journal._read(root, sn)
        start = begin.get("start_ms")
        if (type(start) is not int or not last <= start
                or start+WORKER_MS+CLEANUP_MS > TOTAL_MS
                or begin != {"slot_id": slot["slot_id"], "start_ms": start,
                             "previous_sha256": head}):
            raise ValueError("invalid START")
        seen.add(sn)
        head = h
        started += 1
        if tn not in names:
            dangling = True
            # A durable result without terminal is still blocked, never promoted.
            if on in names:
                journal._read(root, on)
                seen.add(on)
            break
        terminal, h = journal._read(root, tn)
        end, release = terminal.get("worker_end_ms"), terminal.get("release_ms")
        state = terminal.get("state")
        reaped = terminal.get("direct_child_reaped_or_not_spawned")
        obs_hash = terminal.get("observation_sha256")
        if (type(end) is not int or type(release) is not int
                or not start <= end <= release
                or state not in (*control.TERMINAL, "UNKNOWN_RELEASE")
                or type(reaped) is not bool
                or (state == "UNKNOWN_RELEASE") != (not reaped)
                or terminal != {"slot_id": slot["slot_id"], "state": state,
                                "worker_end_ms": end, "release_ms": release,
                                "direct_child_reaped_or_not_spawned": reaped,
                                "observation_sha256": obs_hash, "previous_sha256": head}):
            raise ValueError("invalid terminal")
        if state == "COMPLETE":
            obs, _ = journal._read(root, on)
            if (not reaped or end-start > WORKER_MS or release-end > CLEANUP_MS
                    or release > TOTAL_MS or control.digest(obs) != obs_hash
                    or control.digest(obs) != control.digest(_expected_observation(slot))):
                raise ValueError("invalid completed observation")
            control.exact_endpoints(obs)
            observations.append(obs)
            seen.add(on)
        else:
            if obs_hash is not None or on in names:
                raise ValueError("failed slot has success output")
            failed = state
        seen.add(tn)
        head, last = h, release
        records.append({"slot_id": slot["slot_id"], "start_ms": start,
                        "worker_end_ms": end, "release_ms": release,
                        "state": state, "synthetic_observation_sha256": obs_hash})
    reason, close_ms = "UNCLOSED_BLOCKED", None
    if "close.json" in names:
        close, h = journal._read(root, "close.json")
        close_ms, reason = close.get("close_ms"), close.get("reason")
        if (dangling or type(close_ms) is not int or close_ms < last
                or type(close.get("started")) is not int
                or close != {"reason": reason, "started": started,
                             "close_ms": close_ms, "previous_sha256": head}):
            raise ValueError("invalid close")
        expected = (failed if failed else
                    "COMPLETE" if started == 720 and close_ms <= TOTAL_MS else
                    "BUDGET_EXHAUSTED" if close_ms > TOTAL_MS or
                    (started < 720 and close_ms+WORKER_MS+CLEANUP_MS > TOTAL_MS) else None)
        if reason != expected or expected is None:
            raise ValueError("invalid close disposition")
        seen.add("close.json")
        head = h
    if seen != names or (expected_head is not None and head != expected_head):
        raise ValueError("inventory or detached head mismatch")
    result = {"kind": KIND, "status": reason, "started": started,
              "completed": len(observations), "not_started": 720-started,
              "dangling_start": dangling, "local_head_sha256": head,
              "caller_detached_head_matched": expected_head is not None,
              "external_custody_verified": False, "science_executable": False,
              "gpu_release_verified": False, "analysis": None}
    # Analysis requires a caller-held detached pin and a closed full registry.
    # This is not authentication: same-account writers remain trusted.
    if reason == "COMPLETE" and expected_head is not None:
        envelope = {"kind": control.KIND, "registry": registry, "records": records,
                    "close_ms": close_ms, "close_reason": "COMPLETE"}
        result["analysis"] = control.analyze_closed_synthetic(
            records, close_ms=close_ms, close_reason="COMPLETE",
            expected_detached_digest=control.digest(envelope), observations=observations)
    return result


def inspect_fixture(root: Path, *, expected_head: str | None = None) -> dict[str, Any]:
    """No repair/restart; a complete fixture without detached pin stays unanalyzed."""
    root = Path(root)
    try:
        if not (root/"session.lock").is_file():
            raise ValueError("missing lock")
        with journal._lock(root):
            return _scan(root, expected_head)
    except BlockingIOError:
        status = "BUSY_BLOCKED"
    except (OSError, ValueError, TypeError, KeyError):
        status = "CORRUPT_BLOCKED"
    return {"kind": KIND, "status": status, "science_executable": False,
            "analysis": None}
