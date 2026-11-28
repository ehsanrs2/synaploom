"""Linux-only CPU fixture supervisor, never a scientific/GPU launcher.

Fixed Python child programs, no user command, feature data, RNG or torch.
The journal is a local crash-conservative record, not an authenticated witness.
An unclosed session is permanently blocked, even if all children exited.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterator

KIND = "M13B_CPU_PROCESS_FIXTURE_JOURNAL_V1"
# Small real-clock fixture budgets; NOT approved scientific timeout settings.
TIMEOUT_MS = 350
CLEANUP_MS = 500
BUDGET_MS = 2500
PROGRAMS = {
    "ok": "pass",
    "fail": "raise SystemExit(7)",
    "timeout": "import time; time.sleep(60)",
}
PLAN = ("toy-0", "toy-1")
STATES = ("COMPLETE", "FAILED", "TIMEOUT", "INTERRUPTED", "UNKNOWN_RELEASE")


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=True) + "\n").encode("ascii")


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_new(root: Path, name: str, obj: dict[str, Any]) -> str:
    """Exclusive create; partial writes are deliberately not repaired."""
    raw = _canonical(obj)
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                 os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    _sync_dir(root)
    return sha256(raw).hexdigest()


def _read(root: Path, name: str) -> tuple[dict[str, Any], str]:
    fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as file:
        import stat
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            raise ValueError("journal entry is not a regular file")
        raw = file.read(65537)
    if len(raw) > 65536:
        raise ValueError("oversized fixture record")
    value = json.loads(raw)
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("noncanonical or damaged journal entry")
    return value, sha256(raw).hexdigest()


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    fd = os.open(root / "session.lock", os.O_RDWR | os.O_CREAT |
                 os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def _elapsed(origin: int) -> int:
    # Round elapsed up to milliseconds, conservatively.
    return (time.monotonic_ns() - origin + 999_999) // 1_000_000


def _child(case: str) -> subprocess.Popen:
    # Isolated stdlib-only fixed snippets; child cannot start descendants.
    # Never inherit a CUDA-visible device setting or user Python startup path.
    return subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", PROGRAMS[case]],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, env={"CUDA_VISIBLE_DEVICES": ""},
        start_new_session=True, close_fds=True,
    )


def _reap(proc: subprocess.Popen, deadline_ns: int) -> bool:
    """Only the direct fixture child; no claim of arbitrary tree/GPU release."""
    try:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=max(0, (deadline_ns-time.monotonic_ns())/1e9))
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_fixture(root: Path, cases: tuple[str, str] = ("ok", "ok")) -> dict[str, Any]:
    """One new local session; never reopen/resume/retry an existing directory."""
    if (type(cases) is not tuple or len(cases) != len(PLAN)
            or any(type(x) is not str or x not in PROGRAMS for x in cases)):
        raise ValueError("only two fixed CPU fixture cases accepted")
    root = Path(root)
    root.mkdir(mode=0o700, exist_ok=False)
    _sync_dir(root.parent)
    with _lock(root):
        origin = time.monotonic_ns()
        manifest = {"kind": KIND, "slots": list(PLAN), "cases": list(cases),
                    "budget_ms": BUDGET_MS, "timeout_ms": TIMEOUT_MS,
                    "cleanup_ms": CLEANUP_MS, "science_executable": False}
        head = _write_new(root, "manifest.json", manifest)
        reason = "COMPLETE"
        started = 0
        for i, case in enumerate(cases):
            start = _elapsed(origin)
            if start + TIMEOUT_MS + CLEANUP_MS > BUDGET_MS:
                reason = "BUDGET_EXHAUSTED"
                break
            deadline = origin + (start + TIMEOUT_MS)*1_000_000
            cleanup_deadline = min(origin+BUDGET_MS*1_000_000,
                                   deadline+CLEANUP_MS*1_000_000)
            # START must be durable before any subprocess creation.
            head = _write_new(root, f"{i}.start.json", {
                "slot": PLAN[i], "case": case, "start_ms": start,
                "previous_sha256": head})
            started += 1
            proc = None
            code = None
            state = "FAILED"
            released = True
            try:
                if time.monotonic_ns() >= deadline:
                    state = "TIMEOUT"
                else:
                    proc = _child(case)
                    try:
                        code = proc.wait(timeout=max(0, (deadline-time.monotonic_ns())/1e9))
                        state = "COMPLETE" if code == 0 else "FAILED"
                        if time.monotonic_ns() > deadline:
                            state = "TIMEOUT"
                    except subprocess.TimeoutExpired:
                        state = "TIMEOUT"
            except KeyboardInterrupt:
                state = "INTERRUPTED"
                # Interruption during Popen may leave no returned handle.
                if proc is None:
                    released = False
            except OSError:
                state = "FAILED"
                if proc is None:
                    released = False
            finally:
                if proc is not None:
                    released = _reap(proc, cleanup_deadline)
                    code = proc.returncode
            end = _elapsed(origin)
            if not released:
                state = "UNKNOWN_RELEASE"
            elif end > start + TIMEOUT_MS + CLEANUP_MS or end > BUDGET_MS:
                state = "TIMEOUT"
            head = _write_new(root, f"{i}.terminal.json", {
                "slot": PLAN[i], "state": state, "end_ms": end,
                "direct_child_reaped_or_not_spawned": released,
                "returncode": code, "previous_sha256": head})
            if state != "COMPLETE":
                reason = state
                break
        closed = _elapsed(origin)
        if closed > BUDGET_MS and reason == "COMPLETE":
            reason = "BUDGET_EXHAUSTED"
        _write_new(root, "close.json", {
            "reason": reason, "started": started, "close_ms": closed,
            "previous_sha256": head})
    return inspect_fixture(root)


def _scan(root: Path) -> dict[str, Any]:
    """Read the exact journal prefix. Missing terminal/close is never repaired."""
    names = {p.name for p in root.iterdir()}
    manifest, head = _read(root, "manifest.json")
    cases = manifest.get("cases")
    if (type(cases) is not list or len(cases) != len(PLAN)
            or any(type(x) is not str or x not in PROGRAMS for x in cases)):
        raise ValueError("bad manifest cases")
    expected = {"kind": KIND, "slots": list(PLAN), "cases": cases,
                "budget_ms": BUDGET_MS, "timeout_ms": TIMEOUT_MS,
                "cleanup_ms": CLEANUP_MS, "science_executable": False}
    if _canonical(manifest) != _canonical(expected):
        raise ValueError("manifest mismatch")
    seen = {"session.lock", "manifest.json"}
    states = []
    started = 0
    last = 0
    dangling = False
    for i, slot in enumerate(PLAN):
        sn, tn = f"{i}.start.json", f"{i}.terminal.json"
        if sn not in names:
            break
        if states and states[-1] != "COMPLETE":
            raise ValueError("attempt after failure")
        start, h = _read(root, sn)
        ms = start.get("start_ms")
        if (type(ms) is not int or not last <= ms
                or ms+TIMEOUT_MS+CLEANUP_MS > BUDGET_MS
                or start != {"slot": slot, "case": cases[i], "start_ms": ms,
                             "previous_sha256": head}):
            raise ValueError("invalid START")
        seen.add(sn)
        head = h
        started += 1
        if tn not in names:
            dangling = True
            break
        terminal, h = _read(root, tn)
        end, state = terminal.get("end_ms"), terminal.get("state")
        release, code = terminal.get("direct_child_reaped_or_not_spawned"), terminal.get("returncode")
        if (type(end) is not int or end < ms or state not in STATES
                or type(release) is not bool
                or (code is not None and type(code) is not int)
                or terminal != {"slot": slot, "state": state, "end_ms": end,
                                "direct_child_reaped_or_not_spawned": release,
                                "returncode": code, "previous_sha256": head}
                or (state == "UNKNOWN_RELEASE") != (not release)
                or (state == "COMPLETE" and (code != 0 or not release
                    or end > ms+TIMEOUT_MS+CLEANUP_MS or end > BUDGET_MS))):
            raise ValueError("invalid terminal")
        seen.add(tn)
        head = h
        last = end
        states.append(state)
    reason = "UNCLOSED_BLOCKED"
    close_ms = None
    if "close.json" in names:
        close, h = _read(root, "close.json")
        close_ms, reason = close.get("close_ms"), close.get("reason")
        if (dangling or type(close_ms) is not int or close_ms < last
                or type(close.get("started")) is not int
                or close != {"reason": reason, "started": started,
                             "close_ms": close_ms, "previous_sha256": head}):
            raise ValueError("invalid close")
        if states and states[-1] != "COMPLETE":
            valid_reason = states[-1]
        elif started == len(PLAN) and close_ms <= BUDGET_MS:
            valid_reason = "COMPLETE"
        elif (close_ms+TIMEOUT_MS+CLEANUP_MS > BUDGET_MS
              and started < len(PLAN)) or close_ms > BUDGET_MS:
            valid_reason = "BUDGET_EXHAUSTED"
        else:
            raise ValueError("premature close")
        if reason != valid_reason:
            raise ValueError("wrong close reason")
        seen.add("close.json")
        head = h
    if seen != names:
        raise ValueError("unexpected/missing journal files")
    return {"kind": KIND, "status": reason, "started": started,
            "terminal_states": states, "dangling_start": dangling,
            "not_started": list(PLAN[started:]), "close_ms": close_ms,
            "local_chain_head_sha256": head, "retry_allowed": False,
            "direct_child_release_unknown": dangling or "UNKNOWN_RELEASE" in states,
            "science_executable": False, "gpu_release_verified": False,
            "external_custody_verified": False}


def inspect_fixture(root: Path) -> dict[str, Any]:
    """Read-only after lock acquisition. No process signalling or repair."""
    root = Path(root)
    try:
        # Don't create missing lock while inspecting a damaged session.
        if not (root / "session.lock").is_file():
            raise ValueError("missing session lock")
        with _lock(root):
            return _scan(root)
    except BlockingIOError:
        return {"kind": KIND, "status": "BUSY_BLOCKED",
                "retry_allowed": False, "science_executable": False}
    except (OSError, ValueError, TypeError, KeyError):
        return {"kind": KIND, "status": "CORRUPT_BLOCKED",
                "retry_allowed": False, "science_executable": False}
