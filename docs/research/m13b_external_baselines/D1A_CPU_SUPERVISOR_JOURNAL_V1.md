# M1.3b D1-A — CPU process supervisor and crash-conservative journal V1

**Status: IMPLEMENTED_FOR_CPU_REVIEW / SCIENCE_GPU_NO_GO.**

Base main 5483ba3c382dc02a8e592242fd4abc6db9695df4; [PR #65](https://github.com/ehsanrs2/synaploom/pull/65) merged, [exact-head CI](https://github.com/ehsanrs2/synaploom/actions/runs/45565010557) SUCCESS. The approved D1/D3/D4 numerical design remains unchanged: 360 paired draws, 720 trajectories, fixed-stake bounded-mean IUT, conditional 80% planning objective, 30 occupied GPU hours, 300/120s worker caps and no retries. No execution permission is implied.

## Implementation

src/synaploom/research/m13b_d1a_cpu_supervisor_journal.py implements a Linux-only, stdlib-only two-slot CPU process fixture. Fixed child programs are success, nonzero exit, and sleep; there is no arbitrary worker-command input, training worker, scientific seed generator, dataset or CUDA API. Child uses isolated Python with a restricted environment. Fixture caps (350ms worker, 500ms cleanup, 2500ms total) are **not** proposed or approved scientific caps.

- Fresh exclusive session directory and advisory flock held by the supervisor. Exclusive per-record creation, file fsync and directory fsync; START is committed before attempting Popen. No overwrite or automatic repair. A monotonic clock includes setup and gaps after the origin is established; before starting, reserve a full timeout and cleanup allowance. Run the fixed child with a deadline; on timeout terminate and wait/reap the direct child with a bounded wait. If creation is interrupted or fails without a handle, mark release unknown rather than assume no process exists.
- Persist terminal and close records linked to predecessor byte digests. Stop after any noncomplete attempt, do not start later slots, enumerate not-started slots. Failure, timeout, unknown release and exhaustion are not successful complete sessions.
- Inspection acquires the session lock nonblocking and checks exact canonical record bytes, expected slot order, links, timing/closure consistency and exact inventory. Busy, corrupt and unclosed sessions are blocked. Missing terminal is explicitly dangling; release unknown. Inspection neither repairs nor sends signals nor restarts any process. The original directory cannot be used to launch again. A fresh path is a new fixture, not an approved retry of a science campaign.

## Qualification tests

tests/unit/test_m13b_d1a_cpu_supervisor_journal.py contains 14 pytest cases (count includes parametrized variants). Tests launch actual CPU child success, nonzero exit and timeout processes; a separate supervisor is SIGKILLed after durable START and before child creation, with a marker outside the journal to synchronize the fault injection. Tests verify the lock while live, no repair or restart after crash, deletion, truncation, tampering, unexpected records, symlink rejection, elapsed budget admission, invalid cases and unknown release without a process handle. The crash injection does not create an orphan child; live-worker-parent-death handling remains unqualified.

Run the targeted tests with uv run --locked python -m pytest -q tests/unit/test_m13b_d1a_cpu_supervisor_journal.py. Exact-head hosted CI is required; no local/hosted test PASS is asserted at creation.

## Explicit limits and remaining gates

- This implements a real **direct CPU child** timeout/kill/reap path, not a GPU or arbitrary process-tree supervisor. Popen, filesystem sync, scheduler delays or uninterruptible kernel calls can themselves overrun a nominal deadline; the fixture cannot guarantee real hard-wall occupation. The recorded close timestamp precedes close-record fsync and lock release. Real budget accounting must include these tails and resource reservation/release, prevent new work after deadline, and be independently enforced. A tested termination margin remains unselected.
- A crash during a live worker may leave an orphan. On unclosed state this module does not prove absence of a live child, kill an unidentified process, reset a budget or permit continuation. Future qualification needs durable process identity/cgroup or equivalent lifecycle isolation, an independent watchdog, parent-death fault injection with live workers, and verified resource release. Never infer GPU release from direct CPU child exit.
- Local hash links detect accidental/partial changes, not complete rehashing by a writer. No authenticated detached root, external custody, antirollback, hostile local filesystem protection or actual power-loss/storage guarantees established. The parent directory/filesystem and same-account writers are trusted for the fixture. Advisory lock is cooperative. No automatic reconstruction of corrupt/missing terminal records.
- The two-slot fixture is not yet wired into the 720-slot offline control from PR #65. Next implement lifecycle/journal interface for the full approved registry and sealed outcomes, with independent durable admission/custody and recovery that fails closed. Audit seed population, source/data/mask, exact sample membership, optimizer post-step evidence and final scientific runner. No actual seeds are drawn and no protected data or GPU used.

**No final protocol freeze, science training, or GPU authorization.** Earlier scientific results and numerical design are unchanged. A green CI or merge does not establish scientific execution readiness.
