"""M1.3b D4 full-workload-SHAPE GPU cost-only probe.

Default: describe/read-only, never opens features or initializes CUDA.
Manual execution: separate owner authorization, explicit flags, trusted frozen
feature file, clean exact source SHA, fresh evidence root, and a parent-enforced
strict worker-process wall-time limit are ALL required. No science results.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from synaploom.research.m13b_d4_fullshape_cost import (
    ARMS, KIND, describe_fullshape_cost_probe,
)


MAX_WALL_SECONDS = 7200
MIN_WALL_SECONDS = 60
PARENT_WORKER_SCOPE = "D4_FULLSHAPE_COST_ONLY_NOT_SCIENCE"


def _write_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        f.write("\n")


def _verified_git_head(expected: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("expected-head must be a 40-digit lowercase commit SHA")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
    ).strip()
    if head != expected:
        raise ValueError("the active source commit differs from expected-head")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        text=True, stderr=subprocess.DEVNULL,
    )
    if dirty.strip():
        raise ValueError("the source working tree is not clean")
    return head


def _worker(*, arm: str, feature_path: Path, expected_head: str) -> dict:
    """Internal worker. Never called by describe-only or CI."""
    scope = os.environ.get("SYNAPLOOM_D4_INTERNAL_WORKER_SCOPE")
    check = sha256(
        (PARENT_WORKER_SCOPE + "|" + arm + "|" + str(feature_path) + "|" + expected_head).encode()
    ).hexdigest()
    if scope != check:
        raise ValueError("direct worker execution refused")
    _verified_git_head(expected_head)
    from synaploom.research.m13b_trainonly_cuda_pilot import load_trusted_train_only_features
    from synaploom.research.m13b_d4_fullshape_cost import (
        make_fullshape_trainonly_stream, execute_one_manual_cost_probe,
    )
    x, y, feature_digest = load_trusted_train_only_features(
        feature_path, trusted_source=True
    )
    stream, selection_digest = make_fullshape_trainonly_stream(x,y)
    return execute_one_manual_cost_probe(
        stream, arm=arm, verified_feature_sha256=feature_digest,
        train_subset_sha256=selection_digest,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Default CPU-only D4 full-shape cost plan.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe-only", action="store_true")
    mode.add_argument("--manual-fullshape-cost-probe", action="store_true")
    mode.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--max-wall-seconds", type=int)
    parser.add_argument("--trusted-feature-source", action="store_true")
    parser.add_argument("--owner-approval-acknowledged", action="store_true")
    parser.add_argument("--allow-in-sample-diagnostic-evaluation", action="store_true")
    opts=parser.parse_args(argv)

    inputs = (
        opts.arm,opts.features,opts.output_dir,opts.expected_head,opts.max_wall_seconds,
        opts.trusted_feature_source,opts.owner_approval_acknowledged,
        opts.allow_in_sample_diagnostic_evaluation
    )
    if opts._worker:
        if (opts.arm is None or opts.features is None or opts.expected_head is None
                or opts.output_dir is not None or opts.max_wall_seconds is not None
                or opts.trusted_feature_source or opts.owner_approval_acknowledged
                or opts.allow_in_sample_diagnostic_evaluation):
            parser.error("internal worker requires only arm, features and expected-head")
        print(json.dumps(_worker(
            arm=opts.arm, feature_path=opts.features, expected_head=opts.expected_head
        ),sort_keys=True,ensure_ascii=True,allow_nan=False))
        return 0

    # Default/no flags and --describe-only both mean no training whatsoever.
    if opts.describe_only or (not opts.manual_fullshape_cost_probe and not any(inputs)):
        if any(inputs):
            parser.error("describe-only cannot be mixed with execution arguments")
        print(json.dumps(describe_fullshape_cost_probe(),sort_keys=True,indent=2))
        return 0

    if not opts.manual_fullshape_cost_probe:
        parser.error("execution parameters without explicit manual mode")
    if (opts.arm is None or opts.features is None or opts.output_dir is None
            or opts.expected_head is None or opts.max_wall_seconds is None
            or not opts.trusted_feature_source
            or not opts.owner_approval_acknowledged
            or not opts.allow_in_sample_diagnostic_evaluation):
        parser.error("manual cost probe requires explicit arm, trusted features, fresh "
                     "output directory, expected-head, max-wall-seconds and all three acknowledgments")
    if not MIN_WALL_SECONDS <= opts.max_wall_seconds <= MAX_WALL_SECONDS:
        parser.error("max-wall-seconds must be in the bounded range 60..7200")
    if opts.output_dir.exists():
        parser.error("existing evidence root cannot be overwritten or resumed")
    if not opts.output_dir.parent.is_dir():
        parser.error("evidence parent directory must exist")
    if not opts.features.is_file():
        parser.error("trusted pinned feature file must exist")
    # The parent may be launched outside the repository. The worker runs with
    # cwd=repository root, so pass a canonical absolute path to prevent
    # silently reading a different relative feature file in the child.
    opts.features = opts.features.resolve(strict=True)
    try:
        head = _verified_git_head(opts.expected_head)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error("source identity preflight failed: "+str(exc))
    # Creating the evidence directory happens only AFTER all cheap input/source
    # guards. It is reserved BEFORE any feature deserialization and CUDA use.
    opts.output_dir.mkdir(exist_ok=False)
    intent = {
        **describe_fullshape_cost_probe(),
        "status":"OWNER_INVOKED_MANUAL_COST_PROBE__NOT_CONFIRMATORY_SCIENCE",
        "source_commit_checked":head,
        "selected_arm":opts.arm,
        "maximum_worker_seconds":opts.max_wall_seconds,
        "explicit_owner_acknowledgement_flag":True,
        "trusted_feature_source_flag":True,
        "diagnostic_train_eval_overlap_acknowledged":True,
        "feature_path_owner_local":str(opts.features),
    }
    _write_exclusive(opts.output_dir/"intent.json",intent)
    worker_command=[
        sys.executable,str(Path(__file__).resolve()),"--_worker",
        "--arm",opts.arm,"--features",str(opts.features),
        "--expected-head",head,
    ]
    scope = sha256(
        (PARENT_WORKER_SCOPE+"|"+opts.arm+"|"+str(opts.features)+"|"+head).encode()
    ).hexdigest()
    env = {**os.environ, "SYNAPLOOM_D4_INTERNAL_WORKER_SCOPE": scope}
    try:
        completed = subprocess.run(
            worker_command, cwd=Path(__file__).resolve().parents[2],
            env=env, capture_output=True, text=True,
            timeout=opts.max_wall_seconds, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"cost worker exited {completed.returncode}: " + completed.stderr[-1500:]
            )
        result = json.loads(completed.stdout)
        if (result.get("kind") != KIND
                or result.get("status")!="ENGINEERING_FULLSHAPE_COST_ONLY__NOT_SCIENTIFIC_RESULT"
                or result.get("arm")!=opts.arm
                or result.get("optimizer_steps")!=8000
                or result.get("protected_eval_partition_used") is not False
                or result.get("diagnostic_accuracy_disclosed") is not False
                or result.get("accuracy_and_forgetting_values") is not None):
            raise RuntimeError("worker produced incomplete or scientifically unsafe cost evidence")
        _write_exclusive(opts.output_dir/"cost-only-observation.json", result)
        raw=json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
        (opts.output_dir/"cost-only-observation.sha256").write_text(
            sha256(raw).hexdigest()+"\n",encoding="ascii"
        )
        print(json.dumps({
            "status":"MANUAL_ENGINEERING_COST_ONLY_COMPLETE",
            "arm":opts.arm,"steps":8000,"output_dir":str(opts.output_dir),
            "scientific_claim":"NONE",
        },sort_keys=True))
        return 0
    except subprocess.TimeoutExpired:
        _write_exclusive(opts.output_dir/"failure.json",{
            "status":"MANUAL_COST_PROBE_TIMED_OUT__NO_PARTIAL_SCIENCE_RESULT",
            "max_wall_seconds":opts.max_wall_seconds,"stage":"supervised_cuda_worker",
            "scientific_claim":"NONE",
        })
        return 2
    except Exception as exc:
        _write_exclusive(opts.output_dir/"failure.json",{
            "status":"MANUAL_COST_PROBE_FAILED__NO_PARTIAL_SCIENCE_RESULT",
            "stage":"supervised_cuda_worker","type":type(exc).__name__,
            "message":str(exc)[-1000:],"scientific_claim":"NONE",
        })
        return 2


if __name__=="__main__":
    raise SystemExit(main())
