"""D4 manual full-shape cost probe: CPU-only shape and safety checks.

Never loads trusted feature pickle or invokes the GPU worker. The synthetic
50k training inventory is an expanded CPU view with deterministic row IDs.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from synaploom.protocols.h1a import H1A_CLASS_ORDER
from synaploom.research.m13b_d4_fullshape_cost import (
    ARMS, KIND, ENGINEERING_SEED, EXPECTED_EVAL_CALLS,
    EXPECTED_EVAL_EXAMPLES_PROCESSED, EXPECTED_STEPS,
    describe_fullshape_cost_probe, execute_one_manual_cost_probe,
    make_fullshape_trainonly_stream,
)


ROOT=Path(__file__).resolve().parents[2]
CLI=ROOT/"experiments"/"scripts"/"m13b_d4_fullshape_cost.py"


@pytest.fixture(scope="module")
def fake_train_inventory():
    ids=torch.arange(50000,dtype=torch.float32).reshape(-1,1).expand(-1,512)
    labels=torch.arange(100,dtype=torch.long).repeat_interleave(500)
    return ids, labels


def _cli(*args):
    return subprocess.run([sys.executable,str(CLI),*map(str,args)],
                          cwd=ROOT,capture_output=True,text=True,check=False)


def test_plan_fully_mirrors_training_and_eval_batch_shapes():
    p=describe_fullshape_cost_probe()
    assert p["kind"]==KIND
    assert p["status"]=="MANUAL_COST_PROBE_PLAN_ONLY__OWNER_AUTHORIZATION_MISSING"
    assert p["science_executable"] is False
    assert p["owner_gpu_authorization_recorded"] is False
    assert p["protected_eval_partition_used"] is False
    assert p["original_training_partition_only"] is True
    assert p["training_diagnostic_eval_overlap"] is True
    assert p["diagnostic_accuracy_is_unusable"] is True
    assert p["accuracy_values_reported"] is False
    assert p["arm_options_one_per_invocation"]==list(ARMS)
    assert p["engineering_seed"]==ENGINEERING_SEED==13
    assert ENGINEERING_SEED not in p["excluded_proposed_science_seeds"]
    assert p["tasks"]==10 and p["epochs_per_task"]==20
    assert p["train_examples_per_task"]==5000
    assert p["diagnostic_eval_examples_per_task"]==1000
    assert p["train_batch_size"]==128 and p["eval_batch_size"]==256
    assert p["train_batches_per_epoch"]==40
    assert p["last_training_batch_size"]==8
    assert p["last_diagnostic_eval_batch_size"]==232
    assert p["optimizer_steps_per_complete_trajectory"]==EXPECTED_STEPS==8000
    assert p["current_task_epoch_eval_calls"]==200
    assert p["task_stage_eval_calls"]==55
    assert p["total_task_eval_calls"]==EXPECTED_EVAL_CALLS==255
    assert p["diagnostic_eval_batches"]==1020
    assert p["total_diagnostic_eval_examples_processed"]==EXPECTED_EVAL_EXAMPLES_PROCESSED==255000
    assert p["no_automatic_gpu_dispatch"] is True
    assert p["strict_parent_subprocess_wall_timeout_required"] is True
    assert p["real_science_equivalent_runtime_guarantee"] is False
    assert p==describe_fullshape_cost_probe()


def test_default_describe_is_readonly_without_features_or_cuda():
    first=_cli()
    second=_cli("--describe-only")
    assert first.returncode==second.returncode==0
    assert json.loads(first.stdout)==json.loads(second.stdout)==describe_fullshape_cost_probe()


def test_full_train_only_stream_geometry_no_protected_eval(fake_train_inventory):
    x,y=fake_train_inventory
    initial_rng=torch.random.get_rng_state().clone()
    stream,digest=make_fullshape_trainonly_stream(x,y)
    assert len(digest)==64 and len(stream)==10
    assert torch.equal(initial_rng,torch.random.get_rng_state())
    train_ids=set()
    eval_ids=set()
    for stage,task in enumerate(stream):
        assert task.task_id==stage
        assert task.classes==tuple(H1A_CLASS_ORDER[stage*10:(stage+1)*10])
        assert task.train_x.shape==(5000,512)
        assert task.eval_x.shape==(1000,512)
        assert task.train_y.shape==(5000,)
        assert task.eval_y.shape==(1000,)
        tr=set(map(int,task.train_x[:,0].tolist()))
        ev=set(map(int,task.eval_x[:,0].tolist()))
        assert len(tr)==5000 and len(ev)==1000
        assert ev.issubset(tr)  # deliberate training/evaluation overlap
        assert tr.isdisjoint(train_ids)
        assert ev.isdisjoint(eval_ids)
        train_ids.update(tr)
        eval_ids.update(ev)
        for cls in task.classes:
            assert int((task.train_y==cls).sum())==500
            assert int((task.eval_y==cls).sum())==100
    assert len(train_ids)==50000 and len(eval_ids)==10000


def test_stream_selection_deterministic_and_does_not_mutate(fake_train_inventory):
    x,y=fake_train_inventory
    before_x=x[123,0].clone()
    s1,d1=make_fullshape_trainonly_stream(x,y)
    s2,d2=make_fullshape_trainonly_stream(x,y)
    assert d1==d2
    assert torch.equal(s1[0].eval_x,s2[0].eval_x)
    assert torch.equal(s1[-1].eval_y,s2[-1].eval_y)
    assert torch.equal(x[123,0],before_x)
    assert y.shape==(50000,)


@pytest.mark.parametrize("error", ["wrong_shape","nonfinite","bad_labels","float_labels"])
def test_bad_training_inventory_rejected_before_any_cuda(fake_train_inventory,error):
    x,y=fake_train_inventory
    if error=="wrong_shape":
        x=x[:49999]
    elif error=="nonfinite":
        x=x.clone()
        x[0,0]=float("nan")
    elif error=="bad_labels":
        y=y.clone()
        y[0]=1
    elif error=="float_labels":
        y=y.float()
    with pytest.raises(ValueError):
        make_fullshape_trainonly_stream(x,y)


def test_worker_entry_rejects_unapproved_arm_before_cuda(fake_train_inventory):
    x,y=fake_train_inventory
    stream,digest=make_fullshape_trainonly_stream(x,y)
    with pytest.raises(ValueError,match="unknown primary engineering arm"):
        execute_one_manual_cost_probe(
            stream,arm="unapproved",
            verified_feature_sha256=describe_fullshape_cost_probe()["feature_sha256_required"],
            train_subset_sha256=digest,
        )
    with pytest.raises(ValueError,match="untrusted or incomplete"):
        execute_one_manual_cost_probe(
            stream,arm="progressive_reserve",
            verified_feature_sha256="0"*64,train_subset_sha256=digest,
        )


@pytest.mark.parametrize("arguments", [
    ("--describe-only","--arm","progressive_reserve"),
    ("--arm","progressive_reserve"),
    ("--manual-fullshape-cost-probe",),
    ("--manual-fullshape-cost-probe","--arm","unknown"),
    ("--manual-fullshape-cost-probe","--arm","progressive_reserve","--max-wall-seconds","9000"),
    ("--manual-fullshape-cost-probe","--gpu"),
    ("--manual-fullshape-cost-probe","--execute-science"),
    ("--manual-fullshape-cost-probe","--seed","501"),
    ("--manual-fullshape-cost-probe","--epochs","1"),
    ("--manual-fullshape-cost-probe","--all-arms"),
    ("--manual-fullshape-cost-probe","--resume"),
    ("--_worker",),
    ("--_worker","--arm","fixed_sparse_mlp","--features","/missing","--expected-head","0"*40),
])
def test_unapproved_manual_or_direct_worker_cli_rejected(arguments):
    done=_cli(*arguments)
    assert done.returncode!=0


def test_invalid_full_manual_limit_rejected_without_data_or_root(tmp_path):
    feature=tmp_path/"fake.pt"
    feature.write_bytes(b"intentionally untrusted dummy")
    for duration in ("59","7201"):
        new_root=tmp_path/("not-created-"+duration)
        result=_cli(
            "--manual-fullshape-cost-probe",
            "--arm","progressive_reserve",
            "--features",feature,
            "--output-dir",new_root,
            "--expected-head","0"*40,
            "--max-wall-seconds",duration,
            "--trusted-feature-source",
            "--owner-approval-acknowledged",
            "--allow-in-sample-diagnostic-evaluation",
        )
        assert result.returncode!=0
        assert not new_root.exists()


def test_wrong_source_sha_blocks_worker_and_output_reservation(tmp_path):
    feature=tmp_path/"fake.pt"
    feature.write_bytes(b"untrusted")
    new_root=tmp_path/"no-output"
    result=_cli(
        "--manual-fullshape-cost-probe",
        "--arm","fixed_sparse_mlp",
        "--features",feature,
        "--output-dir",new_root,
        "--expected-head","0"*40,
        "--max-wall-seconds","60",
        "--trusted-feature-source","--owner-approval-acknowledged",
        "--allow-in-sample-diagnostic-evaluation",
    )
    assert result.returncode!=0
    assert "source identity preflight failed" in result.stderr
    assert not new_root.exists()


def test_existing_output_root_refuses_overwrite(tmp_path):
    path=tmp_path/"already"
    path.mkdir()
    feature=tmp_path/"exists.pt"
    feature.write_bytes(b"no trust")
    result=_cli(
        "--manual-fullshape-cost-probe",
        "--arm","all_recruited",
        "--features",feature,
        "--output-dir",path,
        "--expected-head","0"*40,
        "--max-wall-seconds","60",
        "--trusted-feature-source","--owner-approval-acknowledged",
        "--allow-in-sample-diagnostic-evaluation",
    )
    assert result.returncode!=0
    assert "existing evidence root" in result.stderr
