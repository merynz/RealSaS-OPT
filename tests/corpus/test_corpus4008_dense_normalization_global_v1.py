"""Dense 1024 original-source ledger never grants head training authority."""
from __future__ import annotations

import hashlib
import json
import pytest

from tools.corpus.reconcile_corpus4008_dense_normalization_1024_v1 import reconcile
from tools.corpus.plan_corpus4008_dense_normalization_shard_v1 import SCHEMA as PLAN
from tools.corpus.run_corpus4008_dense_normalization_shard_v1 import SCHEMA as BATCH


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,v):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(v))
    return path


def fixture(tmp):
    master={}
    root=tmp/"shards"
    for shard in range(32):
        entries=[]
        status=[]
        for i in range(32):
            val=shard*32+i+1
            h=f"{val:064x}"
            aid=f"asset_{val:024x}"
            c="objaverse_animated_originals:"+h
            master[aid]={"candidate_id":c,"split":"FIT"}
            entries.append({
                "canonical_asset_id":aid,
                "original_source_sha256":h,
                "candidate_id":c,
                "source_recovery_shard":shard,
                "original_source_extension":".glb",
                "split":"FIT",
            })
            status.append({
                "canonical_asset_id":aid,
                "original_source_sha256":h,
                "source_shard":shard,
                "status":"SOURCE_PROCESS_TIMEOUT",
                "materialization_authorized":False,
                "training_authorized":False,
            })
        directory=root/f"shard-{shard}"
        plan=dump(directory/"PLAN.json",{
            "schema":PLAN,"shard":shard,"source_count":32,
            "source_candidates":entries,"training_authorized":False,
        })
        dump(directory/"BATCH_SUMMARY.json",{
            "schema":BATCH,"shard":shard,"plan_sha256":sha(plan),
            "source_count":32,"status_counts":{"SOURCE_PROCESS_TIMEOUT":32},
            "training_authorized":False,"materialization_authorized":False,
        })
        (directory/"RESULTS.jsonl").write_text(
            "".join(json.dumps(x)+"\n" for x in status))
    selection=dump(tmp/"MASTER.json",master)
    return selection,root


def check(master,root,out):
    return reconcile(master,root,out,expected_master_count=1024,
                     expected_master_sha=sha(master))


def test_full_failure_ledger_is_not_fake_source_truth(tmp_path):
    master,root=fixture(tmp_path)
    s=check(master,root,tmp_path/"out")
    assert s["source_count"]==1024
    assert s["imported_source_math_success_count"]==0
    assert s["source_failure_reasons"]["SOURCE_PROCESS_TIMEOUT"]==1024
    assert s["per_head_training_admitted_count"]==0
    assert s["training_authorized"] is False


def test_missing_shard_fails_closed(tmp_path):
    master,root=fixture(tmp_path)
    (root/"shard-31/BATCH_SUMMARY.json").unlink()
    with pytest.raises(ValueError,match="BATCH_SUMMARY.json"):
        check(master,root,tmp_path/"out")


def test_false_authorization_fails_even_if_source_data_claimed_valid(tmp_path):
    master,root=fixture(tmp_path)
    file=root/"shard-2/BATCH_SUMMARY.json"
    j=json.loads(file.read_text())
    j["training_authorized"]=True
    file.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="DENSE_BATCH_CERTIFICATE"):
        check(master,root,tmp_path/"out")


def test_cross_split_source_leak_does_not_enter_master(tmp_path):
    master,root=fixture(tmp_path)
    d=json.loads(master.read_text())
    victim=next(iter(d))
    d[victim]["split"]="EXTERNAL_HOLDOUT"
    master.write_text(json.dumps(d))
    with pytest.raises(ValueError,match="MASTER_ORIGINAL_SPLIT_LINEAGE"):
        check(master,root,tmp_path/"out")
