from __future__ import annotations
import json
from pathlib import Path
from compiler.realsas_compiler_services.orchestrator.mainline import _dependency_blockers,content_sha256,status_text,validate_ledger,validate_plan
ROOT=Path(__file__).resolve().parents[2]
def load(rel): return json.loads((ROOT/rel).read_text(encoding="utf-8"))
def test_plan_is_exact_40_stage_subject_agnostic_contract():
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json")
    assert len(validate_plan(plan))==64
    assert plan["stage_count"]==40
    assert plan["subject_specific_code_forbidden"] is True
    assert plan["performance_contract"]["expected_exact_8view_xyz_reduction_factor"]==8.0
    assert plan["stages"][36]["id"]=="37_RSS_MATERIALIZE_COMPACT"
def test_active_ledger_is_consistent_and_human_resumable():
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json"); ledger=load("canonical/ACTIVE_RUN_V1.json")
    validate_ledger(plan,ledger); text=status_text(plan,ledger)
    assert "progress=0/40" in text and "next=01_SOURCE_BYTES_SEALED" in text
def test_motion_manifest_scope_does_not_touch_iris_or_skin():
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json"); by={x["id"]:x for x in plan["stages"]}
    assert by["10_IRIS_FIT"]["manifest_keys"]==["iris_fit"]
    assert by["21_ARACHNE_FIT"]["manifest_keys"]==["arachne_fit"]
    assert by["33_MOTION_SOURCE_OR_PRESET_SEAL"]["manifest_keys"]==["motion"]
def test_policy_hash_changes_on_semantic_change():
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json"); policy=dict(plan["stages"][0]["policy"]); before=content_sha256(policy); policy["cacheable"]=False
    assert content_sha256(policy)!=before

def test_stage_cannot_execute_past_unpassed_or_stale_dependency(tmp_path):
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json")
    ledger=load("canonical/ACTIVE_RUN_V1.json")
    stage=next(x for x in plan["stages"] if x["id"]=="24_MECHANICAL_PARTITION_QUALIFIED")
    dep=next(x for x in ledger["stages"] if x["id"]=="23_ARACHNE_CHECKPOINT_SEALED")
    assert _dependency_blockers(stage,ledger)==["DEPENDENCY_NOT_PASS:23_ARACHNE_CHECKPOINT_SEALED:PENDING"]
    artifact=tmp_path/"sealed.json"; artifact.write_text("{}\n",encoding="utf-8")
    import hashlib
    dep["status"]="PASS"
    dep["outputs"]=[{"path":str(artifact),"sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()}]
    assert _dependency_blockers(stage,ledger)==[]
    artifact.write_text("{\"drift\":true}\n",encoding="utf-8")
    assert _dependency_blockers(stage,ledger)==["DEPENDENCY_OUTPUT_IDENTITY_INVALID:23_ARACHNE_CHECKPOINT_SEALED"]

def test_active_ledger_rejects_nonprefix_pass_state():
    import copy
    import pytest
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json")
    ledger=copy.deepcopy(load("canonical/ACTIVE_RUN_V1.json"))
    ledger["stages"][1]["status"]="PASS"
    ledger["stages"][1]["outputs"]=[{"path":"/nonexistent","sha256":"0"*64}]
    ledger["completed_count"]=1
    with pytest.raises(RuntimeError,match="NONPREFIX_PASS"):
        validate_ledger(plan,ledger)
