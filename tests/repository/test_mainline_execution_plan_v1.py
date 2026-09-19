from __future__ import annotations
import json
from pathlib import Path
from compiler.realsas_compiler_services.orchestrator.mainline import _dependency_blockers,_invalidate_dependents,content_sha256,status_text,validate_ledger,validate_plan
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
    assert by["32_REST_SOURCE_PRESERVATION_GATE"]["manifest_keys"]==["appearance","observation"]
    assert by["32_REST_SOURCE_PRESERVATION_GATE"]["depends_on"]==[
        "07_OBSERVATION_CONTRACT_QUALIFIED","26_MESH_CANDIDATE_BUILD","31_REST_RENDER_8VIEW"
    ]
    assert by["33_MOTION_SOURCE_OR_PRESET_SEAL"]["manifest_keys"]==["motion"]
    assert by["33_MOTION_SOURCE_OR_PRESET_SEAL"]["depends_on"]==[
        "29_CANONICAL_PUPPET_STATE_SEALED","32_REST_SOURCE_PRESERVATION_GATE"
    ]
    assert by["33_MOTION_SOURCE_OR_PRESET_SEAL"]["adapter"]=="compiler.realsas_compiler_services.orchestrator.adapters.motion_source_v1:seal_motion_source_or_preset_stage"
    assert by["34_MOTION_COMPILE_RUN"]["adapter"]=="UNBOUND"
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

def test_active_ledger_allows_nonprefix_pass_when_declared_dependencies_are_pass():
    import copy
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json")
    ledger=copy.deepcopy(load("canonical/ACTIVE_RUN_V1.json"))
    # Make the prefix through stage 23 valid, leave 24 pending, then retain stage 25
    # only if all of its declared dependencies are PASS. This models a reusable
    # independent branch in the execution DAG.
    for i in range(23):
        ledger["stages"][i]["status"]="PASS"
    stage25=next(row for row in ledger["stages"] if row["id"]=="25_DEFORMATION_CAPABILITY_ENVELOPE")
    stage25["status"]="PASS"
    ledger["completed_count"]=24
    ledger["next_stage"]="24_MECHANICAL_PARTITION_QUALIFIED"
    validate_ledger(plan,ledger)


def test_active_ledger_rejects_pass_with_unpassed_declared_dependency():
    import copy
    import pytest
    plan=load("canonical/MAINLINE_EXECUTION_PLAN_V1.json")
    ledger=copy.deepcopy(load("canonical/ACTIVE_RUN_V1.json"))
    stage25=next(row for row in ledger["stages"] if row["id"]=="25_DEFORMATION_CAPABILITY_ENVELOPE")
    stage25["status"]="PASS"
    ledger["completed_count"]=1
    with pytest.raises(RuntimeError,match="PASS_WITH_UNPASSED_DEPENDENCY"):
        validate_ledger(plan,ledger)


def test_dag_invalidation_preserves_independent_pass_rows():
    import copy
    plan=copy.deepcopy(load("canonical/MAINLINE_EXECUTION_PLAN_V1.json"))
    ledger=copy.deepcopy(load("canonical/ACTIVE_RUN_V1.json"))
    # Synthetic dependency fork: stage 24 depends on 15, stage 25 depends on 18,
    # stage 26 depends on 24, stage 27 consumes both branches.
    by={row["id"]:row for row in plan["stages"]}
    by["24_MECHANICAL_PARTITION_QUALIFIED"]["depends_on"]=["15_RIGGING_SURFACE_QUALIFIED"]
    by["25_DEFORMATION_CAPABILITY_ENVELOPE"]["depends_on"]=["18_SKELETON_QUALIFIED"]
    by["26_MESH_CANDIDATE_BUILD"]["depends_on"]=["15_RIGGING_SURFACE_QUALIFIED","24_MECHANICAL_PARTITION_QUALIFIED"]
    by["27_QUALIFIED_MESH_GATE"]["depends_on"]=[
        "22_SKIN_QUALIFIED","24_MECHANICAL_PARTITION_QUALIFIED",
        "25_DEFORMATION_CAPABILITY_ENVELOPE","26_MESH_CANDIDATE_BUILD"
    ]
    for row in ledger["stages"]:
        row["status"]="PASS"
    invalid=_invalidate_dependents(plan,ledger,"22_SKIN_QUALIFIED","TEST_SKIN_REFIT")
    assert "22_SKIN_QUALIFIED" in invalid
    assert "27_QUALIFIED_MESH_GATE" in invalid
    assert "28_QUALIFIED_MESH_SKIN_TRANSFER" in invalid
    assert "24_MECHANICAL_PARTITION_QUALIFIED" not in invalid
    assert "25_DEFORMATION_CAPABILITY_ENVELOPE" not in invalid
    assert "26_MESH_CANDIDATE_BUILD" not in invalid
    status={row["id"]:row["status"] for row in ledger["stages"]}
    assert status["24_MECHANICAL_PARTITION_QUALIFIED"]=="PASS"
    assert status["26_MESH_CANDIDATE_BUILD"]=="PASS"
    assert status["27_QUALIFIED_MESH_GATE"]=="PENDING"
