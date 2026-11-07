"""Unknown source/2D/control target never becomes training supervision."""
import hashlib
import json
import pytest

from tools.corpus.validate_corpus4008_prematerialization_v1 import (
    validate,HEADS,
)


def write(path,doc):
    path.write_text(json.dumps(doc,sort_keys=True))
    return {"path":path.name,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path,head="IRIS_GRAPHICS",split="FIT"):
    source="a"*64
    aid="asset_sample"
    plan=write(tmp_path/"plan.json",{
        "schema":"RealSaS.Corpus4008ReversibleSourceNormalizationPlan.v1",
        "status":"REVERSIBLE_PLAN_NUMERICALLY_CHECKED__NOT_SOURCE_PARITY",
        "canonical_asset_id":aid,
        "original_source_sha256":source,
        "materialization_authorized":False,
        "training_authorized":False,
        "source_mesh_vertex_roundtrip_courted":False,
        "common_global_scale_for_all_views_and_poses":True,
        "per_view_recenter_permitted":False,
    })
    d={
        "schema":"RealSaS.Corpus4008StandardizedSourceTargetCandidate.v1",
        "canonical_asset_id":aid,
        "source_sha256":source,
        "source_identity_cluster":"source_cluster_original",
        "split":split,
        "model_head":head,
        "normalization_plan":plan,
        "family_courts":{},
        "target_family_status":{"geometry_normals":"UNKNOWN"},
        "target_payloads":{},
        "graphics_target_domain":(
            "STYLIZED_2D_GAME_ART" if head=="IRIS_GRAPHICS"
            else "NOT_A_GRAPHICS_TARGET"),
        "training_authorized":False,
        "materialization_authorized":False,
    }
    f=tmp_path/"candidate.json"
    f.write_text(json.dumps(d))
    return f,d


def test_missing_courts_are_explicitly_blocked(tmp_path):
    file,_=fixture(tmp_path)
    v=validate(file)
    assert v["status"]=="BLOCKED_PENDING_TRUSTED_ADMISSION"
    assert v["materialization_authorized"] is False
    assert v["training_authorized"] is False
    assert "MISSING_REQUIRED_COURT:STYLIZED_2D_GAME_ART" in v["blocked_reasons"]
    assert "MISSING_REQUIRED_COURT:CORE370_2D_ART_DOMAIN_PARITY" in v["blocked_reasons"]


def test_masked_optional_targets_are_allowed_but_not_supervision(tmp_path):
    file,doc=fixture(tmp_path,head="AXIS")
    doc["target_family_status"]={
        "rig_hierarchy":"UNKNOWN","ik_solver":"MASKED","kinematics":"FAIL"}
    file.write_text(json.dumps(doc))
    v=validate(file)
    assert not v["preflight_structurally_complete"]
    assert v["materialization_authorized"] is False


def test_cannot_zero_fill_unknown_or_masked_targets(tmp_path):
    file,doc=fixture(tmp_path)
    ref=write(tmp_path/"fake_target.json",{"fake":0})
    doc["target_payloads"]["geometry_normals"]=ref
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="UNKNOWN_MASKED_TARGET_MUST_BE_ABSENT"):
        validate(file)


def test_forged_training_preapproval_rejected(tmp_path):
    file,doc=fixture(tmp_path)
    doc["training_authorized"]=True
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="PREAUTHORIZED_INPUT_FORBIDDEN"):
        validate(file)


def test_cross_source_normalization_plan_rejected(tmp_path):
    file,doc=fixture(tmp_path)
    plan=tmp_path/"plan.json"
    value=json.loads(plan.read_text())
    value["original_source_sha256"]="b"*64
    doc["normalization_plan"]=write(plan,value)
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="NORMALIZATION_PLAN_NOT_SOURCE_TRUTH"):
        validate(file)


def test_raw_photorealistic_3d_graphics_target_rejected(tmp_path):
    file,doc=fixture(tmp_path)
    doc["graphics_target_domain"]="PHOTOREALISTIC_3D"
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="GRAPHICS_TARGET_DOMAIN"):
        validate(file)


def test_holdout_still_blocked_when_all_independent_court_receipts_claim_pass(tmp_path):
    file,doc=fixture(tmp_path,head="MIRA",split="EXTERNAL_HOLDOUT")
    for court in HEADS["MIRA"]:
        p=tmp_path/(court+".json")
        ref=write(p,{
            "schema":"RealSaS.IndependentSourceCourt.v1",
            "asset_id":"asset_sample","source_sha256":"a"*64,
            "court_kind":court,"status":"PASS",
            "independent_review":True,"training_authorized":False
        })
        doc["family_courts"][court]={"status":"PASS","receipt":ref}
    file.write_text(json.dumps(doc))
    v=validate(file)
    assert v["preflight_structurally_complete"] is True
    assert v["status"]=="BLOCKED_PENDING_TRUSTED_ADMISSION"
    assert "PROTECTED_HOLDOUT_CANNOT_ENTER_TRAIN" in v["blocked_reasons"]
    assert "EXTERNAL_TRUSTED_ADMISSION_FREEZE_REQUIRED" in v["blocked_reasons"]


def test_tampered_source_court_payload_rejected(tmp_path):
    file,doc=fixture(tmp_path,head="AXIS")
    court=HEADS["AXIS"][0]
    ref=write(tmp_path/"receipt.json",{
        "schema":"RealSaS.IndependentSourceCourt.v1",
        "asset_id":"asset_sample",
        "source_sha256":"a"*64,
        "court_kind":court,
        "status":"PASS","independent_review":True,"training_authorized":False,
    })
    doc["family_courts"][court]={"status":"PASS","receipt":ref}
    file.write_text(json.dumps(doc))
    (tmp_path/"receipt.json").write_text('{"status":"PASS","tampered":true}')
    with pytest.raises(ValueError,match="EVIDENCE_SHA_DRIFT"):
        validate(file)
