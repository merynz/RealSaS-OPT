from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tools.corpus.audit_corpus370_v5_head_target_readiness_v1 import (
    REST_SHA, V5_SHA, build, render, verify_actual_court_receipts,
)


def synthetic_352():
    qualified, failures, rows = [], [], {}
    for i in range(352):
        sid = "sub_%03d" % i
        digest = hashlib.sha256(sid.encode()).hexdigest()
        qualified.append({"subject_id": sid, "source_sha256": digest})
        full_face = i < 322
        signed = 8192 if i < 65 else (100 if i < 340 else 0)
        family = {
            "SURFACE_GEOMETRY": "QUALIFIED_CORE_CANDIDATE" if full_face else "QUALIFIED_MASKED_CANDIDATE",
            "SURFACE_TOPOLOGY": "QUALIFIED_CORE_CANDIDATE" if full_face else "QUALIFIED_MASKED_CANDIDATE",
            "ORIENTED_NORMALS": "QUALIFIED_CORE_CANDIDATE" if full_face else "QUALIFIED_MASKED_CANDIDATE",
            "SIGNED_DISTANCE_OCCUPANCY": (
                "QUALIFIED_CORE_CANDIDATE" if signed == 8192 else
                "QUALIFIED_MASKED_CANDIDATE" if signed else "UNAVAILABLE_CANDIDATE"
            ),
        }
        rows[sid] = {
            "status": "PASS_COMPONENT_GEOMETRY_COURT",
            "input": {"source_sha256": digest},
            "family_candidate_outcomes": family,
            "counts": {"query_count": 8192, "signed_query_count": signed, "face_count": 10,
                       "valid_face_count": 10 if full_face else 9},
            "training_authorized": False,
        }
    for i in range(18):
        failures.append({"subject_id": "fail_%03d" % i})
    v5 = {"schema": "RealSaS.Corpus370PreparedObservationPackets.v5",
          "subject_count": 370, "prepared_subject_count": 352,
          "failure_count": 18, "training_authorized": False,
          "subjects": qualified, "failures": failures}
    geometry = {
        "schema": "RealSaS.Corpus370V5ComponentGeometryIndependentSweep.v1",
        "status": "ALL_352_COURTS_ACCOUNTED__NOT_TRAINING",
        "universe_subject_count": 370, "observation_qualified_count": 352,
        "observation_excluded_count": 18,
        "excluded_observation_subject_ids": [r["subject_id"] for r in failures],
        "identity": {"v5_index_sha256": V5_SHA, "rest_index_sha256": REST_SHA},
        "authority": {"model_targets_issued": False, "global_union_sign_proven": False,
                      "family_admission_minted": False, "training_authorized": False,
                      "candidate_only": True},
        "subject_results": rows, "training_authorized": False,
    }
    return v5, geometry


def court(v5, geom):
    return build(v5, geom, v5_sha=V5_SHA, geometry_sha="a" * 64)


def test_measured_counts_preserve_candidate_vs_global_authority():
    v5, geom = synthetic_352()
    r = court(v5, geom)
    assert r["status"] == "PASS_EXACT_STAGE50_COURT_READBACK__TARGET_SUFFICIENCY_NOT_ESTABLISHED"
    assert r["geometry_family_candidates"]["SURFACE_GEOMETRY"] == {
        "QUALIFIED_CORE_CANDIDATE": 322, "QUALIFIED_MASKED_CANDIDATE": 30}
    assert r["geometry_family_candidates"]["SIGNED_DISTANCE_OCCUPANCY"] == {
        "QUALIFIED_CORE_CANDIDATE": 65, "QUALIFIED_MASKED_CANDIDATE": 275,
        "UNAVAILABLE_CANDIDATE": 12}
    assert r["measured_component_geometry_query_samples"] == 352 * 8192
    assert r["measured_component_local_signed_query_samples"] == 65 * 8192 + 275 * 100
    assert r["source_component_sign_is_global_sdf"] is False
    assert r["unknown_head_targets"]["IRIS_GLOBAL_SOLID_SDF"]["qualified_subject_count"] is None
    assert r["unknown_head_targets"]["IRIS_DRAWABLE_OWNERSHIP"]["training_sample_count"] is None
    assert r["cross_family_intersection"] is None
    assert r["training_authorized"] is False
    assert r["materialization_authorized"] is False
    assert "UNKNOWN, not zero" in render(r)


def test_rejects_fabricated_observation_pass():
    v5, geom = synthetic_352()
    v5["subjects"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_IDENTITY"):
        court(v5, geom)


def test_rejects_missing_stage50_court():
    v5, geom = synthetic_352()
    del geom["subject_results"]["sub_001"]
    with pytest.raises(ValueError, match="MISSING_OR_EXTRA_SUBJECT"):
        court(v5, geom)


def test_rejects_component_global_sign_promotion():
    v5, geom = synthetic_352()
    geom["authority"]["global_union_sign_proven"] = True
    with pytest.raises(ValueError, match="FORGED_AUTHORITY"):
        court(v5, geom)


def test_rejects_nonindependent_source_parent():
    v5, geom = synthetic_352()
    geom["identity"]["rest_index_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PARENT_DRIFT"):
        court(v5, geom)


def test_rejects_zero_imputed_sign():
    v5, geom = synthetic_352()
    geom["subject_results"]["sub_000"]["counts"]["signed_query_count"] = 0
    with pytest.raises(ValueError, match="SIGN_MASK"):
        court(v5, geom)


def test_rejects_geometry_not_in_original_v5():
    v5, geom = synthetic_352()
    geom["subject_results"]["sub_001"]["input"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_IDENTITY"):
        court(v5, geom)


def test_rejects_hidden_training_authority():
    v5, geom = synthetic_352()
    geom["subject_results"]["sub_001"]["training_authorized"] = True
    with pytest.raises(ValueError, match="SUBJECT_COURT"):
        court(v5, geom)


def test_independent_court_byte_replay_rejects_modified_court(tmp_path):
    v5, geometry = synthetic_352()
    tiny = copy.deepcopy(geometry)
    tiny["subject_results"] = {"sub_000": tiny["subject_results"]["sub_000"]}
    row = tiny["subject_results"]["sub_000"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"subject_id": "sub_000"}))
    evidence = tmp_path / "court.json"
    payload = {
        "schema": "RealSaS.ComponentGeometryAuthorityCourt.v1",
        "status": "PASS_COMPONENT_GEOMETRY_AUTHORITY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE",
        "subject_id": "sub_000",
        "counts": row["counts"],
        "family_candidate_outcomes": row["family_candidate_outcomes"],
        "authority": {"independent_court_passed": True,
                      "final_subject_disposition_minted": False,
                      "training_authorized": False},
    }
    evidence.write_text(json.dumps(payload))
    for key, path in (("manifest", manifest), ("court", evidence)):
        row[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    verify_actual_court_receipts(tiny)
    payload["authority"]["training_authorized"] = True
    evidence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="READBACK_COURT:sub_000_SHA"):
        verify_actual_court_receipts(tiny)
