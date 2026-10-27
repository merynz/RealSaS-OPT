from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters import source_ray_appearance_platform_v1 as platform_admission


def _raw(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _measurement(subject_id="fit8_fixture_v1"):
    return {
        "schema": platform_admission.MEASUREMENT_SCHEMA,
        "subject_id": subject_id,
        "court_status": platform_admission.PASS_MEASUREMENT,
        "qualification_minted": False,
        "qualified_sampler": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "rays": [{
            "complete": True,
            "has_depth_tie": False,
            "hit_count": 2,
            "canonical_hits": [
                {"face_index": 3, "depth": 1.0, "barycentric": [0.2, 0.3, 0.5]},
                {"face_index": 5, "depth": 2.0, "barycentric": [0.3, 0.3, 0.4]},
            ],
            "measurement": {
                "schema_version": "RealSaS.RayAppearanceMeasurement.v1",
                "status": platform_admission.PASS_FACTORIZATION,
                "valid": True,
                "effective_alpha": [1.0, 1.0],
                "straight_linear_rgb": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            },
        }],
    }


def _bundle(monkeypatch):
    measurement = _measurement()
    measurement_text = _raw(measurement)
    packed_sha = _sha_text(measurement_text)
    witness = "a" * 64
    court = {
        "schema": platform_admission.COURT_SCHEMA,
        "court_status": platform_admission.PASS_COURT,
        "subject_count": 1,
        "ray_count": 1,
        "hit_count": 2,
        "witness_sha256": witness,
        "family_qualification_input": {
            "family": platform_admission.FAMILY,
            "measurement_status": "MEASURED",
            "source_equivalent": True,
            "coverage_qualified": True,
            "replay_deterministic": True,
            "valid_count": 2,
            "total_count": 2,
            "witness_sha256": witness,
        },
        "local_qualification_decision": {
            "family": platform_admission.FAMILY,
            "oracle_eligible": True,
            "deterministic_student_supervision_eligible": False,
        },
        "global_sampler_qualification_claimed": False,
        "registry_admitted": False,
        "training_allowed": False,
        "product_pass_claimed": False,
        "policy": {"coverage_scope": "FIXTURE_SPARSE_DOMAIN"},
        "subjects": [{
            "subject_id": measurement["subject_id"],
            "packed_replay_core_exact": True,
            "source_equivalent_on_declared_hits": True,
            "source_reference_replay_core_exact": True,
            "parents": {"packed_a_sha256": packed_sha, "packed_b_sha256": packed_sha},
        }],
    }
    court_text = _raw(court)
    court_sha = _sha_text(court_text)
    monkeypatch.setattr(platform_admission, "SOURCE_COURT_RAW_SHA256", court_sha)
    return {
        "schema": platform_admission.BUNDLE_SCHEMA,
        "source_artifact": {
            "workflow_run_id": platform_admission.SOURCE_WORKFLOW_RUN_ID,
            "artifact_id": platform_admission.SOURCE_ARTIFACT_ID,
            "artifact_name": platform_admission.SOURCE_ARTIFACT_NAME,
            "artifact_digest": platform_admission.SOURCE_ARTIFACT_DIGEST,
        },
        "court": {"sha256": court_sha, "json_text": court_text},
        "measurements": [{"subject_id": measurement["subject_id"], "sha256": packed_sha, "json_text": measurement_text}],
        "transport_authority": "EXTERNAL_UNQUALIFIED_INPUT_ONLY",
        "registry_qualification_minted": False,
        "canonical_mesh_binding_minted": False,
        "training_authorized": False,
        "product_authority_changed": False,
    }


def _ctx(tmp_path: Path, bundle: dict):
    return {
        "run_manifest": {"source_ray_appearance_court_bundle": bundle},
        "run_root": tmp_path,
        "stage": {"id": "91_SOURCE_RAY_APPEARANCE_PLATFORM_ADMISSION"},
    }


def test_platform_admission_emits_sparse_qualification_and_evidence(tmp_path, monkeypatch):
    result = platform_admission.admit_source_ray_appearance_v1(_ctx(tmp_path, _bundle(monkeypatch)))
    assert result["status"] == "PASS"
    assert result["diagnostics"]["subject_count"] == 1
    assert result["diagnostics"]["ray_count"] == 1
    assert result["diagnostics"]["hit_count"] == 2
    assert result["diagnostics"]["canonical_mesh_binding_minted"] is False
    outputs = {row["schema"]: row for row in result["outputs"]}
    assert set(outputs) == {platform_admission.QUALIFICATION_SCHEMA, platform_admission.EVIDENCE_SCHEMA}
    qualification = json.loads(Path(outputs[platform_admission.QUALIFICATION_SCHEMA]["path"]).read_text())
    evidence = json.loads(Path(outputs[platform_admission.EVIDENCE_SCHEMA]["path"]).read_text())
    assert qualification["oracle_eligible"] is True
    assert qualification["global_sampler_qualified"] is False
    assert qualification["canonical_mesh_binding_minted"] is False
    assert evidence["measurements"][0]["subject_id"] == "fit8_fixture_v1"
    assert evidence["canonical_mesh_binding_minted"] is False


def test_platform_admission_rejects_measurement_raw_byte_substitution(tmp_path, monkeypatch):
    bundle = _bundle(monkeypatch)
    bundle["measurements"][0]["json_text"] += " "
    with pytest.raises(QualificationError, match="MEASUREMENT_HASH_DRIFT"):
        platform_admission.admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_measurement_parent_substitution(tmp_path, monkeypatch):
    bundle = _bundle(monkeypatch)
    bundle["measurements"][0]["sha256"] = "b" * 64
    with pytest.raises(QualificationError, match="MEASUREMENT_PARENT_HASH_DRIFT"):
        platform_admission.admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_student_scope_escalation(tmp_path, monkeypatch):
    bundle = _bundle(monkeypatch)
    court = json.loads(bundle["court"]["json_text"])
    court["local_qualification_decision"]["deterministic_student_supervision_eligible"] = True
    text = _raw(court)
    digest = _sha_text(text)
    monkeypatch.setattr(platform_admission, "SOURCE_COURT_RAW_SHA256", digest)
    bundle["court"] = {"sha256": digest, "json_text": text}
    with pytest.raises(QualificationError, match="STUDENT_SCOPE_ESCALATION"):
        platform_admission.admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_external_prequalification(tmp_path, monkeypatch):
    bundle = _bundle(monkeypatch)
    bundle["registry_qualification_minted"] = True
    with pytest.raises(QualificationError, match="EXTERNAL_PREQUALIFICATION_FORBIDDEN"):
        platform_admission.admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))
