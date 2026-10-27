from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.source_ray_appearance_platform_v1 import (
    BUNDLE_SCHEMA,
    COURT_SCHEMA,
    EVIDENCE_SCHEMA,
    MEASUREMENT_SCHEMA,
    QUALIFICATION_SCHEMA,
    admit_source_ray_appearance_v1,
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _measurement(subject_id="fit8_fixture_v1"):
    return {
        "schema": MEASUREMENT_SCHEMA,
        "subject_id": subject_id,
        "court_status": "PASS_MEASURED_SOURCE_RAY_APPEARANCE_CANDIDATE",
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
                "status": "PASS_MEASURED_RAY_APPEARANCE_FACTORIZATION",
                "valid": True,
                "effective_alpha": [1.0, 1.0],
                "straight_linear_rgb": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            },
        }],
    }


def _bundle():
    measurement = _measurement()
    packed_sha = _digest(measurement)
    witness = "a" * 64
    court = {
        "schema": COURT_SCHEMA,
        "court_status": "PASS_SOURCE_RAY_APPEARANCE_LOCAL_QUALIFICATION_CANDIDATE",
        "subject_count": 1,
        "ray_count": 1,
        "hit_count": 2,
        "witness_sha256": witness,
        "family_qualification_input": {
            "family": "RAY_COMPOSITING",
            "measurement_status": "MEASURED",
            "source_equivalent": True,
            "coverage_qualified": True,
            "replay_deterministic": True,
            "valid_count": 2,
            "total_count": 2,
            "witness_sha256": witness,
        },
        "local_qualification_decision": {
            "family": "RAY_COMPOSITING",
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
    return {
        "schema": BUNDLE_SCHEMA,
        "court": court,
        "court_sha256": _digest(court),
        "measurements": [{"subject_id": measurement["subject_id"], "sha256": packed_sha, "payload": measurement}],
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


def test_platform_admission_emits_sparse_qualification_and_evidence(tmp_path):
    result = admit_source_ray_appearance_v1(_ctx(tmp_path, _bundle()))
    assert result["status"] == "PASS"
    assert result["diagnostics"]["subject_count"] == 1
    assert result["diagnostics"]["ray_count"] == 1
    assert result["diagnostics"]["hit_count"] == 2
    assert result["diagnostics"]["canonical_mesh_binding_minted"] is False
    outputs = {row["schema"]: row for row in result["outputs"]}
    assert set(outputs) == {QUALIFICATION_SCHEMA, EVIDENCE_SCHEMA}
    qualification = json.loads(Path(outputs[QUALIFICATION_SCHEMA]["path"]).read_text())
    evidence = json.loads(Path(outputs[EVIDENCE_SCHEMA]["path"]).read_text())
    assert qualification["oracle_eligible"] is True
    assert qualification["global_sampler_qualified"] is False
    assert qualification["deterministic_student_supervision_eligible"] is False
    assert qualification["canonical_mesh_binding_minted"] is False
    assert evidence["measurements"][0]["subject_id"] == "fit8_fixture_v1"
    assert evidence["canonical_mesh_binding_minted"] is False


def test_platform_admission_rejects_court_hash_drift(tmp_path):
    bundle = _bundle()
    bundle["court"]["court_status"] = "FAIL"
    with pytest.raises(QualificationError, match="COURT_HASH_DRIFT"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_measurement_parent_substitution(tmp_path):
    bundle = _bundle()
    bundle["measurements"][0]["sha256"] = "b" * 64
    with pytest.raises(QualificationError, match="MEASUREMENT_PARENT_HASH_DRIFT"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_student_scope_escalation(tmp_path):
    bundle = _bundle()
    bundle["court"]["local_qualification_decision"]["deterministic_student_supervision_eligible"] = True
    bundle["court_sha256"] = _digest(bundle["court"])
    with pytest.raises(QualificationError, match="STUDENT_SCOPE_ESCALATION"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_external_prequalification(tmp_path):
    bundle = _bundle()
    bundle["registry_qualification_minted"] = True
    with pytest.raises(QualificationError, match="EXTERNAL_PREQUALIFICATION_FORBIDDEN"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))
