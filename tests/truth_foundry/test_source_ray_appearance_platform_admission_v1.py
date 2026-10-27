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
    QUALIFICATION_SCHEMA,
    admit_source_ray_appearance_v1,
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _bundle():
    record_hashes = ["1" * 64, "2" * 64]
    court = {
        "schema": COURT_SCHEMA,
        "status": "PASS_SOURCE_EQUIVALENT_SPARSE_APPEARANCE_CANDIDATE",
        "scope": {
            "domain": "SPARSE_MEASURED_RAYS_ONLY",
            "valid_hit_count": len(record_hashes),
            "global_sampler_qualified": False,
        },
        "qualification_decision": {
            "family": "RAY_COMPOSITING",
            "oracle_eligible": True,
        },
        "training_authorized": False,
        "product_authority_changed": False,
    }
    evidence = {
        "record_hashes": record_hashes,
        "measurement_payload": {"schema": "RealSaS.SourceRayAppearanceMeasurementCandidate.v1"},
    }
    return {
        "schema": BUNDLE_SCHEMA,
        "court": court,
        "court_sha256": _digest(court),
        "evidence": evidence,
        "evidence_sha256": _digest(evidence),
    }


def _ctx(tmp_path: Path, bundle: dict):
    return {
        "run_manifest": {"source_ray_appearance_court_bundle": bundle},
        "run_root": tmp_path,
        "stage": {"id": "91_SOURCE_RAY_APPEARANCE_PLATFORM_ADMISSION"},
    }


def test_platform_admission_emits_separate_qualification_and_evidence(tmp_path):
    result = admit_source_ray_appearance_v1(_ctx(tmp_path, _bundle()))
    assert result["status"] == "PASS"
    assert result["diagnostics"]["record_count"] == 2
    assert result["diagnostics"]["global_sampler_qualified"] is False
    outputs = {row["schema"]: row for row in result["outputs"]}
    assert set(outputs) == {QUALIFICATION_SCHEMA, EVIDENCE_SCHEMA}
    qualification = json.loads(Path(outputs[QUALIFICATION_SCHEMA]["path"]).read_text())
    evidence = json.loads(Path(outputs[EVIDENCE_SCHEMA]["path"]).read_text())
    assert qualification["oracle_eligible"] is True
    assert qualification["global_sampler_qualified"] is False
    assert qualification["training_authorized"] is False
    assert evidence["record_hashes"] == ["1" * 64, "2" * 64]
    assert evidence["global_sampler_qualified"] is False


def test_platform_admission_rejects_court_hash_drift(tmp_path):
    bundle = _bundle()
    bundle["court"]["status"] = "FAIL"
    with pytest.raises(QualificationError, match="COURT_HASH_DRIFT"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_global_sampler_escalation(tmp_path):
    bundle = _bundle()
    bundle["court"]["scope"]["global_sampler_qualified"] = True
    bundle["court_sha256"] = _digest(bundle["court"])
    with pytest.raises(QualificationError, match="GLOBAL_SCOPE_FORBIDDEN"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))


def test_platform_admission_rejects_record_count_drift(tmp_path):
    bundle = _bundle()
    bundle["court"]["scope"]["valid_hit_count"] = 3
    bundle["court_sha256"] = _digest(bundle["court"])
    with pytest.raises(QualificationError, match="RECORD_COUNT_DRIFT"):
        admit_source_ray_appearance_v1(_ctx(tmp_path, bundle))
