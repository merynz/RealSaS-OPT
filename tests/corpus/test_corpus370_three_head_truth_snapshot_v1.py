"""Truth snapshot tests use an explicitly synthetic 370-row fixture, not real truth."""
from __future__ import annotations

import csv
import hashlib
import json

import pytest

from tools.corpus.build_corpus370_three_head_truth_snapshot_v1 import (
    SCHEMA, SOURCE_FAMILIES, HEAD_TARGETS, audit_legacy_matrix, build, write_outputs,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _legacy() -> dict:
    subjects = []
    for i in range(370):
        cells = {
            family: {
                "status": "UNKNOWN",
                "reason_codes": ["NO_CURRENT_COURT_EVIDENCE"],
                "evidence_refs": [],
                "court_refs": [],
                "producer_refs": [],
                "claim_metadata": [],
            } for family in SOURCE_FAMILIES
        }
        subjects.append({
            "subject_id": f"synthetic_{i:03d}",
            "source_sha256": _sha(f"subject-{i}"),
            "source_format": ".gltf",
            "origin": "SYNTHETIC_TEST_ONLY",
            "families": cells,
        })
    subjects[0]["families"]["SOURCE_IMPORT"] = {
        "status": "PASS", "reason_codes": [],
        "evidence_refs": [{"sha256": _sha("evidence"), "schema": "SyntheticEvidence.v1"}],
        "court_refs": [{"sha256": _sha("court"), "status": "PASS", "schema": "SyntheticCourt.v1"}],
    }
    rollup = {f: {"UNKNOWN": 370} for f in SOURCE_FAMILIES}
    rollup["SOURCE_IMPORT"] = {"PASS": 1, "UNKNOWN": 369}
    return {
        "schema": "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1",
        "subject_count": 370, "training_authorized": False,
        "subjects": subjects, "rollup": {"families": rollup},
    }


def _snapshot(legacy: dict) -> dict:
    return build(
        legacy=legacy, upstream_run_id=41838505934,
        upstream_artifact_id=13327879038,
        upstream_sha256=_sha("synthetic-archive-bytes"),
        semantic_policy_sha256=_sha("synthetic-designed-prior"),
    )


def test_370_rows_and_semantic_prior_never_inherit_geometry_or_rig():
    original = _legacy()
    snapshot = _snapshot(original)
    assert snapshot["schema"] == SCHEMA
    assert snapshot["subject_count"] == 370
    assert snapshot["rollup"]["original_families"]["SOURCE_IMPORT"] == {
        "PASS": 1, "UNKNOWN": 369}
    assert snapshot["rollup"]["head_targets"]["IRIS_SEMANTIC"]["semantic_concept"] == {
        "UNKNOWN": 370}
    assert snapshot["rollup"]["head_targets"]["TESSA"]["animation_ready_deformation_consequence"] == {
        "UNKNOWN": 370}
    assert all(x["status"] == "UNKNOWN" for x in
               snapshot["subjects"][0]["heads"]["IRIS_SEMANTIC"].values())
    assert all(x["training_consumable"] is False for x in
               snapshot["subjects"][0]["heads"]["IRIS_GEOMETRY"].values())
    assert original["subjects"][0]["families"]["SOURCE_IMPORT"]["status"] == "PASS"
    assert snapshot["source_authority"]["fresh_revalidation_of_parent_court_files"] is False
    assert snapshot["training_authorized"] is False
    assert snapshot["target_admission_minted"] is False


def test_nonpass_evidence_must_not_disappear():
    legacy = _legacy()
    legacy["subjects"][1]["families"]["RIG_HIERARCHY"] = {
        "status": "FAIL",
        "reason_codes": ["SYNTHETIC_IMPORT_DISCREPANCY"],
        "evidence_refs": [{"sha256": _sha("evidence")}],
        "court_refs": [{"sha256": _sha("court"), "status": "FAIL"}],
    }
    legacy["rollup"]["families"]["RIG_HIERARCHY"] = {"UNKNOWN": 369, "FAIL": 1}
    res = _snapshot(legacy)
    assert res["subjects"][1]["heads"]["AXIS"]["source_rig_hierarchy"]["status"] == "FAIL"
    assert res["subjects"][1]["heads"]["AXIS"]["typed_controllable_rig_program_teacher"]["status"] == "UNKNOWN"


def test_historical_pass_must_be_court_bound():
    fake = _legacy()
    fake["subjects"][0]["families"]["SOURCE_IMPORT"]["court_refs"] = []
    with pytest.raises(ValueError, match="PASS_MISSING_COURT_REFS"):
        _snapshot(fake)


def test_all_370_and_rollup_recheck_is_fail_closed():
    fake = _legacy()
    fake["subjects"].pop()
    with pytest.raises(ValueError, match="EXACT_370_REQUIRED"):
        _snapshot(fake)
    fake = _legacy()
    fake["subjects"][1]["subject_id"] = fake["subjects"][0]["subject_id"]
    with pytest.raises(ValueError, match="SUBJECT_UNIQUE"):
        _snapshot(fake)
    fake = _legacy()
    fake["rollup"]["families"]["SURFACE_GEOMETRY_UDF"] = {"PASS": 370}
    with pytest.raises(ValueError, match="ROLLUP_DRIFT"):
        _snapshot(fake)


def test_output_csv_covers_exact_370_and_all_head_targets(tmp_path):
    doc = _snapshot(_legacy())
    write_outputs(doc, tmp_path)
    with (tmp_path / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 370
    assert len({r["subject_id"] for r in rows}) == 370
    assert set(f"{head}.{target}" for head, ts in HEAD_TARGETS.items()
               for target in ts) <= set(rows[0])
    assert rows[0]["IRIS_SEMANTIC.semantic_concept"] == "UNKNOWN"
    assert "UNKNOWN" in (tmp_path / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.md").read_text()
    data = json.loads((tmp_path / "CORPUS370_THREE_HEAD_TRUTH_STATUS_V1.json").read_text())
    assert data["training_authorized"] is False
    assert len(data["worklists"]["IRIS_SEMANTIC.semantic_concept"]) == 370
