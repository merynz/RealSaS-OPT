from __future__ import annotations

import json

import pytest

from tools.corpus.build_truth139_materialization_work_orders_v1 import build
from tools.corpus.seal_truth139_materialization_receipt_v1 import (
    PASS_STATUS,
    REPO_ROOT,
    checked_out_git_sha,
    file_sha256,
    seal,
)


def _subject() -> dict:
    return {
        "subject_id": "truth139_000",
        "subject_identity_signature": "identity-000",
        "source": {
            "corpus_relative_path": "raw/quaternius/pack/glTF/truth139_000.gltf",
            "sha256": "1" * 64,
            "byte_size": 1234,
            "format": "GLTF",
            "creator": "quaternius",
            "pack_id": "quaternius-pack",
            "dependencies": [],
            "dependency_count": 0,
            "source_page_snapshot": {"path": "provenance/source_pages/pack.html", "sha256": "a" * 64},
            "official_url": "https://example.invalid",
            "drive_folder_id": "folder",
            "source_family_signature": ["SRCF:000"],
        },
        "raw_capabilities": {
            "OBSERVATION_FSTAR": True,
            "SURFACE_NORMAL_TRUTH": True,
            "STAGE47_MECHANICS": True,
            "STAGE48_APPEARANCE_PRESENTATION": True,
            "TESSA_TOPOLOGY_TEACHER": True,
        },
        "development_or_regression_tainted": False,
    }


def _work_orders(tmp_path) -> object:
    manifest = {
        "schema": "RealSaS.Truth139CorpusManifest.v2",
        "corpus_manifest_identity_sha256": "b" * 64,
        "historical_reproduction": {"passed": False, "operational_blocker": False},
        "subjects": [_subject()],
    }
    work = build(manifest)
    path = tmp_path / "work_orders.json"
    path.write_text(json.dumps(work, sort_keys=True, indent=2) + "\n")
    return path, work


def _visibility_outputs(tmp_path, *, court_status="PASS_INDEPENDENT_CPU_RAY_ORACLE_AGREES_WITH_VISIBILITY_OCCLUSION_TEACHER"):
    teacher_npz = tmp_path / "teacher.npz"
    teacher_npz.write_bytes(b"synthetic-teacher")
    producer = {
        "schema": "RealSaS.Truth139VisibilityOcclusionTeacher.v1",
        "status": "PASS_VISIBILITY_OCCLUSION_TEACHER__INDEPENDENT_CPU_ORACLE_COURT_PENDING",
        "producer_revision": "V2_SILHOUETTE_NEIGHBOR_FIRST",
        "subject_id": "truth139_000",
        "teacher_npz": {"path": str(teacher_npz), "sha256": file_sha256(teacher_npz)},
        "teacher_identity_sha256": "c" * 64,
    }
    producer_path = tmp_path / "producer.json"
    producer_path.write_text(json.dumps(producer, sort_keys=True, indent=2) + "\n")
    court = {
        "schema": "RealSaS.Truth139VisibilityOcclusionCourt.v1",
        "status": court_status,
        "subject_id": "truth139_000",
        "court_identity_sha256": "d" * 64,
    }
    court_path = tmp_path / "court.json"
    court_path.write_text(json.dumps(court, sort_keys=True, indent=2) + "\n")
    return producer_path, court_path


def _head() -> str:
    return checked_out_git_sha(REPO_ROOT)


def test_receipt_binds_work_order_source_implementation_outputs_and_court(tmp_path):
    work_path, work = _work_orders(tmp_path)
    producer_path, court_path = _visibility_outputs(tmp_path)
    result = seal(
        work_orders_path=work_path,
        subject_id="truth139_000",
        job_name="VISIBILITY_OCCLUSION_CROSS_VIEW",
        producer_manifest_path=producer_path,
        court_result_path=court_path,
        code_commit_sha=_head(),
        execution_kind="SYNTHETIC_TEST_WORKER",
        wall_seconds=1.25,
    )
    assert result["status"] == PASS_STATUS
    assert result["code_commit_sha"] == result["checked_out_git_sha"] == _head()
    assert result["work_order"]["work_order_identity_sha256"] == work["work_order_identity_sha256"]
    assert result["subject_source"]["sha256"] == "1" * 64
    assert result["implementation"]["declared"]["producer_revision"] == "V2_SILHOUETTE_NEIGHBOR_FIRST"
    assert result["implementation"]["producer_source"]["sha256"]
    assert result["implementation"]["independent_court_source"]["sha256"]
    assert result["producer_output"]["manifest_sha256"] == file_sha256(producer_path)
    assert result["producer_output"]["declared_path_sha_bindings"]["teacher_npz"] == file_sha256(tmp_path / "teacher.npz")
    assert result["independent_court"]["result_sha256"] == file_sha256(court_path)
    assert result["independent_court_passed"] is True
    assert result["derived_truth_admitted"] is False
    assert result["training_performed"] is False
    assert result["split_created"] is False
    assert result["qualification_minted"] is False


def test_receipt_rejects_nonpassing_independent_court(tmp_path):
    work_path, _work = _work_orders(tmp_path)
    producer_path, court_path = _visibility_outputs(tmp_path, court_status="FAIL_SYNTHETIC_COURT")
    with pytest.raises(ValueError, match="TRUTH139_RECEIPT_COURT_STATUS"):
        seal(
            work_orders_path=work_path,
            subject_id="truth139_000",
            job_name="VISIBILITY_OCCLUSION_CROSS_VIEW",
            producer_manifest_path=producer_path,
            court_result_path=court_path,
            code_commit_sha=_head(),
            execution_kind="SYNTHETIC_TEST_WORKER",
        )


def test_receipt_rejects_work_order_identity_drift(tmp_path):
    work_path, work = _work_orders(tmp_path)
    work["policy"]["subject_specific_code_forbidden"] = False
    work_path.write_text(json.dumps(work, sort_keys=True, indent=2) + "\n")
    producer_path, court_path = _visibility_outputs(tmp_path)
    with pytest.raises(ValueError, match="TRUTH139_RECEIPT_WORK_ORDER_IDENTITY_DRIFT"):
        seal(
            work_orders_path=work_path,
            subject_id="truth139_000",
            job_name="VISIBILITY_OCCLUSION_CROSS_VIEW",
            producer_manifest_path=producer_path,
            court_result_path=court_path,
            code_commit_sha=_head(),
            execution_kind="SYNTHETIC_TEST_WORKER",
        )


def test_receipt_rejects_declared_commit_that_is_not_checked_out(tmp_path):
    work_path, _work = _work_orders(tmp_path)
    producer_path, court_path = _visibility_outputs(tmp_path)
    wrong = "0" * 40
    if wrong == _head():
        wrong = "f" * 40
    with pytest.raises(ValueError, match="TRUTH139_RECEIPT_CODE_COMMIT_DRIFT"):
        seal(
            work_orders_path=work_path,
            subject_id="truth139_000",
            job_name="VISIBILITY_OCCLUSION_CROSS_VIEW",
            producer_manifest_path=producer_path,
            court_result_path=court_path,
            code_commit_sha=wrong,
            execution_kind="SYNTHETIC_TEST_WORKER",
        )
