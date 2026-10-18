from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.iris_vnext.audit_fit8_preparation_snapshot_v1 import (
    STATUS_CURRENT,
    STATUS_INCOMPLETE,
    STATUS_STALE,
    audit_snapshot,
)
from tools.iris_vnext.build_fit8_boundary_causality_v1 import _static_rows
from tools.iris_vnext.verify_fit8_candidate_roundtrip_v1 import (
    verify_candidate_roundtrip,
)
from tools.iris_vnext.migrate_v8_relation_head_v2 import (
    V8_BINDING_SHA256,
    V8_CHECKPOINT_FILE,
    V8_CHECKPOINT_SHA256,
    V8_SOURCE_SHA,
)
from tools.iris_vnext.seal_fit8_pretraining_materialization_v1 import seal_materialization
from tools.iris_vnext.audit_fit8_pretraining_readiness_v1 import audit_readiness

ROOT = Path(__file__).resolve().parents[2]
CURRENT_SHA = "b" * 40
STALE_SHA = "9" * 40


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2), encoding="utf-8")


def _snapshot(root: Path, source_sha: str) -> Path:
    _write_json(root / "teachers/TEACHER_BATCH_RECEIPT.json", {"schema": "RealSaS.IRISVNextFit8TeacherBatchReceipt.v1"})
    _write_json(
        root / "prepared/COHORT_MANIFEST.json",
        {"schema": "RealSaS.IRISVNextFit8Manifest.v4", "manifest_value_sha256": "a" * 64},
    )
    _write_json(
        root / "prepared/PREPARATION_RECEIPT.json",
        {
            "schema": "RealSaS.IRISVNextFit8PreparationReceipt.v2",
            "source_commit_sha": source_sha,
            "source_commit_bound": True,
            "scientific_fit_started": False,
            "scientific_fit_pass_claimed": False,
        },
    )
    _write_json(root / "receipts/HYDRATION_RECEIPT.json", {"schema": "RealSaS.VNextFit8LegacyHydration.v1"})
    return root


def test_remote_snapshot_audit_marks_exact_current_source_candidate_but_never_authorizes_consumption(tmp_path: Path):
    report = audit_snapshot(_snapshot(tmp_path, CURRENT_SHA), CURRENT_SHA)
    assert report["status"] == STATUS_CURRENT
    assert report["remote_source_current"] is True
    assert report["remote_consumption_authorized"] is False
    assert "READ_ONLY_AUDIT_NEVER_AUTHORIZES_CONSUMPTION" in report["authorization_note"]
    assert report["remote_preparation_source_commit_sha"] == CURRENT_SHA
    assert report["file_count"] == 4
    assert len(report["preparation_receipt_sha256"]) == 64


def test_remote_snapshot_audit_marks_stale_source_fail_closed(tmp_path: Path):
    report = audit_snapshot(_snapshot(tmp_path, STALE_SHA), CURRENT_SHA)
    assert report["status"] == STATUS_STALE
    assert report["remote_source_current"] is False
    assert report["remote_consumption_authorized"] is False
    assert report["remote_preparation_source_commit_sha"] == STALE_SHA


def test_remote_snapshot_audit_marks_missing_required_files_incomplete(tmp_path: Path):
    _write_json(tmp_path / "prepared/PREPARATION_RECEIPT.json", {"source_commit_sha": STALE_SHA, "source_commit_bound": True})
    report = audit_snapshot(tmp_path, CURRENT_SHA)
    assert report["status"] == STATUS_INCOMPLETE
    assert report["remote_source_current"] is False
    assert report["remote_consumption_authorized"] is False
    assert set(report["required_missing"]) == {"teacher_batch_receipt", "cohort_manifest", "hydration_receipt"}


def test_retired_v1_self_hosted_workflow_cannot_queue_or_write():
    text = (ROOT / ".github/workflows/iris_vnext_fit8_prepare_self_hosted_v1.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "VNEXT_SELF_HOSTED_PREPARE_V1_RETIRED" in text
    assert "rclone " not in text


def test_quarantine_audit_v2_is_manual_read_only_and_current_branch_bound():
    text = (ROOT / ".github/workflows/iris_vnext_fit8_quarantine_audit_self_hosted_v2.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "research/iris-vnext-post-v8-quality-plan-20261018" in text
    assert "QUARANTINED_REMOTE: 'gdrive:RealSaS_Fit8_IRIS_vNext_20261016'" in text
    assert 'rclone copy "$QUARANTINED_REMOTE/teachers" "$AUDIT_ROOT/teachers"' in text
    assert 'rclone copy "$QUARANTINED_REMOTE/prepared" "$AUDIT_ROOT/prepared"' in text
    assert 'rclone copy "$QUARANTINED_REMOTE/receipts/HYDRATION_RECEIPT.json" "$AUDIT_ROOT/receipts"' in text
    assert '"$AUDIT_ROOT/teachers" "$QUARANTINED_REMOTE' not in text
    assert '"$AUDIT_ROOT/prepared" "$QUARANTINED_REMOTE' not in text
    assert "rclone sync" not in text
    assert "--fail-unless-current" in text


def test_candidate_workflow_is_manual_exact_sha_scoped_and_never_writes_quarantined_remote():
    text = (ROOT / ".github/workflows/iris_vnext_fit8_pretraining_candidate_self_hosted_v1.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "gdrive:RealSaS_Fit8_IRIS_vNext_candidates/${{ github.sha }}" in text
    assert "gdrive:RealSaS_Fit8_IRIS_vNext_20261016" in text
    assert "actions/download-artifact@v4" in text
    assert "quarantine_audit_run_id" in text
    assert "VNEXT_CANDIDATE_REMOTE_ALREADY_NONEMPTY__NO_OVERWRITE" in text
    assert 'rclone copy "$RUN_ROOT/candidate" "$CANDIDATE_REMOTE" --checksum' in text
    assert 'rclone copy "$RUN_ROOT/roundtrip" "$QUARANTINED_REMOTE"' not in text
    assert 'rclone copy "$RUN_ROOT/candidate" "$QUARANTINED_REMOTE"' not in text
    assert "run_fit8_vnext_scientific_v1.py" not in text
    assert "training_authorized=false" in text


def test_candidate_roundtrip_requires_exact_files_and_sha_scoped_remote(tmp_path: Path):
    local = _snapshot(tmp_path / "local", CURRENT_SHA)
    _write_json(local / "CANDIDATE.json", {"schema": "RealSaS.IRISVNextFit8PretrainingCandidate.v1"})
    remote = tmp_path / "remote"
    for path in sorted(p for p in local.rglob("*") if p.is_file()):
        dst = remote / path.relative_to(local)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
    report = verify_candidate_roundtrip(
        local,
        remote,
        CURRENT_SHA,
        "gdrive:RealSaS_Fit8_IRIS_vNext_candidates/" + CURRENT_SHA,
    )
    assert report["status"] == "ROUNDTRIP_VERIFIED_CANDIDATE"
    assert report["roundtrip_sha256_equal"] is True
    assert report["candidate_authorized_for_training"] is False
    assert report["canonical_remote_promoted"] is False

    with pytest.raises(ValueError, match="REMOTE_MUST_BE_EXACT_SHA_SCOPED"):
        verify_candidate_roundtrip(
            local,
            remote,
            CURRENT_SHA,
            "gdrive:RealSaS_Fit8_IRIS_vNext_20261016",
        )


def test_boundary_causality_static_builder_uses_only_source_defensible_edge_causes():
    # Two triangles form a square. The shared diagonal changes authored material;
    # the four perimeter edges are geometric terminations.
    faces = np.asarray([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
    owners = np.asarray([0, 0], dtype=np.int64)
    materials = np.asarray([2, 7], dtype=np.int64)
    rows, counts = _static_rows(faces, owners, materials)
    assert len(rows) == 5
    assert counts["GEOMETRIC_TERMINATION"] == 4
    assert counts["MATERIAL_BOUNDARY"] == 1
    assert counts["SOURCE_PRIMITIVE_COMPONENT_SEAM"] == 0
    shared = next(row for row in rows if row["edge_vertex_ids"] == [1, 2])
    assert shared["positive_causes"] == ["MATERIAL_BOUNDARY"]
    assert shared["abstained"] is False


def _relation_manifest_with_support() -> dict:
    zero = {"total": 2, "valid": 0, "abstained": 2, "positive": 0, "negative": 0, "positive_fraction": 0.0}
    balanced = {"total": 2, "valid": 2, "abstained": 0, "positive": 1, "negative": 1, "positive_fraction": 0.5}
    positive_only = {"total": 2, "valid": 2, "abstained": 0, "positive": 2, "negative": 0, "positive_fraction": 1.0}
    return {
        "schema": "RealSaS.IRISVNextFit8RelationTruthManifest.v2",
        "subjects": [{
            "subject_id": "Synthetic",
            "views": [{
                "static": {
                    "same_drawable_affinity": dict(zero),
                    "cut_probability": dict(zero),
                    "attachment_affinity": dict(zero),
                },
                "view": {
                    "front_probability": dict(balanced),
                    "occlusion_probability": dict(positive_only),
                    "visibility_probability_a": dict(balanced),
                    "visibility_probability_b": dict(balanced),
                    "projected_alpha_boundary_probability": dict(balanced),
                    "projected_appearance_boundary_probability": dict(balanced),
                },
            }],
        }],
        "manifest_value_sha256": "1" * 64,
    }


def test_v8_relation_migration_source_contract_is_exact_completed_authority():
    assert V8_SOURCE_SHA == "5083c9915533ec8ec6530c7d229c635887abfea7"
    assert V8_BINDING_SHA256 == "735b37f450e947eaad5928f39fca2fb92c61b4642f71044d5b19da5877a53ec2"
    assert V8_CHECKPOINT_FILE == "step_00012000_ac23a9ea448e465cb668773a131981be.pt"
    assert V8_CHECKPOINT_SHA256 == "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"


def test_materialization_seal_and_readiness_stop_before_training_without_preregistration(tmp_path: Path):
    import hashlib

    source = CURRENT_SHA
    candidate_path = tmp_path / "CANDIDATE_ROUNDTRIP_RECEIPT.json"
    relation_path = tmp_path / "RELATION_TRUTH_V2_MANIFEST.json"
    boundary_path = tmp_path / "BOUNDARY_CAUSALITY_V1_MANIFEST.json"
    audit_path = tmp_path / "QUARANTINE_AUDIT.json"
    migration_path = tmp_path / "V8_RELATION_HEAD_TO_V2_MIGRATION_RECEIPT.json"

    relation = _relation_manifest_with_support()
    _write_json(relation_path, relation)
    relation_sha = hashlib.sha256(relation_path.read_bytes()).hexdigest()
    _write_json(
        candidate_path,
        {
            "schema": "RealSaS.IRISVNextFit8CandidateRoundTripReceipt.v1",
            "status": "ROUNDTRIP_VERIFIED_CANDIDATE",
            "expected_source_sha": source,
            "candidate_remote": "gdrive:RealSaS_Fit8_IRIS_vNext_candidates/" + source,
            "roundtrip_sha256_equal": True,
            "candidate_authorized_for_training": False,
        },
    )
    _write_json(
        boundary_path,
        {
            "schema": "RealSaS.IRISVNextFit8BoundaryCausalityManifest.v1",
            "subjects": [{} for _ in range(8)],
            "relation_truth_v2_manifest_sha256": relation_sha,
            "manifest_value_sha256": "2" * 64,
        },
    )
    relation["subjects"] = relation["subjects"] * 8
    _write_json(relation_path, relation)
    relation_sha = hashlib.sha256(relation_path.read_bytes()).hexdigest()
    boundary = json.loads(boundary_path.read_text())
    boundary["relation_truth_v2_manifest_sha256"] = relation_sha
    _write_json(boundary_path, boundary)

    _write_json(
        audit_path,
        {
            "schema": "RealSaS.IRISVNextFit8RemotePreparationAudit.v1",
            "expected_current_source_sha": source,
            "remote_consumption_authorized": False,
            "status": "STALE_SOURCE_REMOTE",
        },
    )
    _write_json(
        migration_path,
        {
            "schema": "RealSaS.IRISVNextV8RelationHeadToV2ArtifactReceipt.v1",
            "status": "PASS",
            "training_steps_performed": 0,
            "source_v8": {
                "source_commit_sha": V8_SOURCE_SHA,
                "checkpoint_sha256": V8_CHECKPOINT_SHA256,
            },
            "target": {
                "sha256": "3" * 64,
                "new_projected_boundaries_camera_conditioned": True,
            },
        },
    )

    materialized = seal_materialization(
        expected_source_sha=source,
        candidate_roundtrip_receipt=candidate_path,
        relation_manifest=relation_path,
        boundary_manifest=boundary_path,
        quarantine_audit_report=audit_path,
        migration_receipt=migration_path,
    )
    assert materialized["truth_materialized"] is True
    assert materialized["training_authorized"] is False
    materialization_path = tmp_path / "PRETRAINING_MATERIALIZATION_RECEIPT.json"
    _write_json(materialization_path, materialized)

    readiness = audit_readiness(
        expected_source_sha=source,
        materialization_receipt_path=materialization_path,
        relation_manifest_path=relation_path,
        migration_receipt_path=migration_path,
        preregistration_path=None,
    )
    assert readiness["status"] == "BLOCKED_BEFORE_TRAINING"
    assert readiness["training_entry_authorized"] is False
    assert readiness["training_steps_executed"] == 0
    assert "ARTIFACT_BOUND_PREREGISTRATION_REQUIRED" in readiness["blockers"]
    assert readiness["relation_support"]["front_probability"]["binary_court_eligible"] is True
    assert readiness["relation_support"]["occlusion_probability"]["binary_court_eligible"] is False
