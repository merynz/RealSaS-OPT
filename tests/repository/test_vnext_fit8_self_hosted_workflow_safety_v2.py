from __future__ import annotations

import json
from pathlib import Path

from tools.iris_vnext.audit_fit8_preparation_snapshot_v1 import (
    STATUS_CURRENT,
    STATUS_INCOMPLETE,
    STATUS_STALE,
    audit_snapshot,
)

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
    assert "self-hosted" not in text
    assert "VNEXT_SELF_HOSTED_PREPARE_V1_RETIRED" in text
    assert "rclone" not in text


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
