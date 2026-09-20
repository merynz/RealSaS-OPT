from __future__ import annotations

import json
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT = Path(__file__).resolve().parents[2]


def _plan():
    return json.loads(
        (ROOT / "canonical" / "MAINLINE_EXECUTION_PLAN_V2.json").read_text(
            encoding="utf-8"
        )
    )


def test_fresh_run_ledger_is_run_scoped_and_preserves_dag_ready_set():
    plan = _plan()
    ledger = mainline.build_fresh_run_ledger(
        plan,
        run_id="TEST_RUN",
        subject_id="TEST_SUBJECT",
        manifest_ref="/tmp/test/run_manifest.json",
        architecture_scope="TEST_V2_RUN",
    )
    assert ledger["run_id"] == "TEST_RUN"
    assert ledger["subject_id"] == "TEST_SUBJECT"
    assert ledger["architecture_scope"] == "TEST_V2_RUN"
    assert ledger["execution_class"] == "WITNESS"
    assert ledger["completed_count"] == 0
    assert ledger["failed_count"] == 0
    assert ledger["total_count"] == 46
    assert ledger["ready_stage_ids"] == [
        "01_SOURCE_BYTES_SEALED",
        "05_CAMERA_CONTRACT_SOLVED",
    ]
    assert all(row["status"] == "PENDING" for row in ledger["stages"])


def test_run_ledger_path_is_under_authority_run_root(monkeypatch, tmp_path):
    monkeypatch.setenv("REALSAS_AUTHORITY_ROOT", str(tmp_path))
    assert mainline.run_ledger_path("ABC") == (
        tmp_path / "runs" / "ABC" / "ACTIVE_RUN_V2.json"
    ).resolve()
    assert mainline.run_manifest_path("ABC") == (
        tmp_path / "runs" / "ABC" / "run_manifest.json"
    ).resolve()


def test_canonical_assembly_ledger_remains_governance_not_witness():
    ledger = json.loads(
        (ROOT / "canonical" / "ACTIVE_RUN_V2.json").read_text(encoding="utf-8")
    )
    assert ledger["run_id"] == "V2_IMPLEMENTATION_ASSEMBLY"
    assert ledger["subject_id"] == "NONE"
    assert ledger["execution_enabled"] is False


def test_witness_workflow_uses_current_cli_and_run_local_ledger_only():
    text = (
        ROOT / ".github" / "workflows" / "subject2_knight_observation_preflight.yml"
    ).read_text(encoding="utf-8")
    assert "--from-stage" not in text
    assert "--to-stage" not in text
    assert "mainline init-run" in text
    assert "--target 02_SOURCE_LICENSE_PROVENANCE" in text
    assert "--target 08_NORMALIZATION_DOMAIN_QUALIFIED" in text
    assert 'mainline status \\\n            --run-id "$RUN_ID"' in text
    assert 'cp "$RUN_ROOT/ACTIVE_RUN_V2.json" "$evidence/"' in text
    assert "push:" not in text.split("permissions:", 1)[0]


def test_readiness_is_reopened_until_witness_orchestration_proof_passes():
    readiness = json.loads(
        (ROOT / "canonical" / "V2_IMPLEMENTATION_READINESS.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["status"] == "IMPLEMENTATION_AUDIT_REOPENED__WITNESS_FORBIDDEN"
    assert (
        "WITNESS_ORCHESTRATION_AND_ARTIFACT_DRY_RUN"
        in readiness["required_proofs"]
    )
    proof = next(
        row
        for row in readiness["proofs"]
        if row["proof_id"] == "WITNESS_ORCHESTRATION_AND_ARTIFACT_DRY_RUN"
    )
    assert proof["status"] == "IN_PROGRESS"
    assert readiness["readiness_seal"]["status"] == "REVOKED_BY_AUDIT_GAP"
    assert readiness["readiness_seal"]["witness_execution_allowed"] is False


def test_implementation_audit_execution_class_is_explicit_and_subject_free():
    plan = _plan()
    ledger = mainline.build_fresh_run_ledger(
        plan,
        run_id="AUDIT",
        subject_id="SUBJECT_FREE_ORCHESTRATION_FIXTURE",
        manifest_ref="/tmp/audit/run_manifest.json",
        architecture_scope="REALSAS_V2_IMPLEMENTATION_DRY_RUN",
        execution_class="IMPLEMENTATION_AUDIT",
    )
    assert ledger["execution_class"] == "IMPLEMENTATION_AUDIT"
    assert ledger["subject_id"] == "SUBJECT_FREE_ORCHESTRATION_FIXTURE"


def test_implementation_audit_rejects_named_product_subject():
    import pytest

    with pytest.raises(
        RuntimeError,
        match="IMPLEMENTATION_AUDIT_REQUIRES_SUBJECT_FREE_SUBJECT_ID",
    ):
        mainline.build_fresh_run_ledger(
            _plan(),
            run_id="BAD_AUDIT",
            subject_id="SUBJECT2_KNIGHT",
            manifest_ref="/tmp/audit/run_manifest.json",
            architecture_scope="REALSAS_V2_IMPLEMENTATION_DRY_RUN",
            execution_class="IMPLEMENTATION_AUDIT",
        )
