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


def test_readiness_state_machine_is_consistent_with_orchestration_proof():
    readiness = json.loads(
        (ROOT / "canonical" / "V2_IMPLEMENTATION_READINESS.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        "WITNESS_ORCHESTRATION_AND_ARTIFACT_DRY_RUN"
        in readiness["required_proofs"]
    )
    proof = next(
        row
        for row in readiness["proofs"]
        if row["proof_id"] == "WITNESS_ORCHESTRATION_AND_ARTIFACT_DRY_RUN"
    )
    if proof["status"] == "IN_PROGRESS":
        assert (
            readiness["status"]
            == "IMPLEMENTATION_AUDIT_REOPENED__WITNESS_FORBIDDEN"
        )
        assert readiness["readiness_seal"]["status"] == "REVOKED_BY_AUDIT_GAP"
        assert readiness["readiness_seal"]["witness_execution_allowed"] is False
    elif proof["status"] == "PASS":
        assert readiness["status"] == "READY_FOR_WITNESS_EXECUTION"
        assert readiness["readiness_seal"]["status"] == "PASS"
        assert readiness["readiness_seal"]["witness_execution_allowed"] is True
        assert (
            readiness["implementation_closure_sha256"]
            == mainline.implementation_closure_sha256(_plan())
        )
    else:
        raise AssertionError(proof)


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


def test_stage_output_seal_is_confined_to_exact_stage_authority_root(tmp_path):
    import pytest

    allowed = tmp_path / "runs" / "R" / "artifacts" / "01_SOURCE_BYTES_SEALED"
    allowed.mkdir(parents=True)
    good = allowed / "good.json"
    good.write_text("{}\n", encoding="utf-8")
    sealed = mainline._seal_outputs(
        [{"path": str(good), "schema": "test"}],
        allowed_root=allowed,
    )
    assert sealed[0]["path"] == str(good.resolve())

    foreign = tmp_path / "runs" / "OTHER" / "artifacts" / "01_SOURCE_BYTES_SEALED"
    foreign.mkdir(parents=True)
    bad = foreign / "bad.json"
    bad.write_text("{}\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError,
        match="STAGE_OUTPUT_OUTSIDE_STAGE_AUTHORITY_ROOT",
    ):
        mainline._seal_outputs(
            [{"path": str(bad), "schema": "test"}],
            allowed_root=allowed,
        )


def test_run_manifest_identity_is_exact_for_run_and_subject():
    import pytest

    manifest = {"run_id": "R", "subject_id": "SUBJECT_FREE_TEST"}
    mainline._validate_run_manifest_identity(
        manifest,
        run_id="R",
        subject_id="SUBJECT_FREE_TEST",
    )
    with pytest.raises(RuntimeError, match="RUN_MANIFEST_RUN_ID_DRIFT"):
        mainline._validate_run_manifest_identity(
            manifest,
            run_id="OTHER",
            subject_id="SUBJECT_FREE_TEST",
        )
    with pytest.raises(RuntimeError, match="RUN_MANIFEST_SUBJECT_ID_DRIFT"):
        mainline._validate_run_manifest_identity(
            manifest,
            run_id="R",
            subject_id="OTHER",
        )


def test_implementation_closure_covers_all_current_stages_and_excludes_donor_era_modules():
    plan = _plan()
    manifest = mainline.implementation_closure_manifest(plan)
    assert manifest["schema"] == "RealSaS.V2ImplementationClosure.v1"
    assert len(manifest["adapter_implementation_closures"]) == 46
    imported = {
        row["module"]
        for stage in manifest["adapter_implementation_closures"]
        for row in stage["local_python_import_closure"]
    }
    assert imported.isdisjoint(mainline.CURRENT_V2_FORBIDDEN_IMPORT_MODULES)
    critical = {row["path"] for row in manifest["critical_files"]}
    assert "runtime/realsas_cpp/src/runtime_v2_caa_reference.cpp" in critical
    assert ".github/workflows/subject2_knight_observation_preflight.yml" in critical
    assert ".github/workflows/v2_witness_orchestration_subject_free_dry_run.yml" in critical


def test_implementation_closure_hash_changes_when_critical_bytes_change(monkeypatch):
    plan = _plan()
    baseline = mainline.implementation_closure_sha256(plan)
    original = mainline.sha256_file
    target = (
        ROOT / "runtime" / "realsas_cpp" / "src" / "runtime_v2_caa_reference.cpp"
    ).resolve()

    def fake_sha(path):
        resolved = Path(path).resolve()
        if resolved == target:
            return "0" * 64
        return original(Path(path))

    monkeypatch.setattr(mainline, "sha256_file", fake_sha)
    mutated = mainline.implementation_closure_sha256(plan)
    assert mutated != baseline


def test_engineer_facing_navigation_describes_current_v2_not_historical_repair_branch():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    repo_map = (ROOT / "REPOSITORY_MAP.md").read_text(encoding="utf-8")
    canonical_index = (ROOT / "canonical" / "README.md").read_text(encoding="utf-8")
    system_index = (ROOT / "SYSTEM_INDEX.md").read_text(encoding="utf-8")

    assert "Current product architecture — RealSaS V2" in readme
    assert "46-stage" in readme
    assert "repair/mage-full-subject-reclosure-20260912" not in readme
    assert "repair/mage-full-subject-reclosure-20260912" not in repo_map
    assert "Current executable lineage" in repo_map
    assert "V2_IMPLEMENTATION_READINESS.json" in canonical_index
    assert "appearance_authority_v2.py" in system_index
    assert "runtime_authority_v2.py" in system_index
