import json
from pathlib import Path

import pytest

from compiler.realsas_compiler_services.orchestrator.adapters.oracle_z_external_v2 import consume_oracle_z_report
from tools.platform_oracle_z_v2 import STAGE_ID, oracle_plan, stage_parameters
from tools.platform_release_snapshot import snapshot


def _ctx(tmp_path: Path, *, experiment="exp-1", code_sha="a" * 40):
    report = {
        "schema": "RealSaS.OracleZRepresentationCourt.v2",
        "mode": "PLATFORM_SCIENTIFIC",
        "status": "VALID_REPRESENTATION_CAPACITY_CURVES_V2",
        "platform_context": {
            "REALSAS_EXPERIMENT_ID": experiment,
            "REALSAS_ATTEMPT_ID": "attempt-1",
            "REALSAS_ENGINE_RELEASE_ID": "release-1",
            "REALSAS_CANONICAL_CODE_SHA": code_sha,
        },
    }
    path = tmp_path / "bound-report.json"
    path.write_text(json.dumps(report, sort_keys=True))
    import hashlib
    return {
        "run_root": tmp_path / "run",
        "run_manifest": {
            "oracle_z_spec": {
                "schema": "RealSaS.OracleZExecutionSpec.v2",
                "experiment_id": experiment,
                "canonical_code_sha": code_sha,
            },
            "_platform_external_inputs": {
                "oracle_z_report": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            },
        },
    }


def test_oracle_z_external_consumer_preserves_report_and_never_infers_scientific_pass(tmp_path):
    result = consume_oracle_z_report(_ctx(tmp_path))
    assert result["status"] == "PASS"
    assert result["diagnostics"]["scientific_verdict_inferred"] is False
    receipt = json.loads((tmp_path / "run/artifacts" / STAGE_ID / "consumption_receipt.json").read_text())
    assert receipt["report_status"] == "VALID_REPRESENTATION_CAPACITY_CURVES_V2"
    assert receipt["scientific_verdict_inferred"] is False
    assert receipt["product_authority_changed"] is False


def test_oracle_z_external_consumer_rejects_platform_context_drift(tmp_path):
    ctx = _ctx(tmp_path)
    report_path = Path(ctx["run_manifest"]["_platform_external_inputs"]["oracle_z_report"]["path"])
    report = json.loads(report_path.read_text())
    report["platform_context"]["REALSAS_EXPERIMENT_ID"] = "wrong"
    report_path.write_text(json.dumps(report, sort_keys=True))
    import hashlib
    ctx["run_manifest"]["_platform_external_inputs"]["oracle_z_report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="ORACLE_Z_REPORT_PLATFORM_CONTEXT_DRIFT"):
        consume_oracle_z_report(ctx)


def test_oracle_z_plan_is_one_research_only_external_stage():
    plan = oracle_plan()
    assert plan["stage_count"] == 1
    stage = plan["stages"][0]
    assert stage["id"] == STAGE_ID
    assert stage["depends_on"] == []
    assert stage["policy"]["product_pass_authority"] is False
    assert stage["manifest_keys"] == ["oracle_z_spec"]
    params = stage_parameters()
    assert params["stages"][STAGE_ID]["external_evidence_roles"] == ["oracle_z_report"]


def test_oracle_z_research_release_is_reconstructable_and_external_role_is_release_owned():
    plan = oracle_plan()
    released = snapshot(
        plan,
        None,
        name="ORACLE_Z_REPRESENTATION_V2_TEST",
        purpose="RESEARCH",
        created_by="test",
        stage_parameters=stage_parameters(),
    )
    assert released["manifest"]["dag"]["stage_count"] == 1
    row = released["manifest"]["stages"][0]
    assert row["stage_id"] == STAGE_ID
    assert row["semantic_parameters"]["external_evidence_roles"] == ["oracle_z_report"]
    assert released["manifest"]["dag"]["stages"][0]["policy"]["product_pass_authority"] is False
