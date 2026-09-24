from pathlib import Path
import json

from compiler.realsas_compiler_services.orchestrator import mainline

ROOT=Path(__file__).resolve().parents[2]

def test_demo_authority_is_non_product_and_stage13_fail():
    payload=json.loads((ROOT/"canonical/KNIGHT_DEMO_EXECUTION_AUTHORITY_20260924.json").read_text())
    assert payload["schema"]=="RealSaS.DemoExecutionAuthority.v1"
    assert payload["status"]=="APPROVED_DEMO_ONLY"
    assert payload["stage13_scientific_pass"] is False
    assert payload["product_authority_claimed"] is False
    assert payload["product_pass_forbidden"] is True

def test_demo_pass_status_is_dependency_pass_only():
    assert "PASS_DEMO_ONLY" in mainline.PASS_STATUSES
    assert "FAIL" not in mainline.PASS_STATUSES

def test_spatial_risk_run_closed_to_fallback():
    payload=json.loads((ROOT/"canonical/IRIS_V5_C_SPATIAL_RISK_REPLAY_DEMO_CLOSURE_20260924.json").read_text())
    assert payload["status"].startswith("CLOSED__SCIENTIFIC_STAGE13_FAIL")
    assert payload["demo_choice"]=="FROZEN_C_FALLBACK"
    assert payload["product_authority_claimed"] is False
    assert payload["p999_pass_claimed"] is False
