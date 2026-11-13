"""Guard the source-derived machine inventory; no network or trained model required."""
from tools.repository_clarity_audit_v1 import (
    REPO, census, product_readiness, static_model_imports,
    static_return_statuses, workflow_events,
)


def test_workflow_events_are_not_all_product_authority():
    source = ("name: example\non:\n  push:\n    branches: [main]\n"
              "  pull_request:\n  workflow_dispatch:\njobs:\n  check:\n"
              "    runs-on: ubuntu-latest\n")
    assert workflow_events(source) == ["pull_request", "push", "workflow_dispatch"]
    assert workflow_events("name: example\non: [push, workflow_dispatch]\njobs: {}\n") == [
        "push", "workflow_dispatch",
    ]
    assert workflow_events("name: example\njobs: {}\n") == []


def test_legacy_model_imports_are_real_dependencies():
    source = ("from models.arachne.v5 import candidate\n"
              "import models.geppetto.reference_strength_v1 as old_rig\n"
              "from models.axis import current\n")
    assert static_model_imports(source) == {
        "models.arachne.v5", "models.geppetto.reference_strength_v1",
    }


def test_live_catalog_and_ci_policy_are_read_not_duplicated():
    result = census(REPO)
    assert result["unclassified_directories"] == []
    assert result["directory_dispositions"]["canonical"] == (
        "MIXED_HISTORIC_EVIDENCE_AND_PATH_BOUND_MACHINE_POLICIES"
    )
    assert result["directory_dispositions"]["models"] == (
        "PRODUCT_EXPERTS_AND_LEGACY_COMPATIBILITY"
    )
    assert "product.input" in result["active_catalog_roles"]
    assert result["active_product_role_count"] >= 5
    workflows = result["workflows"]
    assert workflows["missing_named_workflows"] == []
    assert workflows["named_fast_count"] == 10
    assert workflows["named_full_count"] == 16
    assert workflows["yaml_count"] >= 26


def test_proof_static_status_extraction_does_not_accept_comments_or_prose():
    assert static_return_statuses(
        'def execute_proof_capability_v1():\n'
        '    # "status": "PASS" is not an execution path\n'
        '    return {"status": "FAIL"}\n',
        "execute_proof_capability_v1",
    ) == ["FAIL"]
    assert static_return_statuses(
        'def execute_proof_capability_v1():\n'
        '    return {"status": decision()}\n',
        "execute_proof_capability_v1",
    ) == []


def test_product_readiness_audit_never_promotes_source_inspection_to_pass():
    result = census(REPO)["product_readiness"]
    assert result["schema"] == "RealSaS.ProductReadinessSourceAudit.v1"
    assert result["scope"] == "SOURCE_STATIC_ONLY_NOT_PRODUCT_QUALIFICATION"
    assert result["readiness_claim"] != "PASS"
    assert result["guarantees"]["static_absence_of_blockers_is_not_deploy_pass"]
    assert result["product_capability_count"] == 9
    issues = {row["id"]: row for row in result["checks"]}
    # These are actual blockers at this source revision, not model failures.
    assert "PRODUCT_CAPABILITIES_DEVELOPER_ONLY" not in issues
    assert "PROOF_HAS_NO_PASS_EXECUTION_PATH" in issues
    assert "PROOF_INDEPENDENT_WITNESSES_MISSING" in issues
    assert "PUBLIC_COMPILE_GATEWAY_DISABLED" in issues
    assert "PRODUCT_HTTP_CONSUMERS_UNBOUND" in issues
    assert "/api/compile/start" in issues[
        "PRODUCT_HTTP_CONSUMERS_UNBOUND"
    ]["source_evidence"]["unhandled_frontend_routes"]
    assert "/api/studio/requalify" in issues[
        "PRODUCT_HTTP_CONSUMERS_UNBOUND"
    ]["source_evidence"]["unhandled_frontend_routes"]
    assert "TWO_VIEW_CAMERA_POLICY_UNBOUND" in issues
    assert "TWO_VIEW_TO_RUNTIME_VIEW_BRIDGE_UNVERIFIED" in issues
    assert result["training_and_unseen"] == "EXTERNAL_NOT_EVALUATED"
