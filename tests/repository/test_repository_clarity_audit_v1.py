"""Guard the source-derived machine inventory; no network or trained model required."""
from tools.repository_clarity_audit_v1 import (
    REPO, census, static_model_imports, workflow_events,
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
