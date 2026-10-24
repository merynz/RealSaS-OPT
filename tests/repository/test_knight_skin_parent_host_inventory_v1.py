from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tools.ops.inspect_knight_skin_parent_host_inventory_v1 as inventory

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/ops/inspect_knight_skin_parent_host_inventory_v1.py"
WORKFLOW = ROOT / ".github/workflows/knight_skin_parent_host_inventory.yml"
PARENT_EXECUTION_WORKFLOW = ROOT / ".github/workflows/knight_current_skin_parent_execution.yml"
TRIGGER = ROOT / "canonical/KNIGHT_SKIN_PARENT_HOST_INVENTORY_TRIGGER_V1.json"


def _row(role: str, value):
    return {
        "role": role,
        "schema": inventory.TARGETS[role],
        "root_role": "test",
        "run_name": "sealed",
        "path": f"/{role}.json",
        "content_sha256": role[0] * 64,
        "size_bytes": 10,
        "value": value,
    }


def _compatible():
    binding = SimpleNamespace(coefficients=(("proposal:v0", 1.0),))
    mesh = SimpleNamespace(
        mesh_lineage_hash="mesh-lineage",
        source_evidence_binding_hash="surface-lineage",
        partition_binding_hash="partition-lineage",
        vertices=(SimpleNamespace(support_binding=binding),),
    )
    rig = SimpleNamespace(
        mesh_lineage_hash="mesh-lineage",
        rig_lineage_hash="rig-lineage",
        feature_evidence_binding_hash="surface-lineage",
    )
    surface = SimpleNamespace(geometry_lineage_hash="surface-lineage")
    support = SimpleNamespace(
        rows=(SimpleNamespace(proposal_vertex_id="proposal:v0"),),
        surface_binding_hash="surface-lineage",
        partition_binding_hash="partition-lineage",
        field_lineage_hash="support-lineage",
    )
    return {
        "mesh": [_row("mesh", mesh)],
        "rig": [_row("rig", rig)],
        "surface": [_row("surface", surface)],
        "support": [_row("support", support)],
    }


def test_exact_bundle_requires_cross_contract_bindings():
    rows = _compatible()
    selected = inventory._select_bundle(rows)
    assert selected is not None
    assert selected["binding_status"] == "PASS_EXACT_TYPED_SKIN_INPUT_BINDINGS"
    assert selected["mesh_lineage_hash"] == "mesh-lineage"
    assert selected["rig_lineage_hash"] == "rig-lineage"

    rows = _compatible()
    rows["rig"][0]["value"].mesh_lineage_hash = "other"
    assert inventory._select_bundle(rows) is None

    rows = _compatible()
    rows["support"][0]["value"].partition_binding_hash = "other"
    assert inventory._select_bundle(rows) is None


def test_source_is_read_only_bounded_and_current_root_scoped():
    text = SOURCE.read_text(encoding="utf-8")
    assert "PRODUCT_NEAREST_KNIGHT_WITNESS_V1.json" in text
    assert "current_host_lineage_names" in text
    assert "MAX_JSON_FILES" in text
    assert "MAX_JSON_BYTES" in text
    assert 'rglob("*.json")' in text
    assert '"artifact_import_performed": False' in text
    assert '"model_inference_executed": False' in text
    assert '"compiler_capability_executed": False' in text
    assert '"qualification_minted": False' in text
    assert "KNIGHT_PARENT_HOST_INVENTORY_PATH_ESCAPE" in text
    assert "/v1/artifacts/import" not in text
    assert "subprocess" not in text


def test_workflow_is_physical_read_only_post_deploy_probe():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflows: [platform-developer-deployment]" in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "KNIGHT_SKIN_PARENT_HOST_INVENTORY_TRIGGER_V1.json" in text
    assert "platform_mutation_performed" not in text
    assert "artifact_import_performed" in text
    assert "product_skin_activated" in text
    assert "git ls-remote origin refs/heads/main" in text


def test_parent_execution_failure_diagnostics_are_managed_and_credential_free():
    text = PARENT_EXECUTION_WORKFLOW.read_text(encoding="utf-8")
    assert "Capture managed Platform diagnostics on parent failure" in text
    assert "if: failure()" in text
    assert "from tools.platform_deploy import startup_diagnostics" in text
    assert "PLATFORM_DIAGNOSTICS.txt" in text
    assert "printenv" not in text
    assert "env |" not in text
    assert "systemctl --user show-environment" not in text
    assert text.index("Execute current Mesh and Rig responsibility closure") < text.index(
        "Capture managed Platform diagnostics on parent failure"
    ) < text.index("Verify current typed parent execution receipt")


def test_trigger_is_explicit_read_only_diagnostic():
    import json

    trigger = json.loads(TRIGGER.read_text(encoding="utf-8"))
    assert trigger["schema"] == "RealSaS.KnightSkinParentHostInventoryTrigger.v1"
    assert trigger["operation"] == "KNIGHT_SKIN_PARENT_HOST_INVENTORY"
    assert trigger["enabled"] is True
    assert "READ_ONLY" in trigger["authority"]
