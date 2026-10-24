from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tools.ops.inspect_knight_skin_readiness_v1 as inspector


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/knight_skin_readiness_inspector.yml"
TRIGGER = ROOT / "canonical/KNIGHT_SKIN_READINESS_INSPECTOR_TRIGGER_V1.json"
SOURCE = ROOT / "tools/ops/inspect_knight_skin_readiness_v1.py"


def _record(role: str, value):
    return {
        "role": role,
        "artifact_id": f"00000000-0000-4000-8000-{len(role):012d}",
        "artifact_type": {
            "mesh": "RealSaS.CanonicalMeshIR",
            "rig": "RealSaS.CanonicalRigIR",
            "surface": "RealSaS.RiggingSurfaceIR",
            "support": "RealSaS.TESSAMaterialSupportFieldIR",
        }[role],
        "schema_version": "v1",
        "content_sha256": role[0] * 64,
        "semantic_sha256": role[-1] * 64,
        "size_bytes": 100,
        "producer_contract": "test",
        "implementation_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "source_kind": "ATTEMPT",
        "source_id": "11111111-1111-4111-8111-111111111111",
        "source_role": role,
        "source_origin": "produced",
        "value": value,
    }


def _compatible_rows():
    support_binding = SimpleNamespace(coefficients=(("proposal:v0", 1.0),))
    vertex = SimpleNamespace(support_binding=support_binding)
    mesh = SimpleNamespace(
        mesh_lineage_hash="mesh-lineage",
        source_evidence_binding_hash="surface-lineage",
        partition_binding_hash="partition-lineage",
        vertices=(vertex,),
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
        "mesh": [_record("mesh", mesh)],
        "rig": [_record("rig", rig)],
        "surface": [_record("surface", surface)],
        "support": [_record("support", support)],
    }


def test_compatible_bundle_requires_all_exact_bindings():
    rows = _compatible_rows()
    selected = inspector._select_bundle(rows)
    assert selected is not None
    assert selected["binding_status"] == "PASS_EXACT_TYPED_SKIN_INPUT_BINDINGS"
    assert selected["reuse_eligibility_claimed"] is False
    assert selected["mesh_lineage_hash"] == "mesh-lineage"
    assert selected["rig_lineage_hash"] == "rig-lineage"

    rows = _compatible_rows()
    rows["rig"][0]["value"].mesh_lineage_hash = "other-mesh"
    assert inspector._select_bundle(rows) is None

    rows = _compatible_rows()
    rows["support"][0]["value"].surface_binding_hash = "other-surface"
    assert inspector._select_bundle(rows) is None

    rows = _compatible_rows()
    rows["mesh"][0]["value"].vertices[0].support_binding.coefficients = (("missing-proposal", 1.0),)
    assert inspector._select_bundle(rows) is None


def test_exact_draft_mira_is_runtime_ready_but_not_research_admitted(monkeypatch):
    release = {
        "id": "25417bd2-863a-405e-ada5-ba9b6a9b364b",
        "family": "MIRA",
        "name": "MIRA_V5_5_RUNTIME_RECOVERY",
        "architecture_id": inspector.MIRA_ARCHITECTURE,
        "checkpoint_artifact_id": "07dbdba8-2bf7-586a-a123-30ab427341af",
        "checkpoint_content_sha256": inspector.MIRA_RUNTIME_SHA256,
        "lifecycle_state": "DRAFT",
        "evaluations": [{"scope": "RUNTIME_PACKAGE_IDENTITY_V1", "result": "PASS"}],
    }

    def fake_get(_api_root, path):
        if path == "/v1/model-releases":
            return [release]
        if path == "/v1/model-channels":
            return []
        raise AssertionError(path)

    monkeypatch.setattr(inspector, "_get_json", fake_get)
    state = inspector._mira_state("http://example.invalid")
    assert state["runtime_identity_ready"] is True
    assert state["lifecycle_state"] == "DRAFT"
    assert state["research_execution_admitted"] is False
    assert state["draft_evaluation_required"] is True


def test_exact_research_channel_changes_admission_not_runtime_identity(monkeypatch):
    release_id = "25417bd2-863a-405e-ada5-ba9b6a9b364b"
    release = {
        "id": release_id,
        "family": "MIRA",
        "name": "MIRA_V5_5_RUNTIME_RECOVERY",
        "architecture_id": inspector.MIRA_ARCHITECTURE,
        "checkpoint_artifact_id": "07dbdba8-2bf7-586a-a123-30ab427341af",
        "checkpoint_content_sha256": inspector.MIRA_RUNTIME_SHA256,
        "lifecycle_state": "RESEARCH_QUALIFIED",
        "evaluations": [],
    }

    def fake_get(_api_root, path):
        if path == "/v1/model-releases":
            return [release]
        if path == "/v1/model-channels":
            return [{"family": "MIRA", "channel": "RESEARCH", "model_release_id": release_id}]
        raise AssertionError(path)

    monkeypatch.setattr(inspector, "_get_json", fake_get)
    state = inspector._mira_state("http://example.invalid")
    assert state["runtime_identity_ready"] is True
    assert state["research_execution_admitted"] is True
    assert state["draft_evaluation_required"] is False
    assert state["bound_channels"] == ["RESEARCH"]


def test_inspector_source_is_read_only_and_fail_closed():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'method="GET"' in text
    assert 'method="POST"' not in text
    assert 'method="PUT"' not in text
    assert "/v1/artifacts/import" not in text
    assert "/promote" not in text
    assert "model_inference_executed\": False" in text
    assert "platform_mutation_performed\": False" in text
    assert "product_skin_activated\": False" in text
    assert "reuse_eligibility_claimed\": False" in text
    assert "READY_FOR_DRAFT_MODEL_EVALUATION" in text
    assert "RUN_DRAFT_MIRA_CURRENT_SKIN_EVALUATION__DO_NOT_ACTIVATE_PRODUCT_SKIN" in text


def test_workflow_only_reaches_physical_host_after_main_deployment_or_manual_dispatch():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "workflow_run:" in text
    assert "workflows: [platform-developer-deployment]" in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "pull_request:" not in text
    assert "WORKFLOW_RUN_CONCLUSION" in text
    assert "WORKFLOW_RUN_HEAD_BRANCH" in text
    assert "git ls-remote origin refs/heads/main" in text
    assert "deployment['code_sha'] == os.environ['EXPECTED_SHA']" in text
    assert "Verify inspection receipt and authority boundaries" in text
    assert "platform_mutation_performed" in text
    assert "product_skin_activated" in text


def test_trigger_is_explicit_read_only_authority():
    trigger = json.loads(TRIGGER.read_text(encoding="utf-8"))
    assert trigger["schema"] == "RealSaS.KnightSkinReadinessInspectorTrigger.v1"
    assert trigger["operation"] == "KNIGHT_SKIN_READINESS_INSPECT"
    assert trigger["enabled"] is True
    assert trigger["authority"] == "READ_ONLY_DIAGNOSTIC__NO_QUALIFICATION_NO_MODEL_PROMOTION_NO_PRODUCT_SKIN_ACTIVATION"
