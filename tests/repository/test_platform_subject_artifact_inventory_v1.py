from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tools/ops/inspect_platform_subject_artifact_inventory_v1.py"
TRIGGER = ROOT / "canonical/PLATFORM_SUBJECT_ARTIFACT_INVENTORY_TRIGGER_V1.json"
WORKFLOW = ROOT / ".github/workflows/platform_subject_artifact_inventory.yml"


def _module():
    spec = importlib.util.spec_from_file_location("platform_subject_artifact_inventory_v1", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trigger_is_bounded_read_only_shape_inventory_request():
    payload = json.loads(TRIGGER.read_text(encoding="utf-8"))
    assert payload["schema"] == "RealSaS.PlatformSubjectArtifactInventoryTrigger.v1"
    assert payload["operation"] == "PLATFORM_SUBJECT_ARTIFACT_INVENTORY"
    assert payload["enabled"] is True
    assert payload["subject_slug"] == "knight"
    assert payload["queries"] == [
        {
            "name": "shape_proposal_v2",
            "artifact_type": "RealSaS.ShapeProposal",
            "schema_version": "v2",
            "limit": 25,
        }
    ]


def test_inspector_reports_exact_platform_rows_without_mutation(monkeypatch):
    module = _module()
    trigger = module.load_trigger(TRIGGER)

    def fake_get(_api_root: str, path: str):
        if path == "/v1/subjects/by-slug/knight":
            return {
                "schema": "RealSaS.SubjectIdentity.v1",
                "subject_id": "subject-knight",
                "slug": "knight",
                "display_name": "Knight",
            }
        if path.startswith("/v1/subjects/subject-knight/artifacts?"):
            return [
                {
                    "artifact_id": "artifact-shape-1",
                    "source_kind": "capability",
                    "source_id": "attempt-1",
                    "role": "capability:product.shape:shape_proposal",
                    "origin": "produced",
                }
            ]
        if path == "/v1/artifacts/artifact-shape-1":
            return {
                "artifact_type": "RealSaS.ShapeProposal",
                "schema_version": "v2",
                "semantic_sha256": "1" * 64,
                "content_sha256": "2" * 64,
                "size_bytes": 123,
                "producer_contract": "product.shape",
                "implementation_sha256": "3" * 64,
                "policy_sha256": "4" * 64,
            }
        raise AssertionError(path)

    monkeypatch.setattr(module, "_get_json", fake_get)
    receipt = module.inspect("http://example.invalid", trigger, "A" * 40)

    assert receipt["schema"] == "RealSaS.PlatformSubjectArtifactInventory.v1"
    assert receipt["status"] == "PASS_PLATFORM_SUBJECT_ARTIFACT_INVENTORY"
    assert receipt["source_code_sha"] == "a" * 40
    assert receipt["queries"]["shape_proposal_v2"]["count"] == 1
    row = receipt["queries"]["shape_proposal_v2"]["artifacts"][0]
    assert row["artifact_id"] == "artifact-shape-1"
    assert row["producer_contract"] == "product.shape"
    for key in (
        "platform_mutation_performed",
        "artifact_import_performed",
        "model_inference_executed",
        "compiler_capability_executed",
        "qualification_minted",
        "model_promotion_performed",
        "product_revision_minted",
        "genericity_claimed",
    ):
        assert receipt[key] is False


def test_workflow_runs_after_exact_main_deployment_and_preserves_receipt():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflows: [platform-developer-deployment]" in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "Require exact deployed canonical main" in text
    assert "inspect_platform_subject_artifact_inventory_v1.py" in text
    assert "platform-subject-artifact-inventory-${{ github.run_id }}" in text
