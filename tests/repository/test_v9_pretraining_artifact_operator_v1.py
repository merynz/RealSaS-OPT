from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/platform_cohort_operator.yml"
OPERATOR = ROOT / "tools/iris_vnext/v9_pretraining_artifact_operator_v1.py"
REQUEST = ROOT / "canonical/V9_PRETRAINING_ARTIFACT_REQUEST_V1.json"
TRIGGER = ROOT / "canonical/RESEARCH_OPERATOR_TRIGGER_V1.json"
TARGET = "7ce4c9f30c7dfe3b67132970cc36a21ef74ed0ff"


def test_v9_request_is_exact_source_and_pretraining_only():
    value = json.loads(REQUEST.read_text(encoding="utf-8"))
    assert value["schema"] == "RealSaS.IRISVNextV9PretrainingArtifactRequest.v1"
    assert value["operation"] == "IRIS_VNEXT_V9_PRETRAINING_ARTIFACT_BUILD"
    assert value["target_branch"] == "research/iris-vnext-post-v8-quality-plan-20261018"
    assert value["target_source_sha"] == TARGET
    assert value["training_allowed"] is False
    assert value["external_storage_write_allowed"] is False
    assert value["claims"] == {
        "scientific_fit_started": False,
        "scientific_fit_pass": False,
        "training_authorized": False,
        "remote_roundtrip_verified": False,
    }
    assert len(value["inputs"]["legacy"]["shards"]) == 8
    assert value["inputs"]["v8"]["checkpoint"]["sha256"] == (
        "0be15df5614a05343cf3850aff747ba4d31f00a322f058819cb5a54d5a5d2fc5"
    )


def test_research_trigger_is_typed_and_cannot_train():
    value = json.loads(TRIGGER.read_text(encoding="utf-8"))
    assert value["schema"] == "RealSaS.ResearchOperatorTrigger.v1"
    assert value["enabled"] is True
    assert value["request"] == "canonical/V9_PRETRAINING_ARTIFACT_REQUEST_V1.json"
    assert value["operation"] == "IRIS_VNEXT_V9_PRETRAINING_ARTIFACT_BUILD"
    assert value["target_source_sha"] == TARGET
    assert value["training_allowed"] is False


def test_operator_never_crosses_remote_roundtrip_or_training_boundary():
    text = OPERATOR.read_text(encoding="utf-8")
    assert "QUARANTINED_REMOTE" in text
    assert "external_storage_write_allowed" in text
    assert "LOCAL_CANDIDATE_BUILT__REMOTE_ROUNDTRIP_PENDING" in text
    assert "remote_roundtrip_verified\": False" in text
    assert "final_materialization_receipt_minted\": False" in text
    assert "training_steps_executed\": 0" in text
    assert "scientific_fit_started\": False" in text
    # The read-only builder may describe the training boundary in prose and may
    # consume the zero-step migration tool, but executable training machinery is
    # forbidden from this operator.
    assert "seal_fit8_pretraining_materialization_v1.py" not in text
    assert "verify_fit8_candidate_roundtrip_v1.py" not in text
    assert "IRISVNextFit8Trainer" not in text
    assert "build_specialist_optimizer_v2" not in text
    assert "torch.optim" not in text
    assert "optimizer.step" not in text.lower()
    assert "rclone copyto" not in text.lower()
    assert "rclone sync" not in text.lower()


def test_registered_main_operator_has_automatic_typed_research_lane():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_run:" in text
    assert "workflows: [platform-developer-deployment]" in text
    assert "research-activation:" in text
    assert "v9-pretraining-artifact:" in text
    assert "canonical/RESEARCH_OPERATOR_TRIGGER_V1.json" in text
    assert "IRIS_VNEXT_V9_PRETRAINING_ARTIFACT_BUILD" in text
    assert "runs-on: [self-hosted, linux, x64, realsas]" in text
    assert "actions/upload-artifact@v4" in text
    assert "compression-level: 0" in text
    assert "external_storage_write_allowed') is False" in text
    # No candidate-remote persistence belongs to this workflow; persistence and
    # true round-trip are performed later by the connector-owned authority step.
    assert "RealSaS_Fit8_IRIS_vNext_candidates" not in text
