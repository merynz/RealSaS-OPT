from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.truth_foundry.source_ray_admission_activation_v1 import REQUEST_PATH, select_admission

ROOT = Path(__file__).resolve().parents[2]


def _select(**overrides):
    request = json.loads((ROOT / REQUEST_PATH).read_text())
    args = dict(event="workflow_run", branch="main", conclusion="success", changed_paths=[REQUEST_PATH])
    args.update(overrides)
    return select_admission(request, **args)


def test_admission_runs_only_after_successful_requested_main_deployment():
    assert _select()[0] == "RUN"
    assert _select(event="workflow_dispatch", changed_paths=[])[0] == "RUN"


@pytest.mark.parametrize("overrides,reason", [
    ({"event": "push"}, "source_ci_only"),
    ({"event": "pull_request"}, "source_ci_only"),
    ({"branch": "review/other"}, "non_main_execution"),
    ({"event": "workflow_dispatch", "branch": "review/other"}, "non_main_execution"),
    ({"conclusion": "failure"}, "deployment_not_successful"),
    ({"conclusion": "cancelled"}, "deployment_not_successful"),
    ({"changed_paths": ["CURRENT_STATE.md"]}, "request_not_changed"),
])
def test_unrelated_or_unready_events_never_submit_admission(overrides, reason):
    assert _select(**overrides) == ("NOOP", reason)


def test_disabled_request_never_submits_even_on_manual_dispatch():
    request = json.loads((ROOT / REQUEST_PATH).read_text())
    request["enabled"] = False
    assert select_admission(request, event="workflow_dispatch", branch="main", conclusion="", changed_paths=[]) == ("NOOP", "request_disabled")


@pytest.mark.parametrize("key", ["training_authorized", "model_promotion_authorized", "product_revision_authorized", "canonical_mesh_binding_authorized"])
def test_request_cannot_expand_the_existing_court_authority(key):
    request = deepcopy(json.loads((ROOT / REQUEST_PATH).read_text()))
    request["claims"][key] = True
    with pytest.raises(ValueError, match="AUTHORITY_ESCALATION"):
        select_admission(request, event="workflow_dispatch", branch="main", conclusion="", changed_paths=[])


def test_workflow_preserves_deployment_and_per_run_receipt_boundaries():
    text = (ROOT / ".github/workflows/source_ray_appearance_platform_admission_fit8_v1.yml").read_text()
    assert "workflows: [platform-developer-deployment]" in text
    assert "needs.source-contract.outputs.decision == 'RUN'" in text
    assert 'test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$EXPECTED_SHA"' in text
    assert "SERVICES_ACTIVE" in text
    assert "${{ github.run_id }}-${{ github.run_attempt }}/receipt.json" in text
    assert "if-no-files-found: error" in text
    assert "for i in $(seq 1 120)" not in text


@pytest.mark.parametrize("recovery", [
    {"attempt_id": "invalid", "source_code_sha": "a" * 40},
    {"attempt_id": "783674b2-8283-498d-ad76-9ed1f09dc330", "source_code_sha": "invalid"},
    {"attempt_id": "783674b2-8283-498d-ad76-9ed1f09dc330", "source_code_sha": "a" * 40, "extra": True},
])
def test_recovery_scope_must_be_explicit_and_well_formed(recovery):
    request = json.loads((ROOT / REQUEST_PATH).read_text())
    request["finalize_existing"] = recovery
    with pytest.raises(ValueError, match="RECOVERY"):
        select_admission(request, event="workflow_run", branch="main", conclusion="success", changed_paths=[REQUEST_PATH])
