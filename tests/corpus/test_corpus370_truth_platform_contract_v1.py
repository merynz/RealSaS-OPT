from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.platform_truth_cohort_bootstrap_v1 import plan_manifest_keys
from tools.truth_foundry.platform_corpus370_truth_operator_v1 import prepare

ROOT = Path(__file__).resolve().parents[2]


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def test_corpus370_truth_campaign_is_finite_stage48_and_training_independent():
    campaign = load("canonical/CORPUS370_TRUTH_CAMPAIGN_V1.json")
    plan = load(campaign["compiler_plan"])
    assert campaign["canonical_branch"] == "main"
    assert campaign["expected_subject_count"] == 370
    assert campaign["required_stage_count"] == 10
    assert campaign["required_target_stage_id"] == "48_PRESENTATION_RESPONSE_TRUTH"
    assert len(plan["stages"]) == 10
    assert plan["stages"][-1]["id"] == "48_PRESENTATION_RESPONSE_TRUTH"
    keys = set(plan_manifest_keys(plan))
    assert "iris_fit" not in keys
    assert keys <= {
        "source_files", "source_license", "source_audit", "admission", "observation",
        "normalization", "truth_foundry", "presentation_truth",
    }
    assert campaign["rules"]["training_forbidden"] is True


def test_operator_request_cannot_run_before_exact_bundle_is_filled():
    request = load("canonical/CORPUS370_TRUTH_OPERATOR_REQUEST_V1.json")
    assert request["enabled"] is False
    assert request["training_allowed"] is False
    assert request["cohort_bundle"]["provider"] is None
    assert request["cohort_bundle"]["sha256"] is None
    assert request["exact_source_roster_sha256"] is None
    with pytest.raises(ValueError, match="CORPUS370_TRUTH_OPERATOR_SCOPE"):
        prepare(
            request,
            repo=ROOT,
            platform_root=ROOT,
            authority_root=ROOT,
            expected_sha="0" * 40,
            output=ROOT / ".test-never-created",
            api_root="http://127.0.0.1:9",
            rclone_remote="gdrive",
            submit=False,
        )


def test_observation_policy_is_generic_corpus370_copy_not_training_authority():
    policy = load("canonical/CORPUS370_IRIS_OBSERVATION_POLICY_V1.json")
    assert policy["schema"] == "RealSaS.IrisObservationPreparationPolicy.v1"
    assert policy["view_count"] == 8
    assert policy["resolution"] == 1024
    assert policy["pose"] == "SOURCE_BIND_REST"
    assert policy["native_frame0_is_bind_assumed"] is False
    assert policy["training_authorized"] is False
