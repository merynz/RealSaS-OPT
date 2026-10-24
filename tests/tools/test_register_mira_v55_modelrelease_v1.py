from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "register_mira_v55_modelrelease_v1",
    ROOT / "tools" / "register_mira_v55_modelrelease_v1.py",
)
assert SPEC and SPEC.loader
registration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registration)

DESCRIPTOR_PATH = ROOT / "canonical" / "MIRA_V55_MODELRELEASE_REGISTRATION_V1.json"
RECOVERY_PATH = ROOT / "canonical" / "MIRA_V55_RUNTIME_RECOVERY_V1.json"
ENGINE_RELEASE_ID = "11111111-1111-4111-8111-111111111111"
MODEL_RELEASE_ID = "22222222-2222-4222-8222-222222222222"
CHECKPOINT_ARTIFACT_ID = "33333333-3333-4333-8333-333333333333"
RECOVERY_ARTIFACT_ID = "44444444-4444-4444-8444-444444444444"
SOURCE_SHA = "a" * 40


def descriptor():
    return json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))


def recovery():
    return json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))


def model_record(value, *, lifecycle="DRAFT", evaluations=None):
    return {
        "id": MODEL_RELEASE_ID,
        "family": registration.EXPECTED_FAMILY,
        "name": registration.EXPECTED_NAME,
        "architecture_id": registration.EXPECTED_ARCHITECTURE,
        "checkpoint_artifact_id": CHECKPOINT_ARTIFACT_ID,
        "checkpoint_content_sha256": registration.EXPECTED_CHECKPOINT_SHA256,
        "engine_release_id": ENGINE_RELEASE_ID,
        "source_code_sha": SOURCE_SHA,
        "lifecycle_state": lifecycle,
        "metadata": registration._release_metadata(value),
        "evaluations": list(evaluations or []),
    }


class FakeAPI:
    def __init__(self, gets=None):
        self.gets = dict(gets or {})
        self.posts = []

    def get(self, path):
        value = self.gets[path]
        return copy.deepcopy(value)

    def post(self, path, payload):
        self.posts.append((path, copy.deepcopy(payload)))
        raise AssertionError(f"unexpected POST {path}")


def test_canonical_descriptor_and_recovery_are_fail_closed():
    value = registration._validate_descriptor(descriptor())
    registration._validate_recovery(recovery(), value)
    assert value["promotion"]["maximum_allowed_channel"] == "RESEARCH"
    assert value["promotion"]["candidate_promotion_allowed"] is False
    assert value["promotion"]["production_promotion_allowed"] is False
    assert value["activation"]["product_skin_activation_performed_by_registration"] is False
    assert value["activation"]["current_platform_skin_smoke_required_before_activation"] is True


def test_descriptor_rejects_channel_or_scientific_overclaim():
    value = descriptor()
    value["promotion"]["maximum_allowed_channel"] = "CANDIDATE"
    with pytest.raises(registration.RegistrationError, match="PROMOTION_BOUNDARY_DRIFT"):
        registration._validate_descriptor(value)

    value = descriptor()
    value["forbidden_claims"]["unseen_generalization"] = True
    with pytest.raises(registration.RegistrationError, match="FORBIDDEN_CLAIM_DRIFT"):
        registration._validate_descriptor(value)


def test_engine_release_requires_exact_canonical_main_source_binding():
    api = FakeAPI(
        {
            f"/v1/releases/{ENGINE_RELEASE_ID}/sources": [
                {
                    "release_id": ENGINE_RELEASE_ID,
                    "code_sha": SOURCE_SHA,
                    "authority": "CANONICAL_MAIN",
                }
            ],
            f"/v1/releases/{ENGINE_RELEASE_ID}/capabilities": {
                "engine_release_id": ENGINE_RELEASE_ID,
                "capability_set_sha256": "b" * 64,
                "capabilities": [{"id": "product.rig"}],
            },
        }
    )
    result = registration._verify_engine_release(api, ENGINE_RELEASE_ID, SOURCE_SHA)
    assert result["canonical_source_binding"]["code_sha"] == SOURCE_SHA
    assert result["product_skin_active_in_supplied_release"] is False

    api.gets[f"/v1/releases/{ENGINE_RELEASE_ID}/sources"][0]["authority"] = "LEGACY_FROZEN"
    with pytest.raises(registration.RegistrationError, match="CURRENT_SOURCE_NOT_BOUND"):
        registration._verify_engine_release(api, ENGINE_RELEASE_ID, SOURCE_SHA)


def test_exact_existing_model_release_is_reused_and_drift_is_rejected():
    value = registration._validate_descriptor(descriptor())
    existing = model_record(value)
    api = FakeAPI({"/v1/model-releases": [existing]})
    out, reused = registration._ensure_model_release(
        api,
        value,
        checkpoint_artifact_id=CHECKPOINT_ARTIFACT_ID,
        engine_release_id=ENGINE_RELEASE_ID,
        source_code_sha=SOURCE_SHA,
        actor="test",
    )
    assert reused is True
    assert out["id"] == MODEL_RELEASE_ID
    assert api.posts == []

    drifted = copy.deepcopy(existing)
    drifted["checkpoint_content_sha256"] = "c" * 64
    api = FakeAPI({"/v1/model-releases": [drifted]})
    with pytest.raises(registration.RegistrationError, match="EXISTING_MODEL_RELEASE_DRIFT"):
        registration._ensure_model_release(
            api,
            value,
            checkpoint_artifact_id=CHECKPOINT_ARTIFACT_ID,
            engine_release_id=ENGINE_RELEASE_ID,
            source_code_sha=SOURCE_SHA,
            actor="test",
        )


def test_not_run_future_scope_never_downgrades_later_observation():
    value = registration._validate_descriptor(descriptor())
    later_smoke = {
        "scope": "CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE",
        "result": "PASS",
        "evidence_artifact_id": "55555555-5555-4555-8555-555555555555",
        "experiment_id": None,
        "details": {"runtime": "observed later"},
    }
    release = model_record(value, evaluations=[later_smoke])

    class EvaluationAPI(FakeAPI):
        def post(self, path, payload):
            assert path == "/v1/model-releases/evaluations"
            self.posts.append((path, copy.deepcopy(payload)))
            row = {
                "scope": payload["scope"],
                "result": payload["result"],
                "evidence_artifact_id": payload.get("evidence_artifact_id"),
                "experiment_id": payload.get("experiment_id"),
                "details": payload["details"],
            }
            release["evaluations"].append(row)
            return copy.deepcopy(release)

    api = EvaluationAPI()
    out, receipts = registration._ensure_evaluations(
        api,
        value,
        release,
        recovery_artifact_id=RECOVERY_ARTIFACT_ID,
        actor="test",
    )
    preserved = [x for x in receipts if x["scope"] == "CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE"]
    assert preserved == [
        {
            "scope": "CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE",
            "action": "PRESERVE_LATER_OBSERVATION",
            "result": "PASS",
        }
    ]
    assert any(
        row["scope"] == "CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE" and row["result"] == "PASS"
        for row in out["evaluations"]
    )
    assert all(
        not (path.endswith("/evaluations") and payload["scope"] == "CURRENT_PLATFORM_SKIN_EXECUTOR_SMOKE")
        for path, payload in api.posts
    )


def test_research_channel_move_requires_explicit_replacement():
    release = {"id": MODEL_RELEASE_ID}
    occupied = {
        "family": "MIRA",
        "channel": "RESEARCH",
        "model_release_id": "66666666-6666-4666-8666-666666666666",
        "lock_version": 7,
    }
    api = FakeAPI({"/v1/model-channels": [occupied]})
    with pytest.raises(registration.RegistrationError, match="RESEARCH_CHANNEL_OCCUPIED"):
        registration._promote_research(api, release, actor="test", replace_existing=False)
    assert api.posts == []

    class PromotionAPI(FakeAPI):
        def post(self, path, payload):
            self.posts.append((path, copy.deepcopy(payload)))
            assert path == "/v1/model-releases/promote"
            return {
                "family": "MIRA",
                "channel": "RESEARCH",
                "model_release_id": MODEL_RELEASE_ID,
                "lock_version": 8,
            }

    api = PromotionAPI({"/v1/model-channels": [occupied]})
    channel, action = registration._promote_research(
        api,
        release,
        actor="test",
        replace_existing=True,
    )
    assert action == "REPLACE_EXPLICIT"
    assert channel["model_release_id"] == MODEL_RELEASE_ID
    _, payload = api.posts[0]
    assert payload["expected_lock_version"] == 7
    assert payload["required_evaluation_scopes"] == list(registration.EXPECTED_PASS_SCOPES)


def test_research_channel_same_release_is_idempotent_without_lock_churn():
    current = {
        "family": "MIRA",
        "channel": "RESEARCH",
        "model_release_id": MODEL_RELEASE_ID,
        "lock_version": 4,
    }
    api = FakeAPI({"/v1/model-channels": [current]})
    channel, action = registration._promote_research(
        api,
        {"id": MODEL_RELEASE_ID},
        actor="test",
        replace_existing=False,
    )
    assert action == "REUSE"
    assert channel == current
    assert api.posts == []


def test_dry_plan_cannot_claim_activation_or_higher_channel():
    value = registration._validate_descriptor(descriptor())
    plan = registration._dry_plan(
        value,
        checkpoint_path=Path("/sealed/MIRA_RUNTIME_CHECKPOINT_V1.pt"),
        recovery_path=RECOVERY_PATH,
        engine_release_id=ENGINE_RELEASE_ID,
        source_code_sha=SOURCE_SHA,
        promote_research=True,
        replace_research=False,
        engine_release={"product_skin_active_in_supplied_release": False},
    )
    assert plan["promotion"]["maximum_allowed_channel"] == "RESEARCH"
    assert plan["product_skin_activation"] == "FORBIDDEN"
    assert plan["candidate_promotion"] == "FORBIDDEN"
    assert plan["production_promotion"] == "FORBIDDEN"
    assert plan["scientific_execution"] == "FORBIDDEN"
