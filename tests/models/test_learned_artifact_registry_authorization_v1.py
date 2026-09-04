from __future__ import annotations

import copy
from pathlib import Path

import pytest

from models.learned_artifact_registry_v1 import (
    LearnedArtifactAuthorizationError,
    authorize_checkpoint_claim_v1,
    load_learned_artifact_registry_v1,
    reject_historical_checkpoint_for_current_claim_v1,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "canonical" / "LEARNED_ARTIFACT_REGISTRY_V1_20260904.json"
ZERO = "0" * 64
ONE = "1" * 64
LIVING = "a" * 64
SELECTION = "b" * 64
TRAINING_COMMIT = "c" * 40


def frozen_registry() -> dict:
    registry = load_learned_artifact_registry_v1(REGISTRY_PATH)
    r = copy.deepcopy(registry)
    r["architecture_source_authority"].update({
        "state": "FROZEN_V2",
        "seal_path": "canonical/ARCHITECTURE_FREEZE_V2.json",
        "living_source_fingerprint_sha256": LIVING,
        "selection_apparatus_fingerprint_sha256": SELECTION,
    })
    return r


def test_pending_architecture_authority_blocks_checkpoint_claims_before_stage_state() -> None:
    registry = load_learned_artifact_registry_v1(REGISTRY_PATH)
    assert registry["architecture_source_authority"]["state"] == "PENDING_V2_REFREEZE"
    with pytest.raises(LearnedArtifactAuthorizationError, match="LEARNED_CHECKPOINT_ARCHITECTURE_AUTHORITY_NOT_FROZEN"):
        authorize_checkpoint_claim_v1(
            registry,
            stage_name="IRIS_V2",
            checkpoint_sha256=ZERO,
            requested_claim="SHIPPING_OBSERVATION_ONLY_INFERENCE",
        )


def test_untrained_current_stage_cannot_be_loaded_after_architecture_is_frozen() -> None:
    registry = frozen_registry()
    with pytest.raises(LearnedArtifactAuthorizationError, match="LEARNED_STAGE_NOT_TRAINED"):
        authorize_checkpoint_claim_v1(
            registry,
            stage_name="IRIS_V2",
            checkpoint_sha256=ZERO,
            requested_claim="SHIPPING_OBSERVATION_ONLY_INFERENCE",
            require_architecture_fingerprint=LIVING,
        )


def test_oracle_rung_checkpoint_can_be_downstream_bounded_but_not_shipping() -> None:
    r = frozen_registry()
    entry = r["current_stages"]["ARACHNE_V2_A1"]
    entry.update({
        "state": "TRAINED_RUNG_BOUNDED",
        "checkpoint": {"sha256": ZERO, "location": "drive:fake", "bytes": 123},
        "authorized_claims": ["DOWNSTREAM_CONSUMER_CEILING"],
        "forbidden_claims": ["SHIPPING_OBSERVATION_ONLY_INFERENCE"],
        "shipping_authority": False,
    })
    entry["produced_by"].update({
        "rung": "ORACLE_S__CURRENT_QUALIFIED_G",
        "teacher_access_level": "L3_CONDITIONING_INPUT",
        "data_manifest_sha256": ONE,
        "optimizer_steps": 17,
        "training_code_commit_hash_kind": "GIT_SHA1_40",
        "training_code_commit": TRAINING_COMMIT,
        "architecture_living_source_fingerprint_sha256": LIVING,
    })
    out = authorize_checkpoint_claim_v1(
        r,
        stage_name="ARACHNE_V2_A1",
        checkpoint_sha256=ZERO,
        requested_claim="DOWNSTREAM_CONSUMER_CEILING",
        require_architecture_fingerprint=LIVING,
    )
    assert out["produced_by"]["rung"] == "ORACLE_S__CURRENT_QUALIFIED_G"
    with pytest.raises(LearnedArtifactAuthorizationError, match="CLAIM_NOT_AUTHORIZED"):
        authorize_checkpoint_claim_v1(
            r,
            stage_name="ARACHNE_V2_A1",
            checkpoint_sha256=ZERO,
            requested_claim="SHIPPING_OBSERVATION_ONLY_INFERENCE",
            require_architecture_fingerprint=LIVING,
        )


def test_checkpoint_architecture_fingerprint_mismatch_fails_closed() -> None:
    r = frozen_registry()
    with pytest.raises(LearnedArtifactAuthorizationError, match="ARCHITECTURE_FINGERPRINT_MISMATCH"):
        authorize_checkpoint_claim_v1(
            r,
            stage_name="IRIS_V2",
            checkpoint_sha256=ZERO,
            requested_claim="DOWNSTREAM_CONSUMER_CEILING",
            require_architecture_fingerprint="d" * 64,
        )


def test_historical_quarantined_checkpoint_is_explicitly_rejected() -> None:
    registry = load_learned_artifact_registry_v1(REGISTRY_PATH)
    sha = registry["historical_artifacts"][0]["checkpoint"]["sha256"]
    with pytest.raises(LearnedArtifactAuthorizationError, match="HISTORICAL_CHECKPOINT_FORBIDDEN"):
        reject_historical_checkpoint_for_current_claim_v1(
            registry,
            checkpoint_sha256=sha,
            requested_claim="SHIPPING_OBSERVATION_ONLY_INFERENCE",
        )
