from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "canonical" / "LEARNED_ARTIFACT_REGISTRY_V1_20260904.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CURRENT_STAGES = (
    "IRIS_V2",
    "GEPPETTO_V2",
    "SKIN_FIELD_CODEC_V1_A0",
    "ARACHNE_V2_A1",
)


def _load() -> dict:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema"] == "RealSaS.LearnedArtifactRegistry.v1"
    assert SHA256.fullmatch(payload["architecture_base_commit"])
    assert payload["architecture_base_ref"] == "first-fit-base/main-20260904"
    assert payload["binary_storage_policy"] == "EXTERNAL_BINARY__REPOSITORY_HASH_AND_LINEAGE_MANIFEST_REQUIRED"
    return payload


def test_current_learned_state_is_explicit_and_fail_closed_before_fit() -> None:
    payload = _load()
    assert payload["fit_authorized_now"] is False
    assert payload["current_product_state"] == "NO_CURRENT_TRAINED_END_TO_END_MODEL_SET"
    assert tuple(payload["dependency_order"]) == CURRENT_STAGES
    assert set(payload["current_stages"]) == set(CURRENT_STAGES)
    for stage_name in CURRENT_STAGES:
        stage = payload["current_stages"][stage_name]
        assert stage["state"] == "UNTRAINED", stage_name
        assert stage["checkpoint"] is None, stage_name
        assert stage["produced_by"]["rung"] == "NONE_UNTRAINED", stage_name
        assert stage["produced_by"]["teacher_access_level"] == "NONE", stage_name
        assert stage["produced_by"]["optimizer_steps"] == 0, stage_name
        assert stage["authorized_claims"] == [], stage_name
        assert stage["shipping_authority"] is False, stage_name
        assert "SHIPPING_OBSERVATION_ONLY_INFERENCE" in stage["forbidden_claims"], stage_name
        assert (ROOT / stage["code_root"]).is_dir(), stage_name
        assert (ROOT / stage["checkpoint_contract"]).is_file(), stage_name


def test_quarantined_weights_live_in_same_registry_and_cannot_ship() -> None:
    payload = _load()
    artifacts = payload["historical_artifacts"]
    assert artifacts
    seen = set()
    for artifact in artifacts:
        assert artifact["state"] == "QUARANTINED"
        assert artifact["shipping_authority"] is False
        assert artifact["authorized_claims"] == ["HISTORICAL_PROVENANCE_ONLY"]
        assert "SHIPPING_OBSERVATION_ONLY_INFERENCE" in artifact["forbidden_claims"]
        assert (ROOT / artifact["quarantine_reason_ref"]).is_file()
        checkpoint = artifact["checkpoint"]
        assert SHA256.fullmatch(checkpoint["sha256"])
        assert int(checkpoint["bytes"]) > 0
        assert int(checkpoint["archive_part_count"]) > 0
        assert checkpoint["name"].endswith(".pt")
        assert checkpoint["sha256"] not in seen
        seen.add(checkpoint["sha256"])


def test_registry_vocabularies_cover_rung_claim_firewall() -> None:
    payload = _load()
    assert "SHIPPING_OBSERVATION_ONLY_INFERENCE" in payload["claim_vocabulary"]
    assert "DOWNSTREAM_CONSUMER_CEILING" in payload["claim_vocabulary"]
    assert "L3_CONDITIONING_INPUT" in payload["teacher_access_vocabulary"]
    rule = payload["registration_rule"]
    assert rule["every_trained_checkpoint_requires_rung"] is True
    assert rule["every_trained_checkpoint_requires_teacher_access_level"] is True
    assert rule["every_trained_checkpoint_requires_authorized_and_forbidden_claims"] is True
    assert rule["every_downstream_checkpoint_requires_upstream_checkpoint_lineage"] is True
    assert rule["shipping_claim_requires_shipping_authority_true"] is True
    assert rule["unregistered_checkpoint_may_be_used_for_product_inference"] is False


def test_git_tree_contains_contracts_not_accidental_model_weight_blobs() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    offenders = [
        str(path.relative_to(ROOT))
        for root in (ROOT / "models", ROOT / "canonical")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert offenders == []
