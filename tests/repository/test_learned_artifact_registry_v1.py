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
    assert set(payload["stages"]) == set(CURRENT_STAGES)
    for stage_name in CURRENT_STAGES:
        stage = payload["stages"][stage_name]
        assert stage["current_checkpoint_status"] == "NO_CURRENT_CHECKPOINT", stage_name
        assert stage["current_checkpoint_sha256"] is None, stage_name
        assert stage["current_checkpoint_locator"] is None, stage_name
        assert stage["optimizer_steps"] == 0, stage_name
        assert stage["training_data_manifest_sha256"] is None, stage_name
        assert stage["inference_authorized"] is False, stage_name
        assert (ROOT / stage["code_root"]).is_dir(), stage_name
        assert (ROOT / stage["checkpoint_contract"]).is_file(), stage_name


def test_historical_external_weights_are_hash_pinned_but_never_current_by_name() -> None:
    payload = _load()
    artifacts = payload["historical_verified_external_artifacts"]
    assert artifacts
    seen = set()
    for artifact in artifacts:
        assert artifact["status"] == "HISTORICAL_NOT_CURRENT_AUTHORITY"
        assert artifact["may_be_loaded_as_current"] is False
        assert SHA256.fullmatch(artifact["source_sha256"])
        assert int(artifact["source_size_bytes"]) > 0
        assert int(artifact["part_count"]) > 0
        assert artifact["source_drive_path"].endswith(".pt")
        assert artifact["source_sha256"] not in seen
        seen.add(artifact["source_sha256"])


def test_registration_policy_requires_hash_data_and_upstream_lineage() -> None:
    rule = _load()["registration_rule"]
    assert rule == {
        "binary_may_live_outside_git": True,
        "every_current_checkpoint_requires_sha256": True,
        "every_current_checkpoint_requires_exact_architecture_base": True,
        "every_current_checkpoint_requires_training_data_manifest": True,
        "every_current_checkpoint_requires_optimizer_step": True,
        "every_downstream_checkpoint_requires_upstream_checkpoint_lineage": True,
        "unregistered_checkpoint_may_be_used_for_product_inference": False,
        "historical_checkpoint_may_be_promoted_by_name_only": False,
    }


def test_git_tree_contains_contracts_not_accidental_model_weight_blobs() -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".safetensors"}
    offenders = [
        str(path.relative_to(ROOT))
        for root in (ROOT / "models", ROOT / "canonical")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert offenders == []
