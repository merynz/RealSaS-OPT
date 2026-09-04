from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "RealSaS.LearnedArtifactRegistry.v1"
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHIPPING_CLAIM = "SHIPPING_OBSERVATION_ONLY_INFERENCE"


class LearnedArtifactAuthorizationError(RuntimeError):
    pass


def _architecture_authority(payload: dict[str, Any]) -> dict[str, Any]:
    authority = payload.get("architecture_source_authority")
    if not isinstance(authority, dict):
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_ARCHITECTURE_AUTHORITY_MISSING")
    state = str(authority.get("state", ""))
    if state not in {"PENDING_V2_REFREEZE", "FROZEN_V2"}:
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_ARCHITECTURE_AUTHORITY_STATE_INVALID")
    if state == "FROZEN_V2":
        living = str(authority.get("living_source_fingerprint_sha256", ""))
        selection = str(authority.get("selection_apparatus_fingerprint_sha256", ""))
        if not _SHA256.fullmatch(living) or not _SHA256.fullmatch(selection):
            raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_ARCHITECTURE_FINGERPRINT_INVALID")
        if not str(authority.get("seal_path", "")):
            raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_ARCHITECTURE_SEAL_PATH_MISSING")
    return authority


def load_learned_artifact_registry_v1(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_SCHEMA_MISMATCH")
    if payload.get("hardening_branch_base_commit_hash_kind") != "GIT_SHA1_40":
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_HARDENING_BASE_HASH_KIND_INVALID")
    if not _GIT_SHA1.fullmatch(str(payload.get("hardening_branch_base_commit", ""))):
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_HARDENING_BASE_COMMIT_INVALID")
    if not str(payload.get("hardening_branch_base_ref", "")):
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_HARDENING_BASE_REF_MISSING")
    _architecture_authority(payload)
    if not isinstance(payload.get("current_stages"), dict):
        raise LearnedArtifactAuthorizationError("LEARNED_ARTIFACT_REGISTRY_CURRENT_STAGES_MISSING")
    return payload


def _checkpoint_sha(entry: dict[str, Any]) -> str | None:
    checkpoint = entry.get("checkpoint")
    if not isinstance(checkpoint, dict):
        return None
    value = checkpoint.get("sha256")
    return str(value) if value is not None else None


def authorize_checkpoint_claim_v1(
    registry: dict[str, Any],
    *,
    stage_name: str,
    checkpoint_sha256: str,
    requested_claim: str,
    require_architecture_fingerprint: str | None = None,
) -> dict[str, Any]:
    authority = _architecture_authority(registry)
    if authority.get("state") != "FROZEN_V2":
        raise LearnedArtifactAuthorizationError("LEARNED_CHECKPOINT_ARCHITECTURE_AUTHORITY_NOT_FROZEN")
    living_fingerprint = str(authority.get("living_source_fingerprint_sha256", ""))
    if require_architecture_fingerprint is not None and living_fingerprint != require_architecture_fingerprint:
        raise LearnedArtifactAuthorizationError("LEARNED_CHECKPOINT_ARCHITECTURE_FINGERPRINT_MISMATCH")

    stages = registry.get("current_stages", {})
    if stage_name not in stages:
        raise LearnedArtifactAuthorizationError(f"UNREGISTERED_LEARNED_STAGE:{stage_name}")
    entry = dict(stages[stage_name])
    if entry.get("state") not in {"TRAINED_RUNG_BOUNDED", "TRAINED_SHIPPING_AUTHORIZED"}:
        raise LearnedArtifactAuthorizationError(f"LEARNED_STAGE_NOT_TRAINED:{stage_name}:{entry.get('state')}")
    registered_sha = _checkpoint_sha(entry)
    if registered_sha is None or not _SHA256.fullmatch(registered_sha):
        raise LearnedArtifactAuthorizationError(f"LEARNED_STAGE_CHECKPOINT_UNREGISTERED:{stage_name}")
    if checkpoint_sha256 != registered_sha:
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_SHA_MISMATCH:{stage_name}")

    forbidden = set(map(str, entry.get("forbidden_claims", ())))
    authorized = set(map(str, entry.get("authorized_claims", ())))
    if requested_claim in forbidden or requested_claim not in authorized:
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_CLAIM_NOT_AUTHORIZED:{stage_name}:{requested_claim}")
    if requested_claim == SHIPPING_CLAIM and entry.get("shipping_authority") is not True:
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_NOT_SHIPPING_AUTHORITY:{stage_name}")

    produced_by = entry.get("produced_by")
    if not isinstance(produced_by, dict) or not produced_by.get("rung") or not produced_by.get("teacher_access_level"):
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_PROVENANCE_INCOMPLETE:{stage_name}")
    data_sha = produced_by.get("data_manifest_sha256")
    if not isinstance(data_sha, str) or not _SHA256.fullmatch(data_sha):
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_DATA_MANIFEST_INVALID:{stage_name}")
    if int(produced_by.get("optimizer_steps", -1)) < 0:
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_OPTIMIZER_STEPS_INVALID:{stage_name}")
    if produced_by.get("training_code_commit_hash_kind") != "GIT_SHA1_40" or not _GIT_SHA1.fullmatch(str(produced_by.get("training_code_commit", ""))):
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_TRAINING_CODE_COMMIT_INVALID:{stage_name}")
    artifact_architecture = str(produced_by.get("architecture_living_source_fingerprint_sha256", ""))
    if not _SHA256.fullmatch(artifact_architecture) or artifact_architecture != living_fingerprint:
        raise LearnedArtifactAuthorizationError(f"LEARNED_CHECKPOINT_ARCHITECTURE_LINEAGE_MISMATCH:{stage_name}")
    return entry


def reject_historical_checkpoint_for_current_claim_v1(
    registry: dict[str, Any],
    *,
    checkpoint_sha256: str,
    requested_claim: str,
) -> None:
    for artifact in registry.get("historical_artifacts", ()):
        checkpoint = artifact.get("checkpoint", {}) if isinstance(artifact, dict) else {}
        if checkpoint.get("sha256") == checkpoint_sha256:
            raise LearnedArtifactAuthorizationError(
                f"HISTORICAL_CHECKPOINT_FORBIDDEN_FOR_CURRENT_CLAIM:{requested_claim}:{artifact.get('state')}"
            )
    raise LearnedArtifactAuthorizationError("CHECKPOINT_NOT_REGISTERED")
