from __future__ import annotations

"""Exact-byte checkpoint loading boundary for the single RSS.vNext neural graph.

The current structural trainer emits a resumable research checkpoint with
model and optimizer. This loader extracts model weights with weights_only=True
and strict shapes. Checkpoint presence is NOT model quality, ProductRevision
authority, corpus admission or deployment permission.

Platform-owned ModelRelease verification is injectable and explicit. Offline
tests/research can load only as UNQUALIFIED_RESEARCH_CHECKPOINT.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import torch

from .training_engine import RSSvNextStructuralTrainer, architecture_digest
from .unified_character_model import (
    RSSvNextPatchEncoderForTests, RSSvNextUnifiedCharacterModel,
)


@dataclass(frozen=True)
class RSSvNextCheckpointRefV1:
    checkpoint_path: str
    checkpoint_sha256: str
    architecture_id: str
    architecture_digest: str
    dataset_manifest_sha256: str
    backbone_artifact_sha256: str
    model_release_id: str = ""
    admission_receipt_hash: str = ""
    checkpoint_format: str = "RealSaS.RSSvNext.StructuralTrainerCheckpoint.v1"

    def validate(self) -> None:
        for label, value in (
            ("CHECKPOINT", self.checkpoint_sha256),
            ("DATASET", self.dataset_manifest_sha256),
            ("BACKBONE", self.backbone_artifact_sha256),
            ("ARCHITECTURE", self.architecture_digest),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"RSS_VNEXT_RELEASE_{label}_SHA256_INVALID")
        if self.architecture_id != RSSvNextUnifiedCharacterModel.architecture_id:
            raise ValueError("RSS_VNEXT_RELEASE_ARCHITECTURE_ID_DRIFT")
        if self.checkpoint_format != RSSvNextStructuralTrainer.checkpoint_schema:
            raise ValueError("RSS_VNEXT_RELEASE_FORMAT_DRIFT")
        if not self.checkpoint_path.strip():
            raise ValueError("RSS_VNEXT_RELEASE_CHECKPOINT_PATH_REQUIRED")
        if bool(self.model_release_id) != bool(self.admission_receipt_hash):
            raise ValueError("RSS_VNEXT_RELEASE_ADMISSION_PAIR_REQUIRED")
        if self.admission_receipt_hash:
            value = self.admission_receipt_hash
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("RSS_VNEXT_RELEASE_ADMISSION_HASH_INVALID")


@dataclass(frozen=True)
class RSSvNextLoadedInferenceV1:
    model: RSSvNextUnifiedCharacterModel
    model_release_id: str
    checkpoint_sha256: str
    dataset_manifest_sha256: str
    qualification: str

    def require_product_model_release(self) -> RSSvNextUnifiedCharacterModel:
        if self.qualification != "PLATFORM_MODEL_RELEASE_VERIFIED":
            raise RuntimeError("RSS_VNEXT_PRODUCT_RELEASE_VERIFICATION_REQUIRED")
        return self.model


def _sha256_file(path: Path, *, max_size_bytes: int) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("RSS_VNEXT_RELEASE_CHECKPOINT_REGULAR_FILE_REQUIRED")
    if path.stat().st_size <= 0 or path.stat().st_size > max_size_bytes:
        raise ValueError("RSS_VNEXT_RELEASE_CHECKPOINT_SIZE_BUDGET")
    sha = hashlib.sha256()
    with path.open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_rss_vnext_runtime_checkpoint_v1(
    model: RSSvNextUnifiedCharacterModel,
    ref: RSSvNextCheckpointRefV1,
    *,
    actual_backbone_sha256: str,
    verify_platform_model_release: Callable[[RSSvNextCheckpointRefV1], bool] | None = None,
    max_size_bytes: int = 24 * 1024**3,
) -> RSSvNextLoadedInferenceV1:
    """Load exact bytes, without minting ProductRevision or model readiness.

    Platform independently checks rights, training lineage, held-out
    qualification and release proof. A predicate acknowledges that verification
    and is never read from the untrusted model checkpoint.
    """
    ref.validate()
    if ref.architecture_digest != architecture_digest(model.cfg):
        raise ValueError("RSS_VNEXT_RELEASE_CONFIG_DIGEST_DRIFT")
    if actual_backbone_sha256 != ref.backbone_artifact_sha256:
        raise ValueError("RSS_VNEXT_RELEASE_BACKBONE_BYTES_DRIFT")
    if model.architecture_id != ref.architecture_id:
        raise ValueError("RSS_VNEXT_RELEASE_MODEL_FAMILY_DRIFT")
    p = Path(ref.checkpoint_path).expanduser()
    if _sha256_file(p, max_size_bytes=max_size_bytes) != ref.checkpoint_sha256:
        raise ValueError("RSS_VNEXT_RELEASE_CHECKPOINT_BYTES_DRIFT")
    if ref.model_release_id:
        if isinstance(model.vision, RSSvNextPatchEncoderForTests):
            raise RuntimeError("RSS_VNEXT_PRODUCT_TEST_BACKBONE_FORBIDDEN")
        if verify_platform_model_release is None or not verify_platform_model_release(ref):
            raise RuntimeError("RSS_VNEXT_PRODUCT_PLATFORM_PROOF_REQUIRED")
        qualification = "PLATFORM_MODEL_RELEASE_VERIFIED"
    else:
        if verify_platform_model_release is not None:
            raise ValueError("RSS_VNEXT_RELEASE_ORPHAN_VERIFIER")
        qualification = "UNQUALIFIED_RESEARCH_CHECKPOINT"

    payload = torch.load(p, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != ref.checkpoint_format:
        raise ValueError("RSS_VNEXT_RELEASE_TRAINER_SCHEMA_DRIFT")
    if payload.get("architecture_id") != ref.architecture_id:
        raise ValueError("RSS_VNEXT_RELEASE_TRAINER_ARCHITECTURE_DRIFT")
    if payload.get("architecture_digest") != ref.architecture_digest:
        raise ValueError("RSS_VNEXT_RELEASE_TRAINER_CONFIG_DRIFT")
    if payload.get("dataset_manifest_sha256") != ref.dataset_manifest_sha256:
        raise ValueError("RSS_VNEXT_RELEASE_TRAINER_DATASET_DRIFT")
    if int(payload.get("global_step", -1)) <= 0:
        raise ValueError("RSS_VNEXT_RELEASE_UNTRAINED_CHECKPOINT_FORBIDDEN")
    if "model" not in payload or not isinstance(payload["model"], dict):
        raise ValueError("RSS_VNEXT_RELEASE_STATE_DICT_MISSING")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return RSSvNextLoadedInferenceV1(
        model=model,
        model_release_id=ref.model_release_id,
        checkpoint_sha256=ref.checkpoint_sha256,
        dataset_manifest_sha256=ref.dataset_manifest_sha256,
        qualification=qualification,
    )


__all__ = [
    "RSSvNextCheckpointRefV1",
    "RSSvNextLoadedInferenceV1",
    "load_rss_vnext_runtime_checkpoint_v1",
]
