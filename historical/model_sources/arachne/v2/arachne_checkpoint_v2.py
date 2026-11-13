from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import torch

from models.skin_field_codec.v1.train_codec_r6_a0_v1 import CodecA0QualificationTokenV1
from .arachne_candidate_v2 import ArachneCandidateV2


SCHEMA = "RealSaS.ArachneCheckpoint.v2"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TEACHER_ACCESS = {
    "L0_TARGET_ONLY",
    "L1_LOSS_ONLY",
    "L2_ORACLE_UPSTREAM_PRODUCT",
    "L3_CONDITIONING_INPUT",
}


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    value = str(value)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"Arachne checkpoint requires exact sha256:{field}")
    return value


def codec_a0_token_hash_v1(token: CodecA0QualificationTokenV1) -> str:
    return _canonical_hash(asdict(token))


def checkpoint_payload_v2(
    model: ArachneCandidateV2,
    *,
    architecture_base_commit: str,
    training_data_manifest_sha256: str,
    optimizer_steps: int,
    rung: str,
    teacher_access_level: str,
    codec_checkpoint_sha256: str,
    codec_a0_token: CodecA0QualificationTokenV1,
    upstream_surface_checkpoint_sha256: str | None,
    upstream_skeleton_checkpoint_sha256: str | None,
    authorized_claims: tuple[str, ...],
    forbidden_claims: tuple[str, ...],
    shipping_authority: bool,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(model, ArachneCandidateV2):
        raise TypeError("Arachne V2 checkpoint requires ArachneCandidateV2")
    if int(optimizer_steps) < 0:
        raise ValueError("Arachne checkpoint optimizer_steps must be non-negative")
    if not rung:
        raise ValueError("Arachne checkpoint rung required")
    if teacher_access_level not in _ALLOWED_TEACHER_ACCESS:
        raise ValueError("Arachne checkpoint teacher_access_level invalid")
    if not authorized_claims or set(authorized_claims).intersection(forbidden_claims):
        raise ValueError("Arachne checkpoint claim sets invalid")
    if shipping_authority and "SHIPPING_OBSERVATION_ONLY_INFERENCE" not in authorized_claims:
        raise ValueError("shipping Arachne checkpoint must explicitly authorize shipping claim")
    codec_a0_token.validate_for(model.codec)
    authority = {
        "schema": SCHEMA,
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "codec_architecture_id": model.codec.config.architecture_id,
        "codec_config_hash": model.codec.config.config_hash,
        "codec_frozen": all(not p.requires_grad for p in model.codec.parameters()),
        "architecture_base_commit": _require_sha256(architecture_base_commit, "architecture_base_commit"),
        "training_data_manifest_sha256": _require_sha256(training_data_manifest_sha256, "training_data_manifest_sha256"),
        "optimizer_steps": int(optimizer_steps),
        "rung": str(rung),
        "teacher_access_level": teacher_access_level,
        "codec_checkpoint_sha256": _require_sha256(codec_checkpoint_sha256, "codec_checkpoint_sha256"),
        "codec_a0_token_hash": codec_a0_token_hash_v1(codec_a0_token),
        "upstream_surface_checkpoint_sha256": None if upstream_surface_checkpoint_sha256 is None else _require_sha256(upstream_surface_checkpoint_sha256, "upstream_surface_checkpoint_sha256"),
        "upstream_skeleton_checkpoint_sha256": None if upstream_skeleton_checkpoint_sha256 is None else _require_sha256(upstream_skeleton_checkpoint_sha256, "upstream_skeleton_checkpoint_sha256"),
        "authorized_claims": tuple(map(str, authorized_claims)),
        "forbidden_claims": tuple(map(str, forbidden_claims)),
        "shipping_authority": bool(shipping_authority),
        "compiler_owns_skin_qualification": True,
        "full_3d_reconstruction_authority": False,
        "metadata": dict(extra_metadata or {}),
    }
    if not authority["codec_frozen"]:
        raise ValueError("Arachne checkpoint requires frozen qualified codec")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return {"authority": authority, "state_dict": state, "codec_a0_token": asdict(codec_a0_token)}


def save_arachne_checkpoint_v2(path: str | Path, model: ArachneCandidateV2, **authority_kwargs: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload_v2(model, **authority_kwargs)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_arachne_checkpoint_v2(
    path: str | Path,
    model: ArachneCandidateV2,
    *,
    expected_architecture_base_commit: str,
    expected_training_data_manifest_sha256: str,
    expected_codec_checkpoint_sha256: str,
    expected_rung: str,
    expected_teacher_access_level: str,
    requested_claim: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "authority" not in payload or "state_dict" not in payload or "codec_a0_token" not in payload:
        raise ValueError("Arachne checkpoint payload malformed")
    authority = dict(payload["authority"])
    expected = {
        "schema": SCHEMA,
        "architecture_id": model.config.architecture_id,
        "config_hash": model.config.config_hash,
        "codec_architecture_id": model.codec.config.architecture_id,
        "codec_config_hash": model.codec.config.config_hash,
        "codec_frozen": True,
        "architecture_base_commit": _require_sha256(expected_architecture_base_commit, "expected_architecture_base_commit"),
        "training_data_manifest_sha256": _require_sha256(expected_training_data_manifest_sha256, "expected_training_data_manifest_sha256"),
        "codec_checkpoint_sha256": _require_sha256(expected_codec_checkpoint_sha256, "expected_codec_checkpoint_sha256"),
        "rung": expected_rung,
        "teacher_access_level": expected_teacher_access_level,
        "compiler_owns_skin_qualification": True,
        "full_3d_reconstruction_authority": False,
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise ValueError(f"Arachne checkpoint authority mismatch:{key}")
    token = CodecA0QualificationTokenV1(**payload["codec_a0_token"])
    token.validate_for(model.codec)
    if authority.get("codec_a0_token_hash") != codec_a0_token_hash_v1(token):
        raise ValueError("Arachne checkpoint codec A0 token hash mismatch")
    authorized = set(map(str, authority.get("authorized_claims", ())))
    forbidden = set(map(str, authority.get("forbidden_claims", ())))
    if requested_claim in forbidden or requested_claim not in authorized:
        raise ValueError(f"Arachne checkpoint claim not authorized:{requested_claim}")
    if requested_claim == "SHIPPING_OBSERVATION_ONLY_INFERENCE" and authority.get("shipping_authority") is not True:
        raise ValueError("Arachne checkpoint is not shipping authority")
    model.load_state_dict(payload["state_dict"], strict=True)
    return authority
