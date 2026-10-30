from __future__ import annotations

"""Self-contained runtime package for the MIRA V6 discrete skin codec."""

from pathlib import Path
from typing import Any, Mapping
import hashlib

from models.mira.skin_fsq_codec_v6 import (
    MIRASkinFSQCodecConfig,
    MIRASkinFSQCodecV6,
    MIRA_V6_CODEC_ID,
)

MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2 = "RealSaS.MIRA.RuntimeCheckpoint.v2"


def _config_dict(config: MIRASkinFSQCodecConfig) -> dict[str, Any]:
    return {
        "vertex_feature_dim": int(config.vertex_feature_dim),
        "joint_feature_dim": int(config.joint_feature_dim),
        "hidden_dim": int(config.hidden_dim),
        "tokens_per_joint": int(config.tokens_per_joint),
        "levels": list(map(int, config.levels)),
        "eps": float(config.eps),
    }


def _config_from_dict(value: Mapping[str, Any]) -> MIRASkinFSQCodecConfig:
    return MIRASkinFSQCodecConfig(
        vertex_feature_dim=int(value["vertex_feature_dim"]),
        joint_feature_dim=int(value["joint_feature_dim"]),
        hidden_dim=int(value.get("hidden_dim", 192)),
        tokens_per_joint=int(value.get("tokens_per_joint", 4)),
        levels=tuple(map(int, value.get("levels") or (5, 5, 5, 5))),
        eps=float(value.get("eps", 1e-8)),
    )


def build_mira_runtime_checkpoint_v2(model: MIRASkinFSQCodecV6) -> dict[str, Any]:
    import torch
    if not isinstance(model, MIRASkinFSQCodecV6):
        raise TypeError("MIRA_V6_RUNTIME_MODEL_TYPE_INVALID")
    return {
        "schema": MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2,
        "architecture_id": MIRA_V6_CODEC_ID,
        "config": _config_dict(model.config),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "codec_contract": {
            "finite_scalar_quantization": True,
            "codebook_size": int(model.config.vocab_size),
            "tokens_per_joint": int(model.config.tokens_per_joint),
            "dense_skin_authority": False,
            "compiler_token_search_allowed": False,
            "compiler_token_repair_allowed": False,
        },
        "packaging_authority": "RUNTIME_PACKAGING_ONLY__NO_SCIENTIFIC_PASS",
    }


def save_mira_runtime_checkpoint_v2(model: MIRASkinFSQCodecV6, path: str | Path) -> str:
    import torch
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_mira_runtime_checkpoint_v2(model), target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest


def load_mira_runtime_model_v2(path: str | Path, *, device) -> tuple[MIRASkinFSQCodecV6, Mapping[str, Any]]:
    import torch
    target = Path(path).resolve()
    if not target.is_file():
        raise RuntimeError("MIRA_V6_RUNTIME_CHECKPOINT_MISSING")
    value = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise RuntimeError("MIRA_V6_RUNTIME_CHECKPOINT_NOT_MAPPING")
    if str(value.get("schema") or "") != MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2:
        raise RuntimeError("MIRA_V6_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")
    if str(value.get("architecture_id") or "") != MIRA_V6_CODEC_ID:
        raise RuntimeError("MIRA_V6_RUNTIME_ARCHITECTURE_DRIFT")
    config_raw = value.get("config")
    state = value.get("state_dict")
    contract = value.get("codec_contract")
    if not isinstance(config_raw, Mapping) or not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise RuntimeError("MIRA_V6_RUNTIME_CHECKPOINT_CONTENT_INVALID")
    config = _config_from_dict(config_raw)
    if int(contract.get("codebook_size") or 0) != config.vocab_size:
        raise RuntimeError("MIRA_V6_RUNTIME_CODEBOOK_DRIFT")
    if int(contract.get("tokens_per_joint") or 0) != config.tokens_per_joint:
        raise RuntimeError("MIRA_V6_RUNTIME_TOKEN_COUNT_DRIFT")
    if contract.get("compiler_token_search_allowed") is not False or contract.get("compiler_token_repair_allowed") is not False:
        raise RuntimeError("MIRA_V6_RUNTIME_COMPILER_TOKEN_AUTHORITY_DRIFT")
    model = MIRASkinFSQCodecV6(config)
    model.load_state_dict(dict(state), strict=True)
    model.to(device=device)
    model.eval()
    return model, value


__all__ = [
    "MIRA_RUNTIME_CHECKPOINT_SCHEMA_V2", "build_mira_runtime_checkpoint_v2",
    "load_mira_runtime_model_v2", "save_mira_runtime_checkpoint_v2",
]
