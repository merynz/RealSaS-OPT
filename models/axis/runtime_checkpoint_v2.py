from __future__ import annotations

"""Self-contained runtime package for AXIS V6 control-ready rig generation."""

from pathlib import Path
from typing import Any, Mapping
import hashlib

from models.axis.rig_program_model_v6 import (
    AXISRigProgramConfigV6,
    AXISRigProgramModelV6,
    AXIS_V6_PRODUCT_ARCHITECTURE_ID,
)
from models.axis.scene_conditioning_v6 import (
    AXIS_V6_SCENE_ADAPTER_ID,
    AXIS_V6_SURFACE_SCENE_DIM,
)

AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2 = "RealSaS.AXIS.RuntimeCheckpoint.v2"


def build_axis_runtime_checkpoint_v2(
    model: AXISRigProgramModelV6,
    *,
    skin_codec_architecture_id: str,
    skin_codec_checkpoint_sha256: str,
    skin_codebook_size: int,
    tokens_per_joint: int,
) -> dict[str, Any]:
    if not isinstance(model, AXISRigProgramModelV6):
        raise TypeError("AXIS_V6_RUNTIME_MODEL_TYPE_INVALID")
    if model.config.scene_dim != AXIS_V6_SURFACE_SCENE_DIM:
        raise RuntimeError("AXIS_V6_RUNTIME_SCENE_ADAPTER_DIM_DRIFT")
    codec_hash = str(skin_codec_checkpoint_sha256).strip().lower()
    if len(codec_hash) != 64 or any(c not in "0123456789abcdef" for c in codec_hash):
        raise RuntimeError("AXIS_V6_RUNTIME_SKIN_CODEC_HASH_INVALID")
    if int(skin_codebook_size) != model.config.skin_vocab_size or int(tokens_per_joint) < 1:
        raise RuntimeError("AXIS_V6_RUNTIME_SKIN_CODEC_DOMAIN_DRIFT")
    return {
        "schema": AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2,
        "architecture_id": AXIS_V6_PRODUCT_ARCHITECTURE_ID,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scene_adapter": {
            "adapter_id": AXIS_V6_SCENE_ADAPTER_ID,
            "scene_dim": AXIS_V6_SURFACE_SCENE_DIM,
            "canonical_parent": False,
        },
        "skin_codec_binding": {
            "architecture_id": str(skin_codec_architecture_id),
            "checkpoint_sha256": codec_hash,
            "codebook_size": int(skin_codebook_size),
            "tokens_per_joint": int(tokens_per_joint),
            "joint_and_skin_generated_in_same_causal_sequence": True,
        },
        "runtime_rules": {
            "compiler_parent_reselection_allowed": False,
            "compiler_control_synthesis_allowed": False,
            "compiler_constraint_synthesis_allowed": False,
            "compiler_code_search_allowed": False,
            "constraint_solver_execution_owner": "MECHANICS",
        },
        "packaging_authority": "RUNTIME_PACKAGING_ONLY__NO_SCIENTIFIC_PASS",
    }


def save_axis_runtime_checkpoint_v2(model: AXISRigProgramModelV6, path: str | Path, **binding) -> str:
    import torch
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_axis_runtime_checkpoint_v2(model, **binding), target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_axis_runtime_model_v2(path: str | Path, *, device) -> tuple[AXISRigProgramModelV6, Mapping[str, Any]]:
    import torch
    target = Path(path).resolve()
    if not target.is_file():
        raise RuntimeError("AXIS_V6_RUNTIME_CHECKPOINT_MISSING")
    value = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise RuntimeError("AXIS_V6_RUNTIME_CHECKPOINT_NOT_MAPPING")
    if str(value.get("schema") or "") != AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2:
        raise RuntimeError("AXIS_V6_RUNTIME_CHECKPOINT_SCHEMA_DRIFT")
    if str(value.get("architecture_id") or "") != AXIS_V6_PRODUCT_ARCHITECTURE_ID:
        raise RuntimeError("AXIS_V6_RUNTIME_ARCHITECTURE_DRIFT")
    config_raw, state = value.get("config"), value.get("state_dict")
    scene, codec, rules = value.get("scene_adapter"), value.get("skin_codec_binding"), value.get("runtime_rules")
    if not all(isinstance(x, Mapping) for x in (config_raw, state, scene, codec, rules)):
        raise RuntimeError("AXIS_V6_RUNTIME_CONTENT_INVALID")
    config = AXISRigProgramConfigV6.from_dict(dict(config_raw))
    if config.scene_dim != AXIS_V6_SURFACE_SCENE_DIM:
        raise RuntimeError("AXIS_V6_RUNTIME_SCENE_DIM_DRIFT")
    if str(scene.get("adapter_id") or "") != AXIS_V6_SCENE_ADAPTER_ID:
        raise RuntimeError("AXIS_V6_RUNTIME_SCENE_ADAPTER_DRIFT")
    if int(scene.get("scene_dim") or 0) != config.scene_dim:
        raise RuntimeError("AXIS_V6_RUNTIME_SCENE_ADAPTER_WIDTH_DRIFT")
    if int(codec.get("codebook_size") or 0) != config.skin_vocab_size:
        raise RuntimeError("AXIS_V6_RUNTIME_SKIN_VOCAB_DRIFT")
    codec_hash = str(codec.get("checkpoint_sha256") or "")
    if len(codec_hash) != 64:
        raise RuntimeError("AXIS_V6_RUNTIME_SKIN_CODEC_BINDING_MISSING")
    for key in (
        "compiler_parent_reselection_allowed",
        "compiler_control_synthesis_allowed",
        "compiler_constraint_synthesis_allowed",
        "compiler_code_search_allowed",
    ):
        if rules.get(key) is not False:
            raise RuntimeError(f"AXIS_V6_RUNTIME_COMPILER_AUTHORITY_DRIFT:{key}")
    if str(rules.get("constraint_solver_execution_owner") or "") != "MECHANICS":
        raise RuntimeError("AXIS_V6_RUNTIME_CONSTRAINT_EXECUTION_OWNER_DRIFT")
    model = AXISRigProgramModelV6(config)
    model.load_state_dict(dict(state), strict=True)
    model.to(device=device)
    model.eval()
    return model, value


__all__ = [
    "AXIS_RUNTIME_CHECKPOINT_SCHEMA_V2", "build_axis_runtime_checkpoint_v2",
    "load_axis_runtime_model_v2", "save_axis_runtime_checkpoint_v2",
]
