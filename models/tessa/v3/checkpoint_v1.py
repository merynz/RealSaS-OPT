from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from .config_v3 import TESSAStructuralizerConfigV3
from .model_v3 import TESSAStructuralizerV3


CHECKPOINT_SCHEMA_V1 = "RealSaS.TESSAStructuralizerCheckpoint.v1"
_CHECKPOINT_KEYS_V1 = {
    "schema",
    "architecture_id",
    "scene_input_dim",
    "surface_input_dim",
    "presentation_input_dim",
    "config",
    "model_state_dict",
}


def _exact_config(value: Any) -> TESSAStructuralizerConfigV3:
    if not isinstance(value, Mapping):
        raise RuntimeError("TESSA_V3_CHECKPOINT_CONFIG_INVALID")
    payload = dict(value)
    expected = {field.name for field in fields(TESSAStructuralizerConfigV3)}
    if set(payload) != expected:
        raise RuntimeError("TESSA_V3_CHECKPOINT_CONFIG_INVALID")
    try:
        config = TESSAStructuralizerConfigV3(**payload)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TESSA_V3_CHECKPOINT_CONFIG_INVALID") from exc
    return config


def _positive_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(code)
    return int(value)


def _validated_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("TESSA_V3_CHECKPOINT_STATE_REQUIRED")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("TESSA_V3_CHECKPOINT_STATE_INVALID")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("TESSA_V3_CHECKPOINT_STATE_NONFINITE")
        state[key] = tensor
    return state


def _input_dims(model: TESSAStructuralizerV3) -> tuple[int, int, int]:
    return (
        int(model.memory.scene.input_dim),
        int(model.memory.surface.input_dim),
        int(model.memory.presentation.input_dim),
    )


def checkpoint_payload_v1(model: TESSAStructuralizerV3) -> dict[str, Any]:
    """Serialize only the current TESSA V3 inference architecture and weights."""

    if not isinstance(model, TESSAStructuralizerV3):
        raise TypeError("TESSA_V3_CHECKPOINT_MODEL_TYPE_INVALID")
    if model.architecture_id != TESSAStructuralizerV3.architecture_id:
        raise RuntimeError("TESSA_V3_CHECKPOINT_ARCHITECTURE_DRIFT")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("TESSA_V3_CHECKPOINT_MODEL_NOT_FROZEN")
    model.config.validate()
    scene_dim, surface_dim, presentation_dim = _input_dims(model)
    state = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in model.state_dict().items()
    }
    return {
        "schema": CHECKPOINT_SCHEMA_V1,
        "architecture_id": TESSAStructuralizerV3.architecture_id,
        "scene_input_dim": scene_dim,
        "surface_input_dim": surface_dim,
        "presentation_input_dim": presentation_dim,
        "config": asdict(model.config),
        "model_state_dict": state,
    }


def write_checkpoint_v1(path: str | Path, model: TESSAStructuralizerV3) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(checkpoint_payload_v1(model), tmp)
    tmp.replace(target)
    return target


def load_checkpoint_v1(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> TESSAStructuralizerV3:
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError("TESSA_V3_CHECKPOINT_MISSING")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("TESSA_V3_CHECKPOINT_DECODE_FAILED") from exc
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS_V1:
        raise RuntimeError("TESSA_V3_CHECKPOINT_PAYLOAD_INVALID")
    if payload.get("schema") != CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("TESSA_V3_CHECKPOINT_SCHEMA_DRIFT")
    if payload.get("architecture_id") != TESSAStructuralizerV3.architecture_id:
        raise RuntimeError("TESSA_V3_CHECKPOINT_ARCHITECTURE_DRIFT")

    config = _exact_config(payload.get("config"))
    scene_dim = _positive_int(payload.get("scene_input_dim"), "TESSA_V3_CHECKPOINT_SCENE_DIM_INVALID")
    surface_dim = _positive_int(payload.get("surface_input_dim"), "TESSA_V3_CHECKPOINT_SURFACE_DIM_INVALID")
    presentation_dim = _positive_int(
        payload.get("presentation_input_dim"),
        "TESSA_V3_CHECKPOINT_PRESENTATION_DIM_INVALID",
    )
    state = _validated_state_dict(payload.get("model_state_dict"))

    model = TESSAStructuralizerV3(
        scene_input_dim=scene_dim,
        surface_input_dim=surface_dim,
        presentation_input_dim=presentation_dim,
        config=config,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("TESSA_V3_CHECKPOINT_STATE_DRIFT") from exc
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("TESSA_V3_CHECKPOINT_INFERENCE_FREEZE_FAILED")
    return model
