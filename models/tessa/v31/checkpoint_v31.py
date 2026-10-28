from __future__ import annotations

"""Inference-only checkpoint envelope for the frozen TESSA V3.1 structuralizer."""

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from .model_v31 import TESSAStructuralizerV31


CHECKPOINT_SCHEMA_V31 = "RealSaS.TESSAStructuralizerCheckpoint.v3_1"
_CHECKPOINT_KEYS_V31 = {
    "schema",
    "architecture_id",
    "attention_semantics",
    "backend",
    "scene_input_dim",
    "surface_input_dim",
    "presentation_input_dim",
    "config",
    "model_state_dict",
}


def _config(value: Any) -> TESSAStructuralizerConfigV3:
    if not isinstance(value, Mapping):
        raise RuntimeError("TESSA_V31_CHECKPOINT_CONFIG_INVALID")
    payload = dict(value)
    expected = {field.name for field in fields(TESSAStructuralizerConfigV3)}
    if set(payload) != expected:
        raise RuntimeError("TESSA_V31_CHECKPOINT_CONFIG_INVALID")
    try:
        config = TESSAStructuralizerConfigV3(**payload)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("TESSA_V31_CHECKPOINT_CONFIG_INVALID") from exc
    return config


def _positive_int(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(code)
    return int(value)


def _state(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("TESSA_V31_CHECKPOINT_STATE_REQUIRED")
    out: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("TESSA_V31_CHECKPOINT_STATE_INVALID")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("TESSA_V31_CHECKPOINT_STATE_NONFINITE")
        out[key] = tensor
    return out


def _dims(model: TESSAStructuralizerV31) -> tuple[int, int, int]:
    return (
        int(model.memory.scene.input_dim),
        int(model.memory.surface.input_dim),
        int(model.memory.presentation.input_dim),
    )


def _backend(model: TESSAStructuralizerV31) -> str:
    values = {
        str(getattr(block.self_attn, "backend", ""))
        for block in model.decoder
    }
    if len(values) != 1 or not next(iter(values)):
        raise RuntimeError("TESSA_V31_CHECKPOINT_BACKEND_DRIFT")
    return next(iter(values))


def checkpoint_payload_v31(model: TESSAStructuralizerV31) -> dict[str, Any]:
    if not isinstance(model, TESSAStructuralizerV31):
        raise TypeError("TESSA_V31_CHECKPOINT_MODEL_TYPE_INVALID")
    if model.architecture_id != TESSAStructuralizerV31.architecture_id:
        raise RuntimeError("TESSA_V31_CHECKPOINT_ARCHITECTURE_DRIFT")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("TESSA_V31_CHECKPOINT_MODEL_NOT_FROZEN")
    model.config.validate()
    scene_dim, surface_dim, presentation_dim = _dims(model)
    state = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in model.state_dict().items()
    }
    return {
        "schema": CHECKPOINT_SCHEMA_V31,
        "architecture_id": TESSAStructuralizerV31.architecture_id,
        "attention_semantics": TESSAStructuralizerV31.attention_semantics,
        "backend": _backend(model),
        "scene_input_dim": scene_dim,
        "surface_input_dim": surface_dim,
        "presentation_input_dim": presentation_dim,
        "config": asdict(model.config),
        "model_state_dict": state,
    }


def write_checkpoint_v31(path: str | Path, model: TESSAStructuralizerV31) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(checkpoint_payload_v31(model), temporary)
    temporary.replace(target)
    return target


def load_checkpoint_v31(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> TESSAStructuralizerV31:
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError("TESSA_V31_CHECKPOINT_MISSING")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("TESSA_V31_CHECKPOINT_DECODE_FAILED") from exc
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS_V31:
        raise RuntimeError("TESSA_V31_CHECKPOINT_PAYLOAD_INVALID")
    if payload.get("schema") != CHECKPOINT_SCHEMA_V31:
        raise RuntimeError("TESSA_V31_CHECKPOINT_SCHEMA_DRIFT")
    if payload.get("architecture_id") != TESSAStructuralizerV31.architecture_id:
        raise RuntimeError("TESSA_V31_CHECKPOINT_ARCHITECTURE_DRIFT")
    if payload.get("attention_semantics") != TESSAStructuralizerV31.attention_semantics:
        raise RuntimeError("TESSA_V31_CHECKPOINT_ATTENTION_SEMANTICS_DRIFT")

    config = _config(payload.get("config"))
    scene_dim = _positive_int(payload.get("scene_input_dim"), "TESSA_V31_CHECKPOINT_SCENE_DIM_INVALID")
    surface_dim = _positive_int(payload.get("surface_input_dim"), "TESSA_V31_CHECKPOINT_SURFACE_DIM_INVALID")
    presentation_dim = _positive_int(
        payload.get("presentation_input_dim"),
        "TESSA_V31_CHECKPOINT_PRESENTATION_DIM_INVALID",
    )
    backend = payload.get("backend")
    if backend not in {"SDPA", "FLEX"}:
        raise RuntimeError("TESSA_V31_CHECKPOINT_BACKEND_INVALID")
    state = _state(payload.get("model_state_dict"))

    model = TESSAStructuralizerV31(
        scene_input_dim=scene_dim,
        surface_input_dim=surface_dim,
        presentation_input_dim=presentation_dim,
        config=config,
        backend=str(backend),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("TESSA_V31_CHECKPOINT_STATE_DRIFT") from exc
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("TESSA_V31_CHECKPOINT_INFERENCE_FREEZE_FAILED")
    return model
