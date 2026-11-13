from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from .model_v31 import IRISConfigV31, IRISV31


CHECKPOINT_SCHEMA_V1 = "RealSaS.IRISDenseObservationEvidenceCheckpoint.v3_1"
_CHECKPOINT_KEYS_V1 = {
    "schema",
    "architecture_id",
    "config",
    "model_state_dict",
}


def _exact_config(value: Any) -> IRISConfigV31:
    if not isinstance(value, Mapping):
        raise RuntimeError("IRIS_V31_CHECKPOINT_CONFIG_INVALID")
    payload = dict(value)
    expected = {field.name for field in fields(IRISConfigV31)}
    if set(payload) != expected:
        raise RuntimeError("IRIS_V31_CHECKPOINT_CONFIG_INVALID")
    if not isinstance(payload.get("mechanical_channels"), tuple) or not isinstance(
        payload.get("pair_channels"), tuple
    ):
        raise RuntimeError("IRIS_V31_CHECKPOINT_CONFIG_INVALID")
    try:
        config = IRISConfigV31(**payload)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("IRIS_V31_CHECKPOINT_CONFIG_INVALID") from exc
    return config


def _validated_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("IRIS_V31_CHECKPOINT_STATE_REQUIRED")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("IRIS_V31_CHECKPOINT_STATE_INVALID")
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("IRIS_V31_CHECKPOINT_STATE_NONFINITE")
        state[key] = tensor
    return state


def checkpoint_payload_v1(model: IRISV31) -> dict[str, Any]:
    """Serialize only V3.1 inference architecture and frozen model weights."""

    if not isinstance(model, IRISV31):
        raise TypeError("IRIS_V31_CHECKPOINT_MODEL_TYPE_INVALID")
    if model.architecture_id != IRISV31.architecture_id:
        raise RuntimeError("IRIS_V31_CHECKPOINT_ARCHITECTURE_DRIFT")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_V31_CHECKPOINT_MODEL_NOT_FROZEN")
    model.config.validate()
    state = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in model.state_dict().items()
    }
    return {
        "schema": CHECKPOINT_SCHEMA_V1,
        "architecture_id": IRISV31.architecture_id,
        "config": asdict(model.config),
        "model_state_dict": state,
    }


def write_checkpoint_v1(path: str | Path, model: IRISV31) -> Path:
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
) -> IRISV31:
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError("IRIS_V31_CHECKPOINT_MISSING")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("IRIS_V31_CHECKPOINT_DECODE_FAILED") from exc
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS_V1:
        raise RuntimeError("IRIS_V31_CHECKPOINT_PAYLOAD_INVALID")
    if payload.get("schema") != CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("IRIS_V31_CHECKPOINT_SCHEMA_DRIFT")
    if payload.get("architecture_id") != IRISV31.architecture_id:
        raise RuntimeError("IRIS_V31_CHECKPOINT_ARCHITECTURE_DRIFT")

    config = _exact_config(payload.get("config"))
    state = _validated_state_dict(payload.get("model_state_dict"))
    model = IRISV31(config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("IRIS_V31_CHECKPOINT_STATE_DRIFT") from exc
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_V31_CHECKPOINT_INFERENCE_FREEZE_FAILED")
    return model
