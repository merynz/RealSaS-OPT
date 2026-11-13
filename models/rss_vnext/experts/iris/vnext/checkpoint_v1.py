from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from .contracts_v1 import IRISVNextConfigV1
from .heads_v3 import GraphicsRelationHeadV3Config
from .model_v3 import IRISVNextEvidenceModelV3


CHECKPOINT_SCHEMA_V1 = "RealSaS.IRISVNextEvidenceCheckpoint.v1"
_CHECKPOINT_KEYS_V1 = {
    "schema",
    "architecture_id",
    "config",
    "relation_config",
    "model_state_dict",
}


def _exact_dataclass_payload(value: Any, cls: type, code: str):
    if not isinstance(value, Mapping):
        raise RuntimeError(code)
    payload = dict(value)
    expected = {field.name for field in fields(cls)}
    if set(payload) != expected:
        raise RuntimeError(code)
    try:
        instance = cls(**payload)
        instance.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc
    return instance


def _validated_state_dict(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_STATE_REQUIRED")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("IRIS_VNEXT_CHECKPOINT_STATE_INVALID")
        if not bool(torch.isfinite(tensor).all()) if torch.is_floating_point(tensor) else False:
            raise RuntimeError("IRIS_VNEXT_CHECKPOINT_STATE_NONFINITE")
        state[key] = tensor
    return state


def checkpoint_payload_v1(model: IRISVNextEvidenceModelV3) -> dict[str, Any]:
    """Create the only admitted checkpoint payload for the current IRIS V3 model.

    This is intentionally an inference/model-state contract, not a training
    checkpoint. Optimizer, RNG, curriculum and stage state are not serialized.
    """

    if not isinstance(model, IRISVNextEvidenceModelV3):
        raise TypeError("IRIS_VNEXT_CHECKPOINT_MODEL_TYPE_INVALID")
    if model.architecture_id != IRISVNextEvidenceModelV3.architecture_id:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_ARCHITECTURE_DRIFT")
    relation_config = model.relations.relation_config
    relation_config.validate()
    model.config.validate()
    state = {
        key: tensor.detach().cpu().contiguous().clone()
        for key, tensor in model.state_dict().items()
    }
    return {
        "schema": CHECKPOINT_SCHEMA_V1,
        "architecture_id": IRISVNextEvidenceModelV3.architecture_id,
        "config": asdict(model.config),
        "relation_config": asdict(relation_config),
        "model_state_dict": state,
    }


def write_checkpoint_v1(path: str | Path, model: IRISVNextEvidenceModelV3) -> Path:
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
) -> IRISVNextEvidenceModelV3:
    """Load exactly one V3 inference checkpoint and freeze it for execution."""

    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_MISSING")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_DECODE_FAILED") from exc
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS_V1:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_PAYLOAD_INVALID")
    if payload.get("schema") != CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_SCHEMA_DRIFT")
    if payload.get("architecture_id") != IRISVNextEvidenceModelV3.architecture_id:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_ARCHITECTURE_DRIFT")

    config = _exact_dataclass_payload(
        payload.get("config"),
        IRISVNextConfigV1,
        "IRIS_VNEXT_CHECKPOINT_CONFIG_INVALID",
    )
    relation_config = _exact_dataclass_payload(
        payload.get("relation_config"),
        GraphicsRelationHeadV3Config,
        "IRIS_VNEXT_CHECKPOINT_RELATION_CONFIG_INVALID",
    )
    state = _validated_state_dict(payload.get("model_state_dict"))

    model = IRISVNextEvidenceModelV3(config, relation_config)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_STATE_DRIFT") from exc
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_VNEXT_CHECKPOINT_INFERENCE_FREEZE_FAILED")
    return model
