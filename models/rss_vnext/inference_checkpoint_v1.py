from __future__ import annotations

"""Pinned RSS.vNext *inference* checkpoint, not ModelRelease/ProductRevision.

The neural graph and its vision encoder are one weight set. Loading is fail-closed:
the caller pins exact checkpoint bytes, weights_only=True forbids pickle object
construction, configuration keys and state tensors must match exactly. The test
encoder can NEVER be loaded in product mode. This module awards no scientific
quality, source admission, or canonical geometry authority.
"""

from dataclasses import asdict, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from .contracts import RSSvNextConfig
from .unified_character_model import (
    RSSvNextUnifiedCharacterModel,
    RSSvNextFrozenDINOv2,
    RSSvNextPatchEncoderForTests,
)

RSS_VNEXT_CHECKPOINT_SCHEMA = "RealSaS.RSSvNext.UnifiedInferenceCheckpoint.v1"
RSS_VNEXT_CHECKPOINT_SCHEMA_V2 = "RealSaS.RSSvNext.UnifiedInferenceCheckpoint.v2"
_V2_KEYS = frozenset({
    "schema", "architecture_id", "config", "vision_backend",
    "model_state_dict", "source_policy", "qualification_status",
    "reconstruction_adapter",
})
_V2_SOURCE_POLICY = "TWO_ORIGINAL_IMAGES_TO_PREDICTED_SIX_VIEW_SHARED_SCENE"
_ADAPTER_KIND = "IRIS_SCENE_ATTENTION_LORA"

_CHECKPOINT_KEYS = frozenset({
    "schema", "architecture_id", "config", "vision_backend",
    "model_state_dict", "source_policy", "qualification_status",
})
_SOURCE_POLICY = "FRONT_BACK_ONLY_NO_TEACHER_NO_PREQUALIFIED_PRODUCT"
_BACKENDS = {
    RSSvNextFrozenDINOv2: "FROZEN_LOCAL_DINOV2",
    RSSvNextPatchEncoderForTests: "TEST_PATCH_ENCODER_NOT_PRODUCT",
}


def _hex_digest(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_config(cls: type, payload: Any, code: str):
    if not isinstance(payload, Mapping) or set(payload) != {f.name for f in fields(cls)}:
        raise RuntimeError(code)
    try:
        return cls(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc


def _load_config(value: Any) -> RSSvNextConfig:
    if not isinstance(value, Mapping) or set(value) != {f.name for f in fields(RSSvNextConfig)}:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_CONFIG_KEYS")
    payload = dict(value)
    payload["iris"] = _exact_config(
        IRISVNextConfigV1, payload["iris"], "RSS_VNEXT_CHECKPOINT_IRIS_CONFIG"
    )
    payload["tessa"] = _exact_config(
        TESSAStructuralizerConfigV3, payload["tessa"], "RSS_VNEXT_CHECKPOINT_TESSA_CONFIG"
    )
    for key in ("semantic_role_names", "skin_fsq_levels"):
        if not isinstance(payload[key], (list, tuple)):
            raise RuntimeError("RSS_VNEXT_CHECKPOINT_CONFIG_SEQUENCE_INVALID")
        payload[key] = tuple(payload[key])
    cfg = RSSvNextConfig(**payload)
    try:
        cfg.validate()
    except (ValueError, TypeError) as exc:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_CONFIG_INVALID") from exc
    return cfg


def _validated_state(value: Any) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_REQUIRED")
    state = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not key or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_TYPE")
        if tensor.layout != torch.strided:
            raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_LAYOUT")
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_NONFINITE")
        state[key] = tensor
    return state


def checkpoint_payload_v1(model: RSSvNextUnifiedCharacterModel) -> dict[str, Any]:
    if type(model) is not RSSvNextUnifiedCharacterModel:
        raise TypeError("RSS_VNEXT_CHECKPOINT_MODEL_TYPE")
    backend = _BACKENDS.get(type(model.vision))
    if backend is None:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_VISION_UNREGISTERED")
    model.cfg.validate()
    if getattr(model, "realsas_stage1_lora_adapter_installed", False):
        raise RuntimeError("RSS_VNEXT_LORA_MODEL_REQUIRES_CHECKPOINT_V2")
    return {
        "schema": RSS_VNEXT_CHECKPOINT_SCHEMA,
        "architecture_id": model.architecture_id,
        "config": asdict(model.cfg),
        "vision_backend": backend,
        "model_state_dict": {
            k: tensor.detach().to("cpu").contiguous().clone()
            for k, tensor in model.state_dict().items()
        },
        "source_policy": _SOURCE_POLICY,
        "qualification_status": "INFERENCE_PACKAGING_ONLY_NOT_PRODUCT_RELEASE",
    }


def checkpoint_payload_v2(model: RSSvNextUnifiedCharacterModel) -> dict[str, Any]:
    """Capture the exact trained scene-attention adapter and shared experts.

    Generated canonical/multiview task models are separately admitted stage
    releases; this reconstruction checkpoint NEVER invents their weights.
    """
    if type(model) is not RSSvNextUnifiedCharacterModel:
        raise TypeError("RSS_VNEXT_CHECKPOINT_MODEL_TYPE")
    recipe = getattr(model, "_realsas_lora_config", None)
    if (
        not getattr(model, "realsas_stage1_lora_adapter_installed", False)
        or not isinstance(recipe, tuple) or len(recipe) != 2
    ):
        raise RuntimeError("RSS_VNEXT_RECONSTRUCTION_ADAPTER_REQUIRED")
    rank, alpha = recipe
    if type(rank) is not int or not 1 <= rank <= 64 or type(alpha) not in (int, float) or not 0 < alpha <= 256:
        raise RuntimeError("RSS_VNEXT_RECONSTRUCTION_ADAPTER_PARAMETERS_INVALID")
    backend = _BACKENDS.get(type(model.vision))
    if backend is None:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_VISION_UNREGISTERED")
    model.cfg.validate()
    if not any("parametrizations" in k and ".down" in k for k in model.state_dict()):
        raise RuntimeError("RSS_VNEXT_RECONSTRUCTION_ADAPTER_STATE_MISSING")
    return {
        "schema": RSS_VNEXT_CHECKPOINT_SCHEMA_V2,
        "architecture_id": model.architecture_id,
        "config": asdict(model.cfg),
        "vision_backend": backend,
        "model_state_dict": {
            k: tensor.detach().cpu().contiguous().clone()
            for k, tensor in model.state_dict().items()
        },
        "source_policy": _V2_SOURCE_POLICY,
        "qualification_status": "INFERENCE_PACKAGING_ONLY_NOT_PRODUCT_RELEASE",
        "reconstruction_adapter": {
            "kind": _ADAPTER_KIND, "rank": rank, "alpha": float(alpha),
        },
    }


def save_inference_checkpoint_v2(
    path: str | Path, model: RSSvNextUnifiedCharacterModel,
) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(checkpoint_payload_v2(model), tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return _sha256_file(target)


def save_inference_checkpoint_v1(
    path: str | Path, model: RSSvNextUnifiedCharacterModel,
) -> str:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(checkpoint_payload_v1(model), tmp)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return _sha256_file(target)


def load_inference_checkpoint_v1(
    path: str | Path,
    *,
    expected_sha256: str,
    vision_encoder: nn.Module,
    device: str | torch.device = "cpu",
    product_mode: bool = True,
) -> RSSvNextUnifiedCharacterModel:
    source = Path(path).resolve()
    if not source.is_file():
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_MISSING")
    if not _hex_digest(expected_sha256):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_PIN_REQUIRED")
    if _sha256_file(source) != expected_sha256:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_BYTES_DRIFT")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_DECODE_FAILED") from exc
    if isinstance(payload, Mapping) and payload.get("schema") == (
        "RealSaS.RSSvNext.UnifiedThreeStageNeuralRelease.v1"
    ):
        from .realsas_product_neural_bundle_v1 import load_product_neural_bundle_v1
        bundle = load_product_neural_bundle_v1(
            source, expected_sha256=expected_sha256,
            vision_encoder=vision_encoder, device=device,
            product_mode=product_mode,
        )
        model = bundle.model
        model._pinned_neural_bundle = bundle
        return model
    if not isinstance(payload, Mapping):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_PAYLOAD_INVALID")
    schema = payload.get("schema")
    modern = schema == RSS_VNEXT_CHECKPOINT_SCHEMA_V2
    if schema not in (RSS_VNEXT_CHECKPOINT_SCHEMA, RSS_VNEXT_CHECKPOINT_SCHEMA_V2):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_SCHEMA_DRIFT")
    if set(payload) != (_V2_KEYS if modern else _CHECKPOINT_KEYS):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_PAYLOAD_INVALID")
    if payload["architecture_id"] != RSSvNextUnifiedCharacterModel.architecture_id:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_ARCHITECTURE_DRIFT")
    if payload["source_policy"] != (_V2_SOURCE_POLICY if modern else _SOURCE_POLICY):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_SOURCE_POLICY_DRIFT")
    if payload["qualification_status"] != "INFERENCE_PACKAGING_ONLY_NOT_PRODUCT_RELEASE":
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_AUTHORITY_DRIFT")
    cfg = _load_config(payload["config"])
    backend = _BACKENDS.get(type(vision_encoder))
    if backend is None or backend != payload["vision_backend"]:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_VISION_BACKEND_DRIFT")
    if product_mode and backend != "FROZEN_LOCAL_DINOV2":
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_TEST_VISION_FORBIDDEN_IN_PRODUCT")
    state = _validated_state(payload["model_state_dict"])
    model = RSSvNextUnifiedCharacterModel(cfg, vision_encoder=vision_encoder)
    if modern:
        recipe = payload["reconstruction_adapter"]
        if (
            type(recipe) is not dict or set(recipe) != {"kind", "rank", "alpha"}
            or recipe["kind"] != _ADAPTER_KIND
            or type(recipe["rank"]) is not int
            or not 1 <= recipe["rank"] <= 64
            or type(recipe["alpha"]) not in (float, int)
            or not 0 < recipe["alpha"] <= 256
        ):
            raise RuntimeError("RSS_VNEXT_RECONSTRUCTION_ADAPTER_MANIFEST_INVALID")
        from .realsas_slrm_stage1_lora_v1 import install_rss_realsas_stage1_attention_lora_v1
        install_rss_realsas_stage1_attention_lora_v1(
            model, rank=recipe["rank"], alpha=recipe["alpha"],
        )
        model.construction_requires_generated_multiview = True
    if set(state) != set(model.state_dict()):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_KEYS_DRIFT")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_STATE_SHAPES_DRIFT") from exc
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("RSS_VNEXT_CHECKPOINT_INFERENCE_NOT_FROZEN")
    return model


__all__ = [
    "RSS_VNEXT_CHECKPOINT_SCHEMA", "RSS_VNEXT_CHECKPOINT_SCHEMA_V2",
    "checkpoint_payload_v1", "checkpoint_payload_v2",
    "save_inference_checkpoint_v1", "save_inference_checkpoint_v2",
    "load_inference_checkpoint_v1",
]
