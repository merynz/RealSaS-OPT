from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, TypeVar

import torch

from .contracts_v1 import IRISVNextConfigV1
from .heads_v3 import GraphicsRelationHeadV3Config
from .model_v3 import IRISVNextEvidenceModelV3


PRODUCT_CHECKPOINT_SCHEMA_V1 = "RealSaS.IRISVNextEvidenceModelV3Checkpoint.v1"
ARCHITECTURE_CONTRACT_SCHEMA_V1 = "RealSaS.IRISVNextEvidenceModelV3ArchitectureContract.v1"

_T = TypeVar("_T")


@dataclass(frozen=True)
class IRISVNextProductCheckpointSealV1:
    content_sha256: str
    state_schema_sha256: str
    architecture_contract_sha256: str
    size_bytes: int
    schema: str = PRODUCT_CHECKPOINT_SCHEMA_V1


@dataclass(frozen=True)
class LoadedIRISVNextProductModelV1:
    model: IRISVNextEvidenceModelV3
    content_sha256: str
    state_schema_sha256: str
    architecture_contract_sha256: str
    schema: str = "RealSaS.LoadedIRISVNextProductModel.v1"

    def assert_inference_sealed(self) -> None:
        if self.model.training:
            raise RuntimeError("IRIS_VNEXT_PRODUCT_MODEL_TRAINING_MODE_DRIFT")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("IRIS_VNEXT_PRODUCT_MODEL_GRADIENT_AUTHORITY_DRIFT")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lower_sha256(value: Any, code: str) -> str:
    raw = str(value or "")
    if len(raw) != 64 or raw.lower() != raw or any(ch not in "0123456789abcdef" for ch in raw):
        raise RuntimeError(code)
    return raw


def _typed_dataclass_from_mapping(cls: type[_T], raw: Any, code: str) -> _T:
    if not isinstance(raw, Mapping):
        raise RuntimeError(code)
    defaults = cls()
    expected_names = {field.name for field in fields(cls)}
    if set(raw) != expected_names:
        raise RuntimeError(code)
    normalized: dict[str, Any] = {}
    for field in fields(cls):
        value = raw[field.name]
        expected_type = type(getattr(defaults, field.name))
        if type(value) is not expected_type:
            raise RuntimeError(code)
        normalized[field.name] = value
    try:
        value = cls(**normalized)
        value.validate()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc
    return value


def _normalized_state_dict(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_STATE_REQUIRED")
    out: dict[str, torch.Tensor] = {}
    for raw_name, raw_tensor in state.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_STATE_KEY_INVALID")
        if not isinstance(raw_tensor, torch.Tensor):
            raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_STATE_TENSOR_REQUIRED:" + raw_name)
        if not bool(torch.isfinite(raw_tensor).all()):
            raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_STATE_NONFINITE:" + raw_name)
        out[raw_name] = raw_tensor
    if len(out) != len(state):
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_STATE_KEY_COLLISION")
    return out


def state_schema_v1(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = _normalized_state_dict(state)
    return [
        {
            "name": name,
            "dtype": str(normalized[name].dtype),
            "shape": list(normalized[name].shape),
            "layout": str(normalized[name].layout),
        }
        for name in sorted(normalized)
    ]


def state_schema_sha256_v1(state: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {"schema": "RealSaS.IRISVNextEvidenceModelV3StateSchema.v1", "tensors": state_schema_v1(state)}
    )


def architecture_contract_payload_v1(
    *,
    config: IRISVNextConfigV1,
    relation_config: GraphicsRelationHeadV3Config,
    state_schema_sha256: str,
) -> dict[str, Any]:
    config.validate()
    relation_config.validate()
    return {
        "schema": ARCHITECTURE_CONTRACT_SCHEMA_V1,
        "architecture_id": IRISVNextEvidenceModelV3.architecture_id,
        "output_semantics": IRISVNextEvidenceModelV3.output_semantics,
        "topology_decoder_embedded": IRISVNextEvidenceModelV3.topology_decoder_embedded,
        "uncalibrated_uncertainty_public_output": IRISVNextEvidenceModelV3.uncalibrated_uncertainty_public_output,
        "config": asdict(config),
        "relation_config": asdict(relation_config),
        "state_schema_sha256": _lower_sha256(
            state_schema_sha256, "IRIS_VNEXT_PRODUCT_STATE_SCHEMA_SHA_INVALID"
        ),
    }


def architecture_contract_sha256_v1(
    *,
    config: IRISVNextConfigV1,
    relation_config: GraphicsRelationHeadV3Config,
    state_schema_sha256: str,
) -> str:
    return _canonical_sha256(
        architecture_contract_payload_v1(
            config=config,
            relation_config=relation_config,
            state_schema_sha256=state_schema_sha256,
        )
    )


def _assert_exact_product_model(model: IRISVNextEvidenceModelV3) -> None:
    if type(model) is not IRISVNextEvidenceModelV3:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_EXPORT_MODEL_TYPE_INVALID")
    if model.training:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_EXPORT_MODEL_MUST_BE_EVAL")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("IRIS_VNEXT_PRODUCT_EXPORT_MODEL_MUST_BE_FROZEN")
    if not isinstance(model.config, IRISVNextConfigV1):
        raise RuntimeError("IRIS_VNEXT_PRODUCT_EXPORT_CONFIG_INVALID")
    if not isinstance(model.relations.relation_config, GraphicsRelationHeadV3Config):
        raise RuntimeError("IRIS_VNEXT_PRODUCT_EXPORT_RELATION_CONFIG_INVALID")


def product_checkpoint_payload_v1(model: IRISVNextEvidenceModelV3) -> dict[str, Any]:
    """Build the only admitted product payload for the current topology-factored V3 model.

    Training state, optimizer state, RNG state, teacher data and legacy V1/V8/V9
    envelopes are intentionally absent. The single ModelRelease artifact therefore
    describes only executable model behavior.
    """

    _assert_exact_product_model(model)
    model_state = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    state_sha = state_schema_sha256_v1(model_state)
    architecture_sha = architecture_contract_sha256_v1(
        config=model.config,
        relation_config=model.relations.relation_config,
        state_schema_sha256=state_sha,
    )
    return {
        "schema": PRODUCT_CHECKPOINT_SCHEMA_V1,
        "architecture_id": IRISVNextEvidenceModelV3.architecture_id,
        "config": asdict(model.config),
        "relation_config": asdict(model.relations.relation_config),
        "state_schema_sha256": state_sha,
        "architecture_contract_sha256": architecture_sha,
        "model_state": model_state,
    }


def save_product_checkpoint_v1(
    model: IRISVNextEvidenceModelV3,
    path: Path,
) -> IRISVNextProductCheckpointSealV1:
    payload = product_checkpoint_payload_v1(model)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return IRISVNextProductCheckpointSealV1(
        content_sha256=_file_sha256(path),
        state_schema_sha256=str(payload["state_schema_sha256"]),
        architecture_contract_sha256=str(payload["architecture_contract_sha256"]),
        size_bytes=path.stat().st_size,
    )


def _validate_payload_v1(payload: Any) -> tuple[IRISVNextConfigV1, GraphicsRelationHeadV3Config, dict[str, torch.Tensor], str, str]:
    if not isinstance(payload, dict):
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_PAYLOAD_INVALID")
    expected_keys = {
        "schema",
        "architecture_id",
        "config",
        "relation_config",
        "state_schema_sha256",
        "architecture_contract_sha256",
        "model_state",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_FIELDS_INVALID")
    if payload.get("schema") != PRODUCT_CHECKPOINT_SCHEMA_V1:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_SCHEMA_INVALID")
    if payload.get("architecture_id") != IRISVNextEvidenceModelV3.architecture_id:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_ARCHITECTURE_INVALID")

    config = _typed_dataclass_from_mapping(
        IRISVNextConfigV1,
        payload.get("config"),
        "IRIS_VNEXT_PRODUCT_CHECKPOINT_CONFIG_INVALID",
    )
    relation_config = _typed_dataclass_from_mapping(
        GraphicsRelationHeadV3Config,
        payload.get("relation_config"),
        "IRIS_VNEXT_PRODUCT_CHECKPOINT_RELATION_CONFIG_INVALID",
    )
    state = _normalized_state_dict(payload.get("model_state"))
    actual_state_sha = state_schema_sha256_v1(state)
    declared_state_sha = _lower_sha256(
        payload.get("state_schema_sha256"),
        "IRIS_VNEXT_PRODUCT_STATE_SCHEMA_SHA_INVALID",
    )
    if declared_state_sha != actual_state_sha:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_STATE_SCHEMA_DRIFT")

    actual_architecture_sha = architecture_contract_sha256_v1(
        config=config,
        relation_config=relation_config,
        state_schema_sha256=actual_state_sha,
    )
    declared_architecture_sha = _lower_sha256(
        payload.get("architecture_contract_sha256"),
        "IRIS_VNEXT_PRODUCT_ARCHITECTURE_CONTRACT_SHA_INVALID",
    )
    if declared_architecture_sha != actual_architecture_sha:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_ARCHITECTURE_CONTRACT_DRIFT")
    return config, relation_config, state, actual_state_sha, actual_architecture_sha


def load_product_checkpoint_v1(
    path: Path,
    *,
    expected_content_sha256: str,
    device: torch.device | str = "cpu",
) -> LoadedIRISVNextProductModelV1:
    """Load only the current V3 product checkpoint format with a strict state court."""

    path = path.resolve()
    expected_content_sha256 = _lower_sha256(
        expected_content_sha256,
        "IRIS_VNEXT_PRODUCT_CHECKPOINT_CONTENT_SHA_INVALID",
    )
    if not path.is_file():
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_MISSING")
    actual_content_sha = _file_sha256(path)
    if actual_content_sha != expected_content_sha256:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_CHECKPOINT_CONTENT_DRIFT")

    payload = torch.load(path, map_location="cpu", weights_only=True)
    config, relation_config, state, state_sha, architecture_sha = _validate_payload_v1(payload)

    model = IRISVNextEvidenceModelV3(config, relation_config)
    fresh_state_sha = state_schema_sha256_v1(model.state_dict())
    if fresh_state_sha != state_sha:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_RUNTIME_STATE_SCHEMA_DRIFT")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError("IRIS_VNEXT_PRODUCT_STRICT_STATE_LOAD_FAILED") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)

    loaded = LoadedIRISVNextProductModelV1(
        model=model,
        content_sha256=actual_content_sha,
        state_schema_sha256=state_sha,
        architecture_contract_sha256=architecture_sha,
    )
    loaded.assert_inference_sealed()
    return loaded
