from __future__ import annotations

"""Lossless, immutable RSS.vNext neural carrier beside Compiler evidence.

This is the unified checkpoint's OWN state, not the legacy independent IRIS V3
ShapeInference worker. Compiler decisions are derived artifacts; they never
replace the original neural carrier. A topology ABSTAIN still preserves this
scene for independent Semantic, Graphics and Mechanics consumers.

Source RGBA and source cameras are stored separately in the canonical source
archive. This record includes every model-owned RSSvNextSceneState tensor
(except those two original source arrays), preserving float/BF16 bytes and
all geometry field names. It is neither source truth nor product authority.
"""

from hashlib import sha256
import io
import json
import re
from typing import Any, Mapping
from zipfile import ZipFile

import numpy as np
import torch
from torch import Tensor

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1,
    validate_inference_evidence_v1,
)

SCHEMA = "RealSaS.RSSvNext.NeuralSceneCarrier.v1"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16384
_REQUIRED = (
    "observed_image_tokens",
    "conditioned_image_tokens",
    "source_semantic_patch_logits",
    "source_semantic_log_uncertainty",
    "scene_triplanes",
    "query_xyz",
    "local_features",
    "semantic_logits",
    "semantic_evidence_logits",
    "presentation_role_logits",
    "presentation_role_probabilities",
    "affordance_logits",
    "graphics_logits",
    "region_slot_features",
    "region_relation_tokens",
    "region_relation_logits",
    "region_relation_valid_mask",
)
_OPTIONAL = (
    "constructed_camera_features",
    "constructed_camera_angles",
)
_KNOWN_DTYPES = {
    "torch.float16": "float16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.bfloat16": "uint16",
    "torch.int64": "int64",
    "torch.int32": "int32",
    "torch.bool": "bool",
}
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")


def _hex(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        x in "0123456789abcdef" for x in value
    )


def _array_from_tensor(name: str, tensor: Tensor) -> tuple[np.ndarray, str]:
    if type(tensor) is not Tensor and not isinstance(tensor, Tensor):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_NON_TENSOR:" + name)
    if tensor.requires_grad:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_UNDETACHED:" + name)
    dtype = str(tensor.dtype)
    if dtype not in _KNOWN_DTYPES:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_DTYPE_NOT_SUPPORTED:" + name)
    raw = tensor.detach().to(device="cpu").contiguous()
    if raw.is_floating_point() and not bool(torch.isfinite(raw).all()):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_NONFINITE:" + name)
    if dtype == "torch.bfloat16":
        # Exact storage of the *bits*; numpy does not support BF16.
        array = raw.view(torch.uint16).numpy().copy()
    else:
        array = raw.numpy().copy()
    return array, dtype


def _tensor_from_array(name: str, raw: np.ndarray, dtype: str) -> Tensor:
    if dtype not in _KNOWN_DTYPES or raw.dtype != np.dtype(_KNOWN_DTYPES[dtype]):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_DTYPE_DRIFT:" + name)
    if raw.size == 0:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_EMPTY_TENSOR:" + name)
    result = torch.from_numpy(raw.copy())
    if dtype == "torch.bfloat16":
        result = result.view(torch.bfloat16)
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_NONFINITE:" + name)
    return result


def _collect(state: Any) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for name in _REQUIRED:
        tensors[name] = getattr(state, name)
    for name in _OPTIONAL:
        value = getattr(state, name)
        if value is not None:
            tensors[name] = value
    geometry = getattr(state, "geometry")
    if not isinstance(geometry, Mapping) or not geometry:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_GEOMETRY_REQUIRED")
    for name, value in sorted(geometry.items()):
        if type(name) is not str or not _NAME.fullmatch(name):
            raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_GEOMETRY_NAME_INVALID")
        tensors["geometry__" + name] = value
    return tensors


def _shapes(tensors: Mapping[str, Tensor]) -> None:
    required = set(_REQUIRED)
    if not required.issubset(tensors) or not any(
        name.startswith("geometry__") for name in tensors
    ):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_MISSING_FIELDS")
    planes = tensors["scene_triplanes"]
    query = tensors["query_xyz"]
    h = tensors["local_features"]
    if (
        planes.ndim != 5 or planes.shape[0:2] != (1, 3)
        or min(planes.shape[2:]) < 1
        or query.ndim != 3 or query.shape[0] != 1 or query.shape[-1] != 3
        or h.ndim != 3 or h.shape[:2] != query.shape[:2]
        or h.shape[-1] < 1 or query.shape[1] < 1
        or tensors["graphics_logits"].shape != (*query.shape[:2], 5)
        or tensors["semantic_logits"].shape[:2] != query.shape[:2]
        or tensors["presentation_role_logits"].shape[:2] != query.shape[:2]
        or tensors["affordance_logits"].shape[:2] != query.shape[:2]
        or tensors["semantic_evidence_logits"].shape[:2] != query.shape[:2]
    ):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_SCENE_DOMAIN_INVALID")
    b, n = query.shape[:2]
    slots = tensors["region_slot_features"]
    mask = tensors["region_relation_valid_mask"]
    logits = tensors["region_relation_logits"]
    if (
        slots.ndim != 3 or slots.shape[:1] != (b,) or slots.shape[2] != h.shape[2]
        or mask.shape != (b, slots.shape[1], slots.shape[1])
        or mask.dtype != torch.bool
        or logits.ndim != 4 or logits.shape[:3] != mask.shape
        or tensors["region_relation_tokens"].shape != slots.shape
        or tensors["presentation_role_probabilities"].shape != tensors["presentation_role_logits"].shape
        or tensors["observed_image_tokens"].shape != tensors["conditioned_image_tokens"].shape
        or tensors["observed_image_tokens"].ndim != 4
        or tensors["observed_image_tokens"].shape[0:2] != (1, 2)
        or tensors["source_semantic_patch_logits"].shape[:3] != tensors["observed_image_tokens"].shape[:3]
        or tensors["source_semantic_log_uncertainty"].shape[:3] != tensors["observed_image_tokens"].shape[:3]
    ):
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_SEMANTIC_DOMAIN_INVALID")
    for name, value in tensors.items():
        if name.startswith("geometry__") and (
            value.ndim < 2 or value.shape[:2] != (b, n)
        ):
            raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_GEOMETRY_ADDRESS_DRIFT:" + name)


def _binding(evidence: RSSvNextInferenceEvidenceIRV1) -> dict[str, str]:
    validate_inference_evidence_v1(evidence)
    return {
        "checkpoint_sha256": evidence.checkpoint_sha256,
        "observed_front_sha256": evidence.observed_front_sha256,
        "observed_back_sha256": evidence.observed_back_sha256,
        "camera_binding_sha256": evidence.camera_binding_sha256,
        "inference_evidence_hash": evidence.evidence_hash,
    }


def encode_neural_scene_carrier_v1(
    state: Any, *, evidence: RSSvNextInferenceEvidenceIRV1,
) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    types: dict[str, str] = {}
    tensors = _collect(state)
    _shapes(tensors)
    for name, tensor in sorted(tensors.items()):
        array, dtype = _array_from_tensor(name, tensor)
        arrays[name], types[name] = array, dtype
    metadata = {
        "schema": SCHEMA,
        **_binding(evidence),
        "tensor_dtypes": types,
        "source_rgba_stored_separately": True,
        "canonical_mesh_authority": False,
        "source_visibility_authority": False,
        "independently_reusable_by_semantic_graphics_mechanics": True,
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray([json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False,
        )]),
        **arrays,
    )
    result = stream.getvalue()
    if len(result) > MAX_ARCHIVE_BYTES:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_ARCHIVE_BYTES_LIMIT")
    return result


def read_neural_scene_carrier_v1(
    payload: bytes, *, evidence: RSSvNextInferenceEvidenceIRV1,
) -> dict[str, Tensor]:
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_ARCHIVE_BYTES:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_ARCHIVE_BYTES_INVALID")
    try:
        with ZipFile(io.BytesIO(payload)) as zf:
            infos = zf.infolist()
            filenames = [row.filename for row in infos]
            if (
                len(infos) < len(_REQUIRED) + 2
                or len(filenames) != len(set(filenames))
                or any(not x.endswith(".npy") for x in filenames)
                or sum(x.file_size for x in infos) > MAX_ARCHIVE_BYTES
                or any(x.file_size > MAX_ARCHIVE_BYTES for x in infos)
            ):
                raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_ARCHIVE_RESOURCE_OR_DUPLICATE")
        with np.load(io.BytesIO(payload), allow_pickle=False) as zf:
            meta_raw = np.asarray(zf["metadata_json"])
            if meta_raw.shape != (1,) or len(str(meta_raw[0])) > _MAX_MANIFEST_BYTES:
                raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_MANIFEST_SHAPE")
            metadata = json.loads(str(meta_raw[0]))
            names = set(zf.files) - {"metadata_json"}
            if not isinstance(metadata, dict) or set(metadata) != {
                "schema", "checkpoint_sha256", "observed_front_sha256",
                "observed_back_sha256", "camera_binding_sha256",
                "inference_evidence_hash", "tensor_dtypes",
                "source_rgba_stored_separately", "canonical_mesh_authority",
                "source_visibility_authority",
                "independently_reusable_by_semantic_graphics_mechanics",
            }:
                raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_MANIFEST_FIELDS")
            if (
                metadata["schema"] != SCHEMA
                or {k: metadata[k] for k in _binding(evidence)} != _binding(evidence)
                or metadata["source_rgba_stored_separately"] is not True
                or metadata["canonical_mesh_authority"] is not False
                or metadata["source_visibility_authority"] is not False
                or metadata["independently_reusable_by_semantic_graphics_mechanics"] is not True
                or not isinstance(metadata["tensor_dtypes"], dict)
                or names != set(metadata["tensor_dtypes"])
                or any(
                    name not in _REQUIRED and name not in _OPTIONAL
                    and not (name.startswith("geometry__") and _NAME.fullmatch(name[10:]))
                    for name in names
                )
            ):
                raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_BINDING_OR_FIELDS_DRIFT")
            tensors = {
                name: _tensor_from_array(
                    name, np.asarray(zf[name]), metadata["tensor_dtypes"][name],
                )
                for name in sorted(names)
            }
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_ARCHIVE_DECODE_FAILED") from exc
    _shapes(tensors)
    # Cross-check all shared tensor fields with the separately qualified JSON.
    # A re-hashed but differently addressed scene may NOT masquerade as the
    # very same source/slot/Graphics prediction.
    if evidence.scene is not None:
        for tensor_key, json_key in (
            ("query_xyz", "region_query_xyz"),
            ("semantic_logits", "region_membership_logits"),
            ("graphics_logits", "region_graphics_logits"),
            ("region_relation_logits", "region_relation_logits"),
            ("region_relation_valid_mask", "region_relation_valid_mask"),
            ("presentation_role_probabilities", "presentation_role_probabilities"),
            ("source_semantic_patch_logits", "source_semantic_patch_logits"),
            ("source_semantic_log_uncertainty", "source_semantic_log_uncertainty"),
        ):
            if json_key in evidence.scene and (
                tensors[tensor_key][0].tolist() != evidence.scene[json_key]
            ):
                raise RuntimeError("RSS_VNEXT_NEURAL_CARRIER_SHARED_EVIDENCE_DRIFT:" + tensor_key)
    return tensors


__all__ = [
    "SCHEMA", "encode_neural_scene_carrier_v1",
    "read_neural_scene_carrier_v1",
]
