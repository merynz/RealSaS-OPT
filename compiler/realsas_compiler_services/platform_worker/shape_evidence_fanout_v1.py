from __future__ import annotations

"""Lossless, multi-consumer learned Shape evidence retention.

The Compiler may mint CanonicalMesh from qualified evidence, but it must never
replace the neural evidence available to other consumers with that Mesh.
This sidecar keeps the IRIS scene planes and original IRIS-frame witnesses
which the existing ShapeProposal.v2 omitted. The ShapeProposal remains an
independent CAS object with TESSA tokens, surface H, normals and mechanics.

Fan-out is an artifact index, NOT a sequential Mesh -> Rig -> Graphics pipe.
It grants no source, semantic, appearance, physics or product authority.
"""

from hashlib import sha256
import io
import json
from zipfile import ZipFile
from typing import Any, Mapping

import numpy as np


SCENE_SCHEMA = "RealSaS.IRISSceneEvidence.v1"
INDEX_SCHEMA = "RealSaS.ShapeEvidenceFanout.v1"
MAX_SCENE_BYTES = 512 * 1024 * 1024
_SCENE_VALUES = ("scene_planes", "surface_xyz_iris", "surface_xyz_tessa")
_DTYPE_RECEIPTS = tuple(f"{name}_source_dtype" for name in _SCENE_VALUES)
_EXTRACTION_COUNTS = ("extracted_vertex_counts", "extracted_face_counts")
_SUPPORTED_SCENE_DTYPE_RECEIPTS = frozenset({
    "float32", "float16", "torch.float32", "torch.float16", "torch.bfloat16",
})


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _tensor_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        detached = value.detach()
        # BF16 has no native NumPy scalar; BF16/FP16 -> FP32 is exact
        # (all finite source values are representable). Never downcast FP64
        # or silently truncate model-produced scene evidence.
        if str(detached.dtype) in ("torch.bfloat16", "torch.float16"):
            detached = detached.float()
        value = detached.cpu().contiguous().numpy()
    array = np.asarray(value)
    if array.dtype == np.float16:
        return array.astype(np.float32)
    return array


def _scene_arrays(raw: Mapping[str, np.ndarray]) -> None:
    planes = np.asarray(raw["scene_planes"])
    iris = np.asarray(raw["surface_xyz_iris"])
    normalized = np.asarray(raw["surface_xyz_tessa"])
    if (planes.ndim != 5 or planes.shape[0:2] != (1, 3)
        or any(d < 1 for d in planes.shape[2:])):
        raise RuntimeError("SHAPE_FANOUT_SCENE_PLANE_SHAPE_INVALID")
    if (
        iris.ndim != 3 or iris.shape[0] != 1 or iris.shape[2] != 3
        or iris.shape[1] < 1 or iris.shape[1] > 500000
        or normalized.shape != iris.shape
    ):
        raise RuntimeError("SHAPE_FANOUT_WITNESS_SHAPE_INVALID")
    if any(v.dtype != np.float32 or not np.isfinite(v).all()
           for v in (planes, iris, normalized)):
        raise RuntimeError("SHAPE_FANOUT_SCENE_NUMERIC_INVALID")
    for name in _EXTRACTION_COUNTS:
        counts = np.asarray(raw[name])
        if (
            counts.shape != (1,) or counts.dtype != np.int64
            or np.any(counts < 1) or np.any(counts > 10_000_000)
        ):
            raise RuntimeError("SHAPE_FANOUT_EXTRACTION_COUNTS_INVALID:" + name)
    for name in _DTYPE_RECEIPTS:
        dtype_receipt = np.asarray(raw[name])
        if (
            dtype_receipt.shape != (1,)
            or str(dtype_receipt[0]) not in _SUPPORTED_SCENE_DTYPE_RECEIPTS
        ):
            raise RuntimeError("SHAPE_FANOUT_SOURCE_DTYPE_RECEIPT_INVALID:" + name)
    if np.any(np.abs(normalized) > 0.500001):
        raise RuntimeError("SHAPE_FANOUT_TESSA_FRAME_INVALID")
    if not np.array_equal(
        normalized, np.clip(iris * np.float32(0.5), -0.5, 0.5)
    ):
        raise RuntimeError("SHAPE_FANOUT_IRIS_TO_TESSA_FRAME_DRIFT")


def encode_iris_scene_evidence_v1(surface: Any) -> bytes:
    arrays = {
        "scene_planes": _tensor_array(surface.scene_planes),
        "surface_xyz_iris": _tensor_array(surface.surface_xyz_iris),
        "surface_xyz_tessa": _tensor_array(surface.surface_xyz_tessa),
        "extracted_vertex_counts": np.asarray(
            surface.extracted_vertex_counts, dtype=np.int64,
        ),
        "extracted_face_counts": np.asarray(
            surface.extracted_face_counts, dtype=np.int64,
        ),
        **{
            f"{name}_source_dtype": np.asarray(
                [str(getattr(getattr(surface, name), "dtype", ""))]
            )
            for name in _SCENE_VALUES
        },
    }
    _scene_arrays(arrays)
    stream = io.BytesIO()
    np.savez_compressed(stream, schema=np.asarray([SCENE_SCHEMA]), **arrays)
    result = stream.getvalue()
    if len(result) > MAX_SCENE_BYTES:
        raise RuntimeError("SHAPE_FANOUT_SCENE_ARCHIVE_LIMIT")
    return result


def decode_iris_scene_evidence_v1(data: bytes, *, shape: Any) -> dict[str, np.ndarray]:
    if type(data) is not bytes or not data or len(data) > MAX_SCENE_BYTES:
        raise RuntimeError("SHAPE_FANOUT_SCENE_BYTES_INVALID")
    names = {"schema", *_SCENE_VALUES, *_DTYPE_RECEIPTS, *_EXTRACTION_COUNTS}
    try:
        with ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if (
                len(entries) != len(names)
                or {e.filename for e in entries} != {n + ".npy" for n in names}
                or sum(e.file_size for e in entries) > MAX_SCENE_BYTES
                or any(e.file_size > MAX_SCENE_BYTES for e in entries)
            ):
                raise RuntimeError("SHAPE_FANOUT_SCENE_ARCHIVE_FIELDS_OR_RESOURCE_LIMIT")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if str(archive["schema"].reshape(-1)[0]) != SCENE_SCHEMA:
                raise RuntimeError("SHAPE_FANOUT_SCENE_SCHEMA_DRIFT")
            arrays = {n: np.asarray(archive[n]).copy() for n in names - {"schema"}}
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SHAPE_FANOUT_SCENE_ARCHIVE_INVALID") from exc
    _scene_arrays(arrays)
    if not np.array_equal(arrays["surface_xyz_tessa"], np.asarray(shape.surface_xyz_tessa)):
        raise RuntimeError("SHAPE_FANOUT_WITNESS_SHAPE_PROPOSAL_DRIFT")
    return arrays


def seal_shape_evidence_fanout_v1(
    *, shape_bytes: bytes, scene_bytes: bytes,
    source_observation_content_sha256: str, shape_input_content_sha256: str,
) -> bytes:
    from .shape_capability_v2 import decode_shape_proposal_v2
    shape = decode_shape_proposal_v2(shape_bytes)
    decode_iris_scene_evidence_v1(scene_bytes, shape=shape)
    if (
        not _sha(source_observation_content_sha256)
        or not _sha(shape_input_content_sha256)
        or shape.metadata.get("source_observation_content_sha256") != source_observation_content_sha256
    ):
        raise RuntimeError("SHAPE_FANOUT_PARENT_BINDING_INVALID")
    index = {
        "schema": INDEX_SCHEMA,
        "source_observation_content_sha256": source_observation_content_sha256,
        "shape_input_content_sha256": shape_input_content_sha256,
        "shape_proposal_content_sha256": _digest(shape_bytes),
        "iris_scene_evidence_content_sha256": _digest(scene_bytes),
        "topology_evidence_sha256": shape.metadata["decoded_topology_sha256"],
        "surface_evidence_sha256": shape.metadata["surface_evidence_sha256"],
        "retained_evidence_families": [
            "TESSA_TOPOLOGY_TOKENS_AND_DECODE",
            "IRIS_SURFACE_XYZ_NORMALS_H_AND_MECHANICS",
            "IRIS_SCENE_PLANES_AND_UNCLIPPED_WITNESS_FRAME",
        ],
        "consumers_are_independent": True,
        "compiler_canonicalization_does_not_delete_parent_evidence": True,
        "source_truth_or_product_pass_authority": False,
    }
    return (json.dumps(index, sort_keys=True, separators=(",", ":")) + "\n").encode("utf8")


def validate_fanout_index_refs_v1(
    index_bytes: bytes, *,
    shape: Any,
    shape_content_sha256: str,
    scene_content_sha256: str,
    source_observation_content_sha256: str,
) -> dict[str, Any]:
    """Cheap Mesh boundary: verify all named CAS parents without loading scene.

    A full IRIS scene can be large. Mesh needs only TESSA topology and the
    bounded witness/Mechanical family already in ShapeProposal, so it should
    NOT decompress the scene merely to establish that its separate CAS object
    remains reachable by downstream Semantic/Graphics/AXIS consumers.
    """
    if type(index_bytes) is not bytes or not 1 <= len(index_bytes) <= 8192:
        raise RuntimeError("SHAPE_FANOUT_INDEX_BYTES_INVALID")
    try:
        value = json.loads(index_bytes)
    except Exception as exc:
        raise RuntimeError("SHAPE_FANOUT_INDEX_JSON_INVALID") from exc
    if type(value) is not dict:
        raise RuntimeError("SHAPE_FANOUT_INDEX_OBJECT_REQUIRED")
    if index_bytes != (json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ) + "\n").encode("utf8"):
        raise RuntimeError("SHAPE_FANOUT_INDEX_NONCANONICAL")
    expected = {
        "schema", "source_observation_content_sha256",
        "shape_input_content_sha256", "shape_proposal_content_sha256",
        "iris_scene_evidence_content_sha256", "topology_evidence_sha256",
        "surface_evidence_sha256", "retained_evidence_families",
        "consumers_are_independent",
        "compiler_canonicalization_does_not_delete_parent_evidence",
        "source_truth_or_product_pass_authority",
    }
    if set(value) != expected or value.get("schema") != INDEX_SCHEMA:
        raise RuntimeError("SHAPE_FANOUT_INDEX_SCHEMA_OR_FIELDS_DRIFT")
    for key in (
        "source_observation_content_sha256", "shape_input_content_sha256",
        "shape_proposal_content_sha256", "iris_scene_evidence_content_sha256",
        "topology_evidence_sha256", "surface_evidence_sha256",
    ):
        if not _sha(value.get(key)):
            raise RuntimeError("SHAPE_FANOUT_INDEX_DIGEST_INVALID:" + key)
    if (
        not all(map(_sha, (
            shape_content_sha256, scene_content_sha256,
            source_observation_content_sha256,
        )))
        or value["shape_proposal_content_sha256"] != shape_content_sha256
        or value["iris_scene_evidence_content_sha256"] != scene_content_sha256
        or value["source_observation_content_sha256"] != source_observation_content_sha256
        or shape.metadata.get("source_observation_content_sha256") != source_observation_content_sha256
        or shape.metadata.get("decoded_topology_sha256") != value["topology_evidence_sha256"]
        or shape.metadata.get("surface_evidence_sha256") != value["surface_evidence_sha256"]
    ):
        raise RuntimeError("SHAPE_FANOUT_INDEX_CAS_PARENT_DRIFT")
    if (
        value["retained_evidence_families"] != [
            "TESSA_TOPOLOGY_TOKENS_AND_DECODE",
            "IRIS_SURFACE_XYZ_NORMALS_H_AND_MECHANICS",
            "IRIS_SCENE_PLANES_AND_UNCLIPPED_WITNESS_FRAME",
        ]
        or value["consumers_are_independent"] is not True
        or value["compiler_canonicalization_does_not_delete_parent_evidence"] is not True
        or value["source_truth_or_product_pass_authority"] is not False
    ):
        raise RuntimeError("SHAPE_FANOUT_INDEX_UNSUPPORTED_AUTHORITY")
    return value


def read_shape_evidence_fanout_v1(
    index_bytes: bytes, *, shape_bytes: bytes, scene_bytes: bytes,
    source_observation_content_sha256: str,
    shape_input_content_sha256: str,
) -> dict[str, Any]:
    if type(index_bytes) is not bytes or len(index_bytes) > 8192:
        raise RuntimeError("SHAPE_FANOUT_INDEX_BYTES_INVALID")
    try:
        index = json.loads(index_bytes)
    except Exception as exc:
        raise RuntimeError("SHAPE_FANOUT_INDEX_JSON_INVALID") from exc
    if not isinstance(index, dict):
        raise RuntimeError("SHAPE_FANOUT_INDEX_OBJECT_REQUIRED")
    canonical = seal_shape_evidence_fanout_v1(
        shape_bytes=shape_bytes, scene_bytes=scene_bytes,
        source_observation_content_sha256=source_observation_content_sha256,
        shape_input_content_sha256=shape_input_content_sha256,
    )
    if index_bytes != canonical:
        raise RuntimeError("SHAPE_FANOUT_INDEX_OR_PARENT_DRIFT")
    return index


__all__ = [
    "SCENE_SCHEMA", "INDEX_SCHEMA",
    "encode_iris_scene_evidence_v1", "decode_iris_scene_evidence_v1",
    "seal_shape_evidence_fanout_v1", "read_shape_evidence_fanout_v1",
    "validate_fanout_index_refs_v1",
]
