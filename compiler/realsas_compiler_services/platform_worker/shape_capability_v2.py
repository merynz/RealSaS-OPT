from __future__ import annotations

"""ShapeProposal v2: learned scene evidence plus deterministic decoded topology.

V1 remains replayable. V2 runs the exact existing IRIS->TESSA learned path once,
then decodes the already-produced token sequence with the frozen TESSA codec.
The decoded topology is learned proposal evidence in the TESSA scene frame. It
is not Compiler/world-frame geometry and cannot mint canonical Mesh authority.

This module is intentionally Torch-free at import time. Learned Shape and TESSA
package imports are performed only when the corresponding runtime work executes.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from typing import Any

import numpy as np


SHAPE_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductShapeInference.v2"
SHAPE_PROPOSAL_SCHEMA_V2 = "RealSaS.ShapeProposal.v2"
SHAPE_OUTPUT_AUTHORITY_V2 = "LEARNED_SHAPE_PROPOSAL_ONLY"
TESSA_EVIDENCE_FRAME_V2 = "RealSaS.TESSA.SequenceNormalizedFrame.v1"
TESSA_COORDINATE_DOMAIN_V2 = "NORMALIZED_HALF_EXTENT__XYZ_IN_MINUS_0P5_PLUS_0P5"


@dataclass(frozen=True)
class ShapeInferenceResultV2:
    payload: bytes
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class DecodedShapeProposalV2:
    metadata: dict[str, Any]
    token_ids: np.ndarray
    decoded_vertices_tessa: np.ndarray
    decoded_faces: np.ndarray
    decoded_vertex_component_indices: np.ndarray
    decoded_face_component_indices: np.ndarray
    surface_xyz_tessa: np.ndarray
    surface_normal: np.ndarray
    surface_h: np.ndarray
    mechanical: dict[str, np.ndarray]
    contact_persistence_valid: np.ndarray


def _run_shape_inference_v1(*args, **kwargs):
    from compiler.realsas_compiler_services.platform_worker.shape_capability_v1 import run_shape_inference_v1

    return run_shape_inference_v1(*args, **kwargs)


def _decode_tessa_asset_sequence_v1(tokens):
    from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1

    return decode_tessa_asset_sequence_v1(tokens)


def _token_sequence_sha256(token_ids: np.ndarray) -> str:
    value = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    return sha256(value.astype("<i8", copy=False).tobytes()).hexdigest()


def _decoded_topology_sha256(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_components: np.ndarray,
    face_components: np.ndarray,
) -> str:
    digest = sha256()
    for array, dtype in (
        (vertices, "<f8"),
        (faces, "<i8"),
        (vertex_components, "<i8"),
        (face_components, "<i8"),
    ):
        value = np.asarray(array).astype(dtype, copy=False)
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _read_v1_payload(payload: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].reshape(-1)[0]))
        return {
            "metadata": metadata,
            "token_ids": np.asarray(bundle["token_ids"], dtype=np.int64).copy(),
            "surface_xyz_tessa": np.asarray(bundle["surface_xyz_tessa"], dtype=np.float32).copy(),
            "surface_normal": np.asarray(bundle["surface_normal"], dtype=np.float32).copy(),
            "surface_h": np.asarray(bundle["surface_h"], dtype=np.float32).copy(),
            "stretch_major": np.asarray(bundle["stretch_major"], dtype=np.float32).copy(),
            "stretch_minor": np.asarray(bundle["stretch_minor"], dtype=np.float32).copy(),
            "tangent_direction_2theta": np.asarray(bundle["tangent_direction_2theta"], dtype=np.float32).copy(),
            "bend_demand": np.asarray(bundle["bend_demand"], dtype=np.float32).copy(),
            "rigidity_probability": np.asarray(bundle["rigidity_probability"], dtype=np.float32).copy(),
            "contact_persistence_probability": np.asarray(bundle["contact_persistence_probability"], dtype=np.float32).copy(),
            "contact_persistence_valid": np.asarray(bundle["contact_persistence_valid"], dtype=np.uint8).copy(),
        }


def encode_shape_proposal_v2(v1: Any) -> bytes:
    source = _read_v1_payload(bytes(v1.payload))
    token_ids = source["token_ids"]
    decoded = _decode_tessa_asset_sequence_v1(token_ids.tolist())
    vertices = np.asarray(decoded.vertices_normalized, dtype=np.float64)
    faces = np.asarray(decoded.faces, dtype=np.int64)
    vertex_components = np.asarray(decoded.vertex_component_indices, dtype=np.int64)
    face_components = np.asarray(decoded.face_component_indices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_FACES_INVALID")
    if len(vertex_components) != len(vertices) or len(face_components) != len(faces):
        raise RuntimeError("SHAPE_V2_DECODED_COMPONENT_ACCOUNTING_INVALID")
    if not np.isfinite(vertices).all():
        raise RuntimeError("SHAPE_V2_DECODED_VERTICES_NONFINITE")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise RuntimeError("SHAPE_V2_DECODED_FACE_INDEX_INVALID")

    topology_sha = _decoded_topology_sha256(vertices, faces, vertex_components, face_components)
    metadata = dict(source["metadata"])
    metadata.update(
        {
            "schema": SHAPE_PROPOSAL_SCHEMA_V2,
            "authority": SHAPE_OUTPUT_AUTHORITY_V2,
            "canonical_mesh_authority": False,
            "mesh_authority": False,
            "compiler_frame_authority": False,
            "compiler_frame_binding_required": True,
            "source_geometry_lineage_claimed": False,
            "decoded_topology_is_proposal_evidence": True,
            "decoded_topology_codec": "RealSaS.TESSASequenceCodec.v1",
            "proposal_evidence_frame": TESSA_EVIDENCE_FRAME_V2,
            "tessa_coordinate_domain": TESSA_COORDINATE_DOMAIN_V2,
            "topology_sequence_sha256": _token_sequence_sha256(token_ids),
            "decoded_topology_sha256": topology_sha,
            "decoded_vertex_count": int(len(vertices)),
            "decoded_face_count": int(len(faces)),
            "decoded_component_count": int(decoded.component_count),
            "decoded_restart_count": int(decoded.restart_count),
            "source_truth_copied_into_shape": False,
        }
    )
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
        token_ids=token_ids,
        decoded_vertices_tessa=vertices,
        decoded_faces=faces,
        decoded_vertex_component_indices=vertex_components,
        decoded_face_component_indices=face_components,
        surface_xyz_tessa=source["surface_xyz_tessa"],
        surface_normal=source["surface_normal"],
        surface_h=source["surface_h"],
        stretch_major=source["stretch_major"],
        stretch_minor=source["stretch_minor"],
        tangent_direction_2theta=source["tangent_direction_2theta"],
        bend_demand=source["bend_demand"],
        rigidity_probability=source["rigidity_probability"],
        contact_persistence_probability=source["contact_persistence_probability"],
        contact_persistence_valid=source["contact_persistence_valid"],
    )
    return stream.getvalue()


def decode_shape_proposal_v2(payload: bytes) -> DecodedShapeProposalV2:
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            required = {
                "metadata_json", "token_ids", "decoded_vertices_tessa", "decoded_faces",
                "decoded_vertex_component_indices", "decoded_face_component_indices",
                "surface_xyz_tessa", "surface_normal", "surface_h", "stretch_major",
                "stretch_minor", "tangent_direction_2theta", "bend_demand",
                "rigidity_probability", "contact_persistence_probability",
                "contact_persistence_valid",
            }
            if set(bundle.files) != required:
                raise RuntimeError("SHAPE_V2_FIELDS_DRIFT")
            metadata = json.loads(str(bundle["metadata_json"].reshape(-1)[0]))
            value = DecodedShapeProposalV2(
                metadata=metadata,
                token_ids=np.asarray(bundle["token_ids"], dtype=np.int64).copy(),
                decoded_vertices_tessa=np.asarray(bundle["decoded_vertices_tessa"], dtype=np.float64).copy(),
                decoded_faces=np.asarray(bundle["decoded_faces"], dtype=np.int64).copy(),
                decoded_vertex_component_indices=np.asarray(bundle["decoded_vertex_component_indices"], dtype=np.int64).copy(),
                decoded_face_component_indices=np.asarray(bundle["decoded_face_component_indices"], dtype=np.int64).copy(),
                surface_xyz_tessa=np.asarray(bundle["surface_xyz_tessa"], dtype=np.float32).copy(),
                surface_normal=np.asarray(bundle["surface_normal"], dtype=np.float32).copy(),
                surface_h=np.asarray(bundle["surface_h"], dtype=np.float32).copy(),
                mechanical={
                    key: np.asarray(bundle[key], dtype=np.float32).copy()
                    for key in (
                        "stretch_major", "stretch_minor", "tangent_direction_2theta",
                        "bend_demand", "rigidity_probability", "contact_persistence_probability",
                    )
                },
                contact_persistence_valid=np.asarray(bundle["contact_persistence_valid"], dtype=np.uint8).copy(),
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("SHAPE_V2_DECODE_FAILED") from exc

    if value.metadata.get("schema") != SHAPE_PROPOSAL_SCHEMA_V2:
        raise RuntimeError("SHAPE_V2_SCHEMA_DRIFT")
    if value.metadata.get("authority") != SHAPE_OUTPUT_AUTHORITY_V2:
        raise RuntimeError("SHAPE_V2_AUTHORITY_DRIFT")
    if value.metadata.get("canonical_mesh_authority") is not False or value.metadata.get("mesh_authority") is not False:
        raise RuntimeError("SHAPE_V2_MESH_AUTHORITY_FORBIDDEN")
    if value.metadata.get("compiler_frame_authority") is not False:
        raise RuntimeError("SHAPE_V2_COMPILER_FRAME_AUTHORITY_FORBIDDEN")
    if value.metadata.get("compiler_frame_binding_required") is not True:
        raise RuntimeError("SHAPE_V2_COMPILER_FRAME_BINDING_REQUIRED")
    if value.metadata.get("source_geometry_lineage_claimed") is not False:
        raise RuntimeError("SHAPE_V2_SOURCE_GEOMETRY_LINEAGE_CLAIM_FORBIDDEN")
    if value.metadata.get("proposal_evidence_frame") != TESSA_EVIDENCE_FRAME_V2:
        raise RuntimeError("SHAPE_V2_EVIDENCE_FRAME_DRIFT")
    if value.metadata.get("topology_sequence_sha256") != _token_sequence_sha256(value.token_ids):
        raise RuntimeError("SHAPE_V2_TOPOLOGY_SEQUENCE_HASH_DRIFT")
    actual_topology_sha = _decoded_topology_sha256(
        value.decoded_vertices_tessa,
        value.decoded_faces,
        value.decoded_vertex_component_indices,
        value.decoded_face_component_indices,
    )
    if value.metadata.get("decoded_topology_sha256") != actual_topology_sha:
        raise RuntimeError("SHAPE_V2_DECODED_TOPOLOGY_HASH_DRIFT")
    return value


def run_shape_inference_v2(*args, **kwargs) -> ShapeInferenceResultV2:
    v1 = _run_shape_inference_v1(*args, **kwargs)
    payload = encode_shape_proposal_v2(v1)
    diagnostics = dict(v1.diagnostics)
    diagnostics.update(
        {
            "schema": "RealSaS.ShapeInferenceDiagnostics.v2",
            "output_schema": SHAPE_PROPOSAL_SCHEMA_V2,
            "decoded_topology_serialized": True,
            "canonical_mesh_authority": False,
            "compiler_frame_authority": False,
        }
    )
    return ShapeInferenceResultV2(payload=payload, diagnostics=diagnostics)


__all__ = [
    "DecodedShapeProposalV2",
    "ShapeInferenceResultV2",
    "SHAPE_EXECUTOR_CONTRACT_V2",
    "SHAPE_PROPOSAL_SCHEMA_V2",
    "decode_shape_proposal_v2",
    "encode_shape_proposal_v2",
    "run_shape_inference_v2",
]
