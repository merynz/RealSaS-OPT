from __future__ import annotations

"""ShapeProposal v2: learned scene evidence plus deterministic decoded topology.

Shape remains proposal-only.  V2 serializes the exact TESSA decoded carrier that
Mesh needs so the later Compiler boundary does not have to reinterpret learned
model internals or depend on a hidden extractor mesh.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from time import perf_counter
from typing import Any

import numpy as np
import torch

from compiler.realsas_compiler_services.platform_worker.iris_inference_v1 import (
    encode_scene_v1,
    load_iris_vnext_inference_v1,
)
from compiler.realsas_compiler_services.platform_worker.tessa_inference_v1 import (
    TESSAInferencePolicyV1,
    build_inputs_from_iris_v1,
    generate_topology_proposal_v1,
    load_tessa_v3_inference_v1,
)
from models.iris.vnext.inference_input_v1 import IRISVNextSceneInputsV1
from models.iris.vnext.surface_v1 import extract_surface_evidence_v1
from models.tessa.v1.sequence_codec_v1 import decode_tessa_asset_sequence_v1


SHAPE_EXECUTOR_CONTRACT_V2 = "RealSaS.ProductShapeInference.v2"
SHAPE_PROPOSAL_SCHEMA_V2 = "RealSaS.ShapeProposal.v2"
SHAPE_OUTPUT_ARTIFACT_TYPE_V2 = "RealSaS.ShapeProposal"
SHAPE_OUTPUT_AUTHORITY_V2 = "LEARNED_SHAPE_PROPOSAL_ONLY"
TESSA_COORDINATE_DOMAIN_V2 = "NORMALIZED_REALSAS_OBJECT_FRAME_XYZ_MINUS_0P5_PLUS_0P5"
MESH_OBJECT_FRAME_TRANSFORM_V2 = "REALSAS_OBJECT_FRAME_XYZ_EQUALS_TESSA_XYZ_TIMES_2"


@dataclass(frozen=True)
class ShapeInferenceResultV2:
    payload: bytes
    diagnostics: dict[str, Any]
    schema: str = "RealSaS.ShapeInferenceResult.v2"


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


def _tensor_numpy(value: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    out = value.detach().cpu().contiguous().numpy()
    return out if dtype is None else out.astype(dtype, copy=False)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(header + b"\x00" + array.tobytes()).hexdigest()


def _decoded_geometry_sha256(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_components: np.ndarray,
    face_components: np.ndarray,
) -> str:
    payload = "|".join(
        (
            _array_sha256(vertices),
            _array_sha256(faces),
            _array_sha256(vertex_components),
            _array_sha256(face_components),
        )
    ).encode("ascii")
    return sha256(payload).hexdigest()


def _token_sequence_sha256(token_ids: np.ndarray) -> str:
    return _array_sha256(np.asarray(token_ids, dtype=np.int64).reshape(-1))


def _metadata(
    *,
    iris_binding: Any,
    tessa_binding: Any,
    proposal: Any,
    decoded: Any,
    geometry_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": SHAPE_PROPOSAL_SCHEMA_V2,
        "authority": SHAPE_OUTPUT_AUTHORITY_V2,
        "canonical_mesh_authority": False,
        "mesh_authority": False,
        "decoded_topology_is_proposal_evidence": True,
        "decoded_topology_codec": "RealSaS.TESSASequenceCodec.v1",
        "tessa_coordinate_domain": TESSA_COORDINATE_DOMAIN_V2,
        "mesh_object_frame_transform": MESH_OBJECT_FRAME_TRANSFORM_V2,
        "topology_sequence_sha256": _token_sequence_sha256(
            np.asarray(proposal.token_ids, dtype=np.int64)
        ),
        "decoded_geometry_sha256": geometry_sha256,
        "decoded_vertex_count": int(len(decoded.vertices_normalized)),
        "decoded_face_count": int(len(decoded.faces)),
        "decoded_component_count": int(decoded.component_count),
        "decoded_restart_count": int(decoded.restart_count),
        "orientation_qualified": bool(decoded.orientation_qualified),
        "topology_court": proposal.topology_court.to_dict(),
        "completed": bool(proposal.completed),
        "stop_reason": str(proposal.stop_reason),
        "source_truth_copied_into_shape": False,
        "temporary_iris_mesh_serialized": False,
        "models": {
            "IRIS": {
                "model_release_id": iris_binding.model_ref.model_release_id,
                "checkpoint_content_sha256": iris_binding.model_ref.checkpoint_content_sha256,
            },
            "TESSA": {
                "model_release_id": tessa_binding.model_ref.model_release_id,
                "checkpoint_content_sha256": tessa_binding.model_ref.checkpoint_content_sha256,
                "architecture_id": tessa_binding.model_ref.architecture_id,
            },
        },
    }


def encode_shape_proposal_v2(
    *,
    iris_binding: Any,
    tessa_binding: Any,
    surface: Any,
    proposal: Any,
    decoded: Any,
) -> bytes:
    vertices = np.asarray(decoded.vertices_normalized, dtype=np.float64)
    faces = np.asarray(decoded.faces, dtype=np.int64)
    vertex_components = np.asarray(decoded.vertex_component_indices, dtype=np.int64)
    face_components = np.asarray(decoded.face_component_indices, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_FACES_INVALID")
    if len(vertex_components) != len(vertices) or len(face_components) != len(faces):
        raise RuntimeError("SHAPE_V2_COMPONENT_ACCOUNTING_INVALID")
    if not np.isfinite(vertices).all() or np.any(vertices < -0.500001) or np.any(vertices > 0.500001):
        raise RuntimeError("SHAPE_V2_DECODED_VERTEX_DOMAIN_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("SHAPE_V2_DECODED_FACE_INDEX_INVALID")

    geometry_sha256 = _decoded_geometry_sha256(
        vertices, faces, vertex_components, face_components
    )
    metadata = _metadata(
        iris_binding=iris_binding,
        tessa_binding=tessa_binding,
        proposal=proposal,
        decoded=decoded,
        geometry_sha256=geometry_sha256,
    )
    mechanical = surface.mechanical
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray(
            [json.dumps(metadata, sort_keys=True, separators=(",", ":"))]
        ),
        token_ids=np.asarray(proposal.token_ids, dtype=np.int64),
        decoded_vertices_tessa=vertices,
        decoded_faces=faces,
        decoded_vertex_component_indices=vertex_components,
        decoded_face_component_indices=face_components,
        surface_xyz_tessa=_tensor_numpy(surface.surface_xyz_tessa, np.float32),
        surface_normal=_tensor_numpy(surface.surface_normal, np.float32),
        surface_h=_tensor_numpy(surface.surface_h, np.float32),
        stretch_major=_tensor_numpy(mechanical["stretch_major"], np.float32),
        stretch_minor=_tensor_numpy(mechanical["stretch_minor"], np.float32),
        tangent_direction_2theta=_tensor_numpy(
            mechanical["tangent_direction_2theta"], np.float32
        ),
        bend_demand=_tensor_numpy(mechanical["bend_demand"], np.float32),
        rigidity_probability=_tensor_numpy(
            mechanical["rigidity_probability"], np.float32
        ),
        contact_persistence_probability=_tensor_numpy(
            mechanical["contact_persistence_probability"], np.float32
        ),
        contact_persistence_valid=_tensor_numpy(
            surface.contact_persistence_valid.to(torch.uint8), np.uint8
        ),
    )
    return stream.getvalue()


def decode_shape_proposal_v2(payload: bytes) -> DecodedShapeProposalV2:
    if not isinstance(payload, (bytes, bytearray, memoryview)) or not payload:
        raise RuntimeError("SHAPE_V2_BYTES_REQUIRED")
    required = {
        "metadata_json",
        "token_ids",
        "decoded_vertices_tessa",
        "decoded_faces",
        "decoded_vertex_component_indices",
        "decoded_face_component_indices",
        "surface_xyz_tessa",
        "surface_normal",
        "surface_h",
        "stretch_major",
        "stretch_minor",
        "tangent_direction_2theta",
        "bend_demand",
        "rigidity_probability",
        "contact_persistence_probability",
        "contact_persistence_valid",
    }
    try:
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as bundle:
            if set(bundle.files) != required:
                raise RuntimeError("SHAPE_V2_FIELDS_DRIFT")
            metadata = json.loads(str(bundle["metadata_json"].reshape(-1)[0]))
            value = DecodedShapeProposalV2(
                metadata=dict(metadata),
                token_ids=np.asarray(bundle["token_ids"], dtype=np.int64).copy(),
                decoded_vertices_tessa=np.asarray(
                    bundle["decoded_vertices_tessa"], dtype=np.float64
                ).copy(),
                decoded_faces=np.asarray(bundle["decoded_faces"], dtype=np.int64).copy(),
                decoded_vertex_component_indices=np.asarray(
                    bundle["decoded_vertex_component_indices"], dtype=np.int64
                ).copy(),
                decoded_face_component_indices=np.asarray(
                    bundle["decoded_face_component_indices"], dtype=np.int64
                ).copy(),
                surface_xyz_tessa=np.asarray(
                    bundle["surface_xyz_tessa"], dtype=np.float32
                ).copy(),
                surface_normal=np.asarray(bundle["surface_normal"], dtype=np.float32).copy(),
                surface_h=np.asarray(bundle["surface_h"], dtype=np.float32).copy(),
                mechanical={
                    key: np.asarray(bundle[key], dtype=np.float32).copy()
                    for key in (
                        "stretch_major",
                        "stretch_minor",
                        "tangent_direction_2theta",
                        "bend_demand",
                        "rigidity_probability",
                        "contact_persistence_probability",
                    )
                },
                contact_persistence_valid=np.asarray(
                    bundle["contact_persistence_valid"], dtype=np.uint8
                ).copy(),
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
    if value.metadata.get("mesh_object_frame_transform") != MESH_OBJECT_FRAME_TRANSFORM_V2:
        raise RuntimeError("SHAPE_V2_OBJECT_FRAME_TRANSFORM_DRIFT")
    if value.metadata.get("topology_sequence_sha256") != _token_sequence_sha256(value.token_ids):
        raise RuntimeError("SHAPE_V2_TOPOLOGY_SEQUENCE_HASH_DRIFT")
    geometry_hash = _decoded_geometry_sha256(
        value.decoded_vertices_tessa,
        value.decoded_faces,
        value.decoded_vertex_component_indices,
        value.decoded_face_component_indices,
    )
    if value.metadata.get("decoded_geometry_sha256") != geometry_hash:
        raise RuntimeError("SHAPE_V2_DECODED_GEOMETRY_HASH_DRIFT")
    vertices = value.decoded_vertices_tessa
    faces = value.decoded_faces
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_VERTICES_INVALID")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise RuntimeError("SHAPE_V2_DECODED_FACES_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("SHAPE_V2_DECODED_FACE_INDEX_INVALID")
    if not np.isfinite(vertices).all() or np.any(vertices < -0.500001) or np.any(vertices > 0.500001):
        raise RuntimeError("SHAPE_V2_DECODED_VERTEX_DOMAIN_INVALID")
    return value


@torch.no_grad()
def run_shape_inference_v2(
    manifest: dict[str, Any],
    inputs: IRISVNextSceneInputsV1,
    *,
    extractor: Any,
    device: torch.device | str = "cpu",
    maximum_new_tokens: int = 131072,
) -> ShapeInferenceResultV2:
    if isinstance(maximum_new_tokens, bool) or int(maximum_new_tokens) <= 0 or int(maximum_new_tokens) > 131072:
        raise ValueError("SHAPE_V2_MAXIMUM_NEW_TOKENS_INVALID")
    inputs.validate()
    timings: dict[str, float] = {}
    total_started = perf_counter()

    started = perf_counter()
    iris_binding = load_iris_vnext_inference_v1(manifest, device=device)
    timings["iris_load_seconds"] = perf_counter() - started
    started = perf_counter()
    scene = encode_scene_v1(iris_binding, inputs)
    timings["iris_scene_seconds"] = perf_counter() - started
    started = perf_counter()
    surface = extract_surface_evidence_v1(
        iris_binding.model,
        scene,
        extractor,
        resolution=int(iris_binding.model.config.extraction_training_resolution),
        witness_count=int(iris_binding.model.config.surface_witness_count),
    )
    timings["surface_seconds"] = perf_counter() - started

    started = perf_counter()
    tessa_binding = load_tessa_v3_inference_v1(manifest, device=device)
    timings["tessa_load_seconds"] = perf_counter() - started
    started = perf_counter()
    tessa_inputs = build_inputs_from_iris_v1(iris_binding.model, surface)
    proposal = generate_topology_proposal_v1(
        tessa_binding,
        tessa_inputs,
        policy=TESSAInferencePolicyV1(maximum_new_tokens=int(maximum_new_tokens)),
    )
    timings["tessa_decode_seconds"] = perf_counter() - started

    if not proposal.completed:
        raise RuntimeError("SHAPE_V2_TOPOLOGY_PROPOSAL_INCOMPLETE:" + str(proposal.stop_reason))
    court = proposal.topology_court
    if not court.grammar_valid or court.decode_failure is not None or not court.finite_geometry or not court.within_scale_policy:
        raise RuntimeError("SHAPE_V2_TOPOLOGY_PROPOSAL_INVALID")
    decoded = decode_tessa_asset_sequence_v1(
        proposal.token_ids,
        cfg=tessa_binding.model.config,
    )
    payload = encode_shape_proposal_v2(
        iris_binding=iris_binding,
        tessa_binding=tessa_binding,
        surface=surface,
        proposal=proposal,
        decoded=decoded,
    )
    timings["total_seconds"] = perf_counter() - total_started
    return ShapeInferenceResultV2(
        payload=payload,
        diagnostics={
            "schema": "RealSaS.ShapeInferenceDiagnostics.v2",
            "authority": SHAPE_OUTPUT_AUTHORITY_V2,
            "canonical_mesh_authority": False,
            "decoded_topology_serialized": True,
            "decoded_vertex_count": int(len(decoded.vertices_normalized)),
            "decoded_face_count": int(len(decoded.faces)),
            "timings": {key: float(value) for key, value in timings.items()},
        },
    )
