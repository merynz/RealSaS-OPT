"""Package Truth139 free-running TESSA output for the existing static product.mesh consequence court.

The stable ShapeProposal.v2 envelope predates the Truth139 mechanics experiment and requires six
legacy mechanics arrays. Truth139 does not fabricate those semantics. For this *static mesh only*
research bridge they are explicit zero placeholders with a metadata abstention and all contact
validity false. The product.mesh static court consumes decoded topology plus IRIS surface witnesses,
normals and H; this artifact is forbidden from mechanics/skin/motion promotion.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_services.platform_worker.shape_capability_v2 import (
    decode_shape_proposal_v2,
    encode_shape_proposal_v2,
)
from .topology_generation_v1 import GeneratedTopologyMeshV1


SCOPE = "TRUTH139_STATIC_MESH_CONSEQUENCE_ONLY__MECHANICS_ABSTAIN"


@dataclass(frozen=True)
class Truth139ShapeProposalBridgeResultV1:
    payload: bytes
    diagnostics: dict


def package_truth139_shape_proposal_v1(
    *,
    generated: GeneratedTopologyMeshV1,
    conditioning_cache: dict,
    iris_checkpoint_sha256: str,
    tessa_checkpoint_sha256: str,
) -> Truth139ShapeProposalBridgeResultV1:
    if len(str(iris_checkpoint_sha256)) != 64 or len(str(tessa_checkpoint_sha256)) != 64:
        raise ValueError("TRUTH139_SHAPE_MODEL_CHECKPOINT_IDENTITY")
    if conditioning_cache.get("schema") != "RealSaS.Truth139TESSAConditioningCache.v1":
        raise ValueError("TRUTH139_SHAPE_CONDITIONING_SCHEMA")
    if conditioning_cache.get("teacher_topology_used") is not False:
        raise ValueError("TRUTH139_SHAPE_CONDITIONING_TEACHER_LEAKAGE")
    if generated.status != "PASS_CANONICAL_TESSA_SEQUENCE_CODEC":
        raise ValueError("TRUTH139_SHAPE_GENERATED_TOPOLOGY_NOT_QUALIFIED")

    surface = np.asarray(conditioning_cache["surface_source"], dtype=np.float32)
    if surface.ndim != 3 or surface.shape[0] != 1 or surface.shape[2] < 10:
        raise ValueError("TRUTH139_SHAPE_SURFACE_SOURCE_LAYOUT")
    # Truth139 surface-source layout: XYZ3, learned normal3, shared H, geometry4.
    query_width = surface.shape[2] - 10
    if query_width <= 0:
        raise ValueError("TRUTH139_SHAPE_SURFACE_H_WIDTH")
    xyz = surface[:, :, 0:3]
    normal = surface[:, :, 3:6]
    h = surface[:, :, 6 : 6 + query_width]
    if not np.isfinite(xyz).all() or not np.isfinite(normal).all() or not np.isfinite(h).all():
        raise ValueError("TRUTH139_SHAPE_SURFACE_NONFINITE")
    count = surface.shape[1]
    placeholder = np.zeros((1, count), dtype=np.float32)
    contact_valid = np.zeros((1, count), dtype=np.uint8)
    metadata = {
        "schema": "RealSaS.ShapeProposal.v1",
        "authority": "LEARNED_SHAPE_PROPOSAL_ONLY",
        "models": {
            "IRIS": {"checkpoint_content_sha256": str(iris_checkpoint_sha256)},
            "TESSA": {"checkpoint_content_sha256": str(tessa_checkpoint_sha256)},
        },
        "truth139_research_scope": SCOPE,
        "mechanics_abstained": True,
        "legacy_mechanics_arrays_are_non_authoritative_zero_placeholders": True,
        "contact_persistence_valid_count": 0,
        "teacher_topology_used_in_generation": False,
        "student_conditioning_teacher_topology_used": False,
        "product_authority_claimed": False,
    }
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))]),
        token_ids=np.asarray(generated.token_ids, dtype=np.int64),
        surface_xyz_tessa=xyz,
        surface_normal=normal,
        surface_h=h,
        stretch_major=placeholder,
        stretch_minor=placeholder,
        tangent_direction_2theta=placeholder,
        bend_demand=placeholder,
        rigidity_probability=placeholder,
        contact_persistence_probability=placeholder,
        contact_persistence_valid=contact_valid,
    )
    v2_payload = encode_shape_proposal_v2(SimpleNamespace(payload=stream.getvalue()))
    decoded = decode_shape_proposal_v2(v2_payload)
    if (
        decoded.metadata.get("truth139_research_scope") != SCOPE
        or decoded.metadata.get("mechanics_abstained") is not True
        or decoded.metadata.get("canonical_mesh_authority") is not False
        or decoded.metadata.get("source_truth_copied_into_shape") is not False
    ):
        raise RuntimeError("TRUTH139_SHAPE_V2_METADATA_DRIFT")
    if not np.array_equal(decoded.token_ids, np.asarray(generated.token_ids, dtype=np.int64)):
        raise RuntimeError("TRUTH139_SHAPE_V2_TOKEN_DRIFT")
    if not np.array_equal(decoded.decoded_faces, generated.faces):
        raise RuntimeError("TRUTH139_SHAPE_V2_FACE_DRIFT")
    diagnostics = {
        "schema": "RealSaS.Truth139ShapeProposalBridgeDiagnostics.v1",
        "scope": SCOPE,
        "shape_schema": decoded.metadata["schema"],
        "decoded_vertex_count": int(len(decoded.decoded_vertices_tessa)),
        "decoded_face_count": int(len(decoded.decoded_faces)),
        "decoded_component_count": int(decoded.metadata["decoded_component_count"]),
        "mechanics_abstained": True,
        "contact_persistence_valid_count": 0,
        "teacher_topology_used": False,
        "canonical_mesh_authority": False,
        "compiler_frame_binding_required": True,
    }
    return Truth139ShapeProposalBridgeResultV1(payload=v2_payload, diagnostics=diagnostics)
