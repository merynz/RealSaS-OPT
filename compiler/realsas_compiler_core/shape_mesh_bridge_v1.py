from __future__ import annotations

"""Compiler frame binding from ShapeProposal.v2 to the existing TESSA proposal court.

Shape owns normalized learned topology evidence only. This bridge owns the exact
Stage08/world-frame conversion and admitted-surface routing hints needed by the
existing TESSA candidate/support courts. Mechanical component identity is not
inferred here; decoded component labels remain proposal-local until the Compiler
material-support court reconciles them against MechanicalPartition.

The TESSA package is imported lazily so generic Platform/Compiler discovery does
not acquire a Torch dependency merely by importing this module.
"""

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .preproduct_authority_v1 import NormalizationDomainIR
from .types import QualificationError, RiggingSurfaceIR


@dataclass(frozen=True)
class ShapeMeshFrameBindingV1:
    shape_proposal_content_sha256: str
    decoded_topology_sha256: str
    normalization_hash: str
    source_geometry_lineage_hash: str
    center_xyz: tuple[float, float, float]
    tessa_world_scale: float
    coordinate_frame: str
    binding_hash: str
    schema_version: str = "RealSaS.ShapeMeshFrameBinding.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_proposal_content_sha256": self.shape_proposal_content_sha256,
            "decoded_topology_sha256": self.decoded_topology_sha256,
            "normalization_hash": self.normalization_hash,
            "source_geometry_lineage_hash": self.source_geometry_lineage_hash,
            "center_xyz": self.center_xyz,
            "tessa_world_scale": self.tessa_world_scale,
            "coordinate_frame": self.coordinate_frame,
            "binding_hash": self.binding_hash,
            "schema_version": self.schema_version,
        }


def shape_mesh_frame_binding_hash_v1(value: ShapeMeshFrameBindingV1) -> str:
    payload = value.to_dict()
    payload.pop("binding_hash", None)
    return content_sha256(payload)


def _nearest_surface_ids(world: np.ndarray, surface: RiggingSurfaceIR) -> tuple[str, ...]:
    ordered = tuple(sorted(surface.surface_nodes, key=lambda row: str(row.surface_id)))
    if not ordered:
        raise QualificationError("SHAPE_MESH_BRIDGE_SURFACE_EMPTY")
    surface_ids = tuple(str(row.surface_id) for row in ordered)
    if any(not value for value in surface_ids) or len(surface_ids) != len(set(surface_ids)):
        raise QualificationError("SHAPE_MESH_BRIDGE_SURFACE_ID_INVALID")
    xyz = np.asarray([tuple(map(float, row.P)) for row in ordered], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise QualificationError("SHAPE_MESH_BRIDGE_SURFACE_XYZ_INVALID")
    # Routing hint only. Mechanical support/component authority is resolved by
    # build_tessa_material_support_field_v1 downstream.
    distance2 = np.sum((world[:, None, :] - xyz[None, :, :]) ** 2, axis=-1)
    nearest = np.argmin(distance2, axis=1)
    return tuple(surface_ids[int(index)] for index in nearest.tolist())


def bind_shape_proposal_to_tessa_v1(
    *,
    shape,
    shape_proposal_content_sha256: str,
    normalization: NormalizationDomainIR,
    surface: RiggingSurfaceIR,
):
    """Return `(TESSAMeshProposalV1, ShapeMeshFrameBindingV1)`.

    `shape` is the decoded `RealSaS.ShapeProposal.v2` value. It is intentionally
    duck-typed here so this Compiler module does not import the Platform worker.
    """
    shape_sha = str(shape_proposal_content_sha256 or "")
    if len(shape_sha) != 64:
        raise QualificationError("SHAPE_MESH_BRIDGE_SHAPE_CONTENT_SHA_INVALID")
    metadata = dict(getattr(shape, "metadata", {}) or {})
    if metadata.get("schema") != "RealSaS.ShapeProposal.v2":
        raise QualificationError("SHAPE_MESH_BRIDGE_SHAPE_SCHEMA_DRIFT")
    if metadata.get("canonical_mesh_authority") is not False:
        raise QualificationError("SHAPE_MESH_BRIDGE_PREMATURE_MESH_AUTHORITY")
    if metadata.get("compiler_frame_authority") is not False:
        raise QualificationError("SHAPE_MESH_BRIDGE_SHAPE_FRAME_AUTHORITY_FORBIDDEN")
    if metadata.get("compiler_frame_binding_required") is not True:
        raise QualificationError("SHAPE_MESH_BRIDGE_FRAME_BINDING_NOT_REQUESTED")
    if metadata.get("source_geometry_lineage_claimed") is not False:
        raise QualificationError("SHAPE_MESH_BRIDGE_SOURCE_LINEAGE_CLAIM_FORBIDDEN")
    if metadata.get("proposal_evidence_frame") != "RealSaS.TESSA.SequenceNormalizedFrame.v1":
        raise QualificationError("SHAPE_MESH_BRIDGE_EVIDENCE_FRAME_DRIFT")

    if str(normalization.coordinate_frame) != "REALSAS_OBJECT_FRAME":
        raise QualificationError("SHAPE_MESH_BRIDGE_NORMALIZATION_FRAME_UNSUPPORTED")
    center = np.asarray(tuple(map(float, normalization.center_xyz)), dtype=np.float64)
    half_extent = float(normalization.half_extent)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise QualificationError("SHAPE_MESH_BRIDGE_NORMALIZATION_CENTER_INVALID")
    if not math.isfinite(half_extent) or half_extent <= 1e-12:
        raise QualificationError("SHAPE_MESH_BRIDGE_NORMALIZATION_HALF_EXTENT_INVALID")
    if not str(normalization.normalization_hash):
        raise QualificationError("SHAPE_MESH_BRIDGE_NORMALIZATION_HASH_MISSING")
    if not str(surface.geometry_lineage_hash):
        raise QualificationError("SHAPE_MESH_BRIDGE_SURFACE_LINEAGE_MISSING")

    vertices_tessa = np.asarray(shape.decoded_vertices_tessa, dtype=np.float64)
    faces = np.asarray(shape.decoded_faces, dtype=np.int64)
    components = np.asarray(shape.decoded_vertex_component_indices, dtype=np.int64).reshape(-1)
    if vertices_tessa.ndim != 2 or vertices_tessa.shape[1] != 3 or len(vertices_tessa) == 0:
        raise QualificationError("SHAPE_MESH_BRIDGE_VERTEX_SHAPE_INVALID")
    if not np.isfinite(vertices_tessa).all():
        raise QualificationError("SHAPE_MESH_BRIDGE_VERTEX_NONFINITE")
    if np.any(vertices_tessa < -0.550001) or np.any(vertices_tessa > 0.550001):
        raise QualificationError("SHAPE_MESH_BRIDGE_TESSA_SCALE_POLICY_EXCEEDED")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise QualificationError("SHAPE_MESH_BRIDGE_FACE_SHAPE_INVALID")
    if np.any(faces < 0) or np.any(faces >= len(vertices_tessa)):
        raise QualificationError("SHAPE_MESH_BRIDGE_FACE_INDEX_INVALID")
    if len(components) != len(vertices_tessa) or np.any(components < 0):
        raise QualificationError("SHAPE_MESH_BRIDGE_COMPONENT_ACCOUNTING_INVALID")

    topology_sha = str(metadata.get("decoded_topology_sha256") or "")
    sequence_sha = str(metadata.get("topology_sequence_sha256") or "")
    if len(topology_sha) != 64 or len(sequence_sha) != 64:
        raise QualificationError("SHAPE_MESH_BRIDGE_TOPOLOGY_HASH_MISSING")

    # Exact inverse of the admitted TESSA frame contract:
    # Stage08 world = center + canonical_normalized * half_extent;
    # TESSA normalized = canonical_normalized * 0.5.
    tessa_world_scale = 2.0 * half_extent
    world = center[None, :] + vertices_tessa * tessa_world_scale
    anchors = _nearest_surface_ids(world, surface)

    proposal_ids = tuple(f"SHAPE_V2:{index:08d}" for index in range(len(world)))
    payload_vertices = [
        {
            "proposal_vertex_id": proposal_ids[index],
            "P": tuple(map(float, world[index].tolist())),
            "primary_surface_id": anchors[index],
            "component_id": f"DECODED_COMPONENT:{int(components[index]):04d}",
            "confidence": 1.0,
            "metadata": {
                "anchor_semantics": "COMPILER_ROUTING_HINT_ONLY_NOT_SUPPORT_BINDING",
                "decoded_component_is_not_mechanical_component_authority": True,
                "shape_vertex_index": index,
                "shape_proposal_content_sha256": shape_sha,
                "decoded_topology_sha256": topology_sha,
            },
        }
        for index in range(len(world))
    ]
    payload_faces = [
        tuple(proposal_ids[int(vertex_index)] for vertex_index in face)
        for face in faces.tolist()
    ]
    model_provenance = content_sha256(
        {
            "schema": "RealSaS.ShapeV2LearnedModelProvenance.v1",
            "shape_proposal_content_sha256": shape_sha,
            "models": metadata.get("models"),
            "topology_sequence_sha256": sequence_sha,
            "decoded_topology_sha256": topology_sha,
        }
    )
    proposal_payload = {
        "schema_version": "RealSaS.TESSAMeshProposal.v1",
        "vertices": payload_vertices,
        "faces": payload_faces,
        "source_geometry_lineage_hash": str(surface.geometry_lineage_hash),
        "model_provenance": model_provenance,
        "topology_sequence_hash": sequence_sha,
        "metadata": {
            "authority_class": "LEARNED_PROPOSAL",
            "origin": "RealSaS.ShapeProposal.v2",
            "shape_proposal_content_sha256": shape_sha,
            "decoded_topology_sha256": topology_sha,
            "normalization_hash": str(normalization.normalization_hash),
            "compiler_frame_binding_applied": True,
            "compiler_frame_authority_owner": "COMPILER",
            "source_geometry_lineage_bound_by_compiler": True,
            "decoded_component_is_not_mechanical_component_authority": True,
            "support_binding_qualified": False,
            "product_authority_claimed": False,
        },
    }
    # Lazy import prevents a TESSA/Torch package dependency on generic discovery.
    from .tessa_candidate_bridge_v1 import tessa_mesh_proposal_from_dict_v1

    proposal = tessa_mesh_proposal_from_dict_v1(proposal_payload)

    binding = ShapeMeshFrameBindingV1(
        shape_proposal_content_sha256=shape_sha,
        decoded_topology_sha256=topology_sha,
        normalization_hash=str(normalization.normalization_hash),
        source_geometry_lineage_hash=str(surface.geometry_lineage_hash),
        center_xyz=tuple(map(float, center.tolist())),
        tessa_world_scale=float(tessa_world_scale),
        coordinate_frame=str(normalization.coordinate_frame),
        binding_hash="",
    )
    binding = ShapeMeshFrameBindingV1(
        **{**binding.to_dict(), "binding_hash": shape_mesh_frame_binding_hash_v1(binding)}
    )
    return proposal, binding


__all__ = [
    "ShapeMeshFrameBindingV1",
    "bind_shape_proposal_to_tessa_v1",
    "shape_mesh_frame_binding_hash_v1",
]
