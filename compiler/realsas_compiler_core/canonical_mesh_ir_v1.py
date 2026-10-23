from __future__ import annotations

"""Static canonical Mesh authority for the product capability graph.

Mesh is born after Compiler topology/orientation/source-fidelity courts.  Rig,
Skin, Mechanics and Motion may later qualify behavior over this exact carrier,
but they may not silently replace its vertex identity or topology.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CanonicalMeshVertexIR:
    canonical_mesh_vertex_id: str
    P: Vec3
    component_id: str
    source_shape_vertex_index: int
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalMeshIR:
    vertices: tuple[CanonicalMeshVertexIR, ...]
    faces: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str], ...]
    source_shape_content_sha256: str
    source_observation_content_sha256: str
    decoded_geometry_sha256: str
    topology_report_sha256: str
    orientation_report_sha256: str
    source_fidelity_policy_sha256: str
    source_fidelity_report_sha256: str
    mesh_lineage_sha256: str
    schema_version: str = "RealSaS.CanonicalMeshIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_mesh_lineage_sha256(value: CanonicalMeshIR) -> str:
    payload = value.to_dict()
    payload.pop("mesh_lineage_sha256", None)
    return content_sha256(payload)


def _vertex_id(source_shape_content_sha256: str, index: int) -> str:
    return "MV:" + content_sha256(
        {
            "source_shape_content_sha256": source_shape_content_sha256,
            "source_shape_vertex_index": int(index),
        }
    )[:24]


def mint_canonical_mesh_v1(
    *,
    vertices_object: Any,
    faces: Any,
    vertex_component_indices: Any,
    source_shape_content_sha256: str,
    source_observation_content_sha256: str,
    decoded_geometry_sha256: str,
    topology_report: Mapping[str, Any],
    orientation_report: Mapping[str, Any],
    source_fidelity_policy_sha256: str,
    source_fidelity_report: Mapping[str, Any],
) -> CanonicalMeshIR:
    import numpy as np

    xyz = np.asarray(vertices_object, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    components = np.asarray(vertex_component_indices, dtype=np.int64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or not np.isfinite(xyz).all():
        raise QualificationError("CANONICAL_MESH_VERTEX_MATRIX_INVALID")
    if tri.ndim != 2 or tri.shape[1] != 3 or len(tri) == 0:
        raise QualificationError("CANONICAL_MESH_FACE_MATRIX_INVALID")
    if len(components) != len(xyz) or np.any(components < 0):
        raise QualificationError("CANONICAL_MESH_COMPONENT_VECTOR_INVALID")
    if np.any(tri < 0) or np.any(tri >= len(xyz)):
        raise QualificationError("CANONICAL_MESH_FACE_INDEX_INVALID")
    if any(len(set(map(int, row))) != 3 for row in tri.tolist()):
        raise QualificationError("CANONICAL_MESH_DEGENERATE_FACE")
    if len(source_shape_content_sha256) != 64 or len(source_observation_content_sha256) != 64:
        raise QualificationError("CANONICAL_MESH_SOURCE_HASH_INVALID")
    if len(decoded_geometry_sha256) != 64:
        raise QualificationError("CANONICAL_MESH_DECODED_GEOMETRY_HASH_INVALID")
    if topology_report.get("passed") is not True:
        raise QualificationError("CANONICAL_MESH_TOPOLOGY_NOT_QUALIFIED")
    if orientation_report.get("passed") is not True:
        raise QualificationError("CANONICAL_MESH_ORIENTATION_NOT_QUALIFIED")
    if source_fidelity_report.get("passed") is not True:
        raise QualificationError("CANONICAL_MESH_SOURCE_FIDELITY_NOT_QUALIFIED")
    if len(source_fidelity_policy_sha256) != 64:
        raise QualificationError("CANONICAL_MESH_SOURCE_POLICY_HASH_INVALID")

    ids = tuple(_vertex_id(source_shape_content_sha256, i) for i in range(len(xyz)))
    if len(ids) != len(set(ids)):
        raise QualificationError("CANONICAL_MESH_VERTEX_ID_COLLISION")
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=ids[i],
            P=tuple(map(float, xyz[i].tolist())),
            component_id=f"C{int(components[i]):04d}",
            source_shape_vertex_index=int(i),
            metadata={"identity_frozen_at_mesh_birth": True},
        )
        for i in range(len(xyz))
    )
    face_ids: list[tuple[str, str, str]] = []
    edge_set: set[tuple[str, str]] = set()
    for row in tri.tolist():
        component_ids = {vertices[int(i)].component_id for i in row}
        if len(component_ids) != 1:
            raise QualificationError("CANONICAL_MESH_FACE_CROSSES_COMPONENT")
        face = tuple(ids[int(i)] for i in row)
        face_ids.append(face)
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_set.add(tuple(sorted((a, b))))

    topology_hash = content_sha256(dict(topology_report))
    orientation_hash = content_sha256(dict(orientation_report))
    source_fidelity_hash = content_sha256(dict(source_fidelity_report))
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=tuple(face_ids),
        edges=tuple(sorted(edge_set)),
        source_shape_content_sha256=source_shape_content_sha256,
        source_observation_content_sha256=source_observation_content_sha256,
        decoded_geometry_sha256=decoded_geometry_sha256,
        topology_report_sha256=topology_hash,
        orientation_report_sha256=orientation_hash,
        source_fidelity_policy_sha256=source_fidelity_policy_sha256,
        source_fidelity_report_sha256=source_fidelity_hash,
        mesh_lineage_sha256="",
        metadata={
            "authority": "COMPILER_STATIC_CANONICAL_MESH",
            "coordinate_frame": "REALSAS_NORMALIZED_OBJECT_FRAME_XYZ_MINUS_1_PLUS_1",
            "source_shape_coordinate_transform": "OBJECT_XYZ_EQUALS_TESSA_XYZ_TIMES_2",
            "dynamic_qualification_claimed": False,
            "rig_authority_claimed": False,
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
            "learned_shape_is_upstream_evidence_only": True,
            "repair_after_mesh_birth_requires_new_mesh_lineage": True,
        },
    )
    return replace(value, mesh_lineage_sha256=canonical_mesh_lineage_sha256(value))


def canonical_mesh_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalMeshIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalMeshIR.v1":
        raise QualificationError("CANONICAL_MESH_SCHEMA_DRIFT")
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            P=tuple(map(float, row["P"])),
            component_id=str(row["component_id"]),
            source_shape_vertex_index=int(row["source_shape_vertex_index"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("vertices") or ()
    )
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=tuple(tuple(map(str, row)) for row in payload.get("faces") or ()),
        edges=tuple(tuple(map(str, row)) for row in payload.get("edges") or ()),
        source_shape_content_sha256=str(payload["source_shape_content_sha256"]),
        source_observation_content_sha256=str(payload["source_observation_content_sha256"]),
        decoded_geometry_sha256=str(payload["decoded_geometry_sha256"]),
        topology_report_sha256=str(payload["topology_report_sha256"]),
        orientation_report_sha256=str(payload["orientation_report_sha256"]),
        source_fidelity_policy_sha256=str(payload["source_fidelity_policy_sha256"]),
        source_fidelity_report_sha256=str(payload["source_fidelity_report_sha256"]),
        mesh_lineage_sha256=str(payload["mesh_lineage_sha256"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.metadata.get("authority") != "COMPILER_STATIC_CANONICAL_MESH":
        raise QualificationError("CANONICAL_MESH_AUTHORITY_DRIFT")
    if value.metadata.get("dynamic_qualification_claimed") is not False:
        raise QualificationError("CANONICAL_MESH_PREMATURE_DYNAMIC_AUTHORITY")
    if canonical_mesh_lineage_sha256(value) != value.mesh_lineage_sha256:
        raise QualificationError("CANONICAL_MESH_LINEAGE_HASH_DRIFT")
    return value
