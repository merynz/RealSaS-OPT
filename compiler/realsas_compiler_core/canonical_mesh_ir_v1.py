from __future__ import annotations

"""Generic static CanonicalMesh authority.

Mesh is born only after Compiler static qualification. Learned systems may supply
upstream proposal evidence, but no learned family is embedded in the canonical
Mesh schema. Rig, Skin, Mechanics and Motion may qualify behaviour over this
exact Mesh; they may not silently replace its geometry/topology basis.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .hashing import content_sha256
from .mechanical_carrier_evidence_v1 import (
    MechanicalCarrierEvidenceIR,
    mechanical_carrier_evidence_hash,
)
from .product_authority_v1 import CanonicalMeshCandidateIR, GeometricRefinementIR
from .surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from .types import QualificationError, SurfaceSupportBinding

Json = dict[str, Any]
Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CanonicalMeshVertexIR:
    canonical_mesh_vertex_id: str
    support_binding: SurfaceSupportBinding
    component_id: str
    P: Vec3
    source_candidate_vertex_id: str
    refinement: GeometricRefinementIR | None = None
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalMeshIR:
    vertices: tuple[CanonicalMeshVertexIR, ...]
    faces: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str], ...]
    source_evidence_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    static_qualification_hash: str
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    proposal_evidence_refs: tuple[str, ...]
    source_candidate_lineage_hash: str
    mesh_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalMeshIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_mesh_lineage_hash_v1(value: CanonicalMeshIR) -> str:
    payload = value.to_dict()
    payload.pop("mesh_lineage_hash", None)
    return content_sha256(payload)


def _stable_vertex_id(candidate: CanonicalMeshCandidateIR, candidate_vertex_id: str) -> str:
    return "MV:" + content_sha256(
        {
            "candidate_lineage_hash": candidate.candidate_lineage_hash,
            "candidate_vertex_id": candidate_vertex_id,
        }
    )[:24]


def _clean_refs(values) -> tuple[str, ...]:
    refs = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if any(len(value) < 8 for value in refs):
        raise QualificationError("CANONICAL_MESH_PROPOSAL_EVIDENCE_REF_INVALID")
    return refs


def _validate_carrier_basis(
    candidate: CanonicalMeshCandidateIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
) -> None:
    if carrier_evidence.carrier_evidence_hash != mechanical_carrier_evidence_hash(carrier_evidence):
        raise QualificationError("CANONICAL_MESH_CARRIER_EVIDENCE_HASH_DRIFT")

    ordered = tuple(sorted(candidate.vertices, key=lambda row: str(row.candidate_vertex_id)))
    expected_ids = tuple(str(row.candidate_vertex_id) for row in ordered)
    expected_positions = tuple(tuple(map(float, row.P)) for row in ordered)
    if carrier_evidence.ordered_vertex_ids != expected_ids:
        raise QualificationError("CANONICAL_MESH_CARRIER_VERTEX_ORDER_DRIFT")
    if carrier_evidence.positions != expected_positions:
        raise QualificationError("CANONICAL_MESH_CARRIER_POSITION_DRIFT")

    index_by_id = {vertex_id: index for index, vertex_id in enumerate(expected_ids)}
    try:
        expected_faces = tuple(
            tuple(index_by_id[str(vertex_id)] for vertex_id in face)
            for face in candidate.faces
        )
    except KeyError as exc:
        raise QualificationError("CANONICAL_MESH_CARRIER_FACE_VERTEX_UNKNOWN") from exc
    if carrier_evidence.face_vertex_indices != expected_faces:
        raise QualificationError("CANONICAL_MESH_CARRIER_TOPOLOGY_BASIS_DRIFT")


def mint_canonical_mesh_v1(
    candidate: CanonicalMeshCandidateIR,
    *,
    static_qualification: StaticCanonicalMeshQualificationIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
    proposal_evidence_refs: tuple[str, ...] = (),
) -> CanonicalMeshIR:
    report = dict(static_qualification.qualification_report or {})
    if static_qualification.qualification_hash != static_mesh_qualification_hash(static_qualification):
        raise QualificationError("CANONICAL_MESH_STATIC_QUALIFICATION_HASH_DRIFT")
    if static_qualification.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("CANONICAL_MESH_STATIC_CANDIDATE_BINDING_DRIFT")
    if static_qualification.partition_binding_hash != candidate.partition_binding_hash:
        raise QualificationError("CANONICAL_MESH_STATIC_PARTITION_BINDING_DRIFT")
    if report.get("status") != "PASS_STATIC_CANONICAL_CARRIER":
        raise QualificationError("CANONICAL_MESH_STATIC_QUALIFICATION_NOT_PASS")
    if report.get("static_quality_policy_passed") is not True:
        raise QualificationError("CANONICAL_MESH_STATIC_QUALITY_NOT_PASS")
    if report.get("source_fidelity_qualification_passed") is not True:
        raise QualificationError("CANONICAL_MESH_SOURCE_FIDELITY_NOT_PASS")
    if report.get("demo_geometry_lineage") is not False:
        raise QualificationError("CANONICAL_MESH_DEMO_LINEAGE_FORBIDDEN")

    if carrier_evidence.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("CANONICAL_MESH_CARRIER_CANDIDATE_DRIFT")
    if carrier_evidence.static_mesh_qualification_binding_hash != static_qualification.qualification_hash:
        raise QualificationError("CANONICAL_MESH_CARRIER_STATIC_BINDING_DRIFT")
    if carrier_evidence.source_surface_binding_hash != candidate.surface_binding_hash:
        raise QualificationError("CANONICAL_MESH_SOURCE_EVIDENCE_BINDING_DRIFT")
    _validate_carrier_basis(candidate, carrier_evidence)

    id_map = {
        str(vertex.candidate_vertex_id): _stable_vertex_id(candidate, str(vertex.candidate_vertex_id))
        for vertex in candidate.vertices
    }
    if len(id_map) != len(candidate.vertices) or not id_map:
        raise QualificationError("CANONICAL_MESH_VERTEX_ID_INVALID")

    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=id_map[str(vertex.candidate_vertex_id)],
            support_binding=vertex.support_binding,
            component_id=str(vertex.component_id),
            P=tuple(map(float, vertex.P)),
            source_candidate_vertex_id=str(vertex.candidate_vertex_id),
            refinement=vertex.refinement,
            metadata=dict(vertex.metadata or {}),
        )
        for vertex in candidate.vertices
    )
    if any(len(vertex.P) != 3 or not all(math.isfinite(value) for value in vertex.P) for vertex in vertices):
        raise QualificationError("CANONICAL_MESH_VERTEX_POSITION_INVALID")

    try:
        faces = tuple(tuple(id_map[str(vertex_id)] for vertex_id in face) for face in candidate.faces)
        edges = tuple(tuple(id_map[str(vertex_id)] for vertex_id in edge) for edge in candidate.edges)
    except KeyError as exc:
        raise QualificationError("CANONICAL_MESH_TOPOLOGY_VERTEX_UNKNOWN") from exc
    if any(len(face) != 3 or len(set(face)) != 3 for face in faces) or not faces:
        raise QualificationError("CANONICAL_MESH_FACE_INVALID")
    if any(len(edge) != 2 or edge[0] == edge[1] for edge in edges):
        raise QualificationError("CANONICAL_MESH_EDGE_INVALID")

    refs = _clean_refs(proposal_evidence_refs)
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        source_evidence_binding_hash=str(candidate.surface_binding_hash),
        partition_binding_hash=str(candidate.partition_binding_hash),
        carrier_policy_binding_hash=str(candidate.carrier_policy_binding_hash),
        static_qualification_hash=str(static_qualification.qualification_hash),
        carrier_evidence_hash=str(carrier_evidence.carrier_evidence_hash),
        carrier_topology_hash=str(carrier_evidence.topology_hash),
        carrier_geometry_hash=str(carrier_evidence.geometry_hash),
        proposal_evidence_refs=refs,
        source_candidate_lineage_hash=str(candidate.candidate_lineage_hash),
        mesh_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_MESH",
            "birth_boundary": "STATIC_MESH_QUALIFIED",
            "geometry_authority": "COMPILER_STATIC_MESH_QUALIFICATION",
            "learned_family_embedded_in_schema": False,
            "learned_proposals_are_upstream_evidence_only": True,
            "dynamic_qualification_claimed": False,
            "rig_authority_claimed": False,
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
            "material_support_is_geometry_authority": False,
            "repair_after_mesh_birth_requires_new_mesh_lineage": True,
        },
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def _support_from_dict(payload: Mapping[str, Any]) -> SurfaceSupportBinding:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.SurfaceSupportBinding.v1":
        raise QualificationError("CANONICAL_MESH_SUPPORT_SCHEMA_DRIFT")
    coefficients = tuple((str(key), float(value)) for key, value in payload.get("coefficients") or ())
    if any(not key or not math.isfinite(value) for key, value in coefficients):
        raise QualificationError("CANONICAL_MESH_SUPPORT_INVALID")
    return SurfaceSupportBinding(
        mode=str(payload["mode"]),
        coefficients=coefficients,
        metadata=dict(payload.get("metadata") or {}),
        schema_version=schema,
    )


def _refinement_from_dict(payload: Mapping[str, Any]) -> GeometricRefinementIR:
    def vec3(key: str) -> Vec3:
        value = tuple(float(x) for x in payload[key])
        if len(value) != 3 or not all(math.isfinite(x) for x in value):
            raise QualificationError("CANONICAL_MESH_REFINEMENT_VEC3_INVALID")
        return value  # type: ignore[return-value]

    scalars = tuple(float(payload[key]) for key in ("local_scale", "normal_component", "tangential_component"))
    if not all(math.isfinite(value) for value in scalars):
        raise QualificationError("CANONICAL_MESH_REFINEMENT_SCALAR_INVALID")
    return GeometricRefinementIR(
        dense_lineage_hash=str(payload["dense_lineage_hash"]),
        base_position=vec3("base_position"),
        refined_position=vec3("refined_position"),
        local_scale=scalars[0],
        normal_component=scalars[1],
        tangential_component=scalars[2],
        method=str(payload["method"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def canonical_mesh_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalMeshIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalMeshIR.v1":
        raise QualificationError("CANONICAL_MESH_SCHEMA_DRIFT")
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            support_binding=_support_from_dict(row["support_binding"]),
            component_id=str(row["component_id"]),
            P=tuple(map(float, row["P"])),
            source_candidate_vertex_id=str(row["source_candidate_vertex_id"]),
            refinement=None if row.get("refinement") is None else _refinement_from_dict(row["refinement"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("vertices") or ()
    )
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=tuple(tuple(map(str, face)) for face in payload.get("faces") or ()),
        edges=tuple(tuple(map(str, edge)) for edge in payload.get("edges") or ()),
        source_evidence_binding_hash=str(payload["source_evidence_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        carrier_policy_binding_hash=str(payload["carrier_policy_binding_hash"]),
        static_qualification_hash=str(payload["static_qualification_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        proposal_evidence_refs=_clean_refs(payload.get("proposal_evidence_refs") or ()),
        source_candidate_lineage_hash=str(payload["source_candidate_lineage_hash"]),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if not value.vertices or not value.faces:
        raise QualificationError("CANONICAL_MESH_EMPTY")
    if value.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_MESH_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("CANONICAL_MESH_AUTHORITY_DRIFT")
    if value.metadata.get("learned_family_embedded_in_schema") is not False:
        raise QualificationError("CANONICAL_MESH_LEARNED_FAMILY_SCHEMA_FORBIDDEN")
    if value.metadata.get("dynamic_qualification_claimed") is not False:
        raise QualificationError("CANONICAL_MESH_PREMATURE_DYNAMIC_AUTHORITY")
    return value


__all__ = [
    "CanonicalMeshIR",
    "CanonicalMeshVertexIR",
    "canonical_mesh_from_dict_v1",
    "canonical_mesh_lineage_hash_v1",
    "mint_canonical_mesh_v1",
]
