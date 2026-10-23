from __future__ import annotations

"""Static canonical Mesh authority minted immediately after Stage19 qualification.

This splits Mesh birth from later AXIS/MIRA dynamic qualification. Geometry and
stable vertex identity are frozen here; downstream Rig/Skin/Mechanics may prove
behavior over this exact Mesh but may not silently mint a different basis.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .mechanical_carrier_evidence_v1 import MechanicalCarrierEvidenceIR
from .product_authority_v1 import CanonicalMeshCandidateIR
from .surface_addressing_v1 import StaticCanonicalMeshQualificationIR, static_mesh_qualification_hash
from .tessa_geometry_qualification_v1 import TESSAStaticCarrierBindingIR
from .types import QualificationError, SurfaceSupportBinding, GeometricRefinementIR

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
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    static_qualification_hash: str
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    tessa_static_binding_hash: str
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


def mint_canonical_mesh_v1(
    candidate: CanonicalMeshCandidateIR,
    *,
    static_qualification: StaticCanonicalMeshQualificationIR,
    carrier_evidence: MechanicalCarrierEvidenceIR,
    tessa_static_binding: TESSAStaticCarrierBindingIR,
) -> CanonicalMeshIR:
    report = dict(static_qualification.qualification_report or {})
    if static_qualification.qualification_hash != static_mesh_qualification_hash(static_qualification):
        raise QualificationError("CANONICAL_MESH_STATIC_QUALIFICATION_HASH_DRIFT")
    if static_qualification.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("CANONICAL_MESH_STATIC_CANDIDATE_BINDING_DRIFT")
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
    if tessa_static_binding.candidate_lineage_hash != candidate.candidate_lineage_hash:
        raise QualificationError("CANONICAL_MESH_TESSA_CANDIDATE_DRIFT")
    if tessa_static_binding.static_mesh_qualification_hash != static_qualification.qualification_hash:
        raise QualificationError("CANONICAL_MESH_TESSA_STATIC_BINDING_DRIFT")
    if tessa_static_binding.partition_binding_hash != candidate.partition_binding_hash:
        raise QualificationError("CANONICAL_MESH_TESSA_PARTITION_DRIFT")

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
    try:
        faces = tuple(tuple(id_map[str(vertex_id)] for vertex_id in face) for face in candidate.faces)
        edges = tuple(tuple(id_map[str(vertex_id)] for vertex_id in edge) for edge in candidate.edges)
    except KeyError as exc:
        raise QualificationError("CANONICAL_MESH_TOPOLOGY_VERTEX_UNKNOWN") from exc

    value = CanonicalMeshIR(
        vertices=vertices,
        faces=faces,
        edges=edges,
        surface_binding_hash=str(candidate.surface_binding_hash),
        partition_binding_hash=str(candidate.partition_binding_hash),
        carrier_policy_binding_hash=str(candidate.carrier_policy_binding_hash),
        static_qualification_hash=str(static_qualification.qualification_hash),
        carrier_evidence_hash=str(carrier_evidence.carrier_evidence_hash),
        carrier_topology_hash=str(carrier_evidence.topology_hash),
        carrier_geometry_hash=str(carrier_evidence.geometry_hash),
        tessa_static_binding_hash=str(tessa_static_binding.binding_hash),
        source_candidate_lineage_hash=str(candidate.candidate_lineage_hash),
        mesh_lineage_hash="",
        metadata={
            "authority": "COMPILER_STATIC_CANONICAL_MESH",
            "geometry_authority": "STAGE19_STATIC_CANONICAL_CARRIER",
            "dynamic_qualification_claimed": False,
            "rig_authority_claimed": False,
            "skin_authority_claimed": False,
            "motion_authority_claimed": False,
            "learned_proposal_is_upstream_evidence": True,
            "material_support_is_geometry_authority": False,
            "repair_after_mesh_birth_requires_new_mesh_lineage": True,
        },
    )
    return replace(value, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(value))


def canonical_mesh_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalMeshIR:
    from .artifact_codec_v2 import surface_support_binding_from_dict, geometric_refinement_from_dict

    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalMeshIR.v1":
        raise QualificationError("CANONICAL_MESH_SCHEMA_DRIFT")
    vertices = tuple(
        CanonicalMeshVertexIR(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            support_binding=surface_support_binding_from_dict(row["support_binding"]),
            component_id=str(row["component_id"]),
            P=tuple(map(float, row["P"])),
            source_candidate_vertex_id=str(row["source_candidate_vertex_id"]),
            refinement=(None if row.get("refinement") is None else geometric_refinement_from_dict(row["refinement"])),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("vertices") or ()
    )
    value = CanonicalMeshIR(
        vertices=vertices,
        faces=tuple(tuple(map(str, face)) for face in payload.get("faces") or ()),
        edges=tuple(tuple(map(str, edge)) for edge in payload.get("edges") or ()),
        surface_binding_hash=str(payload["surface_binding_hash"]),
        partition_binding_hash=str(payload["partition_binding_hash"]),
        carrier_policy_binding_hash=str(payload["carrier_policy_binding_hash"]),
        static_qualification_hash=str(payload["static_qualification_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        tessa_static_binding_hash=str(payload["tessa_static_binding_hash"]),
        source_candidate_lineage_hash=str(payload["source_candidate_lineage_hash"]),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_MESH_LINEAGE_HASH_DRIFT")
    if value.metadata.get("dynamic_qualification_claimed") is not False:
        raise QualificationError("CANONICAL_MESH_PREMATURE_DYNAMIC_AUTHORITY")
    return value
