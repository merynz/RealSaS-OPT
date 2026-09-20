from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class SurfaceVertexAddressIR:
    candidate_vertex_id: str
    address_id: str
    component_id: str
    support_binding_hash: str
    schema_version: str = "RealSaS.SurfaceVertexAddressIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SurfaceAddressingIR:
    candidate_mesh_binding_hash: str
    surface_binding_hash: str
    vertex_addresses: tuple[SurfaceVertexAddressIR, ...]
    face_address_ids: tuple[tuple[str, str, str], ...]
    addressing_hash: str
    schema_version: str = "RealSaS.SurfaceAddressingIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class AppearanceDomainIR:
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    output_direction_set_binding_hash: str
    renderable_face_count: int
    total_appearance_required: bool
    geometry_mutation_forbidden: bool
    domain_hash: str
    schema_version: str = "RealSaS.AppearanceDomainIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StaticCanonicalMeshQualificationIR:
    candidate_mesh_binding_hash: str
    surface_addressing_binding_hash: str
    appearance_domain_binding_hash: str
    geometry_gate_binding_hash: str
    partition_binding_hash: str
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.StaticCanonicalMeshQualificationIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _hash_without(value, field_name: str) -> str:
    payload = value.to_dict()
    payload.pop(field_name, None)
    return content_sha256(payload)


def surface_addressing_hash(value: SurfaceAddressingIR) -> str:
    return _hash_without(value, "addressing_hash")


def appearance_domain_hash(value: AppearanceDomainIR) -> str:
    return _hash_without(value, "domain_hash")


def static_mesh_qualification_hash(value: StaticCanonicalMeshQualificationIR) -> str:
    return _hash_without(value, "qualification_hash")


def build_surface_addressing(candidate) -> SurfaceAddressingIR:
    vertices = {str(v.candidate_vertex_id): v for v in candidate.vertices}
    rows = []
    address_by_vertex = {}
    for vertex_id in sorted(vertices):
        vertex = vertices[vertex_id]
        support_hash = content_sha256(asdict(vertex.support_binding))
        address_id = content_sha256(
            {
                "schema": "RealSaS.CanonicalSurfaceAddress.v1",
                "candidate_mesh": candidate.candidate_lineage_hash,
                "candidate_vertex_id": vertex_id,
                "component_id": str(vertex.component_id),
                "support_binding_hash": support_hash,
            }
        )
        address_by_vertex[vertex_id] = address_id
        rows.append(SurfaceVertexAddressIR(vertex_id, address_id, str(vertex.component_id), support_hash))
    faces = []
    for face in candidate.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or any(x not in address_by_vertex for x in ids) or len(set(ids)) != 3:
            raise QualificationError("SURFACE_ADDRESSING_FACE_INVALID")
        faces.append(tuple(address_by_vertex[x] for x in ids))
    value = SurfaceAddressingIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_binding_hash=candidate.surface_binding_hash,
        vertex_addresses=tuple(rows),
        face_address_ids=tuple(faces),
        addressing_hash="",
        metadata={
            "addressing_kind": "FROZEN_CANONICAL_VERTEX_FACE_DOMAIN_V1",
            "uv_layout_is_not_authority": True,
        },
    )
    return replace(value, addressing_hash=surface_addressing_hash(value))


def build_appearance_domain(candidate, addressing: SurfaceAddressingIR, output_direction_set_hash: str) -> AppearanceDomainIR:
    if addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("APPEARANCE_DOMAIN_MESH_ADDRESSING_DRIFT")
    value = AppearanceDomainIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        output_direction_set_binding_hash=str(output_direction_set_hash),
        renderable_face_count=len(candidate.faces),
        total_appearance_required=True,
        geometry_mutation_forbidden=True,
        domain_hash="",
        metadata={
            "domain": "CANONICAL_MESH_SURFACE_X_DISCRETE_V0_V7",
            "presentation_warp_v1_forbidden": True,
            "appearance_cannot_mutate_geometry": True,
        },
    )
    return replace(value, domain_hash=appearance_domain_hash(value))


def surface_addressing_from_dict(payload: Mapping[str, Any]) -> SurfaceAddressingIR:
    rows = tuple(
        SurfaceVertexAddressIR(
            str(row["candidate_vertex_id"]),
            str(row["address_id"]),
            str(row["component_id"]),
            str(row["support_binding_hash"]),
            schema_version=str(row.get("schema_version") or "RealSaS.SurfaceVertexAddressIR.v1"),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("vertex_addresses") or ()
    )
    value = SurfaceAddressingIR(
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_binding_hash"]),
        rows,
        tuple(tuple(map(str, face)) for face in payload.get("face_address_ids") or ()),
        str(payload["addressing_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.SurfaceAddressingIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.addressing_hash != surface_addressing_hash(value):
        raise QualificationError("SURFACE_ADDRESSING_HASH_MISMATCH")
    return value


def appearance_domain_from_dict(payload: Mapping[str, Any]) -> AppearanceDomainIR:
    value = AppearanceDomainIR(
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["output_direction_set_binding_hash"]),
        int(payload["renderable_face_count"]),
        bool(payload["total_appearance_required"]),
        bool(payload["geometry_mutation_forbidden"]),
        str(payload["domain_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.AppearanceDomainIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.domain_hash != appearance_domain_hash(value):
        raise QualificationError("APPEARANCE_DOMAIN_HASH_MISMATCH")
    return value


def static_mesh_qualification_from_dict(payload: Mapping[str, Any]) -> StaticCanonicalMeshQualificationIR:
    value = StaticCanonicalMeshQualificationIR(
        str(payload["candidate_mesh_binding_hash"]),
        str(payload["surface_addressing_binding_hash"]),
        str(payload["appearance_domain_binding_hash"]),
        str(payload["geometry_gate_binding_hash"]),
        str(payload["partition_binding_hash"]),
        dict(payload.get("qualification_report") or {}),
        str(payload["qualification_hash"]),
        schema_version=str(payload.get("schema_version") or "RealSaS.StaticCanonicalMeshQualificationIR.v1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.qualification_hash != static_mesh_qualification_hash(value):
        raise QualificationError("STATIC_MESH_QUALIFICATION_HASH_MISMATCH")
    return value
