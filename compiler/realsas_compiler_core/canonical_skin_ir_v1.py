from __future__ import annotations

"""Product-facing canonical Skin authority.

Learned systems may propose influences on the exact mechanical carrier, but the
canonical Skin is minted only after Compiler qualification against the exact
CanonicalMesh and CanonicalRig parents. Legacy carrier vertex ids and legacy
qualified-skeleton ids remain provenance only; product authority is expressed in
canonical Mesh vertex ids and canonical Rig joint ids.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v1 import CanonicalRigIR, canonical_rig_lineage_hash_v1
from .carrier_skin_v1 import QualifiedCarrierSkinIR
from .hashing import content_sha256
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CanonicalSkinRowIR:
    canonical_mesh_vertex_id: str
    influences: tuple[tuple[str, float], ...]
    source_carrier_vertex_id: str
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSkinIR:
    rows: tuple[CanonicalSkinRowIR, ...]
    mesh_lineage_hash: str
    rig_lineage_hash: str
    mesh_carrier_evidence_hash: str
    mesh_carrier_topology_hash: str
    mesh_carrier_geometry_hash: str
    structure_qualification_hash: str
    carrier_skin_qualification_hash: str
    proposal_evidence_refs: tuple[str, ...]
    skin_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalSkinIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_skin_lineage_hash_v1(value: CanonicalSkinIR) -> str:
    payload = value.to_dict()
    payload.pop("skin_lineage_hash", None)
    return content_sha256(payload)


def _clean_refs(values) -> tuple[str, ...]:
    refs = tuple(sorted({str(value).strip() for value in values if str(value).strip()}))
    if any(len(value) < 8 for value in refs):
        raise QualificationError("CANONICAL_SKIN_PROPOSAL_EVIDENCE_REF_INVALID")
    return refs


def _qualified_skin_lineage_hash(value: QualifiedCarrierSkinIR) -> str:
    return content_sha256(
        {
            "carrier_evidence_hash": value.carrier_evidence_hash,
            "carrier_topology_hash": value.carrier_topology_hash,
            "carrier_geometry_hash": value.carrier_geometry_hash,
            "skeleton_binding_hash": value.skeleton_binding_hash,
            "rows": [row.to_dict() for row in value.rows],
            "qualification_report": value.qualification_report,
            "metadata": value.metadata,
        }
    )


def _validate_local_rows(rows: tuple[CanonicalSkinRowIR, ...]) -> None:
    if not rows:
        raise QualificationError("CANONICAL_SKIN_EMPTY")
    canonical_ids: set[str] = set()
    source_ids: set[str] = set()
    for row in rows:
        canonical_id = str(row.canonical_mesh_vertex_id)
        source_id = str(row.source_carrier_vertex_id)
        if not canonical_id or canonical_id in canonical_ids:
            raise QualificationError("CANONICAL_SKIN_CANONICAL_VERTEX_ID_INVALID")
        if not source_id or source_id in source_ids:
            raise QualificationError("CANONICAL_SKIN_SOURCE_CARRIER_ID_INVALID")
        canonical_ids.add(canonical_id)
        source_ids.add(source_id)
        if not row.influences:
            raise QualificationError("CANONICAL_SKIN_VERTEX_UNCOVERED")
        joint_ids: set[str] = set()
        total = 0.0
        for joint_id, raw_weight in row.influences:
            jid = str(joint_id)
            weight = float(raw_weight)
            if not jid or jid in joint_ids:
                raise QualificationError("CANONICAL_SKIN_JOINT_ID_INVALID")
            if not math.isfinite(weight) or weight < 0.0:
                raise QualificationError("CANONICAL_SKIN_WEIGHT_INVALID")
            joint_ids.add(jid)
            total += weight
        if abs(total - 1.0) > 1e-6:
            raise QualificationError("CANONICAL_SKIN_SIMPLEX_DRIFT")


def mint_canonical_skin_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    *,
    qualified_skin: QualifiedCarrierSkinIR,
    proposal_evidence_refs: tuple[str, ...] = (),
) -> CanonicalSkinIR:
    if mesh.mesh_lineage_hash != canonical_mesh_lineage_hash_v1(mesh):
        raise QualificationError("CANONICAL_SKIN_MESH_LINEAGE_HASH_DRIFT")
    if mesh.metadata.get("authority") != "COMPILER_CANONICAL_MESH":
        raise QualificationError("CANONICAL_SKIN_MESH_AUTHORITY_DRIFT")
    if rig.rig_lineage_hash != canonical_rig_lineage_hash_v1(rig):
        raise QualificationError("CANONICAL_SKIN_RIG_LINEAGE_HASH_DRIFT")
    if rig.metadata.get("authority") != "COMPILER_CANONICAL_RIG":
        raise QualificationError("CANONICAL_SKIN_RIG_AUTHORITY_DRIFT")
    if rig.mesh_lineage_hash != mesh.mesh_lineage_hash:
        raise QualificationError("CANONICAL_SKIN_RIG_MESH_PARENT_DRIFT")

    qualified_hash = _qualified_skin_lineage_hash(qualified_skin)
    if qualified_skin.skin_lineage_hash != qualified_hash:
        raise QualificationError("CANONICAL_SKIN_QUALIFIED_RECEIPT_HASH_DRIFT")
    if qualified_skin.carrier_evidence_hash != mesh.carrier_evidence_hash:
        raise QualificationError("CANONICAL_SKIN_CARRIER_EVIDENCE_DRIFT")
    if qualified_skin.carrier_topology_hash != mesh.carrier_topology_hash:
        raise QualificationError("CANONICAL_SKIN_CARRIER_TOPOLOGY_DRIFT")
    if qualified_skin.carrier_geometry_hash != mesh.carrier_geometry_hash:
        raise QualificationError("CANONICAL_SKIN_CARRIER_GEOMETRY_DRIFT")
    if qualified_skin.skeleton_binding_hash != rig.structure_qualification_hash:
        raise QualificationError("CANONICAL_SKIN_RIG_QUALIFICATION_BINDING_DRIFT")

    report = dict(qualified_skin.qualification_report or {})
    if report.get("status") != "PASS_DIRECT_CARRIER_SKIN":
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_NOT_PASS")
    if report.get("product_skin_evidence_complete") is not True:
        raise QualificationError("CANONICAL_SKIN_EVIDENCE_INCOMPLETE")
    if report.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("CANONICAL_SKIN_POST_HOC_TRANSFER_FORBIDDEN")
    if report.get("gsa_skin_field_authority_minted") is not False:
        raise QualificationError("CANONICAL_SKIN_GSA_AUTHORITY_FORBIDDEN")

    meta = dict(qualified_skin.metadata or {})
    if meta.get("mesh_mutation_invalidates_skin") is not True:
        raise QualificationError("CANONICAL_SKIN_MESH_MUTATION_INVALIDATION_REQUIRED")
    if str(meta.get("normal_path_skin_transport") or "") != "FORBIDDEN":
        raise QualificationError("CANONICAL_SKIN_TRANSPORT_POLICY_DRIFT")

    mesh_by_source = {
        str(vertex.source_candidate_vertex_id): vertex
        for vertex in mesh.vertices
    }
    if len(mesh_by_source) != len(mesh.vertices):
        raise QualificationError("CANONICAL_SKIN_SOURCE_CARRIER_ID_DUPLICATE")
    rig_joint_ids = {str(joint.canonical_joint_id) for joint in rig.joints}
    if not rig_joint_ids:
        raise QualificationError("CANONICAL_SKIN_RIG_EMPTY")

    rows: list[CanonicalSkinRowIR] = []
    seen_vertices: set[str] = set()
    for row in qualified_skin.rows:
        source_id = str(row.carrier_vertex_id)
        mesh_vertex = mesh_by_source.get(source_id)
        if mesh_vertex is None:
            raise QualificationError("CANONICAL_SKIN_CARRIER_VERTEX_NOT_IN_CANONICAL_MESH")
        canonical_id = str(mesh_vertex.canonical_mesh_vertex_id)
        if canonical_id in seen_vertices:
            raise QualificationError("CANONICAL_SKIN_DUPLICATE_CANONICAL_VERTEX")
        seen_vertices.add(canonical_id)
        influences = tuple((str(jid), float(weight)) for jid, weight in row.influences)
        if any(jid not in rig_joint_ids for jid, _ in influences):
            raise QualificationError("CANONICAL_SKIN_JOINT_NOT_IN_CANONICAL_RIG")
        rows.append(
            CanonicalSkinRowIR(
                canonical_mesh_vertex_id=canonical_id,
                influences=influences,
                source_carrier_vertex_id=source_id,
                metadata={
                    "source_carrier_vertex_id_is_provenance_only": True,
                    "canonical_vertex_authority": "CANONICAL_MESH",
                    "canonical_joint_authority": "CANONICAL_RIG",
                },
            )
        )

    canonical_rows = tuple(rows)
    _validate_local_rows(canonical_rows)
    canonical_vertex_ids = {
        str(vertex.canonical_mesh_vertex_id) for vertex in mesh.vertices
    }
    if len(canonical_rows) != len(mesh.vertices) or seen_vertices != canonical_vertex_ids:
        raise QualificationError("CANONICAL_SKIN_CANONICAL_MESH_COVERAGE_DRIFT")

    refs = _clean_refs(proposal_evidence_refs)
    value = CanonicalSkinIR(
        rows=canonical_rows,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        structure_qualification_hash=rig.structure_qualification_hash,
        carrier_skin_qualification_hash=qualified_hash,
        proposal_evidence_refs=refs,
        skin_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_SKIN",
            "canonical_parents": ["CANONICAL_MESH", "CANONICAL_RIG"],
            "learned_family_embedded_in_schema": False,
            "learned_influences_are_proposal_evidence": True,
            "post_hoc_surface_skin_transfer_performed": False,
            "gsa_skin_authority_minted": False,
            "hidden_repair_performed": False,
            "mesh_mutation_invalidates_skin": True,
            "rig_mutation_invalidates_skin": True,
            "deformation_authority_claimed": False,
            "motion_authority_claimed": False,
        },
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v1(value))


def canonical_skin_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalSkinIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalSkinIR.v1":
        raise QualificationError("CANONICAL_SKIN_SCHEMA_DRIFT")
    rows = tuple(
        CanonicalSkinRowIR(
            canonical_mesh_vertex_id=str(row["canonical_mesh_vertex_id"]),
            influences=tuple((str(jid), float(weight)) for jid, weight in row.get("influences") or ()),
            source_carrier_vertex_id=str(row["source_carrier_vertex_id"]),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in payload.get("rows") or ()
    )
    _validate_local_rows(rows)
    value = CanonicalSkinIR(
        rows=rows,
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        mesh_carrier_evidence_hash=str(payload["mesh_carrier_evidence_hash"]),
        mesh_carrier_topology_hash=str(payload["mesh_carrier_topology_hash"]),
        mesh_carrier_geometry_hash=str(payload["mesh_carrier_geometry_hash"]),
        structure_qualification_hash=str(payload["structure_qualification_hash"]),
        carrier_skin_qualification_hash=str(payload["carrier_skin_qualification_hash"]),
        proposal_evidence_refs=_clean_refs(payload.get("proposal_evidence_refs") or ()),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.skin_lineage_hash != canonical_skin_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_SKIN_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_SKIN":
        raise QualificationError("CANONICAL_SKIN_AUTHORITY_DRIFT")
    if value.metadata.get("learned_family_embedded_in_schema") is not False:
        raise QualificationError("CANONICAL_SKIN_LEARNED_FAMILY_SCHEMA_FORBIDDEN")
    if value.metadata.get("post_hoc_surface_skin_transfer_performed") is not False:
        raise QualificationError("CANONICAL_SKIN_POST_HOC_TRANSFER_FORBIDDEN")
    if value.metadata.get("deformation_authority_claimed") is not False:
        raise QualificationError("CANONICAL_SKIN_PREMATURE_DEFORMATION_AUTHORITY")
    return value


__all__ = [
    "CanonicalSkinIR",
    "CanonicalSkinRowIR",
    "canonical_skin_from_dict_v1",
    "canonical_skin_lineage_hash_v1",
    "mint_canonical_skin_v1",
]
