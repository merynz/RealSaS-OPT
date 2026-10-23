from __future__ import annotations

"""Product-facing canonical Skin authority.

MIRA or another learned provider proposes influences on the exact mechanical
carrier. The Compiler binds those proposal rows to CanonicalMesh vertex IDs and
CanonicalRig joint IDs, then uses the shared Skin numeric court before minting
CanonicalSkinIR.

Legacy carrier/candidate IDs and learned-family names are evidence provenance,
not stable Skin identity.
"""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any, Mapping

from .canonical_mesh_ir_v1 import CanonicalMeshIR, canonical_mesh_lineage_hash_v1
from .canonical_rig_ir_v1 import CanonicalRigIR, canonical_rig_lineage_hash_v1
from .carrier_skin_v1 import CarrierSkinProposalIR
from .hashing import content_sha256
from .skin_simplex_qualification_v1 import qualify_influence_simplex_v1
from .types import QualificationError

Json = dict[str, Any]


@dataclass(frozen=True)
class CanonicalSkinRowIR:
    canonical_mesh_vertex_id: str
    influences: tuple[tuple[str, float], ...]
    simplex_residual_before: float
    correction_l1: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSkinQualificationIR:
    rows: tuple[CanonicalSkinRowIR, ...]
    mesh_lineage_hash: str
    rig_lineage_hash: str
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    proposal_evidence_hash: str
    qualification_report: Json
    qualification_hash: str
    schema_version: str = "RealSaS.CanonicalSkinQualificationIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalSkinIR:
    rows: tuple[CanonicalSkinRowIR, ...]
    mesh_lineage_hash: str
    rig_lineage_hash: str
    carrier_evidence_hash: str
    carrier_topology_hash: str
    carrier_geometry_hash: str
    proposal_evidence_hash: str
    qualification_hash: str
    skin_lineage_hash: str
    schema_version: str = "RealSaS.CanonicalSkinIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _proposal_evidence_hash(proposal: CarrierSkinProposalIR) -> str:
    return content_sha256(
        {
            "schema": "RealSaS.SkinInfluenceProposalEvidence.v1",
            "proposal": proposal.to_dict(),
        }
    )


def canonical_skin_qualification_hash_v1(value: CanonicalSkinQualificationIR) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def canonical_skin_lineage_hash_v1(value: CanonicalSkinIR) -> str:
    payload = value.to_dict()
    payload.pop("skin_lineage_hash", None)
    return content_sha256(payload)


def _validate_parent_authorities(mesh: CanonicalMeshIR, rig: CanonicalRigIR) -> None:
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
    expected = (
        mesh.carrier_evidence_hash,
        mesh.carrier_topology_hash,
        mesh.carrier_geometry_hash,
    )
    actual = (
        rig.mesh_carrier_evidence_hash,
        rig.mesh_carrier_topology_hash,
        rig.mesh_carrier_geometry_hash,
    )
    if actual != expected:
        raise QualificationError("CANONICAL_SKIN_RIG_CARRIER_PARENT_DRIFT")


def qualify_canonical_skin_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    proposal: CarrierSkinProposalIR,
    *,
    max_simplex_repair_l1: float,
    max_total_correction_l1: float,
    negative_tolerance: float,
    max_influences: int | None = None,
) -> CanonicalSkinQualificationIR:
    _validate_parent_authorities(mesh, rig)

    expected_carrier = (
        mesh.carrier_evidence_hash,
        mesh.carrier_topology_hash,
        mesh.carrier_geometry_hash,
    )
    proposal_carrier = (
        str(proposal.carrier_evidence_hash),
        str(proposal.carrier_topology_hash),
        str(proposal.carrier_geometry_hash),
    )
    if proposal_carrier != expected_carrier:
        raise QualificationError("CANONICAL_SKIN_PROPOSAL_CARRIER_BINDING_DRIFT")
    if str(proposal.skeleton_binding_hash) != rig.rig_lineage_hash:
        raise QualificationError("CANONICAL_SKIN_PROPOSAL_RIG_BINDING_DRIFT")

    meta = dict(proposal.metadata or {})
    if str(meta.get("output_domain") or "") not in {
        "MECHANICAL_CARRIER_M",
        "CANONICAL_MESH_CARRIER",
    }:
        raise QualificationError("CANONICAL_SKIN_OUTPUT_DOMAIN_INVALID")
    if meta.get("semantic_skin_transfer_performed") is not False:
        raise QualificationError("CANONICAL_SKIN_TRANSFER_FORBIDDEN")
    if meta.get("gsa_skin_field_authority_minted") is not False:
        raise QualificationError("CANONICAL_SKIN_GSA_AUTHORITY_FORBIDDEN")
    if meta.get("direct_simplex_readout") is not True:
        raise QualificationError("CANONICAL_SKIN_DIRECT_SIMPLEX_REQUIRED")
    if meta.get("source_skin_runtime_authority") is not False:
        raise QualificationError("CANONICAL_SKIN_SOURCE_RUNTIME_AUTHORITY_FORBIDDEN")
    if meta.get("mesh_mutation_invalidates_prediction") is not True:
        raise QualificationError("CANONICAL_SKIN_MESH_MUTATION_INVALIDATION_REQUIRED")

    by_source_candidate: dict[str, str] = {}
    for vertex in mesh.vertices:
        source_id = str(vertex.source_candidate_vertex_id)
        canonical_id = str(vertex.canonical_mesh_vertex_id)
        if not source_id or source_id in by_source_candidate:
            raise QualificationError("CANONICAL_SKIN_MESH_CARRIER_ID_MAP_INVALID")
        by_source_candidate[source_id] = canonical_id
    if len(by_source_candidate) != len(mesh.vertices) or not by_source_candidate:
        raise QualificationError("CANONICAL_SKIN_EMPTY_MESH_DOMAIN")

    mapped_influences: list[tuple[str, str, float]] = []
    for influence in proposal.influences:
        source_vid = str(influence.carrier_vertex_id)
        if source_vid not in by_source_candidate:
            raise QualificationError("CANONICAL_SKIN_UNKNOWN_CARRIER_VERTEX")
        mapped_influences.append(
            (
                by_source_candidate[source_vid],
                str(influence.canonical_joint_id),
                float(influence.weight),
            )
        )

    joint_ids = {str(joint.canonical_joint_id) for joint in rig.joints}
    ordered_vertices = tuple(vertex.canonical_mesh_vertex_id for vertex in mesh.vertices)
    numeric = qualify_influence_simplex_v1(
        ordered_domain_ids=ordered_vertices,
        allowed_joint_ids=joint_ids,
        influences=mapped_influences,
        max_simplex_repair_l1=float(max_simplex_repair_l1),
        max_total_correction_l1=float(max_total_correction_l1),
        negative_tolerance=float(negative_tolerance),
        max_influences=max_influences,
        error_prefix="CANONICAL_SKIN",
        domain_label="VERTEX",
    )
    rows = tuple(
        CanonicalSkinRowIR(
            canonical_mesh_vertex_id=row.domain_id,
            influences=row.influences,
            simplex_residual_before=row.simplex_residual_before,
            correction_l1=row.correction_l1,
        )
        for row in numeric.rows
    )

    report = {
        "status": "PASS_CANONICAL_SKIN_QUALIFIED",
        "canonical_mesh_vertex_count": len(ordered_vertices),
        "canonical_rig_joint_count": len(joint_ids),
        "prediction_carrier_coverage": 1.0,
        "simplex_max_abs_residual_before": numeric.max_simplex_residual_before,
        "max_row_correction_l1": numeric.max_row_correction_l1,
        "total_correction_l1": numeric.total_correction_l1,
        "max_influences": max_influences,
        "semantic_skin_transfer_performed": False,
        "gsa_skin_field_authority_minted": False,
        "source_skin_runtime_authority": False,
    }
    value = CanonicalSkinQualificationIR(
        rows=rows,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        proposal_evidence_hash=_proposal_evidence_hash(proposal),
        qualification_report=report,
        qualification_hash="",
        metadata={
            "authority": "COMPILER_SKIN_QUALIFICATION",
            "canonical_parent_set": ["CANONICAL_MESH", "CANONICAL_RIG"],
            "learned_family_embedded_in_schema": False,
            "legacy_carrier_ids_are_provenance_only": True,
            "feature_evidence_is_canonical_parent": False,
            "post_bind_frame_observability_claimed": False,
        },
    )
    return replace(
        value,
        qualification_hash=canonical_skin_qualification_hash_v1(value),
    )


def mint_canonical_skin_v1(
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIR,
    qualification: CanonicalSkinQualificationIR,
) -> CanonicalSkinIR:
    _validate_parent_authorities(mesh, rig)
    if qualification.qualification_hash != canonical_skin_qualification_hash_v1(qualification):
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_HASH_DRIFT")
    if qualification.qualification_report.get("status") != "PASS_CANONICAL_SKIN_QUALIFIED":
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_NOT_PASS")
    expected = (
        mesh.mesh_lineage_hash,
        rig.rig_lineage_hash,
        mesh.carrier_evidence_hash,
        mesh.carrier_topology_hash,
        mesh.carrier_geometry_hash,
    )
    actual = (
        qualification.mesh_lineage_hash,
        qualification.rig_lineage_hash,
        qualification.carrier_evidence_hash,
        qualification.carrier_topology_hash,
        qualification.carrier_geometry_hash,
    )
    if actual != expected:
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_PARENT_DRIFT")

    value = CanonicalSkinIR(
        rows=qualification.rows,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        carrier_evidence_hash=mesh.carrier_evidence_hash,
        carrier_topology_hash=mesh.carrier_topology_hash,
        carrier_geometry_hash=mesh.carrier_geometry_hash,
        proposal_evidence_hash=qualification.proposal_evidence_hash,
        qualification_hash=qualification.qualification_hash,
        skin_lineage_hash="",
        metadata={
            "authority": "COMPILER_CANONICAL_SKIN",
            "canonical_parent_set": ["CANONICAL_MESH", "CANONICAL_RIG"],
            "learned_family_embedded_in_schema": False,
            "learned_influences_are_proposal_evidence": True,
            "mesh_mutation_invalidates_skin": True,
            "rig_mutation_invalidates_skin": True,
            "semantic_skin_transfer_performed": False,
            "feature_evidence_is_canonical_parent": False,
            "post_bind_frame_observability_claimed": False,
            "mechanics_authority_claimed": False,
            "motion_authority_claimed": False,
        },
    )
    return replace(value, skin_lineage_hash=canonical_skin_lineage_hash_v1(value))


def _row_from_dict(payload: Mapping[str, Any]) -> CanonicalSkinRowIR:
    influences = tuple(
        (str(joint_id), float(weight))
        for joint_id, weight in payload.get("influences") or ()
    )
    if not influences or any(
        not joint_id or not math.isfinite(weight) or weight < 0.0
        for joint_id, weight in influences
    ):
        raise QualificationError("CANONICAL_SKIN_ROW_INVALID")
    if abs(sum(weight for _, weight in influences) - 1.0) > 1e-9:
        raise QualificationError("CANONICAL_SKIN_ROW_SIMPLEX_DRIFT")
    return CanonicalSkinRowIR(
        canonical_mesh_vertex_id=str(payload["canonical_mesh_vertex_id"]),
        influences=influences,
        simplex_residual_before=float(payload["simplex_residual_before"]),
        correction_l1=float(payload["correction_l1"]),
    )


def canonical_skin_qualification_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalSkinQualificationIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalSkinQualificationIR.v1":
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_SCHEMA_DRIFT")
    value = CanonicalSkinQualificationIR(
        rows=tuple(_row_from_dict(row) for row in payload.get("rows") or ()),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        proposal_evidence_hash=str(payload["proposal_evidence_hash"]),
        qualification_report=dict(payload.get("qualification_report") or {}),
        qualification_hash=str(payload["qualification_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if value.qualification_hash != canonical_skin_qualification_hash_v1(value):
        raise QualificationError("CANONICAL_SKIN_QUALIFICATION_HASH_DRIFT")
    return value


def canonical_skin_from_dict_v1(payload: Mapping[str, Any]) -> CanonicalSkinIR:
    schema = str(payload.get("schema_version") or payload.get("schema") or "")
    if schema != "RealSaS.CanonicalSkinIR.v1":
        raise QualificationError("CANONICAL_SKIN_SCHEMA_DRIFT")
    value = CanonicalSkinIR(
        rows=tuple(_row_from_dict(row) for row in payload.get("rows") or ()),
        mesh_lineage_hash=str(payload["mesh_lineage_hash"]),
        rig_lineage_hash=str(payload["rig_lineage_hash"]),
        carrier_evidence_hash=str(payload["carrier_evidence_hash"]),
        carrier_topology_hash=str(payload["carrier_topology_hash"]),
        carrier_geometry_hash=str(payload["carrier_geometry_hash"]),
        proposal_evidence_hash=str(payload["proposal_evidence_hash"]),
        qualification_hash=str(payload["qualification_hash"]),
        skin_lineage_hash=str(payload["skin_lineage_hash"]),
        schema_version=schema,
        metadata=dict(payload.get("metadata") or {}),
    )
    if not value.rows:
        raise QualificationError("CANONICAL_SKIN_EMPTY")
    if value.skin_lineage_hash != canonical_skin_lineage_hash_v1(value):
        raise QualificationError("CANONICAL_SKIN_LINEAGE_HASH_DRIFT")
    if value.metadata.get("authority") != "COMPILER_CANONICAL_SKIN":
        raise QualificationError("CANONICAL_SKIN_AUTHORITY_DRIFT")
    if value.metadata.get("learned_family_embedded_in_schema") is not False:
        raise QualificationError("CANONICAL_SKIN_LEARNED_FAMILY_SCHEMA_FORBIDDEN")
    if value.metadata.get("post_bind_frame_observability_claimed") is not False:
        raise QualificationError("CANONICAL_SKIN_PREMATURE_MECHANICS_AUTHORITY")
    return value


__all__ = [
    "CanonicalSkinIR",
    "CanonicalSkinQualificationIR",
    "CanonicalSkinRowIR",
    "canonical_skin_from_dict_v1",
    "canonical_skin_lineage_hash_v1",
    "canonical_skin_qualification_from_dict_v1",
    "canonical_skin_qualification_hash_v1",
    "mint_canonical_skin_v1",
    "qualify_canonical_skin_v1",
]
