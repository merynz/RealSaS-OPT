from __future__ import annotations

"""Fail-closed learned-geometry qualification for TESSA candidates.

Legacy product G1 proves that deterministic mesh XYZ is an exact/bounded lift of
``RiggingSurfaceIR`` support.  TESSA intentionally solves a different problem:
it predicts a production mesh from observation-grounded GSA evidence.  Requiring
its XYZ to equal a convex GSA lift would collapse the learned geometry model back
into the deterministic substrate.

This module defines the separate G1B evidence contract.  It does *not* replace
legacy G1 and it does not mint a QualifiedMeshIR.  It proves only that:

1. the exact learned candidate is lineage-bound to the typed TESSA bridge;
2. material/mechanical support remains explicitly separate from geometry XYZ;
3. independent eight-view G5 source-fidelity evidence for that exact candidate
   is complete and PASS.

Topology/rest conditioning, deformation stress, skin-topology compatibility and
actual motion remain their existing independent gates.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from .hashing import content_sha256
from .mesh.product_coverage_v1 import g5_coverage_evidence_hash
from .product_authority_v1 import (
    CanonicalMeshCandidateIR,
    ComponentCarrierPolicyIR,
    MechanicalPartitionIR,
    validate_canonical_mesh_candidate,
)
from .tessa_candidate_bridge_v1 import TESSACandidateBridgeEvidenceIR
from .types import QualificationError, RiggingSurfaceIR

Json = dict[str, Any]
_LEARNED_GEOMETRY_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"


@dataclass(frozen=True)
class TESSALearnedGeometryQualificationIR:
    candidate_lineage_hash: str
    bridge_evidence_hash: str
    proposal_geometry_hash: str
    material_support_field_hash: str
    surface_binding_hash: str
    partition_binding_hash: str
    carrier_policy_binding_hash: str
    g5_evidence_hash: str
    g5_cell_count: int
    vertex_count: int
    face_count: int
    qualification_hash: str
    schema_version: str = "RealSaS.TESSALearnedGeometryQualificationIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def tessa_learned_geometry_qualification_hash_v1(
    value: TESSALearnedGeometryQualificationIR,
) -> str:
    payload = value.to_dict()
    payload.pop("qualification_hash", None)
    return content_sha256(payload)


def _validate_candidate_authority_split(candidate: CanonicalMeshCandidateIR) -> None:
    if candidate.producer_id != "RealSaS.TESSALearnedMechanicalCarrierProposal.v1":
        raise QualificationError("TESSA_G1B_PRODUCER_INVALID")
    md = dict(candidate.metadata or {})
    if md.get("geometry_authority_class") != _LEARNED_GEOMETRY_AUTHORITY_CLASS:
        raise QualificationError("TESSA_G1B_AUTHORITY_CLASS_INVALID")
    if md.get("material_support_is_not_geometry_support") is not True:
        raise QualificationError("TESSA_G1B_SUPPORT_SEPARATION_MISSING")
    if md.get("learned_xyz_preserved_exactly") is not True:
        raise QualificationError("TESSA_G1B_LEARNED_XYZ_PRESERVATION_MISSING")
    if md.get("legacy_g1_convex_lift_claimed") is not False:
        raise QualificationError("TESSA_G1B_LEGACY_CONVEX_LIFT_FORBIDDEN")
    if md.get("product_geometry_authority_claimed") is not False:
        raise QualificationError("TESSA_G1B_PREMATURE_PRODUCT_AUTHORITY")

    evidence_hash = str(md.get("candidate_bridge_evidence_hash") or "")
    proposal_hash = str(md.get("proposal_geometry_hash") or "")
    support_hash = str(md.get("material_support_field_hash") or "")
    if not evidence_hash or not proposal_hash or not support_hash:
        raise QualificationError("TESSA_G1B_CANDIDATE_EVIDENCE_BINDING_MISSING")

    for vertex in candidate.vertices:
        vmd = dict(vertex.metadata or {})
        if vmd.get("geometry_authority_class") != _LEARNED_GEOMETRY_AUTHORITY_CLASS:
            raise QualificationError("TESSA_G1B_VERTEX_AUTHORITY_CLASS_INVALID")
        if vmd.get("material_support_is_not_geometry_support") is not True:
            raise QualificationError("TESSA_G1B_VERTEX_SUPPORT_SEPARATION_MISSING")
        if vmd.get("geometry_position_derived_from_material_support") is not False:
            raise QualificationError("TESSA_G1B_VERTEX_GEOMETRY_SUPPORT_CONFLATION")
        if vmd.get("teacher_vertex_index_used") is not False:
            raise QualificationError("TESSA_G1B_TEACHER_VERTEX_INDEX_FORBIDDEN")
        if str(vmd.get("proposal_geometry_hash") or "") != proposal_hash:
            raise QualificationError("TESSA_G1B_VERTEX_PROPOSAL_HASH_DRIFT")
        if str(vmd.get("material_support_field_hash") or "") != support_hash:
            raise QualificationError("TESSA_G1B_VERTEX_SUPPORT_FIELD_HASH_DRIFT")
        bmd = dict(vertex.support_binding.metadata or {})
        if bmd.get("authority_class") != "MATERIAL_SUPPORT_ONLY":
            raise QualificationError("TESSA_G1B_VERTEX_SUPPORT_AUTHORITY_INVALID")
        if bmd.get("geometry_position_derived_from_support") is not False:
            raise QualificationError("TESSA_G1B_VERTEX_SUPPORT_GEOMETRY_CLAIM_FORBIDDEN")
        if bmd.get("teacher_vertex_index_used") is not False:
            raise QualificationError("TESSA_G1B_VERTEX_SUPPORT_TEACHER_INDEX_FORBIDDEN")
        if vertex.refinement is not None:
            raise QualificationError("TESSA_G1B_LEGACY_REFINEMENT_ENCODING_FORBIDDEN")


def validate_tessa_learned_geometry_qualification_v1(
    value: TESSALearnedGeometryQualificationIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    g5_rows: Iterable[dict],
) -> None:
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    _validate_candidate_authority_split(candidate)
    rows = tuple(dict(row) for row in g5_rows)
    if not rows or any(str(row.get("status") or "") != "PASS" for row in rows):
        raise QualificationError("TESSA_G1B_G5_REQUIRES_ALL_CELLS_PASS")
    expected_g5_hash = g5_coverage_evidence_hash(rows)

    cmd = dict(candidate.metadata or {})
    if value.candidate_lineage_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_G1B_CANDIDATE_LINEAGE_MISMATCH")
    if value.bridge_evidence_hash != bridge_evidence.evidence_hash:
        raise QualificationError("TESSA_G1B_BRIDGE_EVIDENCE_MISMATCH")
    if str(cmd.get("candidate_bridge_evidence_hash") or "") != bridge_evidence.evidence_hash:
        raise QualificationError("TESSA_G1B_CANDIDATE_BRIDGE_BINDING_MISMATCH")
    if value.proposal_geometry_hash != bridge_evidence.proposal_geometry_hash:
        raise QualificationError("TESSA_G1B_PROPOSAL_GEOMETRY_MISMATCH")
    if value.material_support_field_hash != bridge_evidence.material_support_field_hash:
        raise QualificationError("TESSA_G1B_SUPPORT_FIELD_MISMATCH")
    if value.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("TESSA_G1B_SURFACE_BINDING_MISMATCH")
    if value.partition_binding_hash != partition.partition_lineage_hash:
        raise QualificationError("TESSA_G1B_PARTITION_BINDING_MISMATCH")
    if value.carrier_policy_binding_hash != carrier_policy.carrier_policy_lineage_hash:
        raise QualificationError("TESSA_G1B_CARRIER_POLICY_BINDING_MISMATCH")
    if value.g5_evidence_hash != expected_g5_hash:
        raise QualificationError("TESSA_G1B_G5_EVIDENCE_HASH_MISMATCH")
    if value.g5_cell_count != len(rows):
        raise QualificationError("TESSA_G1B_G5_CELL_COUNT_MISMATCH")
    if value.vertex_count != len(candidate.vertices) or value.face_count != len(candidate.faces):
        raise QualificationError("TESSA_G1B_GEOMETRY_COUNT_MISMATCH")

    md = dict(value.metadata or {})
    if md.get("status") != "PASS_LEARNED_GEOMETRY_SOURCE_FIDELITY":
        raise QualificationError("TESSA_G1B_STATUS_NOT_PASS")
    if md.get("legacy_g1_replaced") is not False:
        raise QualificationError("TESSA_G1B_LEGACY_G1_REPLACEMENT_FORBIDDEN")
    if md.get("material_support_is_geometry_authority") is not False:
        raise QualificationError("TESSA_G1B_MATERIAL_SUPPORT_AUTHORITY_FORBIDDEN")
    if md.get("g5_is_independent_source_fidelity_authority") is not True:
        raise QualificationError("TESSA_G1B_G5_INDEPENDENCE_MISSING")
    if md.get("motion_capability_claimed") is not False:
        raise QualificationError("TESSA_G1B_MOTION_AUTHORITY_FORBIDDEN")
    if md.get("generalization_claimed") is not False:
        raise QualificationError("TESSA_G1B_GENERALIZATION_AUTHORITY_FORBIDDEN")
    if value.qualification_hash != tessa_learned_geometry_qualification_hash_v1(value):
        raise QualificationError("TESSA_G1B_QUALIFICATION_HASH_MISMATCH")


def qualify_tessa_learned_geometry_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    surface: RiggingSurfaceIR,
    partition: MechanicalPartitionIR,
    carrier_policy: ComponentCarrierPolicyIR,
    g5_rows: Iterable[dict],
) -> TESSALearnedGeometryQualificationIR:
    """Seal G1B only after independent G5 passes on the exact candidate."""
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
    )
    _validate_candidate_authority_split(candidate)
    rows = tuple(dict(row) for row in g5_rows)
    if not rows or any(str(row.get("status") or "") != "PASS" for row in rows):
        raise QualificationError("TESSA_G1B_G5_REQUIRES_ALL_CELLS_PASS")

    provisional = TESSALearnedGeometryQualificationIR(
        candidate_lineage_hash=str(candidate.candidate_lineage_hash),
        bridge_evidence_hash=str(bridge_evidence.evidence_hash),
        proposal_geometry_hash=str(bridge_evidence.proposal_geometry_hash),
        material_support_field_hash=str(bridge_evidence.material_support_field_hash),
        surface_binding_hash=str(surface.geometry_lineage_hash),
        partition_binding_hash=str(partition.partition_lineage_hash),
        carrier_policy_binding_hash=str(carrier_policy.carrier_policy_lineage_hash),
        g5_evidence_hash=g5_coverage_evidence_hash(rows),
        g5_cell_count=len(rows),
        vertex_count=len(candidate.vertices),
        face_count=len(candidate.faces),
        qualification_hash="",
        metadata={
            "status": "PASS_LEARNED_GEOMETRY_SOURCE_FIDELITY",
            "authority_class": "TESSA_LEARNED_GEOMETRY_G1B",
            "legacy_g1_replaced": False,
            "legacy_g1_convex_lift_applicable": False,
            "material_support_is_geometry_authority": False,
            "g5_is_independent_source_fidelity_authority": True,
            "topology_rest_conditioning_owned_elsewhere": True,
            "deformation_stress_owned_elsewhere": True,
            "skin_topology_compatibility_owned_elsewhere": True,
            "motion_capability_claimed": False,
            "generalization_claimed": False,
        },
    )
    value = replace(
        provisional,
        qualification_hash=tessa_learned_geometry_qualification_hash_v1(provisional),
    )
    validate_tessa_learned_geometry_qualification_v1(
        value,
        candidate=candidate,
        bridge_evidence=bridge_evidence,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        g5_rows=rows,
    )
    return value
