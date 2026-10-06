from __future__ import annotations

"""Bind TESSA learned geometry to the existing V2 Stage19 static authority.

This module intentionally creates no parallel geometry metric and no second G1.
The 46-stage V2 DAG already owns static carrier admission at
``19_STATIC_CANONICAL_MESH_QUALIFIED``: it remeasures the exact Stage18 candidate
for rest conditioning and eight-view source fidelity under the frozen Stage13
policy.

TESSA only needs a fail-closed lineage bridge proving that the Stage19 PASS is for
the exact learned proposal/candidate and the exact material-support field.  The
material field remains non-geometric; actual motion remains downstream.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

from .hashing import content_sha256
from .product_authority_v1 import CanonicalMeshCandidateIR
from .surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from .tessa_candidate_bridge_v1 import TESSACandidateBridgeEvidenceIR
from .types import QualificationError

Json = dict[str, Any]
_LEARNED_GEOMETRY_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"


@dataclass(frozen=True)
class TESSAStaticCarrierBindingIR:
    candidate_lineage_hash: str
    bridge_evidence_hash: str
    static_mesh_qualification_hash: str
    proposal_geometry_hash: str
    material_support_field_hash: str
    partition_binding_hash: str
    binding_hash: str
    schema_version: str = "RealSaS.TESSAStaticCarrierBindingIR.v1"
    metadata: Json = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def tessa_static_carrier_binding_from_dict_v1(
    payload: Mapping[str, Any],
) -> TESSAStaticCarrierBindingIR:
    if str(payload.get("schema_version") or payload.get("schema") or "") != "RealSaS.TESSAStaticCarrierBindingIR.v1":
        raise QualificationError("TESSA_STATIC_BINDING_SCHEMA_INVALID")
    return TESSAStaticCarrierBindingIR(
        candidate_lineage_hash=str(payload.get("candidate_lineage_hash") or ""),
        bridge_evidence_hash=str(payload.get("bridge_evidence_hash") or ""),
        static_mesh_qualification_hash=str(payload.get("static_mesh_qualification_hash") or ""),
        proposal_geometry_hash=str(payload.get("proposal_geometry_hash") or ""),
        material_support_field_hash=str(payload.get("material_support_field_hash") or ""),
        partition_binding_hash=str(payload.get("partition_binding_hash") or ""),
        binding_hash=str(payload.get("binding_hash") or ""),
        metadata=dict(payload.get("metadata") or {}),
    )


def tessa_static_carrier_binding_hash_v1(value: TESSAStaticCarrierBindingIR) -> str:
    payload = value.to_dict()
    payload.pop("binding_hash", None)
    return content_sha256(payload)


def _validate_tessa_candidate_boundary(candidate: CanonicalMeshCandidateIR) -> None:
    if candidate.producer_id != "RealSaS.TESSALearnedMechanicalCarrierProposal.v1":
        raise QualificationError("TESSA_STATIC_BINDING_PRODUCER_INVALID")
    md = dict(candidate.metadata or {})
    if md.get("geometry_authority_class") != _LEARNED_GEOMETRY_AUTHORITY_CLASS:
        raise QualificationError("TESSA_STATIC_BINDING_AUTHORITY_CLASS_INVALID")
    if md.get("material_support_is_not_geometry_support") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_SEPARATION_MISSING")
    if md.get("learned_xyz_preserved_exactly") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_LEARNED_XYZ_PRESERVATION_MISSING")
    if md.get("legacy_g1_convex_lift_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_LEGACY_G1_CLAIM_FORBIDDEN")
    if md.get("product_geometry_authority_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_PREMATURE_PRODUCT_AUTHORITY")
    if not str(md.get("candidate_bridge_evidence_hash") or ""):
        raise QualificationError("TESSA_STATIC_BINDING_BRIDGE_HASH_MISSING")
    if not str(md.get("proposal_geometry_hash") or ""):
        raise QualificationError("TESSA_STATIC_BINDING_PROPOSAL_HASH_MISSING")
    if not str(md.get("material_support_field_hash") or ""):
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_FIELD_HASH_MISSING")


def _validate_stage19_pass(
    static_mesh: StaticCanonicalMeshQualificationIR,
    *,
    candidate: CanonicalMeshCandidateIR,
) -> None:
    if static_mesh.qualification_hash != static_mesh_qualification_hash(static_mesh):
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_HASH_MISMATCH")
    if static_mesh.candidate_mesh_binding_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_CANDIDATE_MISMATCH")
    report = dict(static_mesh.qualification_report or {})
    if report.get("status") != "PASS_STATIC_CANONICAL_CARRIER":
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_STATUS_NOT_PASS")
    if report.get("static_quality_policy_passed") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_STATIC_QUALITY_NOT_PASS")
    if report.get("stage13_policy_replayed_on_actual_candidate_mesh") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_POLICY_REPLAY_MISSING")
    if report.get("actual_candidate_source_fidelity_passed") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_SOURCE_FIDELITY_NOT_PASS")
    if report.get("source_fidelity_qualification_passed") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_SOURCE_QUALIFICATION_NOT_PASS")
    if report.get("demo_geometry_lineage") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_DEMO_LINEAGE_FORBIDDEN")
    if report.get("product_authority_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_PREMATURE_PRODUCT_AUTHORITY")
    per_view = tuple(report.get("actual_candidate_source_fidelity_views") or ())
    if len(per_view) != 8 or {int(row.get("view_index", -1)) for row in per_view} != set(range(8)):
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_REQUIRES_EIGHT_VIEWS")
    if any(row.get("passed") is not True for row in per_view):
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_VIEW_NOT_PASS")


def validate_tessa_static_carrier_binding_v1(
    value: TESSAStaticCarrierBindingIR,
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    static_mesh: StaticCanonicalMeshQualificationIR,
) -> None:
    _validate_tessa_candidate_boundary(candidate)
    _validate_stage19_pass(static_mesh, candidate=candidate)
    cmd = dict(candidate.metadata or {})
    if value.candidate_lineage_hash != candidate.candidate_lineage_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_LINEAGE_MISMATCH")
    if value.bridge_evidence_hash != bridge_evidence.evidence_hash:
        raise QualificationError("TESSA_STATIC_BINDING_BRIDGE_EVIDENCE_MISMATCH")
    if str(cmd.get("candidate_bridge_evidence_hash") or "") != bridge_evidence.evidence_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_BRIDGE_DRIFT")
    if value.static_mesh_qualification_hash != static_mesh.qualification_hash:
        raise QualificationError("TESSA_STATIC_BINDING_STAGE19_BINDING_MISMATCH")
    if value.proposal_geometry_hash != bridge_evidence.proposal_geometry_hash:
        raise QualificationError("TESSA_STATIC_BINDING_PROPOSAL_GEOMETRY_MISMATCH")
    if str(cmd.get("proposal_geometry_hash") or "") != value.proposal_geometry_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_PROPOSAL_DRIFT")
    if value.material_support_field_hash != bridge_evidence.material_support_field_hash:
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_FIELD_MISMATCH")
    if str(cmd.get("material_support_field_hash") or "") != value.material_support_field_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_SUPPORT_FIELD_DRIFT")
    if value.partition_binding_hash != candidate.partition_binding_hash:
        raise QualificationError("TESSA_STATIC_BINDING_PARTITION_MISMATCH")
    if value.partition_binding_hash != bridge_evidence.partition_binding_hash:
        raise QualificationError("TESSA_STATIC_BINDING_BRIDGE_PARTITION_MISMATCH")
    md = dict(value.metadata or {})
    if md.get("authority") != "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER":
        raise QualificationError("TESSA_STATIC_BINDING_AUTHORITY_INVALID")
    if md.get("parallel_source_fidelity_metric_created") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_PARALLEL_METRIC_FORBIDDEN")
    if md.get("material_support_is_geometry_authority") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_GEOMETRY_AUTHORITY_FORBIDDEN")
    if md.get("motion_capability_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_MOTION_AUTHORITY_FORBIDDEN")
    if md.get("generalization_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_GENERALIZATION_AUTHORITY_FORBIDDEN")
    if value.binding_hash != tessa_static_carrier_binding_hash_v1(value):
        raise QualificationError("TESSA_STATIC_BINDING_HASH_MISMATCH")


def bind_tessa_to_static_carrier_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    static_mesh: StaticCanonicalMeshQualificationIR,
) -> TESSAStaticCarrierBindingIR:
    """Bind exact TESSA lineage to the already-qualified Stage19 carrier."""
    _validate_tessa_candidate_boundary(candidate)
    _validate_stage19_pass(static_mesh, candidate=candidate)
    cmd = dict(candidate.metadata or {})
    if str(cmd.get("candidate_bridge_evidence_hash") or "") != bridge_evidence.evidence_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_BRIDGE_DRIFT")
    if str(cmd.get("proposal_geometry_hash") or "") != bridge_evidence.proposal_geometry_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_PROPOSAL_DRIFT")
    if str(cmd.get("material_support_field_hash") or "") != bridge_evidence.material_support_field_hash:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_SUPPORT_FIELD_DRIFT")
    if candidate.partition_binding_hash != bridge_evidence.partition_binding_hash:
        raise QualificationError("TESSA_STATIC_BINDING_BRIDGE_PARTITION_MISMATCH")

    provisional = TESSAStaticCarrierBindingIR(
        candidate_lineage_hash=str(candidate.candidate_lineage_hash),
        bridge_evidence_hash=str(bridge_evidence.evidence_hash),
        static_mesh_qualification_hash=str(static_mesh.qualification_hash),
        proposal_geometry_hash=str(bridge_evidence.proposal_geometry_hash),
        material_support_field_hash=str(bridge_evidence.material_support_field_hash),
        partition_binding_hash=str(candidate.partition_binding_hash),
        binding_hash="",
        metadata={
            "authority": "EXISTING_STAGE19_STATIC_CANONICAL_CARRIER",
            "parallel_source_fidelity_metric_created": False,
            "material_support_is_geometry_authority": False,
            "learned_xyz_preserved_exactly": True,
            "stage19_static_source_fidelity_required": True,
            "motion_capability_claimed": False,
            "generalization_claimed": False,
        },
    )
    value = replace(
        provisional,
        binding_hash=tessa_static_carrier_binding_hash_v1(provisional),
    )
    validate_tessa_static_carrier_binding_v1(
        value,
        candidate=candidate,
        bridge_evidence=bridge_evidence,
        static_mesh=static_mesh,
    )
    return value
