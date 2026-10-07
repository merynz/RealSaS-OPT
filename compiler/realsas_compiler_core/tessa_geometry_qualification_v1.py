from __future__ import annotations

"""Bind TESSA learned geometry to the existing V2 Stage19 static authority.

Stage19 remains the sole static mechanical-carrier authority. TESSA owns the
learned proposal; Compiler may deterministically repair it under the frozen
quality policy before Stage19. GSA material support remains non-geometric and
motion authority remains downstream.

The recovered Knight FIT1 research lineage proves a static TESSA carrier and a
separate MIRA Raw41 field, but it does *not* prove that the generic product
``TESSAMaterialSupportFieldIR`` transport is mechanically equivalent to the
research court's source-topology barycentric-adjoint/harmonic binding. Therefore
Stage19 may seal static TESSA geometry while downstream QualifiedMesh minting is
fail-closed until a product-authoritative carrier-field binding is explicitly
qualified.
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
_INITIAL_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_PROPOSAL_V1"
_REPAIRED_AUTHORITY_CLASS = "TESSA_LEARNED_GEOMETRY_COMPILER_REPAIRED_V1"
_TESSA_PRODUCER = "RealSaS.TESSALearnedMechanicalCarrierProposal.v1"
_RESEARCH_FIT1_BINDING_RULE = (
    "GSA_BARYCENTRIC_ADJOINT_ANCHOR__COMPONENT_HARMONIC__"
    "EXACT_RIGID_FALLBACK_V1"
)
_DYNAMIC_FIELD_BLOCKER = "TESSA_MIRA_CARRIER_FIELD_BINDING_NOT_PRODUCT_QUALIFIED"


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


def _validate_tessa_candidate_boundary(candidate: CanonicalMeshCandidateIR) -> str:
    if candidate.producer_id != _TESSA_PRODUCER:
        raise QualificationError("TESSA_STATIC_BINDING_PRODUCER_INVALID")
    md = dict(candidate.metadata or {})
    repaired = md.get("compiler_static_repair_applied") is True
    if repaired:
        if md.get("geometry_authority_class") != _REPAIRED_AUTHORITY_CLASS:
            raise QualificationError("TESSA_STATIC_BINDING_REPAIRED_AUTHORITY_CLASS_INVALID")
        if md.get("learned_proposal_is_immutable_reference") is not True:
            raise QualificationError("TESSA_STATIC_BINDING_IMMUTABLE_REFERENCE_MISSING")
        if md.get("learned_proposal_xyz_preserved_exactly") is not False:
            raise QualificationError("TESSA_STATIC_BINDING_REPAIRED_PROPOSAL_FLAG_INVALID")
        if md.get("learned_xyz_preserved_exactly") is not True:
            raise QualificationError("TESSA_STATIC_BINDING_POST_REPAIR_PRESERVATION_MISSING")
        if md.get("learned_xyz_preservation_scope") != "POST_COMPILER_REPAIR_CANDIDATE_TO_DOWNSTREAM":
            raise QualificationError("TESSA_STATIC_BINDING_PRESERVATION_SCOPE_INVALID")
        if md.get("threshold_relaxation_performed") is not False:
            raise QualificationError("TESSA_STATIC_BINDING_THRESHOLD_RELAXATION_FORBIDDEN")
        if md.get("teacher_geometry_used_to_repair") is not False:
            raise QualificationError("TESSA_STATIC_BINDING_TEACHER_REPAIR_FORBIDDEN")
        if not str(md.get("tessa_static_repair_evidence_hash") or ""):
            raise QualificationError("TESSA_STATIC_BINDING_REPAIR_EVIDENCE_HASH_MISSING")
        if not str(md.get("tessa_reference_binding_hash") or ""):
            raise QualificationError("TESSA_STATIC_BINDING_REFERENCE_HASH_MISSING")
        lane = "COMPILER_REPAIRED"
    else:
        if md.get("geometry_authority_class") != _INITIAL_AUTHORITY_CLASS:
            raise QualificationError("TESSA_STATIC_BINDING_AUTHORITY_CLASS_INVALID")
        if md.get("learned_xyz_preserved_exactly") is not True:
            raise QualificationError("TESSA_STATIC_BINDING_LEARNED_XYZ_PRESERVATION_MISSING")
        lane = "LEARNED_PROPOSAL"

    if md.get("material_support_is_not_geometry_support") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_SEPARATION_MISSING")
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
    return lane


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
    require_dynamic_carrier_field_binding: bool = True,
) -> None:
    lane = _validate_tessa_candidate_boundary(candidate)
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
    if md.get("candidate_lane") != lane:
        raise QualificationError("TESSA_STATIC_BINDING_CANDIDATE_LANE_MISMATCH")
    if md.get("parallel_source_fidelity_metric_created") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_PARALLEL_METRIC_FORBIDDEN")
    if md.get("material_support_is_geometry_authority") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_SUPPORT_GEOMETRY_AUTHORITY_FORBIDDEN")
    if md.get("motion_capability_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_MOTION_AUTHORITY_FORBIDDEN")
    if md.get("generalization_claimed") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_GENERALIZATION_AUTHORITY_FORBIDDEN")
    if md.get("dynamic_carrier_field_binding_required_before_qualified_mesh") is not True:
        raise QualificationError("TESSA_STATIC_BINDING_DYNAMIC_INTERLOCK_MISSING")
    if md.get("research_fit1_binding_rule") != _RESEARCH_FIT1_BINDING_RULE:
        raise QualificationError("TESSA_STATIC_BINDING_RESEARCH_RULE_DRIFT")
    if md.get("research_fit1_binding_is_product_authority") is not False:
        raise QualificationError("TESSA_STATIC_BINDING_RESEARCH_AUTHORITY_OVERCLAIM")
    dynamic_binding_qualified = md.get("dynamic_carrier_field_binding_product_qualified")
    if dynamic_binding_qualified not in {False, True}:
        raise QualificationError("TESSA_STATIC_BINDING_DYNAMIC_QUALIFICATION_FLAG_INVALID")

    if lane == "COMPILER_REPAIRED":
        if md.get("static_repair_evidence_hash") != cmd.get("tessa_static_repair_evidence_hash"):
            raise QualificationError("TESSA_STATIC_BINDING_REPAIR_EVIDENCE_DRIFT")
        if md.get("reference_binding_hash") != cmd.get("tessa_reference_binding_hash"):
            raise QualificationError("TESSA_STATIC_BINDING_REFERENCE_HASH_DRIFT")
    if value.binding_hash != tessa_static_carrier_binding_hash_v1(value):
        raise QualificationError("TESSA_STATIC_BINDING_HASH_MISMATCH")

    # Static Stage19 qualification and dynamic carrier-field qualification are
    # intentionally different authorities. QualifiedMesh/Stage35 callers use the
    # default True and therefore fail closed today. Stage19's own construction
    # validates the same receipt with this requirement explicitly disabled.
    if require_dynamic_carrier_field_binding and dynamic_binding_qualified is not True:
        raise QualificationError(_DYNAMIC_FIELD_BLOCKER)


def bind_tessa_to_static_carrier_v1(
    *,
    candidate: CanonicalMeshCandidateIR,
    bridge_evidence: TESSACandidateBridgeEvidenceIR,
    static_mesh: StaticCanonicalMeshQualificationIR,
    static_repair_evidence_hash: str | None = None,
    reference_binding_hash: str | None = None,
) -> TESSAStaticCarrierBindingIR:
    """Bind exact Stage18 TESSA lineage to the qualified Stage19 carrier.

    Repaired candidates commit to their repair/reference evidence hashes directly
    in candidate metadata, so legacy Stage19 callers need no new positional/data
    dependency merely to preserve the cryptographic closure.

    This receipt is *static authority only*. The historical Knight FIT1
    GSA->source/TESSA binding remains research evidence, not product authority, so
    this function deliberately emits the downstream dynamic interlock as closed.
    """
    lane = _validate_tessa_candidate_boundary(candidate)
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

    if lane == "COMPILER_REPAIRED":
        committed_repair = str(cmd.get("tessa_static_repair_evidence_hash") or "")
        committed_reference = str(cmd.get("tessa_reference_binding_hash") or "")
        if static_repair_evidence_hash is not None and str(static_repair_evidence_hash) != committed_repair:
            raise QualificationError("TESSA_STATIC_BINDING_EXPLICIT_REPAIR_HASH_DRIFT")
        if reference_binding_hash is not None and str(reference_binding_hash) != committed_reference:
            raise QualificationError("TESSA_STATIC_BINDING_EXPLICIT_REFERENCE_HASH_DRIFT")
        static_repair_evidence_hash = committed_repair
        reference_binding_hash = committed_reference
    else:
        static_repair_evidence_hash = ""
        reference_binding_hash = ""

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
            "candidate_lane": lane,
            "parallel_source_fidelity_metric_created": False,
            "material_support_is_geometry_authority": False,
            "learned_xyz_preserved_exactly": True,
            "learned_xyz_preservation_scope": (
                "POST_COMPILER_REPAIR_CANDIDATE_TO_DOWNSTREAM"
                if lane == "COMPILER_REPAIRED"
                else "LEARNED_PROPOSAL_TO_DOWNSTREAM"
            ),
            "compiler_static_repair_applied": lane == "COMPILER_REPAIRED",
            "static_repair_evidence_hash": str(static_repair_evidence_hash or ""),
            "reference_binding_hash": str(reference_binding_hash or ""),
            "stage19_static_source_fidelity_required": True,
            "motion_capability_claimed": False,
            "generalization_claimed": False,
            "dynamic_carrier_field_binding_required_before_qualified_mesh": True,
            "dynamic_carrier_field_binding_product_qualified": False,
            "research_fit1_binding_rule": _RESEARCH_FIT1_BINDING_RULE,
            "research_fit1_binding_is_product_authority": False,
            "dynamic_interlock_blocker": _DYNAMIC_FIELD_BLOCKER,
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
        require_dynamic_carrier_field_binding=False,
    )
    return value
