from __future__ import annotations

"""Fail-closed application seam for Stage35 -> Stage17 mechanical repartition.

The Stage35 diagnostic directive is not execution authority.  A child attempt may
apply it only when a separately sealed trustworthy-skin authorization binds the
exact parent surface/partition/skin evidence that produced the directive.
"""

from dataclasses import replace

from .hashing import content_sha256
from .mechanical_partition_v1 import build_structural_partition
from .product_authority_v1 import (
    ComponentBoundaryConstraintIR,
    mechanical_partition_lineage_hash,
)
from .types import QualificationError

AUTH_SCHEMA="RealSaS.TrustworthySkinRepartitionAuthorization.v1"
AUTH_STATUS="PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION"
DIRECTIVE_SCHEMA="RealSaS.MechanicalRepartitionDirective.v2"


def repartition_authorization_hash_v1(payload:dict)->str:
    value=dict(payload)
    value.pop("authorization_hash",None)
    return content_sha256(value)


def validate_trustworthy_skin_repartition_authorization_v1(
    authorization:dict,
    *,
    directive:dict,
    surface,
    parent_partition,
)->None:
    if str(directive.get("schema") or "")!=DIRECTIVE_SCHEMA:
        raise QualificationError("REPARTITION_DIRECTIVE_SCHEMA_DRIFT")
    if str(directive.get("status") or "")!="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY":
        raise QualificationError("REPARTITION_DIRECTIVE_NOT_APPLICABLE")
    if directive.get("face_deletion_count")!=0 or directive.get("weight_mutation") is not False:
        raise QualificationError("REPARTITION_DIRECTIVE_MUTATION_SCOPE_INVALID")
    if directive.get("requires_trustworthy_skin_reliability_authority") is not True:
        raise QualificationError("REPARTITION_DIRECTIVE_RELIABILITY_GATE_MISSING")
    if str(directive.get("source_surface_lineage_hash") or "")!=str(surface.geometry_lineage_hash):
        raise QualificationError("REPARTITION_DIRECTIVE_SURFACE_LINEAGE_DRIFT")
    if str(directive.get("source_partition_lineage_hash") or "")!=str(parent_partition.partition_lineage_hash):
        raise QualificationError("REPARTITION_DIRECTIVE_PARENT_PARTITION_DRIFT")

    if str(authorization.get("schema") or "")!=AUTH_SCHEMA:
        raise QualificationError("REPARTITION_AUTHORIZATION_SCHEMA_DRIFT")
    if str(authorization.get("status") or "")!=AUTH_STATUS:
        raise QualificationError("REPARTITION_AUTHORIZATION_NOT_PASS")
    if str(authorization.get("directive_hash") or "")!=str(directive.get("directive_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_DIRECTIVE_DRIFT")
    if str(authorization.get("source_surface_lineage_hash") or "")!=str(surface.geometry_lineage_hash):
        raise QualificationError("REPARTITION_AUTHORIZATION_SURFACE_DRIFT")
    if str(authorization.get("source_partition_lineage_hash") or "")!=str(parent_partition.partition_lineage_hash):
        raise QualificationError("REPARTITION_AUTHORIZATION_PARTITION_DRIFT")
    if str(authorization.get("source_skin_lineage_hash") or "")!=str(directive.get("source_skin_lineage_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_SKIN_LINEAGE_DRIFT")
    if str(authorization.get("source_skeleton_lineage_hash") or "")!=str(directive.get("source_skeleton_lineage_hash") or ""):
        raise QualificationError("REPARTITION_AUTHORIZATION_SKELETON_LINEAGE_DRIFT")
    if authorization.get("teacher_inputs_used_by_predictor") is not False:
        raise QualificationError("REPARTITION_AUTHORIZATION_TEACHER_PREDICTOR_INPUT_FORBIDDEN")
    if authorization.get("weight_reliability_closure_passed") is not True:
        raise QualificationError("REPARTITION_AUTHORIZATION_WEIGHT_CLOSURE_NOT_PASS")
    evidence_hash=str(authorization.get("weight_reliability_evidence_hash") or "")
    if len(evidence_hash)!=64:
        raise QualificationError("REPARTITION_AUTHORIZATION_WEIGHT_EVIDENCE_HASH_INVALID")
    if str(authorization.get("authorization_hash") or "")!=repartition_authorization_hash_v1(authorization):
        raise QualificationError("REPARTITION_AUTHORIZATION_HASH_DRIFT")


def build_repartitioned_partition_v2(
    *,
    surface,
    parent_partition,
    directive:dict,
    authorization:dict,
):
    """Build a distinct Stage17 child partition from an authorized Stage35 directive."""
    validate_trustworthy_skin_repartition_authorization_v1(
        authorization,
        directive=directive,
        surface=surface,
        parent_partition=parent_partition,
    )
    proposed=tuple(directive.get("proposed_boundary_overrides") or ())
    if not proposed:
        raise QualificationError("REPARTITION_DIRECTIVE_HAS_NO_BOUNDARY_OVERRIDES")

    # Preserve prior explicit overrides, then let the authorized dynamic SEPARATE
    # evidence replace any same-pair parent override.
    overrides={}
    for row in parent_partition.boundary_constraints:
        md=dict(row.metadata or {})
        if "overrides_relation_default" in md or row.decision=="SEPARATE":
            pair=tuple(sorted((str(row.a_surface_id),str(row.b_surface_id))))
            overrides[pair]=row

    for raw in proposed:
        if str(raw.get("decision") or "")!="SEPARATE":
            raise QualificationError("REPARTITION_DIRECTIVE_NONSEPARATE_OVERRIDE")
        a,b=sorted((str(raw.get("a_surface_id") or ""),str(raw.get("b_surface_id") or "")))
        if not a or not b or a==b:
            raise QualificationError("REPARTITION_DIRECTIVE_PAIR_INVALID")
        refs=tuple(map(str,raw.get("evidence_refs") or ()))
        if not refs:
            raise QualificationError("REPARTITION_DIRECTIVE_EVIDENCE_MISSING")
        overrides[(a,b)]=ComponentBoundaryConstraintIR(
            constraint_id=str(raw.get("constraint_id") or ""),
            a_surface_id=a,
            b_surface_id=b,
            decision="SEPARATE",
            evidence_refs=refs,
            confidence=float(raw.get("confidence",0.0)),
            metadata={
                **dict(raw.get("metadata") or {}),
                "repair_directive_hash":str(directive["directive_hash"]),
                "repartition_authorization_hash":str(authorization["authorization_hash"]),
                "dynamic_parent_skin_lineage_hash":str(directive["source_skin_lineage_hash"]),
            },
        )

    child=build_structural_partition(
        surface,
        boundary_overrides=tuple(overrides[k] for k in sorted(overrides)),
    )
    if child.partition_lineage_hash==parent_partition.partition_lineage_hash:
        raise QualificationError("REPARTITION_CHILD_NOT_DISTINCT")
    parent_separate=sum(x.decision=="SEPARATE" for x in parent_partition.boundary_constraints)
    child_separate=sum(x.decision=="SEPARATE" for x in child.boundary_constraints)
    if child_separate<=parent_separate:
        raise QualificationError("REPARTITION_CHILD_DID_NOT_ADD_SEPARATION")

    child=replace(
        child,
        metadata={
            **dict(child.metadata or {}),
            "repair_child_attempt":True,
            "parent_partition_lineage_hash":str(parent_partition.partition_lineage_hash),
            "repair_directive_hash":str(directive["directive_hash"]),
            "repartition_authorization_hash":str(authorization["authorization_hash"]),
            "source_skin_lineage_hash":str(directive["source_skin_lineage_hash"]),
            "automatic_manual_authoring_used":False,
        },
        partition_lineage_hash="",
    )
    child=replace(child,partition_lineage_hash=mechanical_partition_lineage_hash(child))
    return child
