from __future__ import annotations

"""Typed Stage-32 rest/source-preservation product policy and qualification."""

from dataclasses import asdict, dataclass, field, replace
import math
from typing import Any

from .hashing import content_sha256
from .rest_preservation_v1 import (
    REST_PRESERVATION_METRIC_CONTRACT_HASH,
    RestPreservationMeasurementSetIR,
)
from .types import QualificationError

Json=dict[str,Any]


@dataclass(frozen=True)
class RestSourcePreservationPolicyIR:
    min_alpha_recall:float
    min_alpha_precision:float
    max_largest_coherent_hole_fraction:float
    max_interior_uncovered_fraction:float
    max_silhouette_edge_p95_px:float
    max_overlap_rgba_mismatch_pixel_count:int
    max_overlap_rgba_max_abs_channel_error_u8:int
    min_direct_source_geometry_fraction:float
    max_cross_view_source_geometry_fraction:float
    calibration_result_binding_hash:str
    mesh_policy_binding_hash:str
    policy_lineage_hash:str
    schema_version:str="RealSaS.RestSourcePreservationPolicyIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class RestPreservationViewDecisionIR:
    view_index:int
    status:str
    failed_rules:tuple[str,...]
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class QualifiedRestSourcePreservationIR:
    measurement_set_binding_hash:str
    rest_render_set_binding_hash:str
    observation_set_binding_hash:str
    policy_binding_hash:str
    view_decisions:tuple[RestPreservationViewDecisionIR,...]
    qualification_report:Json
    preservation_lineage_hash:str
    schema_version:str="RealSaS.QualifiedRestSourcePreservationIR.v1"
    metadata:Json=field(default_factory=dict)
    def to_dict(self): return asdict(self)


def rest_source_preservation_policy_hash(value:RestSourcePreservationPolicyIR)->str:
    payload=value.to_dict(); payload.pop("policy_lineage_hash",None)
    return content_sha256(payload)


def qualified_rest_preservation_hash(value:QualifiedRestSourcePreservationIR)->str:
    payload=value.to_dict(); payload.pop("preservation_lineage_hash",None)
    return content_sha256(payload)


def _mesh_threshold(mesh_policy,carrier_class:str):
    matches=[row for row in mesh_policy.coverage_thresholds if row.carrier_class==carrier_class]
    if len(matches)!=1:
        raise QualificationError("REST_POLICY_MESH_CARRIER_THRESHOLD_CARDINALITY")
    return matches[0]


def build_rest_source_preservation_policy(
    *,
    policy_document:dict,
    calibration_result:dict,
    mesh_policy,
)->RestSourcePreservationPolicyIR:
    if policy_document.get("schema")!="RealSaS.RestSourcePreservationProductPolicy.v2":
        raise QualificationError("REST_POLICY_DOCUMENT_SCHEMA_DRIFT")
    if policy_document.get("status")!="FROZEN_AFTER_G5_MESH_V2_REBIND__BEFORE_ANY_NEW_KNIGHT_STAGE32_PASS":
        raise QualificationError("REST_POLICY_DOCUMENT_NOT_FROZEN")
    if policy_document.get("authority")!="NORMATIVE_STAGE32_PRODUCT_ADMISSION_POLICY__NOT_EMPIRICAL_KNIGHT_FIT":
        raise QualificationError("REST_POLICY_DOCUMENT_AUTHORITY_DRIFT")
    if calibration_result.get("schema")!="RealSaS.RestSourcePreservationCalibrationResult.v1" or calibration_result.get("status")!="PASS":
        raise QualificationError("REST_POLICY_CALIBRATION_RESULT_NOT_PASS")
    if calibration_result.get("authority")!="SUBJECT_FREE_SYNTHETIC_ONLY__NO_KNIGHT_OR_MAGE_INPUT":
        raise QualificationError("REST_POLICY_CALIBRATION_AUTHORITY_DRIFT")
    measurement=dict(calibration_result.get("measurement_implementation") or {})
    if measurement.get("metric_contract_hash")!=REST_PRESERVATION_METRIC_CONTRACT_HASH:
        raise QualificationError("REST_POLICY_METRIC_CONTRACT_DRIFT")
    if not all(bool(v) for v in dict(calibration_result.get("sensitivity") or {}).values()):
        raise QualificationError("REST_POLICY_CALIBRATION_SENSITIVITY_FAIL")

    selected=dict(calibration_result.get("silhouette_selection") or {})
    p=float(selected.get("max_positive_identity_p95_px",float("nan")))
    n=float(selected.get("min_negative_one_pixel_shift_p95_px",float("nan")))
    t=float(selected.get("selected_p95_ceiling_px",float("nan")))
    if not (math.isfinite(p) and math.isfinite(n) and math.isfinite(t) and p<n):
        raise QualificationError("REST_POLICY_SILHOUETTE_CALIBRATION_INVALID")
    if abs(t-((p+n)/2.0))>1e-15:
        raise QualificationError("REST_POLICY_SILHOUETTE_SELECTION_RULE_DRIFT")

    result_id=dict(policy_document.get("calibration_result") or {})
    for key in ("source_commit","workflow_run_id","workflow_job_id"):
        if result_id.get(key)!=calibration_result.get(key):
            raise QualificationError(f"REST_POLICY_CALIBRATION_IDENTITY_DRIFT:{key}")

    thresholds=dict(policy_document.get("per_view_thresholds") or {})
    mesh_row=_mesh_threshold(mesh_policy,"MESH")
    expected_mesh={
        "min_alpha_recall":float(mesh_row.min_recall),
        "min_alpha_precision":float(mesh_row.min_precision),
        "max_largest_coherent_hole_fraction":float(mesh_row.max_largest_coherent_hole_fraction),
        "max_interior_uncovered_fraction":float(mesh_row.max_interior_uncovered_fraction),
    }
    for key,expected in expected_mesh.items():
        if abs(float(thresholds.get(key,float("nan")))-expected)>1e-15:
            raise QualificationError(f"REST_POLICY_G5_INHERITANCE_DRIFT:{key}")
    if abs(float(thresholds.get("max_silhouette_edge_p95_px",float("nan")))-t)>1e-15:
        raise QualificationError("REST_POLICY_SILHOUETTE_THRESHOLD_DRIFT")
    exact=dict(calibration_result.get("exact_policy_candidates") or {})
    if int(thresholds.get("max_overlap_rgba_mismatch_pixel_count",-1))!=int(exact.get("max_overlap_rgba_mismatch_pixel_count",-2)):
        raise QualificationError("REST_POLICY_RGBA_MISMATCH_RULE_DRIFT")
    if int(thresholds.get("max_overlap_rgba_max_abs_channel_error_u8",-1))!=int(exact.get("max_overlap_rgba_max_abs_channel_error_u8",-2)):
        raise QualificationError("REST_POLICY_RGBA_CHANNEL_RULE_DRIFT")
    if float(thresholds.get("max_cross_view_source_geometry_fraction",float("nan")))!=float(exact.get("max_cross_view_source_geometry_fraction",float("nan"))):
        raise QualificationError("REST_POLICY_CROSS_VIEW_RULE_DRIFT")
    if float(thresholds.get("min_direct_source_geometry_fraction",float("nan")))!=1.0:
        raise QualificationError("REST_POLICY_DIRECT_SOURCE_RULE_DRIFT")

    calibration_hash=content_sha256(calibration_result)
    value=RestSourcePreservationPolicyIR(
        min_alpha_recall=expected_mesh["min_alpha_recall"],
        min_alpha_precision=expected_mesh["min_alpha_precision"],
        max_largest_coherent_hole_fraction=expected_mesh["max_largest_coherent_hole_fraction"],
        max_interior_uncovered_fraction=expected_mesh["max_interior_uncovered_fraction"],
        max_silhouette_edge_p95_px=t,
        max_overlap_rgba_mismatch_pixel_count=int(thresholds["max_overlap_rgba_mismatch_pixel_count"]),
        max_overlap_rgba_max_abs_channel_error_u8=int(thresholds["max_overlap_rgba_max_abs_channel_error_u8"]),
        min_direct_source_geometry_fraction=1.0,
        max_cross_view_source_geometry_fraction=float(thresholds["max_cross_view_source_geometry_fraction"]),
        calibration_result_binding_hash=calibration_hash,
        mesh_policy_binding_hash=mesh_policy.qualification_policy_lineage_hash,
        policy_lineage_hash="",
        metadata={
            "policy_document_schema":policy_document["schema"],
            "calibration_source_commit":calibration_result["source_commit"],
            "workflow_run_id":calibration_result["workflow_run_id"],
            "workflow_job_id":calibration_result["workflow_job_id"],
            "aggregate_compensation_forbidden":True,
            "knight_result_used_for_threshold_selection":False,
            "alpha_coverage_inherited_from_g5_mesh_v2":True,
        },
    )
    return replace(value,policy_lineage_hash=rest_source_preservation_policy_hash(value))


def _failed_rules(row,policy:RestSourcePreservationPolicyIR)->tuple[str,...]:
    failed=[]
    eps=1e-12
    if row.alpha_recall+eps<policy.min_alpha_recall: failed.append("ALPHA_RECALL")
    if row.alpha_precision+eps<policy.min_alpha_precision: failed.append("ALPHA_PRECISION")
    if row.largest_coherent_hole_fraction>policy.max_largest_coherent_hole_fraction+eps: failed.append("COHERENT_HOLE")
    if row.interior_uncovered_fraction>policy.max_interior_uncovered_fraction+eps: failed.append("INTERIOR_UNCOVERED")
    if row.silhouette_edge_p95_px>policy.max_silhouette_edge_p95_px+eps: failed.append("SILHOUETTE_P95")
    if row.overlap_rgba_mismatch_pixel_count>policy.max_overlap_rgba_mismatch_pixel_count: failed.append("OVERLAP_RGBA_MISMATCH")
    if row.overlap_rgba_max_abs_channel_error_u8>policy.max_overlap_rgba_max_abs_channel_error_u8: failed.append("OVERLAP_RGBA_CHANNEL_ERROR")
    if row.direct_source_geometry_fraction+eps<policy.min_direct_source_geometry_fraction: failed.append("DIRECT_SOURCE_FRACTION")
    if row.cross_view_source_geometry_fraction>policy.max_cross_view_source_geometry_fraction+eps: failed.append("CROSS_VIEW_SOURCE_FRACTION")
    return tuple(failed)


def qualify_rest_source_preservation(
    measurements:RestPreservationMeasurementSetIR,
    *,
    policy:RestSourcePreservationPolicyIR,
    rest_render_set,
    observation_set,
)->QualifiedRestSourcePreservationIR:
    if measurements.rest_render_set_binding_hash!=rest_render_set.render_set_hash:
        raise QualificationError("REST_QUALIFICATION_RENDER_BINDING_DRIFT")
    if measurements.observation_set_binding_hash!=observation_set.observation_set_hash:
        raise QualificationError("REST_QUALIFICATION_OBSERVATION_BINDING_DRIFT")
    rows=tuple(sorted(measurements.views,key=lambda x:x.view_index))
    if len(rows)!=8 or tuple(row.view_index for row in rows)!=tuple(range(8)):
        raise QualificationError("REST_QUALIFICATION_REQUIRES_EXACT_8_VIEWS")
    decisions=tuple(
        RestPreservationViewDecisionIR(
            view_index=row.view_index,
            status="PASS" if not _failed_rules(row,policy) else "FAIL",
            failed_rules=_failed_rules(row,policy),
            metadata={
                "alpha_recall":row.alpha_recall,
                "alpha_precision":row.alpha_precision,
                "silhouette_edge_p95_px":row.silhouette_edge_p95_px,
                "overlap_rgba_mismatch_pixel_count":row.overlap_rgba_mismatch_pixel_count,
                "cross_view_source_geometry_fraction":row.cross_view_source_geometry_fraction,
            },
        )
        for row in rows
    )
    failures=tuple(
        (decision.view_index,rule)
        for decision in decisions
        for rule in decision.failed_rules
    )
    if failures:
        raise QualificationError(f"REST_SOURCE_PRESERVATION_POLICY_FAIL:{failures[:16]}")
    value=QualifiedRestSourcePreservationIR(
        measurement_set_binding_hash=measurements.measurement_set_hash,
        rest_render_set_binding_hash=rest_render_set.render_set_hash,
        observation_set_binding_hash=observation_set.observation_set_hash,
        policy_binding_hash=policy.policy_lineage_hash,
        view_decisions=decisions,
        qualification_report={
            "status":"PASS",
            "view_count":8,
            "every_view_passed_every_rule":True,
            "aggregate_compensation_used":False,
            "motion_authorization_precondition_satisfied":True,
        },
        preservation_lineage_hash="",
        metadata={
            "calibration_result_binding_hash":policy.calibration_result_binding_hash,
            "mesh_policy_binding_hash":policy.mesh_policy_binding_hash,
            "metric_contract_hash":REST_PRESERVATION_METRIC_CONTRACT_HASH,
        },
    )
    return replace(value,preservation_lineage_hash=qualified_rest_preservation_hash(value))
