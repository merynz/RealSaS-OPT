from __future__ import annotations

"""Canonical frozen product-policy document -> typed mesh qualification policy."""

import json
from pathlib import Path

from ..product_authority_v1 import (
    CarrierCoverageThresholdIR,
    MeshQualificationPolicyIR,
    build_mesh_qualification_policy,
)
from ..types import QualificationError


def mesh_policy_from_product_policy_document(payload: dict) -> MeshQualificationPolicyIR:
    if payload.get("schema") != "RealSaS.QualifiedMeshProductPolicy.v1":
        raise QualificationError("MESH_PRODUCT_POLICY_DOCUMENT_SCHEMA_DRIFT")
    if payload.get("status") != "FROZEN_BEFORE_KNIGHT_QUALIFIED_MESH_RESULT":
        raise QualificationError("MESH_PRODUCT_POLICY_DOCUMENT_NOT_FROZEN")
    if payload.get("authority") != "NORMATIVE_PRODUCT_ADMISSION_POLICY__NOT_EMPIRICAL_KNIGHT_FIT":
        raise QualificationError("MESH_PRODUCT_POLICY_DOCUMENT_AUTHORITY_DRIFT")

    try:
        g1=dict(payload["g1"])
        g3=dict(payload["g3"])
        g5=dict(payload["g5"])
        thresholds=tuple(
            CarrierCoverageThresholdIR(
                carrier_class=carrier,
                min_recall=float(g5[carrier]["min_recall"]),
                min_precision=float(g5[carrier]["min_precision"]),
                max_largest_coherent_hole_fraction=float(g5[carrier]["max_largest_coherent_hole_fraction"]),
                max_interior_uncovered_fraction=float(g5[carrier]["max_interior_uncovered_fraction"]),
            )
            for carrier in ("MESH","PLANAR")
        )
        return build_mesh_qualification_policy(
            g1_max_normal_refinement_ratio=float(g1["max_normal_refinement_ratio_of_local_scale"]),
            g1_max_tangential_to_normal_ratio=float(g1["max_tangential_to_normal_ratio"]),
            g3_min_angle_deg=float(g3["minimum_rest_triangle_angle_deg"]),
            g3_max_aspect_longest_over_min_altitude=float(g3["maximum_rest_aspect_longest_edge_over_min_altitude"]),
            g3_min_dynamic_area_ratio=float(g3["minimum_dynamic_area_ratio"]),
            g3_max_dynamic_area_ratio=float(g3["maximum_dynamic_area_ratio"]),
            g3_max_dynamic_condition_number=float(g3["maximum_dynamic_condition_number"]),
            coverage_thresholds=thresholds,
            metadata={
                "policy_document_schema":payload["schema"],
                "policy_document_status":payload["status"],
                "policy_document_authority":payload["authority"],
            },
        )
    except (KeyError,TypeError,ValueError) as exc:
        raise QualificationError("MESH_PRODUCT_POLICY_DOCUMENT_INCOMPLETE") from exc


def load_mesh_policy_document(path: str | Path) -> MeshQualificationPolicyIR:
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    return mesh_policy_from_product_policy_document(payload)
