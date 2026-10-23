from __future__ import annotations

"""Typed static Mesh qualification shared by legacy DAG and product.mesh.

This is the canonical static court for a candidate Mesh: exact addressability,
rest-quality policy, source-view fidelity, and immutable mechanical-carrier
basis. It accepts typed evidence only; filesystem/run-manifest concerns belong
in adapters. The current v1 source-fidelity science is exact-8-view, but that is
an implementation contract, not a Platform invariant.
"""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import numpy as np

from .camera_geometry_v2 import project_points_xyz_v3
from .mechanical_carrier_evidence_v1 import (
    MechanicalCarrierEvidenceIR,
    build_mechanical_carrier_evidence_v1,
)
from .mesh.conditioning_v1 import triangle_rest_metric
from .mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from .silhouette_metrics_v2 import silhouette_distance_metrics
from .surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from .types import QualificationError


@dataclass(frozen=True)
class StaticMeshQualificationResultV1:
    status: str
    qualification: StaticCanonicalMeshQualificationIR | None
    carrier_evidence: MechanicalCarrierEvidenceIR | None
    blockers: tuple[str, ...]
    diagnostics: dict[str, Any]


def relation_parent_quality_report_v1(candidate, policy) -> dict[str, Any]:
    vertices = {
        str(vertex.candidate_vertex_id): tuple(map(float, vertex.P))
        for vertex in candidate.vertices
    }
    rows = []
    for face in candidate.faces:
        try:
            points = tuple(vertices[str(vertex_id)] for vertex_id in face)
        except KeyError as exc:
            raise QualificationError("STATIC_MESH_QUALITY_FACE_VERTEX_UNKNOWN") from exc
        rows.append(triangle_rest_metric(points))
    if not rows:
        raise QualificationError("STATIC_MESH_QUALITY_REQUIRES_FACES")
    bad_angle = sum(
        bool(row["degenerate"])
        or float(row["min_angle_deg"]) + 1e-9 < float(policy.g3_min_angle_deg)
        for row in rows
    )
    bad_aspect = sum(
        bool(row["degenerate"])
        or float(row["aspect_longest_over_min_altitude"]) - 1e-9
        > float(policy.g3_max_aspect_longest_over_min_altitude)
        for row in rows
    )
    bad_union = sum(
        bool(row["degenerate"])
        or float(row["min_angle_deg"]) + 1e-9 < float(policy.g3_min_angle_deg)
        or float(row["aspect_longest_over_min_altitude"]) - 1e-9
        > float(policy.g3_max_aspect_longest_over_min_altitude)
        for row in rows
    )
    finite_aspects = [
        float(row["aspect_longest_over_min_altitude"])
        for row in rows
        if not bool(row["degenerate"])
        and float(row["aspect_longest_over_min_altitude"]) < float("inf")
    ]
    return {
        "schema": "RealSaS.RelationParentQualityReport.v1",
        "face_count": len(rows),
        "degenerate_face_count": sum(bool(row["degenerate"]) for row in rows),
        "min_angle_deg": min(float(row["min_angle_deg"]) for row in rows),
        "max_aspect_longest_over_min_altitude": max(finite_aspects, default=float("inf")),
        "policy_min_angle_deg": float(policy.g3_min_angle_deg),
        "policy_max_aspect_longest_over_min_altitude": float(
            policy.g3_max_aspect_longest_over_min_altitude
        ),
        "below_min_angle_face_count": int(bad_angle),
        "above_max_aspect_face_count": int(bad_aspect),
        "policy_violating_face_count": int(bad_union),
        "below_min_angle_face_fraction": float(bad_angle / len(rows)),
        "above_max_aspect_face_fraction": float(bad_aspect / len(rows)),
        "policy_violating_face_fraction": float(bad_union / len(rows)),
    }


def _double_area(a, b, c) -> float:
    ux, uy, uz = (b[index] - a[index] for index in range(3))
    vx, vy, vz = (c[index] - a[index] for index in range(3))
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def evaluate_candidate_source_fidelity_v1(
    *,
    candidate,
    geometry,
    cameras,
    observation,
    source_foreground: Mapping[int, bytes],
) -> tuple[bool, list[dict[str, Any]]]:
    if geometry.camera_set_binding_hash != cameras.camera_set_hash:
        raise QualificationError("STATIC_MESH_CAMERA_BINDING_DRIFT")
    if geometry.observation_set_binding_hash != observation.observation_set_hash:
        raise QualificationError("STATIC_MESH_OBSERVATION_BINDING_DRIFT")

    ordered_cameras = tuple(sorted(cameras.cameras, key=lambda value: value.view_index))
    observations = {int(view.view_index): view for view in observation.views}
    foreground = {int(key): bytes(value) for key, value in source_foreground.items()}
    if (
        len(ordered_cameras) != 8
        or tuple(int(camera.view_index) for camera in ordered_cameras) != tuple(range(8))
        or set(observations) != set(range(8))
        or set(foreground) != set(range(8))
    ):
        raise QualificationError("STATIC_MESH_SOURCE_FIDELITY_REQUIRES_EXACT_8_VIEWS")

    vertex_ids = [str(value.candidate_vertex_id) for value in candidate.vertices]
    index_by_id = {value: index for index, value in enumerate(vertex_ids)}
    if len(index_by_id) != len(vertex_ids):
        raise QualificationError("STATIC_MESH_DUPLICATE_VERTEX_ID")
    world = np.asarray(
        [tuple(map(float, value.P)) for value in candidate.vertices], dtype=np.float64
    )
    if world.ndim != 2 or world.shape[1] != 3 or not np.isfinite(world).all():
        raise QualificationError("STATIC_MESH_SOURCE_FIDELITY_VERTICES_INVALID")

    faces = []
    for face in candidate.faces:
        try:
            faces.append(tuple(index_by_id[str(vertex_id)] for vertex_id in face))
        except KeyError as exc:
            raise QualificationError("STATIC_MESH_SOURCE_FIDELITY_FACE_ID_INVALID") from exc
    face_array = np.asarray(faces, dtype=np.int64)
    if face_array.ndim != 2 or face_array.shape[1] != 3 or len(face_array) == 0:
        raise QualificationError("STATIC_MESH_SOURCE_FIDELITY_FACES_INVALID")

    policy = dict(geometry.policy or {})
    required_policy = (
        "min_recall",
        "min_precision",
        "max_largest_coherent_hole_fraction",
        "max_interior_uncovered_fraction",
        "min_component_recall",
        "component_min_foreground_fraction",
        "max_silhouette_edge_p95_px",
    )
    if any(key not in policy for key in required_policy):
        raise QualificationError("STATIC_MESH_GEOMETRY_POLICY_INCOMPLETE")

    rows: list[dict[str, Any]] = []
    for camera in ordered_cameras:
        view_index = int(camera.view_index)
        authority = observations[view_index]
        raw_source = foreground[view_index]
        expected_size = int(authority.width) * int(authority.height)
        if len(raw_source) != expected_size or any(value not in (0, 1) for value in raw_source):
            raise QualificationError("STATIC_MESH_FOREGROUND_MASK_INVALID")
        projected = project_points_xyz_v3(world, camera)
        if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0.0):
            raise QualificationError("STATIC_MESH_OUTSIDE_CAMERA_FORWARD_DOMAIN")
        triangles = []
        for face in face_array:
            a, b, c = (projected[int(index)] for index in face)
            triangles.append(
                (
                    (float(a[0]), float(a[1])),
                    (float(b[0]), float(b[1])),
                    (float(c[0]), float(c[1])),
                )
            )
        predicted = rasterize_triangles_half_integer_top_left(
            triangles,
            width=int(authority.width),
            height=int(authority.height),
        )
        metrics = coverage_metrics(
            raw_source,
            predicted,
            width=int(authority.width),
            height=int(authority.height),
        )
        source_mask = np.frombuffer(raw_source, dtype=np.uint8).reshape(
            int(authority.height), int(authority.width)
        ).astype(bool)
        predicted_mask = np.frombuffer(predicted, dtype=np.uint8).reshape(
            int(authority.height), int(authority.width)
        ).astype(bool)
        edge_mean, edge_p95, edge_max = silhouette_distance_metrics(
            source_mask, predicted_mask
        )
        component_metrics = source_connected_component_recall_metrics(
            raw_source,
            predicted,
            width=int(authority.width),
            height=int(authority.height),
            minimum_foreground_fraction=float(policy["component_min_foreground_fraction"]),
        )
        if (
            metrics["foreground_pixel_count"] > 0
            and component_metrics["eligible_component_count"] <= 0
        ):
            raise QualificationError("STATIC_MESH_COMPONENT_POLICY_SELECTS_NO_FOREGROUND")
        component_recall = float(component_metrics["minimum_eligible_component_recall"])
        passed = (
            metrics["recall"] >= float(policy["min_recall"])
            and metrics["precision"] >= float(policy["min_precision"])
            and metrics["largest_coherent_hole_fraction"]
            <= float(policy["max_largest_coherent_hole_fraction"])
            and metrics["interior_uncovered_fraction"]
            <= float(policy["max_interior_uncovered_fraction"])
            and component_recall >= float(policy["min_component_recall"])
            and edge_p95 <= float(policy["max_silhouette_edge_p95_px"])
        )
        rows.append(
            {
                "view_index": view_index,
                "silhouette_recall": float(metrics["recall"]),
                "silhouette_precision": float(metrics["precision"]),
                "largest_coherent_hole_fraction": float(
                    metrics["largest_coherent_hole_fraction"]
                ),
                "interior_uncovered_fraction": float(metrics["interior_uncovered_fraction"]),
                "component_recall": component_recall,
                "silhouette_edge_mean_px": float(edge_mean),
                "silhouette_edge_p95_px": float(edge_p95),
                "silhouette_edge_max_px": float(edge_max),
                "source_foreground_pixel_count": int(metrics["foreground_pixel_count"]),
                "predicted_foreground_pixel_count": int(metrics["predicted_pixel_count"]),
                "eligible_component_count": int(component_metrics["eligible_component_count"]),
                "passed": bool(passed),
            }
        )
    return all(bool(row["passed"]) for row in rows), rows


def qualify_static_mesh_v1(
    *,
    candidate,
    surface_addressing,
    mesh_face_domain,
    geometry,
    mesh_policy,
    partition,
    cameras,
    observation,
    source_foreground: Mapping[int, bytes],
    demo_geometry_lineage: bool = False,
) -> StaticMeshQualificationResultV1:
    if (
        surface_addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
        or mesh_face_domain.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
    ):
        raise QualificationError("STATIC_MESH_DOMAIN_BINDING_DRIFT")
    if mesh_face_domain.surface_addressing_binding_hash != surface_addressing.addressing_hash:
        raise QualificationError("STATIC_MESH_ADDRESSING_DOMAIN_DRIFT")
    if partition.partition_lineage_hash != candidate.partition_binding_hash:
        raise QualificationError("STATIC_MESH_PARTITION_BINDING_DRIFT")

    vertices = {
        str(vertex.candidate_vertex_id): tuple(map(float, vertex.P))
        for vertex in candidate.vertices
    }
    if not vertices or not candidate.faces:
        raise QualificationError("STATIC_MESH_EMPTY")
    if any(not all(math.isfinite(value) for value in point) for point in vertices.values()):
        raise QualificationError("STATIC_MESH_NONFINITE")
    degenerate = 0
    for face in candidate.faces:
        try:
            a, b, c = (vertices[str(vertex_id)] for vertex_id in face)
        except KeyError as exc:
            raise QualificationError("STATIC_MESH_FACE_VERTEX_UNKNOWN") from exc
        if _double_area(a, b, c) <= 1e-12:
            degenerate += 1
    if degenerate:
        raise QualificationError(f"STATIC_MESH_DEGENERATE_FACE_COUNT:{degenerate}")
    if len(surface_addressing.vertex_addresses) != len(candidate.vertices):
        raise QualificationError("STATIC_MESH_VERTEX_ADDRESSABILITY_INCOMPLETE")
    if len(surface_addressing.face_address_ids) != len(candidate.faces):
        raise QualificationError("STATIC_MESH_FACE_ADDRESSABILITY_INCOMPLETE")
    if not all(view.passed for view in geometry.views) and not demo_geometry_lineage:
        raise QualificationError("STATIC_MESH_GEOMETRY_SUBSTRATE_NOT_PASS")

    static_quality = relation_parent_quality_report_v1(candidate, mesh_policy)
    static_quality_passed = int(static_quality["policy_violating_face_count"]) == 0
    if not static_quality_passed and not demo_geometry_lineage:
        return StaticMeshQualificationResultV1(
            status="FAIL",
            qualification=None,
            carrier_evidence=None,
            blockers=("STATIC_MESH_QUALITY_POLICY_FAILED",),
            diagnostics={
                "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
                "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
                "static_quality": static_quality,
            },
        )

    source_fidelity_passed, source_fidelity_rows = evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source_foreground,
    )
    if not source_fidelity_passed and not demo_geometry_lineage:
        return StaticMeshQualificationResultV1(
            status="FAIL",
            qualification=None,
            carrier_evidence=None,
            blockers=("STATIC_MESH_SOURCE_FIDELITY_FAILED",),
            diagnostics={
                "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
                "geometry_substrate_hash": geometry.substrate_hash,
                "per_view": source_fidelity_rows,
                "policy": dict(geometry.policy),
            },
        )

    report = {
        "status": (
            "PASS_STATIC_CANONICAL_CARRIER"
            if source_fidelity_passed
            else "DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL"
        ),
        "vertex_count": len(candidate.vertices),
        "face_count": len(candidate.faces),
        "degenerate_face_count": 0,
        "static_quality_policy_passed": bool(static_quality_passed),
        "static_quality_min_angle_deg": float(static_quality["min_angle_deg"]),
        "static_quality_max_aspect": float(
            static_quality["max_aspect_longest_over_min_altitude"]
        ),
        "static_quality_policy_violating_face_count": int(
            static_quality["policy_violating_face_count"]
        ),
        "mesh_policy_hash": mesh_policy.qualification_policy_lineage_hash,
        "surface_addressability_fraction": 1.0,
        "geometry_substrate_policy_replayed_on_actual_candidate_mesh": True,
        "actual_candidate_source_fidelity_passed": bool(source_fidelity_passed),
        "actual_candidate_source_fidelity_views": source_fidelity_rows,
        "demo_geometry_lineage": bool(demo_geometry_lineage),
        "product_authority_claimed": False,
        "source_fidelity_qualification_passed": bool(source_fidelity_passed),
        "demo_only_source_fidelity_admission": bool(demo_geometry_lineage),
        "silhouette_is_geometry_authority_not_graphics": True,
        "unknown_boundary_policy": "CONSERVATIVE_UNTIL_NEW_MESH_LINEAGE",
    }
    value = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=surface_addressing.addressing_hash,
        appearance_domain_binding_hash=mesh_face_domain.domain_hash,
        geometry_gate_binding_hash=geometry.substrate_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "mesh_frozen_for_branching": True,
            "repair_failure_requires_new_mesh_lineage": True,
            "source_fidelity_policy_owner": "GEOMETRY_SUBSTRATE_POLICY",
            "source_fidelity_measurement_target": "ACTUAL_CANDIDATE_MESH",
            "static_quality_measurement_target": "ACTUAL_CANDIDATE_MESH",
            "static_quality_policy_passed": bool(static_quality_passed),
            "mesh_face_domain_semantics": "STABLE_M_FACE_ADDRESS_SPACE",
        },
    )
    value = replace(value, qualification_hash=static_mesh_qualification_hash(value))
    carrier = build_mechanical_carrier_evidence_v1(
        candidate,
        static_qualification=value,
        surface_addressing=surface_addressing,
    )
    return StaticMeshQualificationResultV1(
        status="PASS_DEMO_ONLY" if demo_geometry_lineage else "PASS",
        qualification=value,
        carrier_evidence=carrier,
        blockers=(),
        diagnostics={
            "qualification_hash": value.qualification_hash,
            "mechanical_carrier_evidence_hash": carrier.carrier_evidence_hash,
            "mechanical_carrier_topology_hash": carrier.topology_hash,
            "mechanical_carrier_geometry_hash": carrier.geometry_hash,
            "mechanical_carrier_invalid_normal_count": int(
                sum(not flag for flag in carrier.normal_valid)
            ),
            **report,
        },
    )


__all__ = [
    "StaticMeshQualificationResultV1",
    "evaluate_candidate_source_fidelity_v1",
    "qualify_static_mesh_v1",
    "relation_parent_quality_report_v1",
]
