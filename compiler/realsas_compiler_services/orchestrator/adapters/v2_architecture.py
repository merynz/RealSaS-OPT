from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from compiler.realsas_compiler_core.output_presentation_v1 import (
    build_output_direction_set,
    output_direction_set_from_dict,
)
from compiler.realsas_compiler_core.geometry_substrate_v2 import (
    geometry_substrate_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mechanical_partition_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    read_json,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    coverage_metrics,
    rasterize_triangles_half_integer_top_left,
    source_connected_component_recall_metrics,
)
from compiler.realsas_compiler_core.silhouette_metrics_v2 import (
    silhouette_distance_metrics,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    appearance_domain_from_dict,
    build_appearance_domain,
    build_surface_addressing,
    static_mesh_qualification_hash,
    surface_addressing_from_dict,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    load_file_ref,
    sha256_file,
    stage_output_payload,
    write_ir,
)
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_v2 import (
    build_canonical_mesh_candidate_stage,
)


def seal_output_presentation_directions_stage(ctx: dict) -> dict:
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1"
        )
    )
    value = build_output_direction_set(cameras)
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "output_presentation_directions.json",
                value,
                authority_class="OUTPUT_PRESENTATION_DIRECTION_AUTHORITY",
            )
        ],
        "diagnostics": {
            "direction_set_hash": value.direction_set_hash,
            "output_direction_count": 8,
            "source_observation_authority_separate": True,
        },
    }


def build_canonical_mesh_addressing_stage(ctx: dict) -> dict:
    base = build_canonical_mesh_candidate_stage(ctx)
    if str(base.get("status")) != "PASS":
        return base
    by_schema = {str(o.get("schema")): o for o in base.get("outputs") or ()}
    candidate = canonical_mesh_candidate_from_dict(
        read_json(by_schema["RealSaS.CanonicalMeshCandidateIR.v1"]["path"])
    )
    directions = output_direction_set_from_dict(
        stage_output_payload(
            ctx,
            "16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED",
            "RealSaS.OutputPresentationDirectionSetIR.v1",
        )
    )
    addressing = build_surface_addressing(candidate)
    domain = build_appearance_domain(
        candidate, addressing, directions.direction_set_hash
    )
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    outputs = list(base["outputs"]) + [
        write_ir(
            root / "surface_addressing.json",
            addressing,
            authority_class="CANONICAL_SURFACE_ADDRESSING",
        ),
        write_ir(
            root / "appearance_domain.json",
            domain,
            authority_class="APPEARANCE_DOMAIN",
        ),
    ]
    diagnostics = dict(base.get("diagnostics") or {})
    diagnostics.update(
        {
            "surface_addressing_hash": addressing.addressing_hash,
            "appearance_domain_hash": domain.domain_hash,
            "appearance_domain_face_count": domain.renderable_face_count,
        }
    )
    return {"status": "PASS", "outputs": outputs, "diagnostics": diagnostics}


def _double_area(a, b, c) -> float:
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx
    return math.sqrt(cx * cx + cy * cy + cz * cz)


def _source_foreground_masks_v1(ctx: dict, observation) -> dict[int, bytes]:
    cfg = dict(ctx["run_manifest"].get("observation") or {})
    rows = tuple(cfg.get("source_foreground_masks") or ())
    if len(rows) != 8 or {int(row["view_index"]) for row in rows} != set(range(8)):
        raise QualificationError("STATIC_MESH_FOREGROUND_MATRIX_INCOMPLETE")
    authority = {int(view.view_index): view for view in observation.views}
    output: dict[int, bytes] = {}
    for row in rows:
        view_index = int(row["view_index"])
        ref = dict(row.get("mask") or {})
        path = load_file_ref(ref, json_required=False)
        if sha256_file(path) != authority[view_index].foreground_mask_sha256:
            raise QualificationError("STATIC_MESH_FOREGROUND_AUTHORITY_DRIFT")
        raw = path.read_bytes()
        expected_size = (
            int(authority[view_index].width) * int(authority[view_index].height)
        )
        if len(raw) != expected_size or any(value not in (0, 1) for value in raw):
            raise QualificationError("STATIC_MESH_FOREGROUND_MASK_INVALID")
        output[view_index] = raw
    return output


def _evaluate_candidate_source_fidelity_v1(
    *,
    candidate,
    geometry,
    cameras,
    observation,
    source_foreground: dict[int, bytes],
) -> tuple[bool, list[dict]]:
    if geometry.camera_set_binding_hash != cameras.camera_set_hash:
        raise QualificationError("STATIC_MESH_CAMERA_BINDING_DRIFT")
    if geometry.observation_set_binding_hash != observation.observation_set_hash:
        raise QualificationError("STATIC_MESH_OBSERVATION_BINDING_DRIFT")

    ordered_cameras = tuple(sorted(cameras.cameras, key=lambda value: value.view_index))
    observations = {int(view.view_index): view for view in observation.views}
    if (
        len(ordered_cameras) != 8
        or tuple(int(camera.view_index) for camera in ordered_cameras) != tuple(range(8))
        or set(observations) != set(range(8))
        or set(source_foreground) != set(range(8))
    ):
        raise QualificationError("STATIC_MESH_SOURCE_FIDELITY_REQUIRES_EXACT_8_VIEWS")

    vertex_ids = [str(value.candidate_vertex_id) for value in candidate.vertices]
    index_by_id = {value: index for index, value in enumerate(vertex_ids)}
    if len(index_by_id) != len(vertex_ids):
        raise QualificationError("STATIC_MESH_DUPLICATE_VERTEX_ID")
    world = np.asarray(
        [tuple(map(float, value.P)) for value in candidate.vertices],
        dtype=np.float64,
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
        raise QualificationError("STATIC_MESH_STAGE13_POLICY_INCOMPLETE")

    rows: list[dict] = []
    for camera in ordered_cameras:
        view_index = int(camera.view_index)
        authority = observations[view_index]
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
        source = source_foreground[view_index]
        metrics = coverage_metrics(
            source,
            predicted,
            width=int(authority.width),
            height=int(authority.height),
        )
        source_mask = np.frombuffer(source, dtype=np.uint8).reshape(
            int(authority.height), int(authority.width)
        ).astype(bool)
        predicted_mask = np.frombuffer(predicted, dtype=np.uint8).reshape(
            int(authority.height), int(authority.width)
        ).astype(bool)
        (
            silhouette_edge_mean_px,
            silhouette_edge_p95_px,
            silhouette_edge_max_px,
        ) = silhouette_distance_metrics(source_mask, predicted_mask)
        component_metrics = source_connected_component_recall_metrics(
            source,
            predicted,
            width=int(authority.width),
            height=int(authority.height),
            minimum_foreground_fraction=float(
                policy["component_min_foreground_fraction"]
            ),
        )
        if (
            metrics["foreground_pixel_count"] > 0
            and component_metrics["eligible_component_count"] <= 0
        ):
            raise QualificationError(
                "STATIC_MESH_COMPONENT_POLICY_SELECTS_NO_FOREGROUND"
            )
        component_recall = float(
            component_metrics["minimum_eligible_component_recall"]
        )
        passed = (
            metrics["recall"] >= float(policy["min_recall"])
            and metrics["precision"] >= float(policy["min_precision"])
            and metrics["largest_coherent_hole_fraction"]
            <= float(policy["max_largest_coherent_hole_fraction"])
            and metrics["interior_uncovered_fraction"]
            <= float(policy["max_interior_uncovered_fraction"])
            and component_recall >= float(policy["min_component_recall"])
            and silhouette_edge_p95_px
            <= float(policy["max_silhouette_edge_p95_px"])
        )
        rows.append(
            {
                "view_index": view_index,
                "silhouette_recall": float(metrics["recall"]),
                "silhouette_precision": float(metrics["precision"]),
                "largest_coherent_hole_fraction": float(
                    metrics["largest_coherent_hole_fraction"]
                ),
                "interior_uncovered_fraction": float(
                    metrics["interior_uncovered_fraction"]
                ),
                "component_recall": component_recall,
                "silhouette_edge_mean_px": float(silhouette_edge_mean_px),
                "silhouette_edge_p95_px": float(silhouette_edge_p95_px),
                "silhouette_edge_max_px": float(silhouette_edge_max_px),
                "source_foreground_pixel_count": int(
                    metrics["foreground_pixel_count"]
                ),
                "predicted_foreground_pixel_count": int(
                    metrics["predicted_pixel_count"]
                ),
                "eligible_component_count": int(
                    component_metrics["eligible_component_count"]
                ),
                "passed": bool(passed),
            }
        )
    return all(bool(row["passed"]) for row in rows), rows


def qualify_static_canonical_mesh_stage(ctx: dict) -> dict:
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    addressing = surface_addressing_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.SurfaceAddressingIR.v1",
        )
    )
    domain = appearance_domain_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.AppearanceDomainIR.v1",
        )
    )
    geometry = geometry_substrate_from_dict(
        stage_output_payload(
            ctx,
            "13_GEOMETRY_SUBSTRATE_QUALIFIED",
            "RealSaS.GeometrySubstrateQualificationIR.v2",
        )
    )
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    cameras = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    if (
        addressing.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
        or domain.candidate_mesh_binding_hash != candidate.candidate_lineage_hash
    ):
        raise QualificationError("STATIC_MESH_DOMAIN_BINDING_DRIFT")
    if domain.surface_addressing_binding_hash != addressing.addressing_hash:
        raise QualificationError("STATIC_MESH_ADDRESSING_DOMAIN_DRIFT")

    vertices = {
        str(v.candidate_vertex_id): tuple(map(float, v.P)) for v in candidate.vertices
    }
    if not vertices or not candidate.faces:
        raise QualificationError("STATIC_MESH_EMPTY")
    if any(not all(math.isfinite(x) for x in p) for p in vertices.values()):
        raise QualificationError("STATIC_MESH_NONFINITE")

    degenerate = 0
    for face in candidate.faces:
        a, b, c = (vertices[str(x)] for x in face)
        if _double_area(a, b, c) <= 1e-12:
            degenerate += 1
    if degenerate:
        raise QualificationError(f"STATIC_MESH_DEGENERATE_FACE_COUNT:{degenerate}")
    if len(addressing.vertex_addresses) != len(candidate.vertices):
        raise QualificationError("STATIC_MESH_VERTEX_ADDRESSABILITY_INCOMPLETE")
    if len(addressing.face_address_ids) != len(candidate.faces):
        raise QualificationError("STATIC_MESH_FACE_ADDRESSABILITY_INCOMPLETE")
    if not all(view.passed for view in geometry.views):
        raise QualificationError("STATIC_MESH_GEOMETRY_SUBSTRATE_NOT_PASS")

    source_foreground = _source_foreground_masks_v1(ctx, observation)
    source_fidelity_passed, source_fidelity_rows = _evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source_foreground,
    )
    if not source_fidelity_passed:
        return {
            "status": "FAIL",
            "blockers": ["STATIC_MESH_SOURCE_FIDELITY_FAILED"],
            "diagnostics": {
                "candidate_mesh_binding_hash": candidate.candidate_lineage_hash,
                "stage13_geometry_substrate_hash": geometry.substrate_hash,
                "per_view": source_fidelity_rows,
                "policy": dict(geometry.policy),
            },
        }

    report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "vertex_count": len(candidate.vertices),
        "face_count": len(candidate.faces),
        "degenerate_face_count": 0,
        "surface_addressability_fraction": 1.0,
        "stage13_geometry_substrate_inherited": False,
        "stage13_policy_replayed_on_actual_candidate_mesh": True,
        "actual_candidate_source_fidelity_passed": True,
        "actual_candidate_source_fidelity_views": source_fidelity_rows,
        "silhouette_is_geometry_authority_not_caa": True,
        "unknown_boundary_policy": (
            "CONSERVATIVE_UNTIL_STAGE35__NO_POST_SKIN_PARTITION_MUTATION_V1"
        ),
    }
    value = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash=addressing.addressing_hash,
        appearance_domain_binding_hash=domain.domain_hash,
        geometry_gate_binding_hash=geometry.substrate_hash,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
        metadata={
            "mesh_frozen_for_branching": True,
            "repair_on_stage35_failure": "NEW_STAGE18_LINEAGE",
            "source_fidelity_policy_owner": "STAGE13_GEOMETRY_SUBSTRATE_POLICY",
            "source_fidelity_measurement_target": "ACTUAL_STAGE18_CANDIDATE_MESH",
        },
    )
    value = replace(value, qualification_hash=static_mesh_qualification_hash(value))
    root = ctx["run_root"] / "artifacts" / ctx["stage"]["id"]
    return {
        "status": "PASS",
        "outputs": [
            write_ir(
                root / "static_canonical_mesh_qualification.json",
                value,
                authority_class="STATIC_CANONICAL_MESH_QUALIFICATION",
            )
        ],
        "diagnostics": {"qualification_hash": value.qualification_hash, **report},
    }
