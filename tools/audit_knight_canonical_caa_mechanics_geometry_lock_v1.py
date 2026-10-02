from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    component_carrier_policy_from_dict,
    deformation_envelope_from_dict,
    mechanical_partition_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.dynamic_geometry_integrity_v2 import (
    unexpected_intersection_pairs,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import (
    run_g3_local_frame_micro_stress_v2,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.product_authority_v1 import (
    validate_canonical_mesh_candidate,
    validate_component_carrier_policy,
    validate_mechanical_partition,
)
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    _evaluate_candidate_source_fidelity_v1,
    _source_foreground_masks_v1,
    _static_geometry_evidence,
)
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    replay_compacted_dense_face_provenance,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _ctx,
    _skin,
    _tracks_for_clip,
)


CLIPS = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")
FULL_MOTION_SAMPLES = 17
VISIBLE_FLIP_SAMPLE_INDICES = {0, 4, 8, 12, 16}
DEMO_VIEWS = {0, 2}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _faces(candidate) -> np.ndarray:
    index = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    if len(index) != len(candidate.vertices):
        raise RuntimeError("LOCK_DUPLICATE_VERTEX_ID")
    out = []
    for face in candidate.faces:
        ids = tuple(map(str, face))
        if len(ids) != 3 or len(set(ids)) != 3 or any(x not in index for x in ids):
            raise RuntimeError("LOCK_FACE_REFERENCE_INVALID")
        out.append(tuple(index[x] for x in ids))
    return np.asarray(out, dtype=np.int64)


def _edge_incidence(faces: np.ndarray) -> dict:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for ia, ib in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[ia]), int(face[ib])
            key = (a, b) if a < b else (b, a)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _triangle_geometry(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> dict:
    r = np.asarray(rest, dtype=np.float64)[faces]
    p = np.asarray(posed, dtype=np.float64)[faces]

    re = np.stack(
        (
            np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
            np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
            np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
        ),
        axis=1,
    )
    pe = np.stack(
        (
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ),
        axis=1,
    )
    edge_ratio = pe / np.maximum(re, 1.0e-12)
    face_edge = np.max(edge_ratio, axis=1)

    rn = np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0])
    pn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    ra = np.linalg.norm(rn, axis=1)
    pa = np.linalg.norm(pn, axis=1)
    valid = ra > 1.0e-12
    posed_degenerate = pa <= 1.0e-12

    area_ratio = np.full((len(faces),), np.nan, dtype=np.float64)
    area_ratio[valid] = pa[valid] / ra[valid]
    normal_dot = np.sum(rn * pn, axis=1)
    normal_reversal = valid & (~posed_degenerate) & (normal_dot < 0.0)

    finite_area = area_ratio[np.isfinite(area_ratio)]
    return {
        "edge_gt_4": int(np.count_nonzero(face_edge > 4.0)),
        "edge_gt_10": int(np.count_nonzero(face_edge > 10.0)),
        "edge_p95": float(np.percentile(face_edge, 95.0)),
        "edge_p99": float(np.percentile(face_edge, 99.0)),
        "edge_max": float(np.max(face_edge)),
        "posed_degenerate_face_count": int(np.count_nonzero(posed_degenerate)),
        "area_ratio_min": float(np.min(finite_area)) if len(finite_area) else None,
        "area_ratio_p01": float(np.percentile(finite_area, 1.0)) if len(finite_area) else None,
        "area_ratio_p99": float(np.percentile(finite_area, 99.0)) if len(finite_area) else None,
        "area_ratio_max": float(np.max(finite_area)) if len(finite_area) else None,
        "rest_normal_reversal_count_diagnostic": int(np.count_nonzero(normal_reversal)),
    }


def _signed_area2(projected_xy: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = np.asarray(projected_xy, dtype=np.float64)[faces]
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def _projected_flip_metrics(
    *,
    candidate,
    faces: np.ndarray,
    rest: np.ndarray,
    posed: np.ndarray,
    cameras,
    do_visible_census: bool,
) -> dict:
    by_view = []
    all_flip_max = 0
    visible_flip_max = 0
    visible_face_max = 0
    for camera in cameras:
        view = int(camera.view_index)
        rest_screen = np.asarray(project_points_xyz_v3(rest, camera), dtype=np.float64)[:, :2]
        posed_screen = np.asarray(project_points_xyz_v3(posed, camera), dtype=np.float64)[:, :2]
        rda = _signed_area2(rest_screen, faces)
        pda = _signed_area2(posed_screen, faces)
        measurable = (np.abs(rda) >= 1.0) & (np.abs(pda) >= 1.0)
        flips = measurable & (rda * pda < 0.0)
        all_count = int(np.count_nonzero(flips))
        all_flip_max = max(all_flip_max, all_count)

        visible_count = None
        visible_flip_count = None
        if do_visible_census and view in DEMO_VIEWS:
            vis = rasterize_visible_owner(
                candidate,
                camera,
                positions=posed,
                coverage_scale=1,
            )
            owner = np.asarray(vis.owner_face_index, dtype=np.int64)
            visible_faces = np.unique(owner[owner >= 0])
            visible_mask = np.zeros((len(faces),), dtype=bool)
            visible_mask[visible_faces] = True
            visible_count = int(len(visible_faces))
            visible_flip_count = int(np.count_nonzero(flips & visible_mask))
            visible_face_max = max(visible_face_max, visible_count)
            visible_flip_max = max(visible_flip_max, visible_flip_count)

        by_view.append(
            {
                "view_index": view,
                "all_measurable_projected_flip_count": all_count,
                "visible_face_count": visible_count,
                "visible_projected_flip_count": visible_flip_count,
            }
        )
    return {
        "by_view": by_view,
        "max_all_measurable_projected_flip_count": int(all_flip_max),
        "max_visible_projected_flip_count": int(visible_flip_max),
        "max_visible_face_count": int(visible_face_max),
    }


def _static_topology(candidate, faces: np.ndarray, explicit_faces, surface) -> dict:
    vertices = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    double_area = np.linalg.norm(cross, axis=1)

    canonical_faces = [tuple(sorted(map(str, face))) for face in candidate.faces]
    duplicate_face_count = len(canonical_faces) - len(set(canonical_faces))

    incidence = _edge_incidence(faces)
    nonmanifold = sum(1 for count in incidence.values() if count > 2)
    boundary = sum(1 for count in incidence.values() if count == 1)

    explicit_set = {tuple(sorted(map(str, row))) for row in explicit_faces}
    identity_face_count = 0
    identity_face_missing_dense_provenance = 0
    seam_or_generated_face_count = 0
    support_simplex_failures = 0
    vertex_by_id = {str(v.candidate_vertex_id): v for v in candidate.vertices}

    for face in candidate.faces:
        surface_ids = []
        all_identity = True
        for vid in face:
            vertex = vertex_by_id[str(vid)]
            coeffs = tuple(vertex.support_binding.coefficients)
            total = sum(float(w) for _sid, w in coeffs)
            if (
                not coeffs
                or not np.isfinite(total)
                or abs(total - 1.0) > 1.0e-8
                or any(float(w) < -1.0e-10 for _sid, w in coeffs)
            ):
                support_simplex_failures += 1
            if (
                str(vertex.support_binding.mode) == "IDENTITY_SURFACE_NODE"
                and len(coeffs) == 1
                and abs(float(coeffs[0][1]) - 1.0) <= 1.0e-10
            ):
                surface_ids.append(str(coeffs[0][0]))
            else:
                all_identity = False
        if all_identity:
            identity_face_count += 1
            if tuple(sorted(surface_ids)) not in explicit_set:
                identity_face_missing_dense_provenance += 1
        else:
            seam_or_generated_face_count += 1

    surface_node_count = len(surface.surface_nodes)
    return {
        "vertex_count": int(len(candidate.vertices)),
        "face_count": int(len(candidate.faces)),
        "surface_node_count": int(surface_node_count),
        "duplicate_face_count": int(duplicate_face_count),
        "degenerate_rest_face_count": int(np.count_nonzero(double_area <= 1.0e-12)),
        "minimum_rest_double_area": float(np.min(double_area)) if len(double_area) else None,
        "nonmanifold_edge_count": int(nonmanifold),
        "boundary_edge_count": int(boundary),
        "identity_face_count": int(identity_face_count),
        "identity_face_missing_dense_provenance_count": int(identity_face_missing_dense_provenance),
        "seam_or_generated_face_count": int(seam_or_generated_face_count),
        "support_simplex_failure_count": int(support_simplex_failures),
        "mechanical_skin_transfer": dict(candidate.metadata or {}).get("mechanical_skin_transfer"),
        "face_provenance_mode": dict(candidate.metadata or {}).get("face_provenance_mode"),
        "face_deletion_count_metadata": dict(candidate.metadata or {}).get("face_deletion_count"),
        "three_clique_face_minting_allowed": dict(candidate.metadata or {}).get("three_clique_face_minting_allowed"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--caa-run-id", required=True)
    ap.add_argument("--corrected-skin-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    authority_root = args.authority_root.expanduser().resolve()
    ctx = _ctx(authority_root, args.run_id)
    rr = Path(ctx["run_root"])

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx,
            "15_RIGGING_SURFACE_QUALIFIED",
            "RealSaS.RiggingSurfaceIR.v1",
        )
    )
    partition = mechanical_partition_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.MechanicalPartitionIR.v1",
        )
    )
    carrier = component_carrier_policy_from_dict(
        stage_output_payload(
            ctx,
            "17_MECHANICAL_PARTITION_QUALIFIED",
            "RealSaS.ComponentCarrierPolicyIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx,
            "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx,
            "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    envelope = deformation_envelope_from_dict(
        stage_output_payload(
            ctx,
            "34_DEFORMATION_CAPABILITY_ENVELOPE",
            "RealSaS.DeformationCapabilityEnvelopeIR.v1",
        )
    )
    policy = mesh_policy_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.MeshQualificationPolicyIR.v1",
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )

    # Fresh current-code validation of exact final CAA parent lineage.
    validate_mechanical_partition(partition, surface)
    validate_component_carrier_policy(carrier, partition)
    validate_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
    )

    faces = _faces(candidate)
    explicit_faces, provenance_replay = replay_compacted_dense_face_provenance(rr, surface)
    static = _static_topology(candidate, faces, explicit_faces, surface)

    # Fresh Stage19-equivalent source-fidelity measurement with no writes.
    geometry, demo_geometry_lineage = _static_geometry_evidence(ctx)
    source_foreground = _source_foreground_masks_v1(ctx, observation)
    source_fidelity_passed, source_fidelity_rows = _evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=camera_set,
        observation=observation,
        source_foreground=source_foreground,
    )

    # Prove this is the corrected Arachne skin consumed by the render lineage.
    run_skin_path = rr / "artifacts/32_SKIN_QUALIFIED/qualified_skin.json"
    corrected_skin_path = args.corrected_skin_json.resolve()
    run_skin_sha = _sha256(run_skin_path)
    corrected_skin_sha = _sha256(corrected_skin_path)
    corrected_skin = qualified_skin_from_dict(_load_json(corrected_skin_path))
    corrected_skin_exact = (
        run_skin_sha == corrected_skin_sha
        and skin.skin_lineage_hash == corrected_skin.skin_lineage_hash
    )
    skin_binding_pass = (
        skin.surface_binding_hash == surface.geometry_lineage_hash
        and skin.skeleton_binding_hash == skeleton.skeleton_lineage_hash
    )

    rest, W, faces_from_support = _candidate_skin_matrix(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    rest = np.asarray(rest, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    faces_support = np.asarray(faces_from_support, dtype=np.int64)
    if not np.array_equal(faces, faces_support):
        raise RuntimeError("LOCK_MECHANICAL_SUPPORT_FACE_ORDER_DRIFT")

    skin_simplex_residual_max = float(np.max(np.abs(W.sum(axis=1) - 1.0)))
    skin_negative_weight_count = int(np.count_nonzero(W < -1.0e-10))
    skin_nonfinite_count = int(np.count_nonzero(~np.isfinite(W)))

    # Fresh current-code G3 on the exact render candidate/skin.
    g3 = run_g3_local_frame_micro_stress_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        envelope=envelope,
        cameras=cameras,
        policy=policy,
    )

    source_report = _load_json(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")
    )
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)

    clip_rows = []
    global_worst_key = (-1, -1, -1.0)
    global_worst_pose = None
    global_worst_locator = None
    max_edge_gt4 = 0
    max_edge_gt10 = 0
    worst_edge = 0.0
    max_visible_projected_flips = 0
    max_all_projected_flips = 0
    max_posed_degenerate = 0

    for clip_id in CLIPS:
        motion_path = (
            rr / "inputs" / "motion" / "quaternius_knight_v1" / f"{clip_id}.motion.json"
        )
        payload = _load_json(motion_path)
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            FULL_MOTION_SAMPLES,
            endpoint=not bool(payload.get("loop")),
        )

        frames = []
        posed_by_index: dict[int, np.ndarray] = {}
        worst_index = 0
        worst_key = (-1, -1, -1.0)

        for frame_index, time_seconds in enumerate(times):
            mats, _joint_pos, frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(time_seconds),
                cameras=cameras,
            )
            posed = np.asarray(_skin(rest, W, joint_ids, mats), dtype=np.float64)
            posed_by_index[frame_index] = posed
            mechanical = _triangle_geometry(rest, posed, faces)
            projected = _projected_flip_metrics(
                candidate=candidate,
                faces=faces,
                rest=rest,
                posed=posed,
                cameras=cameras,
                do_visible_census=frame_index in VISIBLE_FLIP_SAMPLE_INDICES,
            )

            key = (
                int(mechanical["edge_gt_10"]),
                int(mechanical["edge_gt_4"]),
                float(mechanical["edge_max"]),
            )
            if key > worst_key:
                worst_key = key
                worst_index = frame_index
            if key > global_worst_key:
                global_worst_key = key
                global_worst_pose = posed.copy()
                global_worst_locator = {
                    "clip_id": clip_id,
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                }

            max_edge_gt4 = max(max_edge_gt4, int(mechanical["edge_gt_4"]))
            max_edge_gt10 = max(max_edge_gt10, int(mechanical["edge_gt_10"]))
            worst_edge = max(worst_edge, float(mechanical["edge_max"]))
            max_visible_projected_flips = max(
                max_visible_projected_flips,
                int(projected["max_visible_projected_flip_count"]),
            )
            max_all_projected_flips = max(
                max_all_projected_flips,
                int(projected["max_all_measurable_projected_flip_count"]),
            )
            max_posed_degenerate = max(
                max_posed_degenerate,
                int(mechanical["posed_degenerate_face_count"]),
            )

            frames.append(
                {
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                    "motion_frame_hash": str(frame_hash),
                    "mechanical": mechanical,
                    "projected": projected,
                }
            )

        clip_rows.append(
            {
                "clip_id": clip_id,
                "mapping": mapping,
                "sample_count": int(len(times)),
                "worst_frame_index": int(worst_index),
                "frames": frames,
            }
        )

    if global_worst_pose is None:
        raise RuntimeError("LOCK_GLOBAL_WORST_POSE_MISSING")
    rest_intersections = set(
        unexpected_intersection_pairs(vertices=rest, faces=faces)
    )
    global_posed_intersections = set(
        unexpected_intersection_pairs(vertices=global_worst_pose, faces=faces)
    )
    new_intersections = sorted(global_posed_intersections - rest_intersections)
    max_new_intersections = int(len(new_intersections))

    caa_ctx = _ctx(authority_root, args.caa_run_id)
    caa_stage23 = stage_output_payload(
        caa_ctx,
        "23_COMPLETE_APPEARANCE_ASSET_BAKED",
        "RealSaS.CompleteAppearanceAssetIR.v2",
    )
    caa_candidate_binding_pass = (
        str(caa_stage23["candidate_mesh_binding_hash"])
        == str(candidate.candidate_lineage_hash)
    )
    caa_is_mechanical_mode = (
        dict(caa_stage23.get("metadata") or {}).get("source_owned_visual_mesh_mode")
        is not True
    )

    geometry_pass = bool(
        static["duplicate_face_count"] == 0
        and static["degenerate_rest_face_count"] == 0
        and static["nonmanifold_edge_count"] == 0
        and static["identity_face_missing_dense_provenance_count"] == 0
        and static["support_simplex_failure_count"] == 0
        and provenance_replay["position_error_max"] < 1.0e-9
        and source_fidelity_passed
    )
    mechanics_pass = bool(
        corrected_skin_exact
        and skin_binding_pass
        and skin_simplex_residual_max <= 1.0e-8
        and skin_negative_weight_count == 0
        and skin_nonfinite_count == 0
        and bool(g3.passed)
        and max_edge_gt4 == 0
        and max_edge_gt10 == 0
        and max_posed_degenerate == 0
        and max_visible_projected_flips == 0
        and max_new_intersections == 0
    )
    exact_render_binding_pass = bool(
        caa_candidate_binding_pass and caa_is_mechanical_mode
    )

    status = (
        "PASS_EXACT_CANONICAL_CAA_MECHANICS_GEOMETRY_LOCK"
        if geometry_pass and mechanics_pass and exact_render_binding_pass
        else "FAIL_EXACT_CANONICAL_CAA_MECHANICS_GEOMETRY_LOCK"
    )

    report = {
        "schema": "RealSaS.KnightCanonicalCAAMechanicsGeometryLock.v1",
        "status": status,
        "measured_with_current_repo_code": True,
        "run_id": args.run_id,
        "caa_run_id": args.caa_run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "static_topology": static,
        "dense_face_provenance_replay": provenance_replay,
        "stage19_source_fidelity": {
            "passed": bool(source_fidelity_passed),
            "demo_geometry_lineage": bool(demo_geometry_lineage),
            "views": source_fidelity_rows,
        },
        "corrected_skin_identity": {
            "run_skin_path": str(run_skin_path),
            "run_skin_sha256": run_skin_sha,
            "corrected_skin_path": str(corrected_skin_path),
            "corrected_skin_sha256": corrected_skin_sha,
            "exact_bytes": bool(run_skin_sha == corrected_skin_sha),
            "same_skin_lineage_hash": bool(
                skin.skin_lineage_hash == corrected_skin.skin_lineage_hash
            ),
            "surface_binding_pass": bool(
                skin.surface_binding_hash == surface.geometry_lineage_hash
            ),
            "skeleton_binding_pass": bool(
                skin.skeleton_binding_hash == skeleton.skeleton_lineage_hash
            ),
            "candidate_weight_simplex_residual_max": skin_simplex_residual_max,
            "candidate_negative_weight_count": skin_negative_weight_count,
            "candidate_nonfinite_weight_count": skin_nonfinite_count,
        },
        "g3": {
            "passed": bool(g3.passed),
            "failure_invariants": list(g3.failure_invariants),
            "probe_count": int(g3.probe_count),
            "face_count": int(g3.face_count),
            "minimum_area_ratio": float(g3.minimum_area_ratio),
            "maximum_area_ratio": float(g3.maximum_area_ratio),
            "maximum_condition_number": float(g3.maximum_condition_number),
            "minimum_edge_ratio": float(g3.minimum_edge_ratio),
            "maximum_edge_ratio": float(g3.maximum_edge_ratio),
            "report_hash": str(g3.report_hash),
        },
        "actual_motion": {
            "clips": clip_rows,
            "aggregate": {
                "max_edge_gt_4": int(max_edge_gt4),
                "max_edge_gt_10": int(max_edge_gt10),
                "worst_edge_max": float(worst_edge),
                "max_posed_degenerate_face_count": int(max_posed_degenerate),
                "max_all_measurable_projected_flip_count_diagnostic": int(
                    max_all_projected_flips
                ),
                "max_visible_projected_flip_count_demo_views": int(
                    max_visible_projected_flips
                ),
                "rest_unexpected_self_intersection_pair_count": int(
                    len(rest_intersections)
                ),
                "global_worst_actual_motion_frame": global_worst_locator,
                "global_worst_frame_total_self_intersection_pair_count": int(
                    len(global_posed_intersections)
                ),
                "global_worst_frame_new_self_intersection_pair_count": int(
                    max_new_intersections
                ),
                "global_worst_frame_new_self_intersection_pairs_sample": [
                    [int(a), int(b)] for a, b in new_intersections[:32]
                ],
            },
        },
        "exact_caa_render_binding": {
            "candidate_binding_hash": str(
                caa_stage23["candidate_mesh_binding_hash"]
            ),
            "candidate_binding_pass": bool(caa_candidate_binding_pass),
            "source_owned_visual_mesh_mode": dict(
                caa_stage23.get("metadata") or {}
            ).get("source_owned_visual_mesh_mode"),
            "mechanical_caa_mode_pass": bool(caa_is_mechanical_mode),
            "appearance_asset_hash": str(caa_stage23["asset_hash"]),
        },
        "verdict": {
            "geometry_pass": geometry_pass,
            "mechanics_pass": mechanics_pass,
            "exact_render_binding_pass": exact_render_binding_pass,
            "claim": (
                "For this exact Knight corrected-weight/topology CAA render lineage, "
                "current-code geometry/mechanics gates are closed; any remaining "
                "visible smear must be investigated downstream in appearance/visibility/"
                "runtime-presentation rather than attributed to catastrophic mechanical "
                "stretch/topology failure."
                if status.startswith("PASS_")
                else "Geometry/mechanics are not yet clean enough to isolate runtime presentation."
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(
        "KNIGHT_CANONICAL_CAA_MECHANICS_GEOMETRY_LOCK="
        + json.dumps(
            {
                "status": status,
                "candidate": candidate.candidate_lineage_hash,
                "skin": skin.skin_lineage_hash,
                "source_fidelity": bool(source_fidelity_passed),
                "g3": bool(g3.passed),
                "motion_gt4": int(max_edge_gt4),
                "motion_gt10": int(max_edge_gt10),
                "motion_worst": float(worst_edge),
                "visible_projected_flips": int(max_visible_projected_flips),
                "new_self_intersections_global_worst_frame": int(max_new_intersections),
                "corrected_skin_exact": bool(corrected_skin_exact),
                "caa_binding": bool(exact_render_binding_pass),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not status.startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()