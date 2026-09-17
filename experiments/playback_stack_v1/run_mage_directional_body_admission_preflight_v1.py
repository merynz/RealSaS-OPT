from __future__ import annotations

"""Render-free Mage directional BODY admission preflight.

This adapter exercises only current generic RealSaS playback authorities:
- rebuild a source-backed directional observation mesh from the corrected FIT2 surface;
- apply deterministic coverage-preserving local edge flips;
- bind fresh FIT2 skin through exact SurfaceSupportBinding;
- compile the historical idle/run motion through the sealed D0 axis contract;
- build Runtime-v4 directional BODY assets;
- certify continuous triangle orientation + continuous boundary nonintersection.

No raster render, GIF, completion, inpainting, source synthesis, or PRODUCT_PASS occurs.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.mesh.directional_edge_flip_repair import (
    repair_directional_mesh_by_edge_flips_v1,
)
from compiler.realsas_compiler_core.mesh.mesh_binding import qualify_supported_mesh
from compiler.realsas_compiler_core.mesh.mwb2_skin import bind_mwb2_mesh_skin
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    evaluate_mesh_quality,
    mesh_raster_quality_report,
)
from compiler.realsas_compiler_core.motion_3d_adapter_v1 import compile_motion_clip_to_d1_v1
from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_core.playback_directional_body_v1 import (
    build_directional_body_asset_v1,
    build_directional_body_runtime_v4_clip_v1,
    build_directional_body_runtime_v4_contract_v1,
)
from compiler.realsas_compiler_core.playback_directional_motion_cert_v1 import (
    certify_directional_body_motion_v1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import qualify_camera_v3

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1_b2_g10_v1 as mat
import experiments.mage_full_subject_reclosure_v1.run_fit2_baseline_preserving_adaptive_patch_cdt_v1 as patch_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v1 as ceiling_v1
import experiments.mage_full_subject_reclosure_v1.run_fit2_legal_steiner_ceiling_v2 as ceiling_v2

SCHEMA = "RealSaS.MageDirectionalBodyAdmissionPreflight.v1"
VIEW_IDS = tuple(f"V{i}" for i in range(8))
TREATMENT = {
    "name": "P2_B1_G8_DIRECTIONAL_BODY_ADMISSION",
    "repair_hops": 2,
    "boundary_stride": 1,
    "interior_spacing": 8,
}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"EXPECTED_JSON_OBJECT:{path}")
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _uniform_times(duration: float, sample_count: int) -> tuple[float, ...]:
    n = int(sample_count)
    if n < 3 or n % 2 == 0:
        raise RuntimeError("DIRECTIONAL_PREFLIGHT_SAMPLE_COUNT_MUST_BE_ODD_AND_AT_LEAST_3")
    return tuple(map(float, np.linspace(0.0, float(duration), n)))


def _root_joint_id(skeleton) -> str:
    roots = [
        str(j.canonical_joint_id)
        for j in skeleton.joints
        if j.parent_canonical_id is None
    ]
    if len(roots) != 1:
        raise RuntimeError(f"DIRECTIONAL_PREFLIGHT_EXACTLY_ONE_ROOT_REQUIRED:{roots}")
    return roots[0]


def _camera_rows(paths: tuple[Path, ...]) -> tuple[dict[str, dict], dict[str, object]]:
    if len(paths) != 8:
        raise RuntimeError("DIRECTIONAL_PREFLIGHT_REQUIRES_8_CAMERAS")
    raw: dict[str, dict] = {}
    qualified = {}
    for i, path in enumerate(paths):
        if _sha(path) != ceiling_v1.CAMERA_SHA256[i]:
            raise RuntimeError(f"DIRECTIONAL_PREFLIGHT_CAMERA_SHA_DRIFT_V{i}")
        row = _load_json(path)
        view_id = f"V{i}"
        raw[view_id] = row
        qualified[view_id] = qualify_camera_v3(row, view_id=view_id, view_index=i)
    return raw, qualified


def _observation_rows(paths: tuple[Path, ...]) -> dict[str, dict]:
    if len(paths) != 8:
        raise RuntimeError("DIRECTIONAL_PREFLIGHT_REQUIRES_8_OBSERVATIONS")
    out = {}
    for i, path in enumerate(paths):
        digest = _sha(path)
        if digest != ceiling_v1.OBSERVATION_SHA256[i]:
            raise RuntimeError(f"DIRECTIONAL_PREFLIGHT_OBSERVATION_SHA_DRIFT_V{i}")
        out[f"V{i}"] = {
            "path": str(path),
            "sha256": digest,
            "width": 1024,
            "height": 1024,
        }
    return out


def _rebuild_directional_meshes(args, surface):
    obs_paths = tuple(Path(x).resolve() for x in args.observations)
    cam_paths = tuple(Path(x).resolve() for x in args.cameras)
    baselines, domains = mat._baseline_inputs(surface, obs_paths, cam_paths)
    meshes = []
    rows = []
    for view in range(8):
        candidate = patch_v1._build_hybrid_candidate(
            surface,
            baselines[view],
            view=view,
            camera_hash=ceiling_v1.CAMERA_SHA256[view],
            domain=domains[view],
            treatment=TREATMENT,
        )
        mesh = qualify_supported_mesh(surface, candidate)
        before = evaluate_mesh_quality(
            coverage=dict(candidate.residual_report),
            raster_report=mesh_raster_quality_report(
                mesh,
                surface=surface,
                view_index=view,
            ),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        repaired = repair_directional_mesh_by_edge_flips_v1(
            mesh,
            surface,
            min_angle_deg=float(FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.min_raster_triangle_angle_deg),
            max_aspect=float(FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.max_raster_triangle_aspect_ratio),
        )
        after = evaluate_mesh_quality(
            coverage=dict(candidate.residual_report),
            raster_report=mesh_raster_quality_report(
                repaired,
                surface=surface,
                view_index=view,
            ),
            policy=FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
        )
        if float(after["source_alpha_recall"]) != float(before["source_alpha_recall"]):
            raise RuntimeError(f"DIRECTIONAL_PREFLIGHT_EDGE_FLIP_RECALL_DRIFT_V{view}")
        if float(after["precision_inside_alpha"]) != float(before["precision_inside_alpha"]):
            raise RuntimeError(f"DIRECTIONAL_PREFLIGHT_EDGE_FLIP_PRECISION_DRIFT_V{view}")
        flip = repaired.qualification_report.get("directional_edge_flip_repair") or {}
        rows.append({
            "view": view,
            "mesh_lineage_hash_before": mesh.mesh_lineage_hash,
            "mesh_lineage_hash_after": repaired.mesh_lineage_hash,
            "source_alpha_recall": float(after["source_alpha_recall"]),
            "precision_inside_alpha": float(after["precision_inside_alpha"]),
            "alpha_iou": float(after["alpha_iou"]),
            "largest_uncovered_component_fraction": float(after["largest_uncovered_component_fraction"]),
            "min_raster_triangle_angle_deg_before": float(before["min_raster_triangle_angle_deg"]),
            "min_raster_triangle_angle_deg_after": float(after["min_raster_triangle_angle_deg"]),
            "max_raster_triangle_aspect_ratio_before": float(before["max_raster_triangle_aspect_ratio"]),
            "max_raster_triangle_aspect_ratio_after": float(after["max_raster_triangle_aspect_ratio"]),
            "frozen_quality_pass_after": bool(after["passed"]),
            "frozen_quality_failures_after": list(after["failure_invariants"]),
            "accepted_edge_flips": int(flip.get("accepted_flip_count", 0)),
            "bad_face_count_before": int(flip.get("bad_face_count_before", -1)),
            "bad_face_count_after": int(flip.get("bad_face_count_after", -1)),
            "vertex_count": len(repaired.vertices),
            "face_count": len(repaired.faces),
        })
        meshes.append(repaired)
    return tuple(meshes), rows


def run(args) -> dict:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Exact source observation/camera identity.
    camera_paths = tuple(Path(x).resolve() for x in args.cameras)
    observation_paths = tuple(Path(x).resolve() for x in args.observations)
    _camera_raw, cameras = _camera_rows(camera_paths)
    observations = _observation_rows(observation_paths)

    # Reconstruct the current corrected FIT2 surface used by the directional mesh
    # lineage and independently load the exact persisted FIT2 mechanics.
    surface, _tensor, replay = ceiling_v2._preflight_surface(args)
    exact_surface, skeleton, skin, mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface).resolve(),
        Path(args.skeleton).resolve(),
        Path(args.fit2_skin).resolve(),
    )
    if not replay.get("gsa_lineage_exact_match"):
        raise RuntimeError("DIRECTIONAL_PREFLIGHT_GSA_REPLAY_FAIL")
    if surface.geometry_lineage_hash != exact_surface.geometry_lineage_hash:
        raise RuntimeError("DIRECTIONAL_PREFLIGHT_SURFACE_LINEAGE_MISMATCH")

    meshes, mesh_rows = _rebuild_directional_meshes(args, surface)

    # Fresh FIT2 skin is rebound after topology repair; no historical mesh-skin rows
    # survive an edge flip.
    joint_ids = tuple(str(j.canonical_joint_id) for j in skeleton.joints)
    directional_assets = []
    binding_rows = []
    for view, mesh in enumerate(meshes):
        mesh_skin = bind_mwb2_mesh_skin(surface, skeleton, skin, mesh)
        asset = build_directional_body_asset_v1(
            mesh,
            mesh_skin,
            view_id=f"V{view}",
            view_index=view,
            joint_ids=joint_ids,
            source_width=int(observations[f"V{view}"]["width"]),
            source_height=int(observations[f"V{view}"]["height"]),
            source_texture_sha256=str(observations[f"V{view}"]["sha256"]),
        )
        directional_assets.append(asset)
        binding_rows.append({
            "view": view,
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "mesh_skin_lineage_hash": mesh_skin.mesh_skin_lineage_hash,
            "transfer_method": mesh_skin.transfer_method,
            "vertex_count": len(mesh.vertices),
        })

    contract = build_directional_body_runtime_v4_contract_v1(
        tuple(directional_assets),
        cameras,
        root_bone_id=_root_joint_id(skeleton),
        required_view_ids=VIEW_IDS,
    )

    axis_contract_path = Path(args.axis_contract).resolve()
    axis_contract = _load_json(axis_contract_path)
    authored_motion = build_mage_historical_phase_motion(mechanical)
    motion = compile_motion_quality(authored_motion, mechanical)

    certificates = []
    clip_rows = []
    for clip_id, display_name, intent in (
        ("mage_fit1_idle_v2", "Idle Directional BODY", "IDLE"),
        ("mage_fit1_run_v2", "Run Directional BODY", "RUN"),
    ):
        source_clip = next(c for c in motion.clips if str(c.clip_id) == clip_id)
        times = _uniform_times(float(source_clip.duration_sec), int(args.sample_count))
        d1 = compile_motion_clip_to_d1_v1(
            skeleton=skeleton,
            axis_contract=axis_contract,
            motion_state=motion,
            clip_id=clip_id,
            sample_times=times,
        )
        if tuple(d1.joint_ids) != joint_ids:
            raise RuntimeError("DIRECTIONAL_PREFLIGHT_D1_JOINT_ORDER_DRIFT")
        nominal_fps = float(len(d1.times) - 1) / float(d1.duration_seconds)
        runtime_clip = build_directional_body_runtime_v4_clip_v1(
            contract,
            tuple(directional_assets),
            d1,
            display_name=display_name,
            intent=intent,
            nominal_fps=nominal_fps,
            required_view_ids=VIEW_IDS,
            runtime_qualified=False,
        )
        cert = certify_directional_body_motion_v1(
            contract,
            runtime_clip,
            tuple(directional_assets),
            min_source_alpha_recall=float(args.min_source_recall),
            min_precision_inside_alpha=float(args.min_source_precision),
            min_signed_area2=float(args.min_signed_area2),
            required_view_ids=VIEW_IDS,
        )
        certificates.append(cert)
        clip_rows.append({
            "clip_id": clip_id,
            "frame_count": len(runtime_clip.frames),
            "duration_seconds": runtime_clip.duration_seconds,
            "runtime_qualified": runtime_clip.runtime_qualified,
            "d1_compile_hash": d1.compile_hash,
            "certificate_hash": cert.certificate_hash,
            "continuous_orientation_certified": cert.continuous_orientation_certified,
            "boundary_nonintersection_certified": cert.boundary_nonintersection_certified,
            "global_embedding_certified": cert.global_embedding_certified,
            "min_signed_area2_margin": cert.min_signed_area2_margin,
            "min_boundary_distance_at_critical_times": cert.min_boundary_distance_at_critical_times,
            "source_alpha_recall_floor": cert.source_alpha_recall_floor,
            "precision_inside_alpha_floor": cert.precision_inside_alpha_floor,
        })

    rest_floor = min(float(row["source_alpha_recall"]) for row in mesh_rows)
    rest_mean = sum(float(row["source_alpha_recall"]) for row in mesh_rows) / 8.0
    precision_floor = min(float(row["precision_inside_alpha"]) for row in mesh_rows)
    status = "PASS__DIRECTIONAL_BODY_SOURCE_AND_CONTINUOUS_MOTION_CERTIFIED"
    report = {
        "schema": SCHEMA,
        "status": status,
        "subject_id": "MAGE_FIT2",
        "scope": "RENDER_FREE_DIRECTIONAL_BODY_ADMISSION",
        "treatment": dict(TREATMENT),
        "surface_lineage_hash": surface.geometry_lineage_hash,
        "skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "motion_state_hash": motion.motion_state_hash,
        "axis_contract_sha256": _sha(axis_contract_path),
        "rest_source_alpha_recall_floor": rest_floor,
        "rest_source_alpha_recall_mean": rest_mean,
        "rest_source_precision_floor": precision_floor,
        "view_meshes": mesh_rows,
        "mesh_skin_bindings": binding_rows,
        "playback_contract_hash": contract.contract_hash,
        "clips": clip_rows,
        "completion_used": False,
        "render_executed": False,
        "gif_executed": False,
        "product_pass_claimed": False,
        "admission_rule": (
            "RENDER_MAY_START_ONLY_IF_THIS_REPORT_STATUS_IS_PASS_AND_ALL_CLIP_"
            "CERTIFICATES_HAVE_GLOBAL_EMBEDDING"
        ),
    }
    report_path = out / "MAGE_DIRECTIONAL_BODY_ADMISSION_PREFLIGHT_V1.json"
    _write_json(report_path, report)
    print("MAGE_DIRECTIONAL_BODY_ADMISSION_PASS")
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": _sha(report_path),
        "rest_recall_floor": rest_floor,
        "rest_recall_mean": rest_mean,
        "precision_floor": precision_floor,
        "clips": clip_rows,
    }, indent=2, sort_keys=True))
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zero-surface", required=True)
    p.add_argument("--cameras", nargs=8, required=True)
    p.add_argument("--observations", nargs=8, required=True)
    p.add_argument("--fit2-surface", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--fit2-skin", required=True)
    p.add_argument("--axis-contract", required=True)
    p.add_argument("--sample-count", type=int, default=17)
    p.add_argument("--min-source-recall", type=float, default=0.97)
    p.add_argument("--min-source-precision", type=float, default=0.995)
    p.add_argument("--min-signed-area2", type=float, default=1.0e-4)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
