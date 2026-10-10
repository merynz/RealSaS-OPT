from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
)
from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.carrier_skin_v1 import qualified_carrier_skin_from_dict
from compiler.realsas_compiler_core.deformation_envelope_derivation_v2 import derive_deformation_envelope_v2
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_post_bind_v2
from compiler.realsas_compiler_core import motion_dynamic_proof_v2 as motion_proof
from compiler.realsas_compiler_core import visual_material_render_v1 as visual_render
from compiler.realsas_compiler_core.skin_topology_compatibility_carrier_v1 import run_skin_topology_compatibility_carrier_v1
from compiler.realsas_compiler_core.visibility_v2 import rasterize_visible_owner
from compiler.realsas_compiler_core.visual_material_v1 import load_visual_material
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    bind_region_visual_vertices_to_mechanical_affine_v1,
    build_visual_mesh_from_region_labels_v1,
    evaluate_region_visual_binding_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import _load_source_inputs
from compiler.realsas_compiler_core.types import QualificationError
from tools.demo import render_knight_motion_preview_v1 as motion_preview
from tools.demo.render_knight_motion_preview_v1 import _skin, _tracks_for_clip
from tools.ops import render_knight_latest_current_visual_v1 as base

CAA_RUN = "KNIGHT_LATEST_CURRENT_CAA_VISUAL_20261010"
MAT_RUN = "KNIGHT_LATEST_CARRIER_MATERIALIZATION_20261010"
SKELETON_NAME = "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_QUALIFIED_SKELETON_V541.json"
VIEW_INDEX = 1
CLIP_ID = "demo_idle_v1"
FRAME_INDEX = 0


def _load_stage_payload(ledger: dict, stage_id: str, schema: str) -> dict:
    row = next(x for x in ledger["stages"] if x["id"] == stage_id)
    out = next(x for x in row.get("outputs") or () if x.get("schema") == schema)
    return json.loads(Path(out["path"]).read_text())


def _classify_depth(kwargs: dict) -> dict:
    original = base.render_visual_material
    try:
        rr = original(**kwargs)
        return {
            "sealed_pass": True,
            "tie": False,
            "overflow": False,
            "minimum_fragment_layers": 4,
            "overlap_pixel_count": int(rr.overlap_pixel_count),
            "rgba": rr.straight_rgba_u8,
        }
    except QualificationError as exc:
        if str(exc) != "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW":
            raise
    old_eps = visual_render.DEPTH_TIE_EPSILON
    old_layers = visual_render.MAXIMUM_FRAGMENT_LAYERS
    tie = False
    overflow = False
    minimum = None
    overlap = None
    rgba = None
    try:
        visual_render.MAXIMUM_FRAGMENT_LAYERS = 1_000_000
        try:
            rr = original(**kwargs)
            overlap = int(rr.overlap_pixel_count)
            rgba = rr.straight_rgba_u8
        except QualificationError as diag:
            tie = str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW"
        visual_render.DEPTH_TIE_EPSILON = -1.0
        visual_render.MAXIMUM_FRAGMENT_LAYERS = old_layers
        try:
            original(**kwargs)
        except QualificationError as diag:
            overflow = str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW"
        if overflow:
            for layers in range(int(old_layers) + 1, 33):
                visual_render.MAXIMUM_FRAGMENT_LAYERS = layers
                try:
                    rr = original(**kwargs)
                except QualificationError as diag:
                    if str(diag) == "VISUAL_DEPTH_TIE_OR_FRAGMENT_OVERFLOW":
                        continue
                    raise
                minimum = layers
                overlap = int(rr.overlap_pixel_count)
                rgba = rr.straight_rgba_u8
                break
    finally:
        visual_render.DEPTH_TIE_EPSILON = old_eps
        visual_render.MAXIMUM_FRAGMENT_LAYERS = old_layers
    return {
        "sealed_pass": False,
        "tie": bool(tie),
        "overflow": bool(overflow),
        "minimum_fragment_layers": minimum,
        "overlap_pixel_count": overlap,
        "rgba": rgba,
    }


def _region_affine(rest: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = np.empty_like(target)
    rows = []
    for rid in sorted(set(map(int, region.tolist()))):
        ids = np.flatnonzero(region == rid)
        x = np.asarray(rest[ids], dtype=np.float64)
        y = np.asarray(target[ids], dtype=np.float64)
        X = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
        rank = int(np.linalg.matrix_rank(X))
        if len(ids) >= 3 and rank == 3:
            B, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ B
            linear = B[:2, :]
        else:
            delta = np.mean(y - x, axis=0)
            pred = x + delta
            linear = np.eye(2, dtype=np.float64)
        det = float(np.linalg.det(linear))
        if det <= 1.0e-9:
            # Deterministic orientation-preserving fallback: similarity fit.
            pred, sim_row = _fit_similarity(x, y)
            linear = np.asarray(sim_row.pop("linear"), dtype=np.float64)
            det = float(np.linalg.det(linear))
        residual = np.linalg.norm(pred - y, axis=1)
        out[ids] = pred
        rows.append({
            "region_id": rid,
            "vertex_count": int(len(ids)),
            "determinant": det,
            "target_rms_residual_px": float(np.sqrt(np.mean(residual * residual))),
            "target_max_residual_px": float(np.max(residual, initial=0.0)),
        })
    return out, rows


def _fit_similarity(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    if len(x) == 0:
        raise RuntimeError("SIMILARITY_EMPTY_REGION")
    xm = np.mean(x, axis=0)
    ym = np.mean(y, axis=0)
    xc = x - xm
    yc = y - ym
    denom = float(np.sum(xc * xc))
    if len(x) < 2 or denom <= 1.0e-12:
        linear = np.eye(2, dtype=np.float64)
        pred = x + (ym - xm)
    else:
        H = xc.T @ yc
        U, s, Vt = np.linalg.svd(H)
        R = U @ Vt
        if float(np.linalg.det(R)) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
            s[-1] *= -1.0
        scale = max(1.0e-9, float(np.sum(s) / denom))
        linear = scale * R
        pred = xc @ linear + ym
    residual = np.linalg.norm(pred - y, axis=1)
    return pred, {
        "linear": linear.tolist(),
        "determinant": float(np.linalg.det(linear)),
        "target_rms_residual_px": float(np.sqrt(np.mean(residual * residual))),
        "target_max_residual_px": float(np.max(residual, initial=0.0)),
    }


def _region_similarity(rest: np.ndarray, target: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    out = np.empty_like(target)
    rows = []
    for rid in sorted(set(map(int, region.tolist()))):
        ids = np.flatnonzero(region == rid)
        pred, row = _fit_similarity(rest[ids], target[ids])
        out[ids] = pred
        rows.append({"region_id": rid, "vertex_count": int(len(ids)), **{k: v for k, v in row.items() if k != "linear"}})
    return out, rows


def _method_metrics(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray, rows: list[dict]) -> dict:
    geom = base.visual_metrics(rest, posed, faces)
    rms = [float(row["target_rms_residual_px"]) for row in rows]
    maxr = [float(row["target_max_residual_px"]) for row in rows]
    det = [float(row["determinant"]) for row in rows]
    return {
        **geom,
        "region_count": int(len(rows)),
        "maximum_region_target_rms_residual_px": max(rms, default=0.0),
        "maximum_region_target_residual_px": max(maxr, default=0.0),
        "minimum_region_determinant": min(det, default=1.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-root", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resolution", type=int, default=320)
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()
    auth = a.authority_root.resolve()
    input_root = a.input_root.resolve()
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    caa = auth / "runs" / CAA_RUN
    mat = auth / "runs" / MAT_RUN

    ledger = json.loads((caa / "ACTIVE_RUN_V2.json").read_text())
    manifest = json.loads((caa / "run_manifest.json").read_text())
    candidate = canonical_mesh_candidate_from_dict(json.loads((mat / "rebound_candidate.json").read_text()))
    skin = qualified_carrier_skin_from_dict(json.loads((mat / "qualified_carrier_skin.json").read_text()))
    skeleton = qualified_skeleton_from_dict(json.loads((input_root / SKELETON_NAME).read_text()))
    cam_raw = _load_stage_payload(ledger, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")
    camera_set = qualified_camera_set_from_dict(cam_raw)
    cameras = tuple(sorted(camera_set.cameras, key=lambda x: int(x.view_index)))
    obs_raw = _load_stage_payload(ledger, "07_OBSERVATION_CONTRACT_QUALIFIED", "RealSaS.QualifiedObservationSetIR.v1")
    observation = qualified_observation_set_from_dict(obs_raw)
    policy_raw = _load_stage_payload(ledger, "18_CANONICAL_MESH_ADDRESSING_BUILD", "RealSaS.MeshQualificationPolicyIR.v1")
    policy = mesh_policy_from_dict(policy_raw)
    asset_raw = _load_stage_payload(ledger, "23_COMPLETE_APPEARANCE_ASSET_BAKED", "RealSaS.CompleteAppearanceAssetIR.v2")
    asset = complete_appearance_asset_from_dict(asset_raw)

    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)
    mech_faces = base.mechanical_faces(candidate)
    _axis_payload, envelope = derive_deformation_envelope_v2(skeleton=skeleton, camera_set=camera_set)
    compatibility = run_skin_topology_compatibility_carrier_v1(candidate, skeleton=skeleton, skin=skin, envelope=envelope, cameras=cameras, policy=policy)
    unsafe = set(map(int, compatibility.get("unsafe_face_indices") or ()))

    ctx = {"repo_root": Path(".").resolve(), "authority_root": auth, "run_root": caa, "run_id": CAA_RUN, "run_manifest_path": caa / "run_manifest.json", "run_manifest": manifest, "ledger": ledger, "stage": {"id": "F6_MICRO"}}
    rgba_by_view, mask_by_view = _load_source_inputs(ctx, observation)
    vi = VIEW_INDEX
    camera = cameras[vi]
    mask = np.asarray(mask_by_view[vi], dtype=bool)
    h, w = mask.shape
    visibility = rasterize_visible_owner(candidate, camera, positions=rest, width=w, height=h, max_layers=4)
    final_labels, seed_labels, charts = partition_source_mask_by_safe_face_adjacency_v1(mask, visibility.owner_face_index, mech_faces, unsafe, minimum_seed_pixels=64)
    region_value = build_visual_mesh_from_region_labels_v1(mask, final_labels, target_edge_px=16)
    vm = region_value.mesh
    region = np.asarray(region_value.vertex_region_id, dtype=np.int32)
    binding = bind_region_visual_vertices_to_mechanical_affine_v1(
        points_source_xy=vm.positions,
        vertex_region_id=region,
        seed_region_labels=np.asarray(seed_labels, dtype=np.int32),
        owner_face_index=np.asarray(visibility.owner_face_index, dtype=np.int64),
        mechanical_positions_xyz=rest,
        mechanical_faces=mech_faces,
        camera=camera,
        candidate_seed_count=16,
        max_seed_distance_px=float(np.hypot(w, h)),
    )
    rest_eval = evaluate_region_visual_binding_v1(binding, posed_mechanical_positions_xyz=rest, camera=camera)
    rest_err = np.linalg.norm(rest_eval - np.asarray(vm.positions, dtype=np.float64), axis=1)
    if float(np.max(rest_err, initial=0.0)) > 1.0e-7:
        raise RuntimeError("F6_MICRO_REST_BIND_DRIFT")

    frames, qualification = derive_joint_frames_post_bind_v2(skeleton, carrier_skin=skin, cameras=cameras)
    if qualification["status"] != "PASS":
        raise RuntimeError("F6_MICRO_POST_BIND_FRAME_FAIL")
    def _canonical_frames(_skeleton, *, cameras):
        return frames
    motion_preview.derive_joint_frames_from_skeleton = _canonical_frames
    motion_proof.derive_joint_frames_from_skeleton = _canonical_frames

    motions = base.motion_payloads(input_root)
    payload, _motion_path = motions[CLIP_ID]
    source_report = json.loads(base.SOURCE_REPORT.read_text())
    tracks, _mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
    mats, _, _frame_hash = motion_proof._joint_pose_v2(skeleton=skeleton, tracks=tracks, time_seconds=0.0, cameras=cameras)
    with np.load(input_root / "MIRA_AXIS41_TESSA_M_SKIN_FIELD_V55.npz", allow_pickle=False) as z:
        W = np.asarray(z["weights_axis41_canonical"], dtype=np.float64)
        joint_ids = tuple(map(str, np.asarray(z["canonical_joint_ids"]).tolist()))
    posed_mechanical = _skin(rest, W, joint_ids, mats)
    raw_posed = evaluate_region_visual_binding_v1(binding, posed_mechanical_positions_xyz=posed_mechanical, camera=camera)
    affine_posed, affine_rows = _region_affine(np.asarray(vm.positions, dtype=np.float64), raw_posed, region)
    similarity_posed, similarity_rows = _region_similarity(np.asarray(vm.positions, dtype=np.float64), raw_posed, region)

    projected = np.asarray(base.project_points_xyz_v3(posed_mechanical, camera), dtype=np.float64)
    depths = np.sum(projected[np.asarray(binding["mechanical_face_indices"], dtype=np.int64), 2] * np.asarray(binding["mechanical_barycentric"], dtype=np.float64), axis=1)
    texture = {int(r.direction_index): r for r in asset.textures}[vi]
    rgba = np.asarray(Image.open(Path(texture.transport_png_path)).convert("RGBA"), dtype=np.uint8)
    prov, donor = load_visual_material(asset, view_index=vi, rgba=rgba)

    common = {
        "depths": depths,
        "faces": np.asarray(vm.faces, dtype=np.int64),
        "uv": np.asarray(vm.uv, dtype=np.float64),
        "texture": rgba,
        "provenance": prov,
        "source_view": donor,
        "view_index": vi,
        "resolution": int(a.resolution),
    }
    raw_depth = _classify_depth({**common, "positions": raw_posed})
    affine_depth = _classify_depth({**common, "positions": affine_posed})
    similarity_depth = _classify_depth({**common, "positions": similarity_posed})

    raw_metrics = base.visual_metrics(vm.positions, raw_posed, vm.faces)
    affine_metrics = _method_metrics(np.asarray(vm.positions, dtype=np.float64), affine_posed, np.asarray(vm.faces, dtype=np.int64), affine_rows)
    similarity_metrics = _method_metrics(np.asarray(vm.positions, dtype=np.float64), similarity_posed, np.asarray(vm.faces, dtype=np.int64), similarity_rows)

    candidates = []
    for name, depth, metrics, image in (
        ("REGION_AFFINE", affine_depth, affine_metrics, affine_depth.get("rgba")),
        ("REGION_SIMILARITY", similarity_depth, similarity_metrics, similarity_depth.get("rgba")),
    ):
        candidates.append({"name": name, "depth": {k: v for k, v in depth.items() if k != "rgba"}, "metrics": metrics})
        if image is not None:
            Image.fromarray(np.asarray(image, dtype=np.uint8), "RGBA").save(out / f"F6_MICRO_{name}.png")

    admissible = [c for c in candidates if c["depth"]["sealed_pass"] and c["metrics"]["flipped_triangles"] == 0 and c["metrics"]["minimum_region_determinant"] > 0.0]
    if admissible:
        selected = min(admissible, key=lambda c: (c["metrics"]["maximum_region_target_rms_residual_px"], c["metrics"]["p95_edge_ratio"], c["name"]))
        status = "PASS_F6_MICRO_COURT"
    else:
        selected = None
        status = "FAIL_F6_MICRO_COURT"

    report = {
        "schema": "RealSaS.F6VisualDeformationMicroCourt.v1",
        "status": status,
        "product_authority_claimed": False,
        "view_index": vi,
        "clip_id": CLIP_ID,
        "frame_index": FRAME_INDEX,
        "rest_binding_max_error_px": float(np.max(rest_err, initial=0.0)),
        "visual_vertex_count": int(len(vm.positions)),
        "visual_face_count": int(len(vm.faces)),
        "region_count": int(len(charts)),
        "raw": {"depth": {k: v for k, v in raw_depth.items() if k != "rgba"}, "metrics": raw_metrics},
        "candidates": candidates,
        "selected_operator": None if selected is None else selected["name"],
        "selection_rule": "SEALED_4_LAYER_PASS__ZERO_FLIPS__POSITIVE_REGION_DETERMINANT__MIN_SAMPLE_RMS_THEN_P95_STRETCH",
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("F6_MICRO_COURT", json.dumps(report, sort_keys=True), flush=True)
    if selected is None:
        raise RuntimeError("F6_MICRO_COURT_NO_ADMISSIBLE_OPERATOR")


if __name__ == "__main__":
    main()
