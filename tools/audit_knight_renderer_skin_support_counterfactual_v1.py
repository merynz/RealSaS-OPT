from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index,
    load_face_uv,
    load_provenance_atlas,
    render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import (
    _candidate_skin_matrix,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _candidate_skin_weights,
    _ctx,
    _joint_pose_v2,
    _load_texture_pages,
    _skin,
    _tracks_for_clip,
)


CLIPS = ("demo_idle_v1", "demo_run_v1", "demo_slash_v1")


def _face_edge_metric(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> dict:
    r = rest[faces]
    p = posed[faces]
    rl = np.stack(
        (
            np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
            np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
            np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
        ),
        axis=1,
    )
    pl = np.stack(
        (
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ),
        axis=1,
    )
    e = (pl / np.maximum(rl, 1e-15)).max(axis=1)
    return {
        "edge_p95": float(np.quantile(e, 0.95)),
        "edge_p99": float(np.quantile(e, 0.99)),
        "edge_max": float(np.max(e)),
        "edge_gt_4": int(np.count_nonzero(e > 4.0)),
        "edge_gt_10": int(np.count_nonzero(e > 10.0)),
    }


def _source_alpha_metrics(render, source_mask: np.ndarray) -> dict:
    alpha = np.asarray(render.final_alpha, dtype=bool)
    source = np.asarray(source_mask, dtype=bool)
    overlap = alpha & source
    extra = alpha & ~source
    missing = source & ~alpha
    ac = int(np.count_nonzero(alpha))
    sc = int(np.count_nonzero(source))
    oc = int(np.count_nonzero(overlap))
    ec = int(np.count_nonzero(extra))
    return {
        "rendered_alpha_pixel_count": ac,
        "source_foreground_pixel_count": sc,
        "overlap_pixel_count": oc,
        "extra_outside_source_pixel_count": ec,
        "missing_source_pixel_count": int(np.count_nonzero(missing)),
        "rendered_to_source_area_ratio": None if sc == 0 else float(ac / sc),
        "source_alpha_precision": None if ac == 0 else float(oc / ac),
        "source_alpha_recall": None if sc == 0 else float(oc / sc),
        "extra_fraction_of_rendered_alpha": None if ac == 0 else float(ec / ac),
    }


def _compose(left: Image.Image, right: Image.Image, title: str) -> Image.Image:
    a = left.convert("RGBA")
    b = right.convert("RGBA")
    header = 36
    canvas = Image.new("RGBA", (a.width + b.width, max(a.height, b.height) + header), (0, 0, 0, 0))
    canvas.alpha_composite(a, (0, header))
    canvas.alpha_composite(b, (a.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "WRONG: geometry support as skin", fill=(255, 255, 255, 255))
    draw.text((a.width + 8, 8), "CANONICAL: mechanical skin support", fill=(255, 255, 255, 255))
    draw.text((8, 22), title, fill=(255, 255, 255, 255))
    return canvas


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    a = p.parse_args()

    ctx = _ctx(a.authority_root.expanduser().resolve(), a.run_id)
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
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    _source_rgba, source_masks = _load_source_inputs(ctx, observation)
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}

    wrong_joint_ids, wrong_w = _candidate_skin_weights(candidate, skin, skeleton)
    rest, correct_w, faces_tuple = _candidate_skin_matrix(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
    )
    correct_joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    faces = np.asarray(faces_tuple, dtype=np.int64)
    rest = np.asarray(rest, dtype=np.float64)
    correct_w = np.asarray(correct_w, dtype=np.float64)
    wrong_w = np.asarray(wrong_w, dtype=np.float64)

    wrong_index = {str(jid): i for i, jid in enumerate(wrong_joint_ids)}
    aligned_wrong = np.stack(
        [wrong_w[:, wrong_index[str(jid)]] for jid in correct_joint_ids],
        axis=1,
    )
    l1 = np.sum(np.abs(aligned_wrong - correct_w), axis=1)
    seam = np.asarray(
        [str(v.support_binding.mode) == "SEAM_GEOMETRY_INTERPOLATION" for v in candidate.vertices],
        dtype=bool,
    )
    nonseam = ~seam

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    report = {
        "schema": "RealSaS.KnightRendererMechanicalSkinSupportCounterfactual.v1",
        "status": "MEASURED_CAUSAL_COUNTERFACTUAL__NO_PRODUCT_AUTHORITY",
        "run_id": a.run_id,
        "candidate_lineage_hash": str(candidate.candidate_lineage_hash),
        "appearance_asset_hash": str(appearance.asset_hash),
        "contract_difference": {
            "renderer_current": "vertex.support_binding.coefficients",
            "mechanical_canonical": "product_mesh_skin_v1._skin_support_coefficients(vertex)",
            "seam_mode": "SEAM_GEOMETRY_INTERPOLATION",
            "canonical_seam_field": "support_binding.metadata.skin_support_coefficients",
        },
        "weight_matrix_difference": {
            "vertex_count": int(len(candidate.vertices)),
            "seam_vertex_count": int(np.count_nonzero(seam)),
            "nonseam_vertex_count": int(np.count_nonzero(nonseam)),
            "differing_vertex_count_gt_1e_9": int(np.count_nonzero(l1 > 1e-9)),
            "differing_seam_vertex_count_gt_1e_9": int(np.count_nonzero(seam & (l1 > 1e-9))),
            "differing_nonseam_vertex_count_gt_1e_9": int(np.count_nonzero(nonseam & (l1 > 1e-9))),
            "l1_mean_all": float(np.mean(l1)),
            "l1_p95_all": float(np.quantile(l1, 0.95)),
            "l1_max_all": float(np.max(l1)),
            "l1_mean_seam": float(np.mean(l1[seam])) if np.any(seam) else None,
            "l1_p95_seam": float(np.quantile(l1[seam], 0.95)) if np.any(seam) else None,
            "l1_max_seam": float(np.max(l1[seam])) if np.any(seam) else None,
        },
        "clips": [],
        "claim_boundary": [
            "Both arms use the exact same corrected Stage18 candidate, Stage32 QualifiedSkinIR, skeleton, cameras, motion tracks, CAA asset and renderer.",
            "The only changed variable is how candidate-vertex skin weights are transferred from surface rows.",
            "WRONG reproduces the frozen renderer implementation.",
            "CANONICAL uses the same mechanical-support contract used by G3/product mesh-skin qualification.",
        ],
    }

    a.image_dir.mkdir(parents=True, exist_ok=True)

    for clip_id in CLIPS:
        motion_path = (
            ctx["run_root"] / "inputs" / "motion" / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        times = np.linspace(
            0.0,
            float(payload["duration_seconds"]),
            4,
            endpoint=not bool(payload.get("loop")),
        )
        frame_rows = []
        posed_wrong_by_frame = {}
        posed_correct_by_frame = {}
        for frame_index, t in enumerate(times):
            mats, _joint_pos, frame_hash = _joint_pose_v2(
                skeleton=skeleton,
                tracks=tracks,
                time_seconds=float(t),
                cameras=cameras,
            )
            posed_wrong = _skin(rest, wrong_w, wrong_joint_ids, mats)
            posed_correct = _skin(rest, correct_w, correct_joint_ids, mats)
            posed_wrong_by_frame[frame_index] = posed_wrong
            posed_correct_by_frame[frame_index] = posed_correct
            wrong_metric = _face_edge_metric(rest, posed_wrong, faces)
            correct_metric = _face_edge_metric(rest, posed_correct, faces)
            frame_rows.append(
                {
                    "frame_index": int(frame_index),
                    "time_seconds": float(t),
                    "motion_frame_hash": str(frame_hash),
                    "wrong_renderer_support": wrong_metric,
                    "canonical_mechanical_support": correct_metric,
                }
            )

        worst_index = int(
            max(
                range(len(frame_rows)),
                key=lambda i: (
                    frame_rows[i]["wrong_renderer_support"]["edge_gt_10"],
                    frame_rows[i]["wrong_renderer_support"]["edge_gt_4"],
                    frame_rows[i]["wrong_renderer_support"]["edge_max"],
                ),
            )
        )
        visual_rows = []
        for view in (0, 2):
            texture = _load_texture_pages(texture_by_view[view])
            wrong_render = render_caa_reference(
                mesh=candidate,
                camera=cameras[view],
                face_uv=face_uv,
                texture_rgba_u8=texture,
                provenance_atlas=provenance[view],
                face_page_index=face_page,
                positions=posed_wrong_by_frame[worst_index],
            )
            correct_render = render_caa_reference(
                mesh=candidate,
                camera=cameras[view],
                face_uv=face_uv,
                texture_rgba_u8=texture,
                provenance_atlas=provenance[view],
                face_page_index=face_page,
                positions=posed_correct_by_frame[worst_index],
            )
            wrong_metrics = _source_alpha_metrics(wrong_render, source_masks[view])
            correct_metrics = _source_alpha_metrics(correct_render, source_masks[view])
            wrong_img = Image.fromarray(np.asarray(wrong_render.straight_rgba_u8, dtype=np.uint8), "RGBA")
            correct_img = Image.fromarray(np.asarray(correct_render.straight_rgba_u8, dtype=np.uint8), "RGBA")
            out_image = a.image_dir / f"{clip_id}_frame{worst_index}_view{view}_AB.png"
            _compose(
                wrong_img,
                correct_img,
                f"{clip_id} frame={worst_index} view={view}",
            ).save(out_image)
            visual_rows.append(
                {
                    "view_index": int(view),
                    "wrong_renderer_support": wrong_metrics,
                    "canonical_mechanical_support": correct_metrics,
                    "ab_image": str(out_image),
                }
            )

        report["clips"].append(
            {
                "clip_id": clip_id,
                "mapping": mapping,
                "frames": frame_rows,
                "selected_visual_frame_index": worst_index,
                "selected_visual_time_seconds": float(times[worst_index]),
                "visual_ab": visual_rows,
            }
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = {
        "weight_difference": report["weight_matrix_difference"],
        "clips": [
            {
                "clip_id": clip["clip_id"],
                "wrong_max_gt4": max(f["wrong_renderer_support"]["edge_gt_4"] for f in clip["frames"]),
                "correct_max_gt4": max(f["canonical_mechanical_support"]["edge_gt_4"] for f in clip["frames"]),
                "wrong_max_gt10": max(f["wrong_renderer_support"]["edge_gt_10"] for f in clip["frames"]),
                "correct_max_gt10": max(f["canonical_mechanical_support"]["edge_gt_10"] for f in clip["frames"]),
                "wrong_worst_edge": max(f["wrong_renderer_support"]["edge_max"] for f in clip["frames"]),
                "correct_worst_edge": max(f["canonical_mechanical_support"]["edge_max"] for f in clip["frames"]),
                "visual_ab": clip["visual_ab"],
            }
            for clip in report["clips"]
        ],
    }
    print("KNIGHT_RENDERER_SKIN_SUPPORT_COUNTERFACTUAL=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
