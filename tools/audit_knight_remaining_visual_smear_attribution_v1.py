from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    CAA_PROVENANCE_BY_CODE,
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
    _ctx,
    _joint_pose_v2,
    _load_texture_pages,
    _skin,
    _tracks_for_clip,
)


def _face_indices(candidate) -> np.ndarray:
    vi = {str(v.candidate_vertex_id): i for i, v in enumerate(candidate.vertices)}
    return np.asarray(
        [[vi[str(x)] for x in face] for face in candidate.faces], dtype=np.int64
    )


def _face_pixel_counts(render, pixel_mask: np.ndarray, face_count: int) -> np.ndarray:
    mask = np.asarray(pixel_mask, dtype=bool)
    contrib = np.asarray(render.contributing_layer_mask, dtype=bool)
    owner = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    selected = mask[:, :, None] & contrib & (owner >= 0)
    if not np.any(selected):
        return np.zeros(face_count, dtype=np.int64)
    pixel_id = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    px = np.broadcast_to(pixel_id[:, :, None], owner.shape)[selected]
    fi = owner[selected]
    base = np.int64(face_count + 1)
    pairs = np.unique(px * base + fi)
    face = pairs % base
    return np.bincount(face.astype(np.int64), minlength=face_count).astype(np.int64)


def _provenance_hist(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    codes, counts = np.unique(values.astype(np.uint8), return_counts=True)
    out = {}
    for code, count in zip(codes.tolist(), counts.tolist()):
        name = CAA_PROVENANCE_BY_CODE.get(int(code), f"UNKNOWN_{int(code)}")
        out[str(name)] = int(count)
    return out


def _edge_ratio(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray) -> np.ndarray:
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
    return (pl / np.maximum(rl, 1e-15)).max(axis=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--counterfactual-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    cf = json.loads(a.counterfactual_json.read_text())
    ctx = _ctx(a.authority_root.expanduser().resolve(), a.run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    surface = rigging_surface_from_dict(
        stage_output_payload(
            ctx, "15_RIGGING_SURFACE_QUALIFIED", "RealSaS.RiggingSurfaceIR.v1"
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1"
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1"
        )
    )
    cameras = tuple(
        sorted(
            qualified_camera_set_from_dict(
                stage_output_payload(
                    ctx, "05_CAMERA_CONTRACT_SOLVED",
                    "RealSaS.QualifiedCameraSetIR.v1",
                )
            ).cameras,
            key=lambda x: int(x.view_index),
        )
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx, "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    _source_rgba, source_masks = _load_source_inputs(ctx, observation)
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx, "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {int(row.direction_index): row for row in appearance.textures}

    rest, weights, faces_tuple = _candidate_skin_matrix(
        candidate, surface=surface, skeleton=skeleton, skin=skin
    )
    rest = np.asarray(rest, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    faces = np.asarray(faces_tuple, dtype=np.int64)
    joint_ids = tuple(j.canonical_joint_id for j in skeleton.joints)
    seam_vertex = np.asarray(
        [str(v.support_binding.mode) == "SEAM_GEOMETRY_INTERPOLATION" for v in candidate.vertices],
        dtype=bool,
    )
    seam_face = np.any(seam_vertex[faces], axis=1)

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    rest_renders = {}
    rest_face_pixels = {}
    for view in (0, 2):
        texture = _load_texture_pages(texture_by_view[view])
        rr = render_caa_reference(
            mesh=candidate, camera=cameras[view], face_uv=face_uv,
            texture_rgba_u8=texture, provenance_atlas=provenance[view],
            face_page_index=face_page, positions=rest,
        )
        rest_renders[view] = rr
        rest_face_pixels[view] = _face_pixel_counts(
            rr, np.asarray(rr.final_alpha, dtype=bool), len(candidate.faces)
        )

    report = {
        "schema": "RealSaS.KnightRemainingVisualSmearAttribution.v1",
        "status": "MEASURED_AUDIT_ONLY__NO_PRODUCT_AUTHORITY",
        "run_id": a.run_id,
        "candidate_lineage_hash": str(candidate.candidate_lineage_hash),
        "appearance_asset_hash": str(appearance.asset_hash),
        "mechanical_support_contract": "product_mesh_skin_v1._skin_support_coefficients",
        "frames": [],
        "claim_boundary": [
            "All posed geometry uses the canonical mechanical seam support that closes G3/actual-motion catastrophic stretch.",
            "The audit attributes only remaining pixels outside the exact same-view source foreground mask.",
            "Appearance provenance follows alpha-contributing renderer layers.",
        ],
    }

    cf_by_clip = {row["clip_id"]: row for row in cf["clips"]}
    for clip_id in ("demo_idle_v1", "demo_run_v1", "demo_slash_v1"):
        selected = cf_by_clip[clip_id]
        time_seconds = float(selected["selected_visual_time_seconds"])
        motion_path = (
            ctx["run_root"] / "inputs" / "motion" / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        mats, _joint_positions, frame_hash = _joint_pose_v2(
            skeleton=skeleton, tracks=tracks, time_seconds=time_seconds, cameras=cameras
        )
        posed = _skin(rest, weights, joint_ids, mats)
        face_edge = _edge_ratio(rest, posed, faces)
        frame_row = {
            "clip_id": clip_id,
            "time_seconds": time_seconds,
            "frame_hash": str(frame_hash),
            "edge_gt_4": int(np.count_nonzero(face_edge > 4.0)),
            "edge_gt_10": int(np.count_nonzero(face_edge > 10.0)),
            "edge_max": float(np.max(face_edge)),
            "views": [],
            "mapping": mapping,
        }

        for view in (0, 2):
            texture = _load_texture_pages(texture_by_view[view])
            render = render_caa_reference(
                mesh=candidate, camera=cameras[view], face_uv=face_uv,
                texture_rgba_u8=texture, provenance_atlas=provenance[view],
                face_page_index=face_page, positions=posed,
            )
            source = np.asarray(source_masks[view], dtype=bool)
            alpha = np.asarray(render.final_alpha, dtype=bool)
            extra = alpha & ~source
            provenance_code = np.asarray(render.provenance_code, dtype=np.uint8)
            owner = np.asarray(render.owner_face_index, dtype=np.int64)
            rep_valid = extra & (owner >= 0)
            rep_face = owner[rep_valid]

            rep_seam = np.zeros_like(extra)
            if np.any(rep_valid):
                rep_seam[rep_valid] = seam_face[rep_face]

            rest_visible_face = rest_face_pixels[view] > 0
            rep_rest_visible = np.zeros_like(extra)
            if np.any(rep_valid):
                rep_rest_visible[rep_valid] = rest_visible_face[rep_face]

            contrib = np.asarray(render.contributing_layer_mask, dtype=bool)
            layer_owner = np.asarray(render.layer_owner_face_index, dtype=np.int64)
            valid_layer = contrib & (layer_owner >= 0)
            layer_seam = np.zeros_like(valid_layer)
            if np.any(valid_layer):
                layer_seam[valid_layer] = seam_face[layer_owner[valid_layer]]
            extra_any_seam = extra & np.any(layer_seam, axis=2)
            extra_any_identity = extra & np.any(valid_layer & ~layer_seam, axis=2)

            face_extra = _face_pixel_counts(render, extra, len(candidate.faces))
            ranked = np.argsort(face_extra)[::-1]
            top = []
            for fi in ranked[:80]:
                count = int(face_extra[int(fi)])
                if count <= 0:
                    break
                top.append({
                    "face_index": int(fi),
                    "extra_pixel_count": count,
                    "seam_face": bool(seam_face[int(fi)]),
                    "rest_contributing_pixel_count": int(rest_face_pixels[view][int(fi)]),
                    "rest_visible": bool(rest_visible_face[int(fi)]),
                    "canonical_motion_edge_ratio": float(face_edge[int(fi)]),
                    "candidate_vertex_ids": [str(x) for x in candidate.faces[int(fi)]],
                })

            total_extra = int(np.count_nonzero(extra))
            frame_row["views"].append({
                "view_index": int(view),
                "source_foreground_pixel_count": int(np.count_nonzero(source)),
                "rendered_alpha_pixel_count": int(np.count_nonzero(alpha)),
                "extra_outside_source_pixel_count": total_extra,
                "extra_fraction_of_rendered_alpha": (
                    None if not np.count_nonzero(alpha)
                    else float(total_extra / np.count_nonzero(alpha))
                ),
                "extra_provenance_histogram": _provenance_hist(provenance_code[extra]),
                "extra_representative_owner_seam_face_pixel_count": int(np.count_nonzero(rep_seam)),
                "extra_representative_owner_nonseam_face_pixel_count": int(np.count_nonzero(rep_valid & ~rep_seam)),
                "extra_representative_owner_rest_visible_pixel_count": int(np.count_nonzero(rep_rest_visible)),
                "extra_representative_owner_rest_invisible_pixel_count": int(np.count_nonzero(rep_valid & ~rep_rest_visible)),
                "extra_pixels_with_any_seam_face_contribution": int(np.count_nonzero(extra_any_seam)),
                "extra_pixels_with_any_nonseam_face_contribution": int(np.count_nonzero(extra_any_identity)),
                "top_extra_faces": top,
            })

        report["frames"].append(frame_row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_REMAINING_VISUAL_SMEAR_ATTRIBUTION="
        + json.dumps(
            {
                "frames": [
                    {
                        "clip": row["clip_id"],
                        "edge_gt4": row["edge_gt_4"],
                        "edge_max": row["edge_max"],
                        "views": [
                            {
                                "view": v["view_index"],
                                "extra": v["extra_outside_source_pixel_count"],
                                "provenance": v["extra_provenance_histogram"],
                                "rep_seam": v["extra_representative_owner_seam_face_pixel_count"],
                                "rep_nonseam": v["extra_representative_owner_nonseam_face_pixel_count"],
                                "rep_rest_visible": v["extra_representative_owner_rest_visible_pixel_count"],
                                "rep_rest_invisible": v["extra_representative_owner_rest_invisible_pixel_count"],
                            }
                            for v in row["views"]
                        ],
                    }
                    for row in report["frames"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
