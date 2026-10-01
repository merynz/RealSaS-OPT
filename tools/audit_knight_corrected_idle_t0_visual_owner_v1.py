from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
from tools.demo.frozen.render_knight_motion_preview_v1_7917be02 import (
    _ctx,
    _load_texture_pages,
    _candidate_skin_weights,
    _joint_pose_v2,
    _skin,
    _tracks_for_clip,
)


def _face_indices(candidate) -> np.ndarray:
    vertex_index = {
        str(vertex.candidate_vertex_id): index
        for index, vertex in enumerate(candidate.vertices)
    }
    return np.asarray(
        [
            [vertex_index[str(vertex_id)] for vertex_id in face]
            for face in candidate.faces
        ],
        dtype=np.int64,
    )


def _face_pixel_counts(render, pixel_mask: np.ndarray, face_count: int) -> np.ndarray:
    mask = np.asarray(pixel_mask, dtype=bool)
    contribution = np.asarray(render.contributing_layer_mask, dtype=bool)
    owners = np.asarray(render.layer_owner_face_index, dtype=np.int64)
    selected = mask[:, :, None] & contribution & (owners >= 0)
    if not np.any(selected):
        return np.zeros((face_count,), dtype=np.int64)
    flat_pixel = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    pixel_values = np.broadcast_to(flat_pixel[:, :, None], owners.shape)[selected]
    selected_faces = owners[selected]
    base = np.int64(face_count + 1)
    unique_pairs = np.unique(pixel_values * base + selected_faces)
    face_values = unique_pairs % base
    return np.bincount(
        face_values.astype(np.int64), minlength=face_count
    ).astype(np.int64)


def _edge_area_ratios(rest: np.ndarray, posed: np.ndarray, faces: np.ndarray):
    r = rest[faces]
    p = posed[faces]
    rest_edges = np.stack(
        [
            np.linalg.norm(r[:, 1] - r[:, 0], axis=1),
            np.linalg.norm(r[:, 2] - r[:, 1], axis=1),
            np.linalg.norm(r[:, 0] - r[:, 2], axis=1),
        ],
        axis=1,
    )
    posed_edges = np.stack(
        [
            np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
            np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
        ],
        axis=1,
    )
    eps = 1e-12
    forward = posed_edges / np.maximum(rest_edges, eps)
    inverse = rest_edges / np.maximum(posed_edges, eps)
    symmetric_edge_ratio = np.maximum(forward, inverse).max(axis=1)
    rest_area = 0.5 * np.linalg.norm(
        np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]), axis=1
    )
    posed_area = 0.5 * np.linalg.norm(
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
    )
    area_ratio = posed_area / np.maximum(rest_area, eps)
    return symmetric_edge_ratio, area_ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ctx = _ctx(args.authority_root.expanduser().resolve(), args.run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
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
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx,
            "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(
        sorted(camera_set.cameras, key=lambda camera: int(camera.view_index))
    )
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
        )
    )
    _source_rgba, source_masks = _load_source_inputs(ctx, observation)
    face_uv = load_face_uv(appearance)
    face_page = load_face_page_index(appearance)
    provenance = load_provenance_atlas(appearance)
    texture_by_view = {
        int(row.direction_index): row for row in appearance.textures
    }

    rest = np.asarray([vertex.P for vertex in candidate.vertices], dtype=np.float64)
    faces = _face_indices(candidate)
    joint_ids, weights = _candidate_skin_weights(candidate, skin, skeleton)
    dominant_joint_col = np.argmax(weights, axis=1)
    dominant_joint_id = np.asarray(
        [str(joint_ids[int(index)]) for index in dominant_joint_col],
        dtype=object,
    )
    dominant_weight = np.max(weights, axis=1)

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    motion_path = (
        ctx["run_root"]
        / "inputs"
        / "motion"
        / "quaternius_knight_v1"
        / "demo_idle_v1.motion.json"
    )
    motion = json.loads(motion_path.read_text())
    tracks, mapping = _tracks_for_clip(motion, skeleton, cameras, source_report)
    skin_matrices, _joint_positions, frame_hash = _joint_pose_v2(
        skeleton=skeleton,
        tracks=tracks,
        time_seconds=0.0,
        cameras=cameras,
    )
    posed = _skin(rest, weights, joint_ids, skin_matrices)
    edge_ratio, area_ratio = _edge_area_ratios(rest, posed, faces)

    report = {
        "schema": "RealSaS.KnightCorrectedIdleT0VisualOwnerAudit.v1",
        "status": "MEASURED_AUDIT_ONLY__NO_PRODUCT_AUTHORITY",
        "run_id": args.run_id,
        "candidate_lineage_hash": str(candidate.candidate_lineage_hash),
        "appearance_asset_hash": str(appearance.asset_hash),
        "clip_id": "demo_idle_v1",
        "time_seconds": 0.0,
        "frame_hash": str(frame_hash),
        "rest_bbox_extent_xyz": (
            np.max(rest, axis=0) - np.min(rest, axis=0)
        ).tolist(),
        "posed_bbox_extent_xyz": (
            np.max(posed, axis=0) - np.min(posed, axis=0)
        ).tolist(),
        "max_vertex_displacement": float(
            np.max(np.linalg.norm(posed - rest, axis=1))
        ),
        "mean_vertex_displacement": float(
            np.mean(np.linalg.norm(posed - rest, axis=1))
        ),
        "retarget_mapping": mapping,
        "views": [],
        "claim_boundary": [
            "This audit attributes the first idle frame visual expansion after the corrected mechanical repair.",
            "It does not mutate product state or grant product authority.",
            "Per-face screen pixel counts use alpha-contributing renderer layers.",
            "3D edge ratios are symmetric rest/posed ratios and therefore distinguish screen-space exposure from intrinsic mechanical stretch.",
        ],
    }

    for view in (0, 2):
        texture = _load_texture_pages(texture_by_view[view])
        rest_render = render_caa_reference(
            mesh=candidate,
            camera=cameras[view],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance[view],
            face_page_index=face_page,
            positions=rest,
        )
        posed_render = render_caa_reference(
            mesh=candidate,
            camera=cameras[view],
            face_uv=face_uv,
            texture_rgba_u8=texture,
            provenance_atlas=provenance[view],
            face_page_index=face_page,
            positions=posed,
        )
        source_mask = np.asarray(source_masks[view], dtype=bool)
        rest_alpha = np.asarray(rest_render.final_alpha, dtype=bool)
        posed_alpha = np.asarray(posed_render.final_alpha, dtype=bool)
        if source_mask.shape != rest_alpha.shape or source_mask.shape != posed_alpha.shape:
            raise RuntimeError("IDLE_OWNER_SOURCE_RENDER_SHAPE_DRIFT")

        posed_extra = posed_alpha & ~source_mask
        rest_face_pixels = _face_pixel_counts(
            rest_render, rest_alpha, len(candidate.faces)
        )
        posed_face_pixels = _face_pixel_counts(
            posed_render, posed_alpha, len(candidate.faces)
        )
        extra_face_pixels = _face_pixel_counts(
            posed_render, posed_extra, len(candidate.faces)
        )

        ranked = np.argsort(extra_face_pixels)[::-1]
        top_faces = []
        for face_index in ranked[:150]:
            extra_count = int(extra_face_pixels[int(face_index)])
            if extra_count <= 0:
                break
            vertex_rows = faces[int(face_index)]
            joints = [str(dominant_joint_id[int(v)]) for v in vertex_rows]
            weights_row = [float(dominant_weight[int(v)]) for v in vertex_rows]
            rest_pixels = int(rest_face_pixels[int(face_index)])
            posed_pixels = int(posed_face_pixels[int(face_index)])
            top_faces.append(
                {
                    "face_index": int(face_index),
                    "extra_outside_source_pixel_count": extra_count,
                    "rest_contributing_pixel_count": rest_pixels,
                    "posed_contributing_pixel_count": posed_pixels,
                    "posed_to_rest_screen_pixel_ratio": (
                        None
                        if rest_pixels == 0
                        else float(posed_pixels / rest_pixels)
                    ),
                    "symmetric_3d_max_edge_ratio": float(
                        edge_ratio[int(face_index)]
                    ),
                    "three_d_area_ratio": float(area_ratio[int(face_index)]),
                    "dominant_joint_ids_by_vertex": joints,
                    "dominant_joint_weights_by_vertex": weights_row,
                    "all_vertices_same_dominant_joint": len(set(joints)) == 1,
                    "candidate_vertex_ids": [
                        str(candidate.faces[int(face_index)][corner])
                        for corner in range(3)
                    ],
                }
            )

        source_count = int(np.count_nonzero(source_mask))
        rest_count = int(np.count_nonzero(rest_alpha))
        posed_count = int(np.count_nonzero(posed_alpha))
        posed_overlap = int(np.count_nonzero(posed_alpha & source_mask))
        posed_extra_count = int(np.count_nonzero(posed_extra))
        report["views"].append(
            {
                "view_index": int(view),
                "source_foreground_pixel_count": source_count,
                "rest_final_alpha_pixel_count": rest_count,
                "posed_final_alpha_pixel_count": posed_count,
                "rest_to_source_area_ratio": float(rest_count / source_count),
                "posed_to_source_area_ratio": float(posed_count / source_count),
                "posed_to_rest_area_ratio": float(posed_count / rest_count),
                "posed_source_alpha_precision": float(
                    posed_overlap / posed_count
                ),
                "posed_source_alpha_recall": float(
                    posed_overlap / source_count
                ),
                "posed_extra_outside_source_pixel_count": posed_extra_count,
                "posed_extra_fraction_of_rendered_alpha": float(
                    posed_extra_count / posed_count
                ),
                "top_extra_contributing_faces": top_faces,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "KNIGHT_CORRECTED_IDLE_T0_VISUAL_OWNER_AUDIT="
        + json.dumps(
            {
                "max_vertex_displacement": report["max_vertex_displacement"],
                "views": [
                    {
                        "view": row["view_index"],
                        "source": row["source_foreground_pixel_count"],
                        "rest": row["rest_final_alpha_pixel_count"],
                        "posed": row["posed_final_alpha_pixel_count"],
                        "posed_to_rest": row["posed_to_rest_area_ratio"],
                        "precision": row["posed_source_alpha_precision"],
                        "extra": row["posed_extra_outside_source_pixel_count"],
                        "top_face": row["top_extra_contributing_faces"][0]
                        if row["top_extra_contributing_faces"]
                        else None,
                    }
                    for row in report["views"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
