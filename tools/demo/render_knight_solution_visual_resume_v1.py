from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import (
    complete_appearance_asset_from_dict,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_points_barycentric,
    evaluate_bone_handles,
    load_visual_mesh_view,
    visual_mesh_set_from_dict,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    resolved_path,
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _tracks_for_clip,
)
from tools.demo.render_knight_visual_mesh_arap_witness_v1 import (
    _candidate_face_indices,
    _bind_visual_to_mechanical_surface,
    _presentation_joint_positions,
    _project_joint_dict,
    _fit_projected_template_to_visual_bbox,
    _presentation_handle_specs,
    _retarget_template_direction_character_length,
    _write_visual_mesh_binary,
    _write_visual_positions_binary,
    _render_native,
    _gif_frame,
)


CLIPS = (
    ("demo_idle_v1", "IDLE", 833),
    ("demo_run_v1", "RUN", 208),
    ("demo_slash_v1", "SLASH", 278),
)


def run(*, authority_root: Path, run_id: str, out_dir: Path, native_player: Path, render_resolution: int):
    t0 = perf_counter()
    ctx = _ctx(authority_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    visual_set = visual_mesh_set_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.VisualMeshSetIR.v1",
        )
    )
    appearance = complete_appearance_asset_from_dict(
        stage_output_payload(
            ctx,
            "23_COMPLETE_APPEARANCE_ASSET_BAKED",
            "RealSaS.CompleteAppearanceAssetIR.v2",
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
    observation = qualified_observation_set_from_dict(
        stage_output_payload(
            ctx,
            "07_OBSERVATION_CONTRACT_QUALIFIED",
            "RealSaS.QualifiedObservationSetIR.v1",
        )
    )

    appearance_meta = dict(appearance.metadata or {})
    if appearance_meta.get("source_owned_visual_mesh_mode") is not True:
        raise RuntimeError("RENDER_RESUME_APPEARANCE_MODE_DRIFT")
    if str(appearance_meta.get("visual_mesh_set_binding_hash") or "") != visual_set.set_hash:
        raise RuntimeError("RENDER_RESUME_VISUAL_SET_BINDING_DRIFT")
    if str(appearance.candidate_mesh_binding_hash) != candidate.candidate_lineage_hash:
        raise RuntimeError("RENDER_RESUME_CANDIDATE_BINDING_DRIFT")
    if str(visual_set.observation_set_binding_hash) != observation.observation_set_hash:
        raise RuntimeError("RENDER_RESUME_OBSERVATION_BINDING_DRIFT")

    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    _rgba_by_view, mask_by_view = _load_source_inputs(ctx, observation)
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    texture_by_view = {int(row.direction_index): row for row in appearance.textures}
    visual_by_view = {int(row.view_index): row for row in visual_set.views}
    if set(texture_by_view) != set(range(8)) or set(visual_by_view) != set(range(8)):
        raise RuntimeError("RENDER_RESUME_VIEW_SET_DRIFT")

    mechanical_joint_ids, mechanical_weights = _candidate_skin_weights(
        candidate, skin, skeleton
    )
    rest_mechanical_xyz = np.asarray(
        [vertex.P for vertex in candidate.vertices], dtype=np.float64
    )
    mechanical_face_indices = _candidate_face_indices(candidate)
    parent_by_joint = {
        str(joint.canonical_joint_id): (
            None if joint.parent_canonical_id is None
            else str(joint.parent_canonical_id)
        )
        for joint in skeleton.joints
    }
    _rest_skin_matrices, rest_joint_positions, _rest_frame_hash = _joint_pose_v2(
        skeleton=skeleton,
        tracks={},
        time_seconds=0.0,
        cameras=cameras,
    )

    clip_payloads = {}
    for clip_id, _short, _ms in CLIPS:
        payload = json.loads(
            (
                ctx["run_root"]
                / "inputs"
                / "motion"
                / "quaternius_knight_v1"
                / f"{clip_id}.motion.json"
            ).read_text()
        )
        tracks, mapping = _tracks_for_clip(
            payload, skeleton, cameras, source_report
        )
        clip_payloads[clip_id] = (payload, tracks, mapping)

    render_inputs = {}
    report_views = []
    for view_index in (0, 2):
        t_view = perf_counter()
        camera = cameras[view_index]
        mask = np.asarray(mask_by_view[view_index], dtype=bool)

        visual_row = visual_by_view[view_index]
        mesh = load_visual_mesh_view(visual_row)
        if int(mesh.width) != int(mask.shape[1]) or int(mesh.height) != int(mask.shape[0]):
            raise RuntimeError("RENDER_RESUME_VISUAL_DIMENSION_DRIFT")

        texture_row = texture_by_view[view_index]
        texture_path = resolved_path(texture_row.transport_png_path)
        if (
            not texture_path.is_file()
            or sha256_file(texture_path) != texture_row.transport_png_sha256
        ):
            raise RuntimeError("RENDER_RESUME_TEXTURE_BYTES_DRIFT")
        if (
            str(dict(texture_row.metadata or {}).get("visual_mesh_set_binding_hash") or "")
            != visual_set.set_hash
        ):
            raise RuntimeError("RENDER_RESUME_TEXTURE_VISUAL_BINDING_DRIFT")

        mechanical_binding = _bind_visual_to_mechanical_surface(
            visual_mesh=mesh,
            candidate=candidate,
            camera=camera,
            rest_mechanical_xyz=rest_mechanical_xyz,
            mechanical_face_indices=mechanical_face_indices,
            mechanical_weights=mechanical_weights,
            source_mask=mask,
        )
        rest_visible_xy = np.asarray(mesh.positions, dtype=np.float64)[
            mechanical_binding["visual_vertex_indices"]
        ]
        rest_presentation_joints, joint_support = _presentation_joint_positions(
            point_xy=rest_visible_xy,
            visual_skin_weights=mechanical_binding["visual_skin_weights"],
            joint_ids=tuple(mechanical_joint_ids),
            parent_by_joint=parent_by_joint,
        )

        # The fitted joints are the existing source-owned visual presentation
        # contract; exact Stage18 visual geometry and Stage23 texture remain fixed.
        handle_specs = _presentation_handle_specs(
            joint_ids=tuple(mechanical_joint_ids),
            parent_by_joint=parent_by_joint,
            rest_joint_xy=rest_presentation_joints,
            joint_support=joint_support,
            step=0.25,
            min_bone_px=2.0,
            min_support=0.05,
        )
        rest_handles = evaluate_bone_handles(
            handle_specs, rest_presentation_joints
        )
        bindings = bind_points_barycentric(mesh, rest_handles)
        arap = Arap2D(mesh, bindings)

        view_root = out_dir / f"V{view_index}"
        mesh_binary = view_root / "visual_mesh.bin"
        _write_visual_mesh_binary(mesh_binary, mesh)

        clip_position_files = {}
        clip_frame_counts = {}
        qa_by_clip = {}
        for clip_id, _short, _ms in CLIPS:
            payload, tracks, _mapping = clip_payloads[clip_id]
            duration = float(payload["duration_seconds"])
            times = np.linspace(
                0.0,
                duration,
                4,
                endpoint=not bool(payload.get("loop")),
                dtype=np.float64,
            )
            arap.reset()
            posed_pixel_frames = []
            qa_rows = []
            for time_seconds in times:
                _skin_matrices, posed_joint_positions, _frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras,
                )
                posed_presentation_joints = (
                    _retarget_template_direction_character_length(
                        rest_presentation=rest_presentation_joints,
                        rest_joint_xyz=rest_joint_positions,
                        posed_joint_xyz=posed_joint_positions,
                        camera=camera,
                        joint_ids=tuple(mechanical_joint_ids),
                        parent_by_joint=parent_by_joint,
                    )
                )
                target_xy = evaluate_bone_handles(
                    handle_specs, posed_presentation_joints
                )
                deformed, qa = arap.solve(target_xy, iterations=3)
                posed_pixel_frames.append(
                    np.asarray(deformed, dtype=np.float64)
                )
                qa_rows.append({
                    "time_seconds": float(time_seconds),
                    "flipped_triangles": int(qa.flipped_triangles),
                    "max_handle_residual_px": float(qa.max_handle_residual_px),
                    "p95_edge_stretch": float(qa.p95_edge_stretch),
                    "max_edge_stretch": float(qa.max_edge_stretch),
                })
            positions_path = view_root / f"{clip_id}.positions.bin"
            _write_visual_positions_binary(
                positions_path,
                np.stack(posed_pixel_frames, axis=0),
            )
            clip_position_files[clip_id] = positions_path
            clip_frame_counts[clip_id] = len(posed_pixel_frames)
            qa_by_clip[clip_id] = qa_rows

        render_inputs[view_index] = {
            "mesh_path": mesh_binary,
            "texture_path": texture_path,
            "clip_position_files": clip_position_files,
            "clip_frame_counts": clip_frame_counts,
        }
        report_views.append({
            "view_index": view_index,
            "visual_mesh_hash": visual_row.mesh_hash,
            "visual_mesh_npz_sha256": visual_row.mesh_npz_sha256,
            "texture_sha256": texture_row.transport_png_sha256,
            "visual_vertex_count": len(mesh.positions),
            "visual_face_count": len(mesh.faces),
            "valid_visual_vertex_count": mechanical_binding[
                "valid_visual_vertex_count"
            ],
            "unbound_visual_vertex_count": mechanical_binding[
                "unbound_visual_vertex_count"
            ],
            "qa_by_clip": qa_by_clip,
            "seconds": float(perf_counter() - t_view),
        })
        print(
            "RENDER_RESUME_VIEW_DONE="
            + json.dumps(
                {
                    "view_index": view_index,
                    "seconds": report_views[-1]["seconds"],
                    "visual_mesh_hash": visual_row.mesh_hash,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    render_root = out_dir / "native_frames"
    jobs = []
    for view_index, row in render_inputs.items():
        for clip_id, frame_count in row["clip_frame_counts"].items():
            for frame in range(frame_count):
                jobs.append((
                    view_index,
                    clip_id,
                    frame,
                    row["mesh_path"],
                    row["clip_position_files"][clip_id],
                    row["texture_path"],
                ))

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(
                lambda row: (
                    row[0], row[1], row[2],
                    _render_native(
                        native_player,
                        mesh_path=row[3],
                        positions_path=row[4],
                        texture_path=row[5],
                        clip_id=row[1],
                        view_index=row[0],
                        frame=row[2],
                        root=render_root,
                        resolution=render_resolution,
                    ),
                ),
                jobs,
            )
        )

    frames = {}
    for view_index, clip_id, frame, path in paths:
        raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        expected = render_resolution * render_resolution * 4
        if raw.size != expected:
            raise RuntimeError("RENDER_RESUME_RGBA_SIZE_DRIFT")
        frames[(view_index, clip_id, frame)] = Image.fromarray(
            raw.reshape(render_resolution, render_resolution, 4),
            mode="RGBA",
        )

    outputs = []
    for clip_id, short, duration_ms in CLIPS:
        gif_frames = [
            _gif_frame(
                frames[(0, clip_id, frame)],
                frames[(2, clip_id, frame)],
            )
            for frame in range(4)
        ]
        gif_path = out_dir / f"KNIGHT_{short}_SOLVED_EXACT_LINEAGE.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
            transparency=0,
        )
        outputs.append({
            "clip_id": clip_id,
            "path": str(gif_path),
            "sha256": sha256_file(gif_path),
        })

    report = {
        "schema": "RealSaS.KnightSolvedExactLineageRenderResume.v1",
        "status": "PASS_RENDER_ONLY_RESUME",
        "run_id": run_id,
        "candidate_lineage_hash": candidate.candidate_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "visual_mesh_set_hash": visual_set.set_hash,
        "appearance_asset_hash": appearance.asset_hash,
        "appearance_candidate_binding_hash": appearance.candidate_mesh_binding_hash,
        "views": report_views,
        "outputs": outputs,
        "render_only_resume": True,
        "stage17_25_recomputed": False,
        "source_owned_visual_mesh_rebuilt": False,
        "source_texture_rebuilt": False,
        "mechanical_candidate_rendered_directly": False,
        "renderer": "CXX_VISUAL_MESH_V1__ARAP_2D",
        "elapsed_seconds": float(perf_counter() - t0),
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        "KNIGHT_RENDER_ONLY_RESUME_PASS="
        + json.dumps({
            "elapsed_seconds": report["elapsed_seconds"],
            "candidate": report["candidate_lineage_hash"],
            "skin": report["skin_lineage_hash"],
            "visual_mesh_set": report["visual_mesh_set_hash"],
            "appearance": report["appearance_asset_hash"],
            "gifs": {
                row["clip_id"]: row["sha256"] for row in outputs
            },
        }, sort_keys=True),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--native-player", required=True)
    p.add_argument("--render-resolution", type=int, default=512)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir).expanduser().resolve(),
        native_player=Path(a.native_player).expanduser().resolve(),
        render_resolution=int(a.render_resolution),
    )


if __name__ == "__main__":
    main()
