from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_camera_set_from_dict,
    qualified_observation_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.camera_geometry_v2 import project_points_xyz_v3
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_core.runtime_authority_v2 import (
    RuntimeClipV2IR,
    RuntimeProjectionV2IR,
    RuntimeViewV2IR,
    runtime_projection_hash,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_rss_v2_entries,
    write_rss_v2,
)
from compiler.realsas_compiler_core.visibility_v2 import (
    VISIBILITY_CONTRACT_V2_HASH,
    rasterize_visible_owner,
)
from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_visual_vertex_handles,
    build_visual_mesh_from_mask,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file,
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_source_inputs,
)
from tools.demo.render_knight_motion_preview_v1 import (
    _candidate_skin_weights,
    _ctx,
    _skin,
    _tracks_for_clip,
)


def _pixel_to_world(xy: np.ndarray, width: int, height: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if width != height:
        raise RuntimeError("VISUAL_WITNESS_CURRENTLY_REQUIRES_SQUARE_SOURCE")
    out = np.zeros((len(xy), 3), dtype=np.float64)
    out[:, 0] = 2.0 * xy[:, 0] / max(1.0, float(width - 1)) - 1.0
    out[:, 1] = -(2.0 * xy[:, 1] / max(1.0, float(height - 1)) - 1.0)
    return out


def _clean_source_texture(rgba: np.ndarray, mask: np.ndarray, path: Path) -> str:
    value = np.asarray(rgba, dtype=np.uint8).copy()
    m = np.asarray(mask, dtype=bool)
    value[~m, :3] = 0
    value[..., 3] = np.where(m, value[..., 3], 0).astype(np.uint8)
    # Some source sheets are opaque-background RGB; the qualified foreground mask
    # is the presentation silhouette authority in that case.
    inside = m & (value[..., 3] == 0)
    value[inside, 3] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGBA").save(path, format="PNG", compress_level=1)
    return sha256_file(path)


def _synthetic_camera(view_index: int, resolution: int) -> dict:
    return {
        "view_index": int(view_index),
        "origin": [0.0, 0.0, -2.0],
        "right": [1.0, 0.0, 0.0],
        "screen_up": [0.0, 1.0, 0.0],
        "forward": [0.0, 0.0, 1.0],
        "half_extent": 1.0,
        "resolution": int(resolution),
    }


def _candidate_face_indices(candidate) -> np.ndarray:
    index = {
        str(vertex.candidate_vertex_id): i
        for i, vertex in enumerate(candidate.vertices)
    }
    faces = np.asarray(
        [
            [index[str(vertex_id)] for vertex_id in face]
            for face in candidate.faces
        ],
        dtype=np.int64,
    )
    if faces.shape != (len(candidate.faces), 3):
        raise RuntimeError("VISUAL_MECHANICAL_FACE_INDEX_DRIFT")
    return faces


def _bind_visual_to_mechanical_surface(
    *,
    visual_mesh,
    candidate,
    camera,
    rest_mechanical_xyz: np.ndarray,
    mechanical_face_indices: np.ndarray,
    mechanical_weights: np.ndarray,
    source_mask: np.ndarray,
):
    """Bind source-owned visual vertices to first-hit mechanical surface.

    The mechanical relation mesh never becomes render authority. It contributes
    only source-view displacement and Arachne skin influence evidence.
    """
    visibility = rasterize_visible_owner(
        candidate,
        camera,
        positions=rest_mechanical_xyz,
        width=int(source_mask.shape[1]),
        height=int(source_mask.shape[0]),
        max_layers=4,
    )
    positions = np.asarray(visual_mesh.positions, dtype=np.float64)
    h, w = source_mask.shape
    x = np.clip(np.rint(positions[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.rint(positions[:, 1]).astype(np.int64), 0, h - 1)
    owner = np.asarray(visibility.owner_face_index[y, x], dtype=np.int64)
    bary = np.asarray(visibility.barycentric[y, x], dtype=np.float64)
    valid = (
        np.asarray(source_mask[y, x], dtype=bool)
        & (owner >= 0)
        & np.isfinite(bary).all(axis=1)
        & (bary.min(axis=1) >= -1.0e-4)
    )
    selected = np.asarray(np.flatnonzero(valid), dtype=np.int64)
    if len(selected) < 8:
        raise RuntimeError("VISUAL_MECHANICAL_BINDING_COVERAGE_TOO_LOW")

    selected_owner = owner[selected]
    selected_bary = bary[selected]
    selected_faces = mechanical_face_indices[selected_owner]
    projected_rest = np.asarray(visibility.projected_vertices, dtype=np.float64)
    rest_bound_xy = np.sum(
        projected_rest[selected_faces, :2] * selected_bary[:, :, None],
        axis=1,
    )
    rest_visual_xy = positions[selected]
    rest_offset = rest_visual_xy - rest_bound_xy

    point_weights = np.sum(
        np.asarray(mechanical_weights[selected_faces], dtype=np.float64)
        * selected_bary[:, :, None],
        axis=1,
    )
    point_weights = np.maximum(point_weights, 0.0)
    row_sum = point_weights.sum(axis=1, keepdims=True)
    good = row_sum[:, 0] > 1.0e-12
    point_weights[good] /= row_sum[good]

    return {
        "visual_vertex_indices": selected,
        "mechanical_face_indices": selected_faces,
        "mechanical_barycentric": selected_bary,
        "rest_projection_offset_xy": rest_offset,
        "visual_skin_weights": point_weights,
        "valid_visual_vertex_count": int(len(selected)),
        "unbound_visual_vertex_count": int(len(positions) - len(selected)),
        "visibility_layer_overflow_pixel_count": int(
            np.count_nonzero(visibility.layer_overflow)
        ),
    }


def _mechanical_targets(
    *,
    binding: dict,
    posed_mechanical_xyz: np.ndarray,
    camera,
) -> np.ndarray:
    projected = project_points_xyz_v3(posed_mechanical_xyz, camera)[:, :2]
    faces = np.asarray(binding["mechanical_face_indices"], dtype=np.int64)
    bary = np.asarray(binding["mechanical_barycentric"], dtype=np.float64)
    posed_bound_xy = np.sum(
        projected[faces] * bary[:, :, None],
        axis=1,
    )
    return posed_bound_xy + np.asarray(
        binding["rest_projection_offset_xy"],
        dtype=np.float64,
    )


def _write_direct_provenance(path: Path, height: int, width: int) -> str:
    provenance = np.zeros((8, height, width), dtype=np.uint8)
    source_view = np.broadcast_to(
        np.arange(8, dtype=np.int16)[:, None, None],
        provenance.shape,
    ).copy()
    np.savez_compressed(path, provenance=provenance, source_view=source_view)
    return sha256_file(path)


def _write_visual_mesh_binary(path: Path, mesh) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uv = np.asarray(mesh.uv, dtype="<f8")
    faces = np.asarray(mesh.faces, dtype="<u4")
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<8sIIII",
                b"RSVM1\\0\\0\\0",
                int(mesh.width),
                int(mesh.height),
                int(len(mesh.positions)),
                int(len(mesh.faces)),
            )
        )
        handle.write(uv.tobytes(order="C"))
        handle.write(faces.tobytes(order="C"))


def _write_visual_positions_binary(path: Path, frames_xy: np.ndarray) -> None:
    frames = np.asarray(frames_xy, dtype="<f8")
    if frames.ndim != 3 or frames.shape[2] != 2:
        raise RuntimeError("VISUAL_POSITIONS_SHAPE_INVALID")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(
            struct.pack(
                "<8sII",
                b"RSVP1\\0\\0\\0",
                int(frames.shape[0]),
                int(frames.shape[1]),
            )
        )
        handle.write(frames.tobytes(order="C"))


def _render_native(
    player: Path,
    *,
    mesh_path: Path,
    positions_path: Path,
    texture_path: Path,
    clip_id: str,
    view_index: int,
    frame: int,
    root: Path,
    resolution: int,
):
    root.mkdir(parents=True, exist_ok=True)
    stem = f"V{view_index}_{clip_id}_{frame}"
    rgba = root / f"{stem}.rgba"
    cmd = [
        str(player),
        "--mesh", str(mesh_path),
        "--positions", str(positions_path),
        "--texture", str(texture_path),
        "--frame", str(int(frame)),
        "--width", str(int(resolution)),
        "--height", str(int(resolution)),
        "--out-rgba", str(rgba),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"VISUAL_WITNESS_NATIVE_RENDER_FAIL:V{view_index}:{clip_id}:{frame}\n"
            + proc.stdout
        )
    return rgba


def _gif_frame(left: Image.Image, right: Image.Image, target_height: int = 512):
    canvas = Image.new("RGBA", (left.width + right.width, max(left.height, right.height)))
    canvas.alpha_composite(left, (0, 0))
    canvas.alpha_composite(right, (left.width, 0))
    if canvas.height != target_height:
        w = max(1, round(canvas.width * target_height / canvas.height))
        canvas = canvas.resize((w, target_height), Image.Resampling.LANCZOS)
    return canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=255)


def run(*, authority_root: Path, run_id: str, out_dir: Path, native_player: Path, render_resolution: int):
    ctx = _ctx(authority_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx,
            "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(ctx, "28_SKELETON_QUALIFIED", "RealSaS.QualifiedSkeletonIR.v1")
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(ctx, "32_SKIN_QUALIFIED", "RealSaS.QualifiedSkinIR.v1")
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(ctx, "05_CAMERA_CONTRACT_SOLVED", "RealSaS.QualifiedCameraSetIR.v1")
    )
    observation = qualified_observation_set_from_dict(
        stage_output_payload(ctx, "07_OBSERVATION_CONTRACT_QUALIFIED", "RealSaS.QualifiedObservationSetIR.v1")
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
    rgba_by_view, mask_by_view = _load_source_inputs(ctx, observation)
    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )

    mechanical_joint_ids, mechanical_weights = _candidate_skin_weights(
        candidate,
        skin,
        skeleton,
    )
    rest_mechanical_xyz = np.asarray(
        [vertex.P for vertex in candidate.vertices],
        dtype=np.float64,
    )
    mechanical_face_indices = _candidate_face_indices(candidate)

    clip_specs = [
        ("demo_idle_v1", "IDLE", 833),
        ("demo_run_v1", "RUN", 208),
        ("demo_slash_v1", "SLASH", 278),
    ]
    clip_payloads = {}
    for clip_id, _short, _ms in clip_specs:
        motion_path = (
            ctx["run_root"]
            / "inputs"
            / "motion"
            / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
        tracks, mapping = _tracks_for_clip(
            payload,
            skeleton,
            cameras,
            source_report,
        )
        clip_payloads[clip_id] = (payload, tracks, mapping)

    render_inputs = {}
    report_views = []
    for view_index in (0, 2):
        camera = cameras[view_index]
        rgba = np.asarray(rgba_by_view[view_index], dtype=np.uint8)
        mask = np.asarray(mask_by_view[view_index], dtype=bool)
        h, w = mask.shape

        mesh = build_visual_mesh_from_mask(mask, target_edge_px=20)
        mechanical_binding = _bind_visual_to_mechanical_surface(
            visual_mesh=mesh,
            candidate=candidate,
            camera=camera,
            rest_mechanical_xyz=rest_mechanical_xyz,
            mechanical_face_indices=mechanical_face_indices,
            mechanical_weights=mechanical_weights,
            source_mask=mask,
        )
        visual_handle_indices = np.asarray(
            mechanical_binding["visual_vertex_indices"],
            dtype=np.int64,
        )
        bindings = bind_visual_vertex_handles(mesh, visual_handle_indices)
        arap = Arap2D(mesh, bindings)
        print(
            "VISUAL_MECHANICAL_BINDING",
            json.dumps(
                {
                    "view_index": int(view_index),
                    "visual_vertex_count": int(len(mesh.positions)),
                    "valid_visual_vertex_count": int(
                        mechanical_binding["valid_visual_vertex_count"]
                    ),
                    "unbound_visual_vertex_count": int(
                        mechanical_binding["unbound_visual_vertex_count"]
                    ),
                    "presentation_handle_count": int(len(bindings)),
                    "presentation_driver": "DIRECT_POSED_MECHANICAL_SURFACE_TARGETS",
                    "rest_mechanical_visibility_layer_overflow_pixel_count": int(
                        mechanical_binding["visibility_layer_overflow_pixel_count"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )


        view_root = out_dir / f"V{view_index}"
        texture_path = view_root / "source_art.png"
        texture_sha = _clean_source_texture(rgba, mask, texture_path)
        mesh_binary = view_root / "visual_mesh.bin"
        _write_visual_mesh_binary(mesh_binary, mesh)
        clip_position_files = {}
        clip_frame_counts = {}
        qa_by_clip = {}
        for clip_id, _short, _ms in clip_specs:
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
                skin_matrices, _posed_joint_positions, _frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras,
                )
                posed_mechanical_xyz = _skin(
                    rest_mechanical_xyz,
                    mechanical_weights,
                    mechanical_joint_ids,
                    skin_matrices,
                )
                target_xy = _mechanical_targets(
                    binding=mechanical_binding,
                    posed_mechanical_xyz=posed_mechanical_xyz,
                    camera=camera,
                )
                deformed, qa = arap.solve(target_xy, iterations=3)
                posed_pixel_frames.append(np.asarray(deformed, dtype=np.float64))
                qa_rows.append({
                    "time_seconds": float(time_seconds),
                    "flipped_triangles": int(qa.flipped_triangles),
                    "max_handle_residual_px": float(qa.max_handle_residual_px),
                    "p95_edge_stretch": float(qa.p95_edge_stretch),
                    "max_edge_stretch": float(qa.max_edge_stretch),
                })
            positions_path = view_root / f"{clip_id}.positions.bin"
            posed_pixel_array = np.stack(posed_pixel_frames, axis=0)
            _write_visual_positions_binary(positions_path, posed_pixel_array)
            clip_position_files[clip_id] = positions_path
            clip_frame_counts[clip_id] = int(len(posed_pixel_frames))
            qa_by_clip[clip_id] = qa_rows
            print(
                "VISUAL_ARAP_QA",
                json.dumps(
                    {
                        "view_index": int(view_index),
                        "clip_id": clip_id,
                        "max_flipped_triangles": max(row["flipped_triangles"] for row in qa_rows),
                        "max_handle_residual_px": max(row["max_handle_residual_px"] for row in qa_rows),
                        "max_p95_edge_stretch": max(row["p95_edge_stretch"] for row in qa_rows),
                        "max_edge_stretch": max(row["max_edge_stretch"] for row in qa_rows),
                        "visual_vertex_count": int(len(mesh.positions)),
                        "visual_face_count": int(len(mesh.faces)),
                        "handle_count": int(len(bindings)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        render_inputs[view_index] = {
            "mesh_path": mesh_binary,
            "texture_path": texture_path,
            "clip_position_files": dict(clip_position_files),
            "clip_frame_counts": dict(clip_frame_counts),
        }
        report_views.append({
            "view_index": int(view_index),
            "visual_vertex_count": int(len(mesh.positions)),
            "visual_face_count": int(len(mesh.faces)),
            "handle_count": int(len(bindings)),
            "source_foreground_pixel_count": int(np.count_nonzero(mask)),
            "qa_by_clip": qa_by_clip,
        })

    render_root = out_dir / "native_frames"
    jobs = []
    for view_index, row in render_inputs.items():
        for clip_id, frame_count in row["clip_frame_counts"].items():
            for frame in range(int(frame_count)):
                jobs.append(
                    (
                        int(view_index),
                        clip_id,
                        int(frame),
                        row["mesh_path"],
                        row["clip_position_files"][clip_id],
                        row["texture_path"],
                    )
                )
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(
            pool.map(
                lambda row: (
                    row[0],
                    row[1],
                    row[2],
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
            raise RuntimeError("VISUAL_WITNESS_RGBA_SIZE_DRIFT")
        frames[(view_index, clip_id, frame)] = Image.fromarray(
            raw.reshape(render_resolution, render_resolution, 4),
            mode="RGBA",
        )

    outputs = []
    for clip_id, short, duration_ms in clip_specs:
        gif_frames = [
            _gif_frame(
                frames[(0, clip_id, frame)],
                frames[(2, clip_id, frame)],
            )
            for frame in range(4)
        ]
        gif_path = out_dir / f"KNIGHT_{short}_VISUAL_MESH_ARAP_WITNESS_V1.gif"
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=int(duration_ms),
            loop=0,
            disposal=2,
            optimize=False,
            transparency=0,
        )
        outputs.append(str(gif_path))

    report = {
        "schema": "RealSaS.KnightVisualMeshArapWitness.v1",
        "status": "MEASURED_DEMO_ONLY",
        "run_id": run_id,
        "renderer": "CXX_VISUAL_MESH_V1",
        "ownership_contract": {
            "mechanical_mesh": "NOT_RENDERED",
            "visual_mesh": "SOURCE_FOREGROUND_MASK_DERIVED",
            "texture": "ORIGINAL_SOURCE_RGBA_WITH_FOREGROUND_ALPHA",
            "deformation_driver": "STAGE18_MECHANICAL_FIRST_HIT_BINDING__STAGE32_ARACHNE_SKIN__STAGE40_POSED_MECHANICAL_SURFACE_PROJECTED_TO_SOURCE_VIEW",
            "deformer": "LOCAL_GLOBAL_ARAP_2D",
        },
        "views": report_views,
        "outputs": outputs,
        "product_authority_claimed": False,
    }
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("KNIGHT_VISUAL_MESH_ARAP_WITNESS_PASS", json.dumps({
        "gif_count": len(outputs),
        "views": [0, 2],
        "renderer": "CXX_VISUAL_MESH_V1",
    }, sort_keys=True))


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
        out_dir=Path(a.out_dir),
        native_player=Path(a.native_player).expanduser().resolve(),
        render_resolution=int(a.render_resolution),
    )


if __name__ == "__main__":
    main()
