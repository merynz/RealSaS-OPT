from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    bind_points_barycentric,
    build_visual_mesh_from_mask,
    evaluate_bone_handles,
    sample_bone_handles,
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


def _presentation_joint_positions(
    *,
    point_xy: np.ndarray,
    visual_skin_weights: np.ndarray,
    joint_ids: tuple[str, ...],
    parent_by_joint: dict[str, str | None],
    minimum_total_weight: float = 0.05,
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Derive source-aligned per-view 2D joint controls from Arachne evidence."""
    points = np.asarray(point_xy, dtype=np.float64)
    weights = np.asarray(visual_skin_weights, dtype=np.float64)
    if weights.shape != (len(points), len(joint_ids)):
        raise RuntimeError("PRESENTATION_JOINT_WEIGHT_SHAPE_DRIFT")

    result: dict[str, tuple[float, float]] = {}
    support: dict[str, float] = {}
    global_center = tuple(map(float, points.mean(axis=0)))
    for column, joint_id in enumerate(joint_ids):
        w = np.maximum(weights[:, column], 0.0)
        total = float(w.sum())
        support[joint_id] = total
        if total >= float(minimum_total_weight):
            xy = (points * w[:, None]).sum(axis=0) / total
            result[joint_id] = tuple(map(float, xy))

    # Unsupported/fully hidden controls inherit their nearest available parent
    # rather than forcing an off-art projected 3D joint into the 2D solver.
    unresolved = set(joint_ids) - set(result)
    for _ in range(len(joint_ids) + 1):
        if not unresolved:
            break
        progressed = False
        for joint_id in tuple(unresolved):
            parent = parent_by_joint.get(joint_id)
            if parent is None:
                result[joint_id] = global_center
                unresolved.remove(joint_id)
                progressed = True
            elif parent in result:
                result[joint_id] = result[parent]
                unresolved.remove(joint_id)
                progressed = True
        if not progressed:
            break
    for joint_id in unresolved:
        result[joint_id] = global_center
    return result, support


def _topological_joint_ids(
    joint_ids: tuple[str, ...],
    parent_by_joint: dict[str, str | None],
) -> tuple[str, ...]:
    remaining = set(joint_ids)
    order = []
    while remaining:
        progressed = False
        for joint_id in joint_ids:
            if joint_id not in remaining:
                continue
            parent = parent_by_joint.get(joint_id)
            if parent is None or parent in order:
                order.append(joint_id)
                remaining.remove(joint_id)
                progressed = True
        if not progressed:
            raise RuntimeError("PRESENTATION_SKELETON_NOT_TOPOLOGICAL")
    return tuple(order)


def _project_joint_dict(
    positions: dict[str, tuple[float, float, float]],
    camera,
) -> dict[str, np.ndarray]:
    ids = tuple(positions)
    xyz = np.asarray([positions[joint_id] for joint_id in ids], dtype=np.float64)
    xy = project_points_xyz_v3(xyz, camera)[:, :2]
    return {
        joint_id: np.asarray(xy[i], dtype=np.float64)
        for i, joint_id in enumerate(ids)
    }


def _retarget_presentation_joints(
    *,
    rest_presentation: dict[str, tuple[float, float]],
    rest_joint_xyz: dict[str, tuple[float, float, float]],
    posed_joint_xyz: dict[str, tuple[float, float, float]],
    camera,
    joint_ids: tuple[str, ...],
    parent_by_joint: dict[str, str | None],
) -> dict[str, tuple[float, float]]:
    """Retarget 3D motion deltas onto a source-fitted 2D rest rig.

    Absolute projected 3D joint positions never become presentation authority.
    For each bone we take only the view-projected angular delta from rest to
    pose, while preserving the source-fitted 2D bone vector/length. Root
    translation is transferred as a view-projected delta.
    """
    rest_projected = _project_joint_dict(rest_joint_xyz, camera)
    posed_projected = _project_joint_dict(posed_joint_xyz, camera)
    order = _topological_joint_ids(joint_ids, parent_by_joint)
    result: dict[str, tuple[float, float]] = {}

    for joint_id in order:
        parent = parent_by_joint.get(joint_id)
        rest_visual = np.asarray(rest_presentation[joint_id], dtype=np.float64)
        if parent is None:
            delta = posed_projected[joint_id] - rest_projected[joint_id]
            target = rest_visual + delta
            result[joint_id] = tuple(map(float, target))
            continue

        parent_target = np.asarray(result[parent], dtype=np.float64)
        parent_rest_visual = np.asarray(rest_presentation[parent], dtype=np.float64)
        visual_bone = rest_visual - parent_rest_visual
        visual_length = float(np.linalg.norm(visual_bone))
        if visual_length <= 1.0e-9:
            result[joint_id] = tuple(map(float, parent_target))
            continue

        rest_bone = rest_projected[joint_id] - rest_projected[parent]
        posed_bone = posed_projected[joint_id] - posed_projected[parent]
        rest_len = float(np.linalg.norm(rest_bone))
        posed_len = float(np.linalg.norm(posed_bone))
        if rest_len <= 1.0e-6 or posed_len <= 1.0e-6:
            delta_angle = 0.0
        else:
            rest_angle = float(np.arctan2(rest_bone[1], rest_bone[0]))
            posed_angle = float(np.arctan2(posed_bone[1], posed_bone[0]))
            delta_angle = float(
                np.arctan2(
                    np.sin(posed_angle - rest_angle),
                    np.cos(posed_angle - rest_angle),
                )
            )
        cosine = float(np.cos(delta_angle))
        sine = float(np.sin(delta_angle))
        rotated = np.asarray(
            (
                cosine * visual_bone[0] - sine * visual_bone[1],
                sine * visual_bone[0] + cosine * visual_bone[1],
            ),
            dtype=np.float64,
        )
        result[joint_id] = tuple(map(float, parent_target + rotated))
    return result


def _presentation_handle_specs(
    *,
    joint_ids: tuple[str, ...],
    parent_by_joint: dict[str, str | None],
    rest_joint_xy: dict[str, tuple[float, float]],
    joint_support: dict[str, float],
    step: float = 0.25,
    min_bone_px: float = 2.0,
    min_support: float = 0.05,
) -> tuple[tuple[str, str, float], ...]:
    rows = []
    root_added = False
    for joint_id in _topological_joint_ids(joint_ids, parent_by_joint):
        parent = parent_by_joint.get(joint_id)
        if parent is None:
            if float(joint_support.get(joint_id, 0.0)) >= min_support:
                rows.append((joint_id, joint_id, 1.0))
                root_added = True
            continue
        if float(joint_support.get(joint_id, 0.0)) < min_support:
            continue
        a = np.asarray(rest_joint_xy[parent], dtype=np.float64)
        b = np.asarray(rest_joint_xy[joint_id], dtype=np.float64)
        if float(np.linalg.norm(b - a)) < float(min_bone_px):
            continue
        if not root_added and float(joint_support.get(parent, 0.0)) >= min_support:
            rows.append((joint_id, parent, 0.0))
            root_added = True
        t = float(step)
        while t < 1.0 - 1.0e-9:
            rows.append((joint_id, parent, t))
            t += float(step)
        rows.append((joint_id, parent, 1.0))
    if not rows:
        raise RuntimeError("PRESENTATION_HANDLE_SET_EMPTY")
    return tuple(rows)


def _write_direct_provenance(path: Path, height: int, width: int) -> str:
    provenance = np.zeros((8, height, width), dtype=np.uint8)
    source_view = np.broadcast_to(
        np.arange(8, dtype=np.int16)[:, None, None],
        provenance.shape,
    ).copy()
    np.savez_compressed(path, provenance=provenance, source_view=source_view)
    return sha256_file(path)


def _render_native(player: Path, package: Path, view_id: str, clip_id: str, frame: int, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{view_id}_{clip_id}_{frame}"
    rgba = root / f"{stem}.rgba"
    cmd = [
        str(player), str(package),
        "--clip", clip_id,
        "--view", view_id,
        "--frame", str(int(frame)),
        "--out-rgba", str(rgba),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"VISUAL_WITNESS_NATIVE_RENDER_FAIL:{view_id}:{clip_id}:{frame}\n" + proc.stdout)
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
    parent_by_joint = {
        str(joint.canonical_joint_id): (
            None
            if joint.parent_canonical_id is None
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


    clip_specs = [
        ("demo_idle_v1", "IDLE", 833),
        ("demo_run_v1", "RUN", 208),
        ("demo_slash_v1", "SLASH", 278),
    ]
    clip_payloads = {}
    for clip_id, _short, _ms in clip_specs:
        motion_path = (
            ctx["run_root"] / "inputs" / "motion" / "quaternius_knight_v1" / f"{clip_id}.motion.json"
        )
        payload = json.loads(motion_path.read_text())
        tracks, mapping = _tracks_for_clip(payload, skeleton, cameras, source_report)
        clip_payloads[clip_id] = (payload, tracks, mapping)

    packages = {}
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
        rest_visible_xy = np.asarray(mesh.positions, dtype=np.float64)[
            mechanical_binding["visual_vertex_indices"]
        ]
        rest_presentation_joints, joint_support = _presentation_joint_positions(
            point_xy=rest_visible_xy,
            visual_skin_weights=mechanical_binding["visual_skin_weights"],
            joint_ids=tuple(mechanical_joint_ids),
            parent_by_joint=parent_by_joint,
        )
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
            handle_specs,
            rest_presentation_joints,
        )
        bindings = bind_points_barycentric(mesh, rest_handles)
        arap = Arap2D(mesh, bindings)
        supported_joint_count = sum(
            1 for value in joint_support.values() if float(value) >= 0.05
        )
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
                    "supported_presentation_joint_count": int(
                        supported_joint_count
                    ),
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
        provenance_path = view_root / "provenance.npz"
        provenance_sha = _write_direct_provenance(provenance_path, h, w)

        arrays = {
            "vertices": _pixel_to_world(mesh.positions, mesh.width, mesh.height),
            "faces": np.asarray(mesh.faces, dtype=np.uint32),
            "face_uv": np.asarray(mesh.uv[mesh.faces], dtype=np.float64),
        }
        runtime_clips = []
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
            posed_frames = []
            qa_rows = []
            for time_seconds in times:
                _skin_matrices, _posed_joint_positions, _frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras,
                )
                posed_presentation_joints = _retarget_presentation_joints(
                    rest_presentation=rest_presentation_joints,
                    rest_joint_xyz=rest_joint_positions,
                    posed_joint_xyz=_posed_joint_positions,
                    camera=camera,
                    joint_ids=tuple(mechanical_joint_ids),
                    parent_by_joint=parent_by_joint,
                )
                target_xy = evaluate_bone_handles(
                    handle_specs,
                    posed_presentation_joints,
                )
                deformed, qa = arap.solve(target_xy, iterations=3)
                posed_frames.append(_pixel_to_world(deformed, mesh.width, mesh.height))
                qa_rows.append({
                    "time_seconds": float(time_seconds),
                    "flipped_triangles": int(qa.flipped_triangles),
                    "max_handle_residual_px": float(qa.max_handle_residual_px),
                    "p95_edge_stretch": float(qa.p95_edge_stretch),
                    "max_edge_stretch": float(qa.max_edge_stretch),
                })
            prefix = clip_id.replace("demo_", "").replace("_v1", "")
            arrays[f"{prefix}_times"] = times
            arrays[f"{prefix}_positions"] = np.stack(posed_frames, axis=0)
            runtime_clips.append(
                RuntimeClipV2IR(
                    clip_id=clip_id,
                    duration_seconds=duration,
                    loop=bool(payload.get("loop")),
                    frame_count=4,
                    array_prefix=prefix,
                    metadata={"visual_mesh_arap_witness": True},
                )
            )
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

        projection_npz = view_root / "projection.npz"
        np.savez(projection_npz, **arrays)
        projection_sha = sha256_file(projection_npz)

        runtime_views = tuple(
            RuntimeViewV2IR(
                view_index=i,
                view_id=f"V{i}",
                camera=_synthetic_camera(i, render_resolution),
                texture_path=str(texture_path.resolve()),
                texture_sha256=texture_sha,
                metadata={"visual_mesh_source_view": int(view_index)},
            )
            for i in range(8)
        )
        projection = RuntimeProjectionV2IR(
            complete_puppet_binding_hash="visual-witness",
            mechanical_state_binding_hash="visual-witness",
            mesh_binding_hash=hashlib.sha256(
                np.asarray(mesh.positions, dtype="<f8").tobytes()
                + np.asarray(mesh.faces, dtype="<u4").tobytes()
            ).hexdigest(),
            dynamic_motion_binding_hash="visual-witness",
            appearance_asset_binding_hash=texture_sha,
            appearance_qualification_binding_hash="visual-witness",
            camera_set_binding_hash="visual-witness",
            visibility_contract_hash=VISIBILITY_CONTRACT_V2_HASH,
            projection_npz_path=str(projection_npz.resolve()),
            projection_npz_sha256=projection_sha,
            provenance_npz_path=str(provenance_path.resolve()),
            provenance_npz_sha256=provenance_sha,
            views=runtime_views,
            clips=tuple(runtime_clips),
            projection_hash="",
            metadata={
                "diagnostic_only": True,
                "source_owned_visual_mesh": True,
                "mechanical_relation_faces_rendered": False,
                "product_authority_claimed": False,
            },
        )
        projection = replace(projection, projection_hash=runtime_projection_hash(projection))
        package = view_root / "visual_mesh_arap_witness.rss"
        meta = write_rss_v2(package, build_rss_v2_entries(projection))
        packages[view_index] = (package, runtime_clips)
        report_views.append({
            "view_index": int(view_index),
            "visual_vertex_count": int(len(mesh.positions)),
            "visual_face_count": int(len(mesh.faces)),
            "handle_count": int(len(bindings)),
            "source_foreground_pixel_count": int(np.count_nonzero(mask)),
            "rss_sha256": str(meta["archive_sha256"]),
            "qa_by_clip": qa_by_clip,
        })

    render_root = out_dir / "native_frames"
    jobs = []
    for view_index, (package, clips) in packages.items():
        for clip in clips:
            for frame in range(clip.frame_count):
                jobs.append((view_index, package, clip.clip_id, frame))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(
            lambda row: (
                row[0],
                row[2],
                row[3],
                _render_native(
                    native_player,
                    row[1],
                    f"V{row[0]}",
                    row[2],
                    row[3],
                    render_root,
                ),
            ),
            jobs,
        ))

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
        "ownership_contract": {
            "mechanical_mesh": "NOT_RENDERED",
            "visual_mesh": "SOURCE_FOREGROUND_MASK_DERIVED",
            "texture": "ORIGINAL_SOURCE_RGBA_WITH_FOREGROUND_ALPHA",
            "deformation_driver": "STAGE32_ARACHNE_FITS_SOURCE_2D_REST_RIG__STAGE28_VIEW_PROJECTED_ROTATION_DELTAS_RETARGETED_WITH_2D_BONE_LENGTH_PRESERVATION",
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
        "renderer": "CXX_RUNTIME_V2",
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
