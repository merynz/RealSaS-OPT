from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

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
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from compiler.realsas_compiler_core.joint_frames_v1 import (
    derive_joint_frames_from_rows,
    derive_joint_frames_from_skeleton,
    object_vector_to_joint_local,
)
from compiler.realsas_compiler_core.motion_compile_v2 import (
    CanonicalJointTrack3DIR,
    MotionKeyframe3DIR,
    _target_body_scale,
    _tree,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import (
    _joint_pose_v2,
    _quat_matrix_xyzw,
    _slerp,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    _load_texture_pages,
)


def _ctx(authority_root: Path, run_id: str) -> dict:
    root = (authority_root / "runs" / run_id).resolve()
    return {
        "repo_root": Path(".").resolve(),
        "authority_root": authority_root.resolve(),
        "run_root": root,
        "run_id": run_id,
        "run_manifest_path": root / "run_manifest.json",
        "run_manifest": json.loads((root / "run_manifest.json").read_text()),
        "ledger": json.loads((root / "ACTIVE_RUN_V2.json").read_text()),
        "stage": {"id": "DEMO_MOTION_VISUAL_CONTINUATION"},
    }


def _matrix_to_quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    q = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    if q[3] < 0.0:
        q = -q
    return tuple(map(float, q))


def _mapping(payload: dict, skeleton, source_report: dict):
    srows = tuple(payload.get("source_skeleton") or ())
    sids, spar, schildren, spos, sfeat, sroot = _tree(
        srows,
        id_key="source_joint_id",
        parent_key="parent_source_joint_id",
        pos_key="rest_position",
    )
    roles = {
        str(row["source_joint_id"]): row
        for row in source_report.get("bone_roles") or ()
    }
    candidates = [
        sid for sid in sids
        if sid == sroot or bool((roles.get(sid) or {}).get("has_mesh_influence"))
    ]

    trows = tuple({
        "joint_id": j.canonical_joint_id,
        "parent_id": j.parent_canonical_id,
        "position": j.position,
    } for j in skeleton.joints)
    tids, tpar, tchildren, tpos, tfeat, troot = _tree(
        trows,
        id_key="joint_id",
        parent_key="parent_id",
        pos_key="position",
    )

    mapping = {troot: sroot}
    details = []
    for tid in tids:
        if tid == troot:
            continue
        tf = tfeat[tid]
        ranked = []
        for sid in candidates:
            if sid == sroot:
                continue
            sf = sfeat[sid]
            c = 2.0 * float(np.linalg.norm(tf[:3] - sf[:3]))
            c += 0.45 * abs(float(tf[3] - sf[3]))
            c += 0.35 * abs(float(tf[4] - sf[4]))
            c += 0.35 * abs(float(tf[5] - sf[5]))
            if (
                abs(float(tf[0])) > 0.05
                and abs(float(sf[0])) > 0.05
                and math.copysign(1.0, float(tf[0]))
                != math.copysign(1.0, float(sf[0]))
            ):
                c += 4.0
            ranked.append((c, sid))
        ranked.sort(key=lambda row: (row[0], row[1]))
        mapping[tid] = ranked[0][1]
        details.append({
            "target_joint_id": tid,
            "source_joint_id": ranked[0][1],
            "cost": float(ranked[0][0]),
            "next_cost": float(ranked[1][0]) if len(ranked) > 1 else None,
        })

    return mapping, details, srows, trows, tpar


def _same_source_chain_group_sizes(mapping: dict, tpar: dict):
    # A source delta may be distributed along a denser target chain, but
    # sibling branches must not dilute one another. For each connected
    # same-source target subtree, use the maximum root-to-leaf chain length
    # as the divisor.
    children = {tid: [] for tid in mapping}
    for tid, parent in tpar.items():
        if parent in children:
            children[parent].append(tid)

    cache = {}
    visited_roots = set()
    for tid, sid in mapping.items():
        root = _same_source_chain_root(tid, sid, mapping, tpar)
        key = (root, sid)
        if key in visited_roots:
            continue
        visited_roots.add(key)

        members = []
        max_depth = 0
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            if mapping.get(node) != sid:
                continue
            members.append(node)
            max_depth = max(max_depth, depth)
            for child in children.get(node, ()):
                if mapping.get(child) == sid:
                    stack.append((child, depth + 1))
        divisor = max(1, max_depth)
        for member in members:
            cache[member] = divisor
    return cache


def _same_source_chain_root(tid, sid, mapping, tpar):
    root = tid
    while tpar.get(root) is not None and mapping.get(tpar[root]) == sid:
        root = tpar[root]
    return root


def _tracks_for_clip(payload, skeleton, cameras, source_report):
    mapping, mapping_details, srows, trows, tpar = _mapping(
        payload, skeleton, source_report
    )
    source_frames = derive_joint_frames_from_rows(
        srows,
        joint_id_key="source_joint_id",
        parent_id_key="parent_source_joint_id",
        position_key="rest_position",
    )
    target_frames = derive_joint_frames_from_skeleton(
        skeleton, cameras=cameras
    )
    group_size = _same_source_chain_group_sizes(mapping, tpar)
    source_tracks = {
        str(row["source_joint_id"]): row
        for row in payload.get("tracks") or ()
    }
    target_scale = _target_body_scale(skeleton)
    out = {}

    for tid in sorted(mapping):
        sid = mapping[tid]
        raw = source_tracks.get(sid)
        if raw is None:
            raise RuntimeError(f"DEMO_MOTION_SOURCE_TRACK_MISSING:{sid}")
        Rs = np.asarray(source_frames[sid].rotation_matrix, dtype=np.float64)
        Rt = np.asarray(target_frames[tid].rotation_matrix, dtype=np.float64)
        n = int(group_size.get(tid, 1))
        keys = []
        previous = None
        for row in raw.get("keyframes") or ():
            q_source = tuple(map(float, row["local_rotation_quat_xyzw"]))
            Qs = _quat_matrix_xyzw(q_source)
            Qobj = Rs @ Qs @ Rs.T
            Qt_full = Rt.T @ Qobj @ Rt
            qt_full = _matrix_to_quat_xyzw(Qt_full)
            qt = _slerp((0.0, 0.0, 0.0, 1.0), qt_full, 1.0 / n)
            if previous is not None and float(np.dot(previous, qt)) < 0.0:
                qt = tuple(-float(x) for x in qt)
            previous = np.asarray(qt, dtype=np.float64)

            source_local = np.asarray(
                row.get("local_translation_xyz") or (0.0, 0.0, 0.0),
                dtype=np.float64,
            )
            object_delta = Rs @ (source_local * float(target_scale))
            target_local = np.asarray(
                object_vector_to_joint_local(target_frames[tid], object_delta),
                dtype=np.float64,
            ) / float(n)
            keys.append(MotionKeyframe3DIR(
                time_seconds=float(row["time_seconds"]),
                local_rotation_quat_xyzw=tuple(map(float, qt)),
                local_translation_xyz=tuple(map(float, target_local)),
                local_scale_xyz=(1.0, 1.0, 1.0),
                metadata={
                    "demo_only": True,
                    "source_joint_id": sid,
                    "shared_chain_divisor": n,
                    "frame_rebased": True,
                },
            ))
        out[tid] = CanonicalJointTrack3DIR(
            canonical_joint_id=tid,
            source_joint_id=sid,
            channel_contract=(
                "LOCAL_ROTATION_QUAT_XYZW",
                "LOCAL_TRANSLATION_XYZ",
                "LOCAL_SCALE_XYZ",
            ),
            keyframes=tuple(keys),
            metadata={
                "demo_only": True,
                "retarget_kind": "NONINJECTIVE_REST_GEOMETRY_V1",
                "source_delta_rebased_through_object_frame": True,
                "shared_source_chain_rotation_distributed": True,
            },
        )
    return out, mapping_details


def _candidate_skin_weights(candidate, skin, skeleton):
    joint_ids = tuple(sorted(j.canonical_joint_id for j in skeleton.joints))
    ji = {jid: i for i, jid in enumerate(joint_ids)}
    skin_by_surface = {row.surface_id: row for row in skin.rows}
    W = np.zeros((len(candidate.vertices), len(joint_ids)), dtype=np.float64)
    for vi, vertex in enumerate(candidate.vertices):
        total_support = 0.0
        for sid, coeff in _skin_support_coefficients(vertex):
            c = float(coeff)
            total_support += c
            row = skin_by_surface.get(sid)
            if row is None:
                raise RuntimeError(f"DEMO_SKIN_SURFACE_MISSING:{sid}")
            for jid, w in row.influences:
                if jid not in ji:
                    raise RuntimeError(f"DEMO_SKIN_JOINT_UNKNOWN:{jid}")
                W[vi, ji[jid]] += c * float(w)
        if abs(total_support - 1.0) > 1e-6:
            raise RuntimeError("DEMO_SKIN_SUPPORT_NOT_SIMPLEX")
        mass = float(W[vi].sum())
        if mass <= 1e-12:
            raise RuntimeError("DEMO_SKIN_ZERO_MASS")
        W[vi] /= mass
    if not np.allclose(W.sum(axis=1), 1.0, atol=1e-8, rtol=0.0):
        raise RuntimeError("DEMO_SKIN_WEIGHT_NOT_SIMPLEX")
    return joint_ids, W


def _skin(rest, W, joint_ids, skin_matrices):
    hom = np.concatenate(
        [rest, np.ones((len(rest), 1), dtype=np.float64)], axis=1
    )
    per = np.stack(
        [(hom @ np.asarray(skin_matrices[jid]).T)[:, :3] for jid in joint_ids],
        axis=1,
    )
    return np.sum(per * W[:, :, None], axis=1)


def _composite_sheet(images, *, columns: int, label: str) -> Image.Image:
    if not images:
        raise RuntimeError("DEMO_PREVIEW_EMPTY")
    cell_w, cell_h = images[0][1].size
    rows = math.ceil(len(images) / columns)
    header = 28
    sheet = Image.new("RGBA", (columns * cell_w, rows * (cell_h + header)), (0,0,0,0))
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(images):
        row, col = divmod(index, columns)
        x = col * cell_w
        y = row * (cell_h + header)
        sheet.alpha_composite(image, (x, y + header))
        draw.text((x + 6, y + 6), title, fill=(255,255,255,255))
    return sheet


def run(*, authority_root: Path, run_id: str, out_dir: Path):
    ctx = _ctx(authority_root, run_id)
    candidate = canonical_mesh_candidate_from_dict(
        stage_output_payload(
            ctx, "18_CANONICAL_MESH_ADDRESSING_BUILD",
            "RealSaS.CanonicalMeshCandidateIR.v1",
        )
    )
    skeleton = qualified_skeleton_from_dict(
        stage_output_payload(
            ctx, "28_SKELETON_QUALIFIED",
            "RealSaS.QualifiedSkeletonIR.v1",
        )
    )
    skin = qualified_skin_from_dict(
        stage_output_payload(
            ctx, "32_SKIN_QUALIFIED",
            "RealSaS.QualifiedSkinIR.v1",
        )
    )
    camera_set = qualified_camera_set_from_dict(
        stage_output_payload(
            ctx, "05_CAMERA_CONTRACT_SOLVED",
            "RealSaS.QualifiedCameraSetIR.v1",
        )
    )
    cameras = tuple(sorted(camera_set.cameras, key=lambda c: int(c.view_index)))
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

    source_report = json.loads(
        Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text()
    )
    joint_ids, W = _candidate_skin_weights(candidate, skin, skeleton)
    rest = np.asarray([v.P for v in candidate.vertices], dtype=np.float64)

    clips = [
        ("demo_idle_v1", "idle"),
        ("demo_run_v1", "run"),
        ("demo_slash_v1", "slash"),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "RealSaS.KnightDemoMotionVisualContinuation.v1",
        "status": "MEASURED_DEMO_ONLY",
        "run_id": run_id,
        "mechanical_carrier": "STAGE18_CANONICAL_MESH_CANDIDATE",
        "skeleton_authority": "STAGE28_QUALIFIED_SKELETON",
        "skin_authority": "STAGE32_QUALIFIED_SKIN",
        "appearance_authority": "STAGE23_COMPLETE_APPEARANCE",
        "stage35_failure_preserved": True,
        "stage35_product_pass_claimed": False,
        "retarget": "NONINJECTIVE_REST_GEOMETRY_OBJECT_FRAME_REBASE_V1",
        "clips": [],
    }

    for clip_id, short in clips:
        path = (
            ctx["run_root"] / "inputs" / "motion" / "quaternius_knight_v1"
            / f"{clip_id}.motion.json"
        )
        payload = json.loads(path.read_text())
        tracks, mapping_details = _tracks_for_clip(
            payload, skeleton, cameras, source_report
        )
        duration = float(payload["duration_seconds"])
        sample_times = np.linspace(
            0.0,
            duration,
            4,
            endpoint=not bool(payload.get("loop")),
        )
        frames = []
        metrics = []
        for view in (0, 2):
            texture = _load_texture_pages(texture_by_view[view])
            for frame_index, time_seconds in enumerate(sample_times):
                skin_mats, joint_pos, frame_hash = _joint_pose_v2(
                    skeleton=skeleton,
                    tracks=tracks,
                    time_seconds=float(time_seconds),
                    cameras=cameras,
                )
                posed = _skin(rest, W, joint_ids, skin_mats)
                displacement = np.linalg.norm(posed - rest, axis=1)
                render = render_caa_reference(
                    mesh=candidate,
                    camera=cameras[view],
                    face_uv=face_uv,
                    texture_rgba_u8=texture,
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,
                    positions=posed,
                )
                rgba = Image.fromarray(
                    np.asarray(render.straight_rgba_u8, dtype=np.uint8),
                    mode="RGBA",
                )
                title = f"{short} V{view} t={time_seconds:.2f}"
                frames.append((title, rgba))
                metrics.append({
                    "view_index": view,
                    "frame_index": int(frame_index),
                    "time_seconds": float(time_seconds),
                    "max_vertex_displacement": float(np.max(displacement)),
                    "mean_vertex_displacement": float(np.mean(displacement)),
                    "visible_pixel_count": int(np.count_nonzero(render.geometry_visible)),
                    "final_alpha_pixel_count": int(np.count_nonzero(render.final_alpha)),
                    "undefined_visible_pixel_count": int(
                        np.count_nonzero(
                            np.asarray(render.geometry_visible)
                            & (np.asarray(render.provenance_code) == 255)
                        )
                    ),
                })
        sheet = _composite_sheet(frames, columns=4, label=short)
        out_path = out_dir / f"KNIGHT_{short.upper()}_DEMO_PREVIEW_V1.png"
        sheet.save(out_path)
        report["clips"].append({
            "clip_id": clip_id,
            "duration_seconds": duration,
            "preview_path": str(out_path),
            "mapping": mapping_details,
            "frame_metrics": metrics,
        })

    report_path = out_dir / "KNIGHT_DEMO_MOTION_VISUAL_CONTINUATION_REPORT_V1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("DEMO_MOTION_VISUAL_CONTINUATION_PASS", json.dumps({
        "preview_count": 3,
        "stage35_failure_preserved": True,
        "product_authority_claimed": False,
    }, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authority-root", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    run(
        authority_root=Path(a.authority_root).expanduser().resolve(),
        run_id=a.run_id,
        out_dir=Path(a.out_dir),
    )


if __name__ == "__main__":
    main()
