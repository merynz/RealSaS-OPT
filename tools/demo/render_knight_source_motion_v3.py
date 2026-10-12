#!/usr/bin/env python3
"""Compile exact source tracks on unchanged Knight M/G/W and render all views.

Demo/source-review evidence only. No Attempt or ProductRevision is minted.
The old witness is used solely to recover and verify its original correspondence,
never as the new animation's pose input.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_rows
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_pre_bind_v2
from compiler.realsas_compiler_core.motion_retarget_frames_v3 import compile_retargeted_keys
from compiler.realsas_compiler_core.single_m_caa_v1 import array_hash, load_material_bundle, render_single_m_caa, write_material_bundle
from tools.demo.render_knight_m_caa_v1 import load_inputs
from tools.demo.knight_preview_media_v1 import save_png_checked, save_npz_checked, package_frames
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import compile_skin_holder_bindings, apply_surface_rigid_3d
from compiler.realsas_compiler_core.presentation_part_order_v1 import compile_part_ids, part_depth_order

CONTEXT = {}


def frame_motion_cameras(poses, cameras, resolution):
    """Fit the full motion using an output viewport; source addressing is fixed."""
    bounds = []
    for camera in cameras:
        q = poses - np.asarray(camera.origin)
        x = q @ np.asarray(camera.right)
        y = q @ np.asarray(camera.screen_up)
        bounds.append((x.min(), x.max(), y.min(), y.max()))
    extent = max(max(b[1]-b[0], b[3]-b[2]) / 2 for b in bounds) * 1.12
    result = []
    for camera, (xmin,xmax,ymin,ymax) in zip(cameras,bounds):
        origin = (np.asarray(camera.origin) + np.asarray(camera.right)*(xmin+xmax)/2
                  + np.asarray(camera.screen_up)*(ymin+ymax)/2)
        result.append(type(camera)(**{**vars(camera), 'origin': origin.tolist(),
                      'half_extent': float(extent), 'resolution': int(resolution)}))
    return result


def preview_samples(duration, *, loop, minimum_duration=4., fps=30):
    if not np.isfinite(duration) or duration <= 0 or minimum_duration < 4 or fps < 1:
        raise ValueError("PREVIEW_DURATION_OR_FPS_INVALID")
    count = int(np.ceil(max(float(duration), float(minimum_duration)) * fps))
    playback = np.arange(count, dtype=float) / fps
    requested = np.mod(playback, duration) if loop else np.minimum(playback, duration)
    if loop:
        requested[np.isclose(requested,duration,rtol=0,atol=1e-10)]=0.
    unique, indices = np.unique(np.round(requested, 12), return_inverse=True)
    unique = np.clip(unique, 0., duration)
    return playback, unique, indices


def sampled_keys(raw, times, translation_scale=1.):
    result = {}
    for jid, keys in raw.items():
        ts = np.asarray([k["time_seconds"] for k in keys], float)
        rots = Slerp(ts, Rotation.from_quat([k["local_rotation_quat_xyzw"] for k in keys]))(times).as_quat()
        tr = np.asarray([k.get("local_translation_xyz", [0, 0, 0]) for k in keys])
        translations = np.column_stack([np.interp(times, ts, tr[:, i]) for i in range(3)]) * translation_scale
        result[jid] = [dict(time_seconds=float(t), local_rotation_quat_xyzw=q.tolist(), local_translation_xyz=v.tolist(), local_scale_xyz=[1., 1., 1.]) for t, q, v in zip(times, rots, translations)]
    return result


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rest_transforms(frames):
    result = {}
    for jid, f in frames.items():
        result[jid] = np.eye(4)
        result[jid][:3, :3] = f.rotation_matrix
        result[jid][:3, 3] = f.rest_position
    return result


def palette(frames, keys, times):
    rest = rest_transforms(frames)
    order, pending = [], set(frames)
    while pending:
        ready = sorted(j for j in pending if frames[j].parent_id is None or frames[j].parent_id in order)
        if not ready:
            raise RuntimeError("TARGET_TREE_INVALID")
        order += ready
        pending.difference_update(ready)
    all_mats = []
    for fi, _ in enumerate(times):
        pose = {}
        for jid in order:
            pid = frames[jid].parent_id
            D = np.eye(4)
            if jid in keys:
                k = keys[jid][fi]
                D[:3, :3] = Rotation.from_quat(k["local_rotation_quat_xyzw"]).as_matrix()
                D[:3, 3] = k["local_translation_xyz"]
            pose[jid] = (rest[jid] if pid is None else pose[pid] @ np.linalg.inv(rest[pid]) @ rest[jid]) @ D
        all_mats.append({j: pose[j] @ np.linalg.inv(rest[j]) for j in frames})
    return all_mats


def recover_correspondence(witness, source, target_frames, index):
    """Identify copied local rotations by full-sequence equality, fail on ambiguity.

    Zero-rotation tracks remain inherited, as in the original witness. Translation
    at the first body control supplies a separate exact identity check below.
    """
    mats = witness[f"clip_{index}_skin_matrices_source"]
    parents = witness["axis_parents"]
    tracks = {t["source_joint_id"]: t["keyframes"] for t in source["tracks"]}
    qs = {sid: Rotation.from_quat([k["local_rotation_quat_xyzw"] for k in ks]).as_matrix() for sid, ks in tracks.items()}
    source_root = next(r["source_joint_id"] for r in source["source_skeleton"] if r["parent_source_joint_id"] is None)
    mapping = {"0": source_root}
    evidence = []
    for i, parent in enumerate(parents):
        if parent < 0:
            continue
        F = np.asarray(target_frames[str(i)].rotation_matrix)
        actual = F.T @ np.swapaxes(mats[:, parent, :3, :3], 1, 2) @ mats[:, i, :3, :3] @ F
        if np.max(np.abs(actual - np.eye(3))) < 1e-7:
            continue
        errors = sorted((float(np.max(np.abs(actual - q))), sid) for sid, q in qs.items())
        if errors[0][0] > 1e-9 or (len(errors) > 1 and errors[1][0] < 1e-8):
            raise RuntimeError(f"OLD_CORRESPONDENCE_NOT_EXACT_OR_UNIQUE:{i}:{errors[:2]}")
        mapping[str(i)] = errors[0][1]
        evidence.append(dict(target_index=i, source_joint_id=errors[0][1], max_copy_residual=errors[0][0], next_residual=errors[1][0]))
    # Recover the animated, rotation-identity body track from its translation.
    target_scale = float(np.max(np.linalg.norm(witness["axis_positions_source"] - witness["axis_positions_source"][0], axis=1)))
    for i, parent in enumerate(parents):
        if parent < 0 or str(i) in mapping:
            continue
        p = witness["axis_positions_source"][i]
        posed = np.einsum("fij,j->fi", mats[:, i], np.r_[p, 1.])
        delta = np.einsum("fij,fj->fi", np.linalg.inv(mats[:, parent]), posed)[:, :3] - p
        local = delta @ np.asarray(target_frames[str(i)].rotation_matrix)
        if np.max(np.abs(local)) < 1e-7:
            continue
        errors = sorted((float(np.max(np.abs(local - np.array([k["local_translation_xyz"] for k in ks]) * target_scale))), sid) for sid, ks in tracks.items())
        if errors[0][0] > 1e-8 or errors[1][0] < 1e-8:
            raise RuntimeError(f"OLD_TRANSLATION_CORRESPONDENCE_AMBIGUOUS:{i}:{errors[:2]}")
        mapping[str(i)] = errors[0][1]
        evidence.append(dict(target_index=i, source_joint_id=errors[0][1], max_copy_residual=errors[0][0], next_residual=errors[1][0], channel="TRANSLATION"))
    return mapping, evidence, target_scale


def render_view(view):
    m, poses, out = CONTEXT["material"], CONTEXT["poses"], CONTEXT["out"]
    prefix = CONTEXT["prefix"]
    results = []
    for fi, posed in enumerate(poses):
        ranks=(CONTEXT["part_orders"][view][fi][CONTEXT["face_part"]]
               if CONTEXT["overlap_policy"]=="kinematic" else None)
        r = render_single_m_caa(**{k: m[k] for k in ("rest", "faces", "source", "maps", "cameras", "caa")}, posed=posed, view=view, resolution=CONTEXT["resolution"],face_draw_rank=ranks, output_camera=CONTEXT['output_cameras'][view])
        save_png_checked(Image.fromarray(r.straight_rgba_u8),out / f"{prefix}_v{view}_{fi:03d}.png")
        results.append(dict(frame=fi, rgba_sha256=array_hash(r.straight_rgba_u8), depth_ties=r.unresolved_depth_tie_count, fragment_overflow=r.fragment_overflow_count))
    return view, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--witness", type=Path, required=True)
    ap.add_argument("--bind-source", type=Path, help="Exact previously sealed SLASH source used only to identify correspondence")
    ap.add_argument("--material", type=Path, help="Existing compiled M+CAA material to reuse")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=4.)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--attachment-policy", choices=["mechanical","slot_anchor"], default="slot_anchor")
    ap.add_argument("--overlap-policy", choices=["geometric","kinematic"], default="geometric")
    args = ap.parse_args()
    source = json.loads(args.source.read_text())
    native = json.loads((ROOT / "canonical/MOTION_PRESET_SOURCE_QUATERNIUS_KNIGHT_NATIVE_V1.json").read_text())
    expected = {r["clip_id"]: r for r in native["clips"]}.get(source["clip_id"])
    if expected is None or source["clip_kind"] != expected["clip_kind"] or not source["metadata"]["source_take"].endswith("|" + expected["source_take"].split("|")[-1]):
        raise RuntimeError("EXACT_KNIGHT_NATIVE_SOURCE_REQUIRED")
    if source["metadata"]["source_fbx_sha256"] != native["source"]["fbx_sha256"]:
        raise RuntimeError("KNIGHT_SOURCE_FBX_IDENTITY_DRIFT")
    clip_kind = expected["clip_kind"]
    prefix = clip_kind.lower()
    bind_path = args.bind_source or args.source
    reference = json.loads(bind_path.read_text())
    if digest(bind_path) != "6422209f1291dbb3dbfdffb6b238e63214f8aeddfc31e12f1b105ecc1da85298":
        raise RuntimeError("SEALED_SLASH_CORRESPONDENCE_SOURCE_REQUIRED")
    source_rows = {r["source_joint_id"]: r for r in source["source_skeleton"]}
    reference_rows = {r["source_joint_id"]: r for r in reference["source_skeleton"]}
    if set(source_rows) != set(reference_rows):
        raise RuntimeError("NATIVE_SOURCE_SKELETON_IDS_DRIFT")
    rest_residual = max(float(np.max(np.abs(np.asarray(source_rows[j]["rest_position"]) - reference_rows[j]["rest_position"]))) for j in source_rows)
    if rest_residual > 1e-5 or any(source_rows[j]["parent_source_joint_id"] != reference_rows[j]["parent_source_joint_id"] for j in source_rows):
        raise RuntimeError("NATIVE_SOURCE_SKELETON_REST_OR_TOPOLOGY_DRIFT")
    fixture = ROOT / "tests/fixtures/single_m_caa_knight_v1/baseline.json"
    manifest, baseline, rgba, cameras, caa = load_inputs(fixture)
    if digest(args.witness) != manifest["parent_witness_sha256"]:
        raise RuntimeError("OLD_WITNESS_BYTES_DRIFT")
    with np.load(args.witness, allow_pickle=False) as z:
        witness = {k: z[k] for k in z.files}
    if not np.array_equal(witness["vertices"], baseline["rest"]) or not np.array_equal(witness["faces"], baseline["faces"]):
        raise RuntimeError("M_DOMAIN_DRIFT")
    axis, parents = witness["axis_positions_source"], witness["axis_parents"]
    skeleton = SimpleNamespace(root_id="0", joints=tuple(SimpleNamespace(canonical_joint_id=str(i), parent_canonical_id=None if parents[i] < 0 else str(parents[i]), position=p) for i, p in enumerate(axis)))
    tf, frame_report = derive_joint_frames_pre_bind_v2(skeleton, cameras=cameras)
    mapping, fingerprint, target_scale = recover_correspondence(witness, reference, tf, 2)
    sf = derive_joint_frames_from_rows(source["source_skeleton"], joint_id_key="source_joint_id", parent_id_key="parent_source_joint_id", position_key="rest_position")
    source_root = mapping["0"]
    C = np.asarray(tf["0"].rotation_matrix) @ np.asarray(sf[source_root].rotation_matrix).T
    playback_times, times, frame_indices = preview_samples(float(source["duration_seconds"]), loop=bool(source["loop"]), minimum_duration=args.duration, fps=args.fps)
    keys = compile_retargeted_keys(source, target_frames=tf, target_to_source=mapping, object_rotation=C, target_body_scale=target_scale, sample_times=times)
    target_palettes = palette(tf, keys, times)
    matrices = np.array([[p[str(i)] for i in range(len(axis))] for p in target_palettes])
    # Independent source FK gives an orientation proof against artist motion.
    skeys = sampled_keys({t["source_joint_id"]: t["keyframes"] for t in source["tracks"]}, times, source["metadata"]["source_body_scale"])
    source_palettes = palette(sf, skeys, times)
    orientation_errors = {tid: max(float(np.max(np.abs(target_palettes[f][tid][:3, :3] - C @ source_palettes[f][sid][:3, :3] @ C.T))) for f in range(len(times))) for tid, sid in mapping.items()}
    if max(orientation_errors.values()) > 1e-9:
        raise RuntimeError("SOURCE_ORIENTATION_NOT_PRESERVED")
    rest, faces = baseline["rest"], baseline["faces"]
    W = witness["canonical_motion_weights"]
    hom = np.c_[rest, np.ones(len(rest))]
    posed = np.einsum("vj,fjkl,vl->fvk", W, matrices[:, :, :3], hom, optimize=True)
    mechanical_poses=posed.copy()
    owners=witness["presentation_attachment_vertex_owner"]
    bindings=compile_skin_holder_bindings(rest=rest,faces=faces,weights=W,parents=parents,owners=owners)
    attachment_matrices=np.zeros((len(times),len(bindings),4,4))
    attachment_proofs=[]
    for fi in range(len(times)):
        frame_proofs=[]
        for bi,row in enumerate(bindings):
            b=row["binding"]
            if args.attachment_policy=="slot_anchor":
                posed[fi],proof=apply_surface_rigid_3d(posed[fi],rest=rest,faces=faces,binding=b,rotation=matrices[fi,row["slot"],:3,:3])
                attachment_matrices[fi,bi]=np.eye(4)
                attachment_matrices[fi,bi,:3,:3]=proof["rotation"]
                attachment_matrices[fi,bi,:3,3]=proof["translation"]
                frame_proofs.append(dict(owner=row["owner"],slot=row["slot"],**proof))
            else:attachment_matrices[fi,bi]=matrices[fi,row["slot"]]
        attachment_proofs.append(frame_proofs)
    vertex_part,face_part,part_keys=compile_part_ids(len(rest),faces,owners,W)
    part_orders=np.array([part_depth_order(posed,vertex_part,camera) for camera in cameras])
    output_cameras=frame_motion_cameras(posed,cameras,args.resolution)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_npz_checked(out / f"compiled_{prefix}_m_poses.npz", poses=posed, mechanical_poses=mechanical_poses,times=times, skin_matrices=matrices,attachment_matrices=attachment_matrices,attachment_owners=owners,vertex_part=vertex_part,face_part=face_part,part_orders=part_orders,playback_times=playback_times, playback_frame_indices=frame_indices, rest=rest, faces=faces, weights=W, axis=axis, parents=parents)
    (out / f"compiled_{prefix}_tracks.json").write_text(json.dumps(dict(source_sha256=digest(args.source), object_rotation=C.tolist(), target_to_source=mapping, tracks=keys), indent=2) + "\n")
    report = dict(schema="RealSaS.KnightSourceMotionFrameFix.v3", product_qualification_claimed=False, source_code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source_take=source["metadata"]["source_take"], source_clip_sha256=digest(args.source), old_witness_sha256=digest(args.witness), source_tracks_compiled=True, old_poses_used_as_new_animation=False, mesh_sha256=array_hash(rest), topology_sha256=array_hash(faces), weights_sha256=array_hash(W), hierarchy_sha256=array_hash(parents), axis_sha256=array_hash(axis), correspondence=fingerprint, source_object_to_target_rotation=C.tolist(), max_orientation_residual=max(orientation_errors.values()), orientation_residual_by_joint=orientation_errors, frame_count=len(times), duration_seconds=float(times[-1]), source_loop=source["loop"], views={}, attachment_qualification_claimed=False, depth_order_fixed=False)
    report.update(clip_id=source["clip_id"], clip_kind=clip_kind, motion_provenance="KNIGHT_NATIVE", bind_source_sha256=digest(bind_path), source_rest_reference_max_residual=rest_residual, source_duration_seconds=float(source["duration_seconds"]), preview_fps=args.fps, preview_duration_seconds=len(playback_times)/args.fps, preview_frame_count=len(playback_times), unique_source_sample_count=len(times), per_view_resolution=args.resolution, preview_repeat_policy="SOURCE_LOOP" if source["loop"] else "HOLD_TERMINAL_POSE", source_motion_speed_changed=False)
    report.update(attachment_policy=args.attachment_policy,overlap_policy=args.overlap_policy,
                  attachment_proofs=attachment_proofs,part_keys=part_keys,
                  semantic_overlap_qualified=False,
                  attachment_bindings=[dict(owner=r["owner"],slot=r["slot"],holder_joint_indices=r["holder_joint_indices"],held_face=r["binding"].held.face_index,held_barycentric=r["binding"].held.barycentric,holder_face=r["binding"].holder.face_index,holder_barycentric=r["binding"].holder.barycentric,setup_surface_gap_world=r["binding"].setup_surface_gap) for r in bindings])
    report['output_cameras']=[vars(c) for c in output_cameras]
    report['source_addressing_cameras_changed']=False
    (out / "source_motion_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.compile_only:
        print(json.dumps(dict(clip=clip_kind, samples=len(times), preview_frames=len(playback_times), source_orientation_residual=report["max_orientation_residual"])), flush=True)
        return
    if args.material:
        ref = dict(path=str(args.material.resolve()), sha256=digest(args.material))
    else:
        ref = write_material_bundle(out / "single_m_caa_material.npz", rest=rest, faces=faces, source=rgba, cameras=cameras, caa_arrays=caa, input_identities=manifest["inputs"])
    material = load_material_bundle(ref["path"], expected_sha256=ref["sha256"], rest=rest, faces=faces)
    CONTEXT.update(material=material, poses=posed, out=out, prefix=prefix, resolution=args.resolution,part_orders=part_orders,face_part=face_part,overlap_policy=args.overlap_policy,output_cameras=output_cameras)
    with ProcessPoolExecutor(max_workers=min(8, max(1, args.workers)), mp_context=multiprocessing.get_context("fork")) as pool:
        jobs = [pool.submit(render_view, v) for v in range(8)]
        for job in as_completed(jobs):
            view, frames = job.result()
            report["views"][f"V{view}"] = frames
            print(f"V{view}: {len(frames)} fresh source-compiled frames", flush=True)
    ffprobe = package_frames(out, prefix, clip_kind, frame_indices, args.resolution, args.fps)
    gif = out / f"Knight_{clip_kind}_V0-V7.gif"
    report["mp4_stream"] = ffprobe
    report["gif_sha256"] = digest(gif)
    report["fresh_render_frames"] = len(times) * 8
    (out / "source_motion_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(frame_count=len(times), views=8, max_orientation_residual=report["max_orientation_residual"], source_take=report["source_take"], gif=str(gif)), indent=2))


if __name__ == "__main__":
    main()
