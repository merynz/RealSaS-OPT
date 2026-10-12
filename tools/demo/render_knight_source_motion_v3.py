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
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_rows
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_pre_bind_v2
from compiler.realsas_compiler_core.motion_retarget_frames_v3 import compile_retargeted_keys
from compiler.realsas_compiler_core.single_m_caa_v1 import array_hash, load_material_bundle, render_single_m_caa, write_material_bundle
from tools.demo.render_knight_m_caa_v1 import load_inputs, tile

CONTEXT = {}


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
    results = []
    for fi, posed in enumerate(poses):
        r = render_single_m_caa(**{k: m[k] for k in ("rest", "faces", "source", "maps", "cameras", "caa")}, posed=posed, view=view, resolution=256)
        Image.fromarray(r.straight_rgba_u8).save(out / f"slash_v{view}_{fi:03d}.png")
        results.append(dict(frame=fi, rgba_sha256=array_hash(r.straight_rgba_u8), depth_ties=r.unresolved_depth_tie_count, fragment_overflow=r.fragment_overflow_count))
    return view, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--witness", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    source = json.loads(args.source.read_text())
    if source["clip_id"] != "demo_slash_v1" or source["clip_kind"] != "SLASH" or not source["metadata"]["source_take"].endswith("|Run_swordAttack"):
        raise RuntimeError("EXACT_ATTACK_SOURCE_REQUIRED")
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
    mapping, fingerprint, target_scale = recover_correspondence(witness, source, tf, 2)
    sf = derive_joint_frames_from_rows(source["source_skeleton"], joint_id_key="source_joint_id", parent_id_key="parent_source_joint_id", position_key="rest_position")
    source_root = mapping["0"]
    C = np.asarray(tf["0"].rotation_matrix) @ np.asarray(sf[source_root].rotation_matrix).T
    keys = compile_retargeted_keys(source, target_frames=tf, target_to_source=mapping, object_rotation=C, target_body_scale=target_scale)
    times = np.array([k["time_seconds"] for k in keys["0"]])
    target_palettes = palette(tf, keys, times)
    matrices = np.array([[p[str(i)] for i in range(len(axis))] for p in target_palettes])
    # Independent source FK gives an orientation proof against artist motion.
    skeys = {t["source_joint_id"]: [dict(k, local_translation_xyz=(np.array(k["local_translation_xyz"]) * source["metadata"]["source_body_scale"]).tolist()) for k in t["keyframes"]] for t in source["tracks"]}
    source_palettes = palette(sf, skeys, times)
    orientation_errors = {tid: max(float(np.max(np.abs(target_palettes[f][tid][:3, :3] - C @ source_palettes[f][sid][:3, :3] @ C.T))) for f in range(len(times))) for tid, sid in mapping.items()}
    if max(orientation_errors.values()) > 1e-9:
        raise RuntimeError("SOURCE_ORIENTATION_NOT_PRESERVED")
    rest, faces = baseline["rest"], baseline["faces"]
    W = witness["canonical_motion_weights"]
    hom = np.c_[rest, np.ones(len(rest))]
    posed = np.einsum("vj,fjkl,vl->fvk", W, matrices[:, :, :3], hom, optimize=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "compiled_slash_m_poses.npz", poses=posed, times=times, skin_matrices=matrices, rest=rest, faces=faces, weights=W, axis=axis, parents=parents)
    (out / "compiled_slash_tracks.json").write_text(json.dumps(dict(source_sha256=digest(args.source), object_rotation=C.tolist(), target_to_source=mapping, tracks=keys), indent=2) + "\n")
    report = dict(schema="RealSaS.KnightSourceMotionFrameFix.v3", product_qualification_claimed=False, source_code_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), source_take=source["metadata"]["source_take"], source_clip_sha256=digest(args.source), old_witness_sha256=digest(args.witness), source_tracks_compiled=True, old_poses_used_as_new_animation=False, mesh_sha256=array_hash(rest), topology_sha256=array_hash(faces), weights_sha256=array_hash(W), hierarchy_sha256=array_hash(parents), axis_sha256=array_hash(axis), correspondence=fingerprint, source_object_to_target_rotation=C.tolist(), max_orientation_residual=max(orientation_errors.values()), orientation_residual_by_joint=orientation_errors, frame_count=len(times), duration_seconds=float(times[-1]), source_loop=source["loop"], views={}, attachment_qualification_claimed=False, depth_order_fixed=False)
    (out / "source_motion_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    ref = write_material_bundle(out / "single_m_caa_material.npz", rest=rest, faces=faces, source=rgba, cameras=cameras, caa_arrays=caa, input_identities=manifest["inputs"])
    material = load_material_bundle(ref["path"], expected_sha256=ref["sha256"], rest=rest, faces=faces)
    CONTEXT.update(material=material, poses=posed, out=out)
    with ProcessPoolExecutor(max_workers=min(8, max(1, args.workers)), mp_context=multiprocessing.get_context("fork")) as pool:
        jobs = [pool.submit(render_view, v) for v in range(8)]
        for job in as_completed(jobs):
            view, frames = job.result()
            report["views"][f"V{view}"] = frames
            print(f"V{view}: {len(frames)} fresh source-compiled frames", flush=True)
    # GIF has a centisecond clock; round cumulative times to avoid a 25fps
    # animation when the authored source uses 24fps.
    boundaries = np.r_[times - times[0], times[-1] - times[0] + times[-1] - times[-2]]
    durations = (np.diff(np.rint(boundaries * 100)).astype(int) * 10).tolist()
    sheets, individual = [], [[] for _ in range(8)]
    for fi in range(len(times)):
        sheet = Image.new("RGB", (1024, 576), (23, 28, 37))
        for v in range(8):
            image = tile(np.asarray(Image.open(out / f"slash_v{v}_{fi:03d}.png").convert("RGBA")), f"SLASH V{v}")
            individual[v].append(image.resize((512, 576), Image.Resampling.NEAREST))
            sheet.paste(image, ((v % 4) * 256, (v // 4) * 288))
        sheets.append(sheet)
        sheet.save(out / f"slash_grid_{fi:03d}.png")
    gif = out / "Knight_SLASH_V0-V7.gif"
    sheets[0].save(gif, save_all=True, append_images=sheets[1:], duration=durations, loop=0)
    for v, frames in enumerate(individual):
        frames[0].save(out / f"Knight_SLASH_V{v}.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    from PIL import ImageSequence
    for path in [gif, *(out / f"Knight_SLASH_V{v}.gif" for v in range(8))]:
        with Image.open(path) as decoded:
            if decoded.n_frames != len(times):
                raise RuntimeError("GIF_FRAME_COUNT_DRIFT:" + path.name)
            if sum(f.info.get("duration", 0) for f in ImageSequence.Iterator(decoded)) != sum(durations):
                raise RuntimeError("GIF_TIMING_DRIFT:" + path.name)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", "24", "-i", str(out / "slash_grid_%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "Knight_SLASH_V0-V7.mp4")], check=True)
    report["gif_sha256"] = digest(gif)
    report["fresh_render_frames"] = len(times) * 8
    (out / "source_motion_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(frame_count=len(times), views=8, max_orientation_residual=report["max_orientation_residual"], source_take=report["source_take"], gif=str(gif)), indent=2))


if __name__ == "__main__":
    main()
