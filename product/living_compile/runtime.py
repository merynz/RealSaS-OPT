from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import Json, LivingCompileError, MOTION_BAKE_SCHEMA, json_load, load_product, load_proof


def motion_bakes(root: Path) -> dict[str, Json]:
    candidates = []
    for base in (root / "proof" / "motion_bakes", root / "motion_bakes"):
        if base.is_dir():
            candidates.extend(sorted(base.glob("*.json")))
    out: dict[str, Json] = {}
    for path in candidates:
        raw = json_load(path)
        if raw.get("schema_version") != MOTION_BAKE_SCHEMA:
            continue
        clip_id = str(raw.get("clip_id") or "")
        if clip_id:
            out[clip_id] = raw
    return out


def runtime_clips(root: Path) -> Json:
    product = load_product(root)
    proof = load_proof(root, product)
    proof_status = str(proof.get("overall_status") or "UNKNOWN")
    if proof_status != "PASS":
        return {
            "schema_version": "RealSaS.LivingCompileRuntimeIndex.v1",
            "clips": [],
            "proof_hash": proof.get("proof_bundle_hash"),
            "proof_status": proof_status,
            "preview_available": False,
            "authority": "RUNTIME_PREVIEW_WITHHELD_UNTIL_CURRENT_PASS_PROOF",
        }
    bakes = motion_bakes(root)
    product_clips = {str(row.get("clip_id")): row for row in list((product.get("motion_state") or {}).get("clips") or [])}
    clips = []
    for clip_id, bake in sorted(bakes.items()):
        clip = product_clips.get(clip_id) or {}
        clips.append({
            "clip_id": clip_id,
            "display_name": str((clip.get("metadata") or {}).get("display_name") or clip_id),
            "intent": str(clip.get("clip_kind") or "custom"),
            "duration_seconds": float(bake.get("duration_seconds") or clip.get("duration_sec") or 1.0),
            "fps": float(bake.get("fps") or 1.0),
            "loop": bool(bake.get("loop", clip.get("loop", False))),
            "frame_count": len(list(bake.get("frames") or [])),
            "bake_hash": bake.get("bake_hash"),
            "authority": "QUALIFICATION_OWNED_MOTION_BAKE",
        })
    return {
        "schema_version": "RealSaS.LivingCompileRuntimeIndex.v1",
        "clips": clips,
        "proof_hash": proof.get("proof_bundle_hash"),
        "proof_status": proof_status,
        "preview_available": True,
        "authority": "QUALIFICATION_OWNED_MOTION_BAKE",
    }


def _frame_mesh_dict(frame: Mapping[str, Any]) -> dict[str, list[list[float]]]:
    raw = frame.get("mesh_vertices_by_id") or []
    if isinstance(raw, dict):
        return {str(k): [[float(p[0]), float(p[1])] for p in v] for k, v in raw.items()}
    out = {}
    for item in list(raw):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out[str(item[0])] = [[float(p[0]), float(p[1])] for p in list(item[1])]
    return out


def _frame_orders(frame: Mapping[str, Any]) -> dict[str, list[str]]:
    raw = frame.get("render_order_by_view") or []
    if isinstance(raw, dict):
        return {str(k): [str(x) for x in v] for k, v in raw.items()}
    return {str(item[0]): [str(x) for x in item[1]] for item in list(raw) if isinstance(item, (list, tuple)) and len(item) == 2}


def runtime_frame(root: Path, clip_id: str, time_seconds: float) -> Json:
    product = load_product(root)
    proof = load_proof(root, product)
    if proof.get("overall_status") != "PASS":
        raise LivingCompileError("runtime preview requires current PASS proof")
    bake = motion_bakes(root).get(str(clip_id))
    if bake is None:
        raise LivingCompileError(f"motion bake not found for clip: {clip_id}")
    if bake.get("source_product_state_hash") != product.get("product_state_hash"):
        raise LivingCompileError("stale motion bake product binding")
    frames = list(bake.get("frames") or [])
    if not frames:
        raise LivingCompileError("motion bake has no frames")
    duration = float(bake.get("duration_seconds") or 1.0)
    loop = bool(bake.get("loop", False))
    t = float(time_seconds)
    if loop and duration > 0:
        t %= duration
    else:
        t = max(0.0, min(duration, t))
    before = frames[0]
    after = frames[-1]
    for frame in frames:
        ft = float(frame.get("time_seconds") or 0.0)
        if ft <= t:
            before = frame
        if ft >= t:
            after = frame
            break
    ta = float(before.get("time_seconds") or 0.0)
    tb = float(after.get("time_seconds") or ta)
    u = 0.0 if tb <= ta else (t - ta) / (tb - ta)
    a_mesh = _frame_mesh_dict(before)
    b_mesh = _frame_mesh_dict(after)
    meshes: dict[str, list[list[float]]] = {}
    for mesh_id in sorted(set(a_mesh) | set(b_mesh)):
        a, b = a_mesh.get(mesh_id), b_mesh.get(mesh_id)
        if a is None or b is None or len(a) != len(b):
            raise LivingCompileError(f"motion bake mesh identity drift: {mesh_id}")
        meshes[mesh_id] = [[(1.0 - u) * pa[0] + u * pb[0], (1.0 - u) * pa[1] + u * pb[1]] for pa, pb in zip(a, b)]
    orders = _frame_orders(before if u < 0.5 else after)
    return {
        "schema_version": "RealSaS.LivingCompileRuntimeFrame.v1",
        "frame": {"mesh_vertices_by_id": meshes, "render_order_by_view": orders, "control_states": {}},
        "session": {"clip_id": clip_id, "time_seconds": t, "duration_seconds": duration, "loop": loop},
        "authority": {"source": "QUALIFICATION_OWNED_MOTION_BAKE", "bake_hash": bake.get("bake_hash")},
    }
