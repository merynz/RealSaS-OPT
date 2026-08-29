#!/usr/bin/env python3
"""RealSaS Stage-B7 selective native-1024 repair publication renderer.

This is a ZERO-DINO-OPTIMIZER data-plane utility.

Fail-closed workflow:
  1. Re-render one unchanged historical sentinel from its existing
     primary_geometry.npz and compare against persisted production bytes/arrays.
  2. Only if sentinel parity passes, render the exact nine Stage-B6 repair
     assets from their frozen normalized.npz into a separate staging tree.
  3. Audit staging. This script NEVER mutates master/assets.

Promotion from staging into master is intentionally a separate action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

try:
    import nvdiffrast.torch as dr
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "nvdiffrast is required. Install the preregistered revision before running: "
        "pip install 'git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'"
    ) from exc

SCHEMA = "RealSaS.StageB7SelectiveRepairRenderer.v1"
BUILD_ID = "REALSAS_STAGE_B7_TRAIN512_REPAIR_PUBLICATION_20260829"
NATIVE = 1024
DERIVED = 512
NVDIFFRAST_EXPECTED_REV = "253ac4fcea7de5f396371124af597e6cc957bfae"
LIGHT = np.asarray([0.35, -0.45, 0.82], dtype=np.float64)
LIGHT /= np.linalg.norm(LIGHT)
DARK_THRESHOLD = 0.35
LIGHT_THRESHOLD = 0.72
YAW_DEG = tuple(range(0, 360, 45))
REPAIR_ASSETS = (
    "asset_1d6d3b17fe506463b8840c51",
    "asset_4970fb8cc7c69df071970dc0",
    "asset_66c8c63c7a97e830b89d0ba8",
    "asset_674fb6e5571ca86dd0e5c858",
    "asset_681fb76277c10d2307f7509f",
    "asset_88fe0971a994597f3866473b",
    "asset_a873bb17b9a66e6e980845a5",
    "asset_ce3577373d7cc71b7025cc74",
    "asset_f56aff9ecf13f7ddf0afcd55",
)
DEFAULT_SENTINEL = "asset_d7d4192f9dac8b146a17bc41"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_geometry(path: Path):
    z = np.load(path, allow_pickle=False)
    required = {"vertices", "faces", "vertex_normals"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(f"{path}: missing geometry keys {sorted(missing)}")
    vertices = np.asarray(z["vertices"], dtype=np.float32)
    faces = np.asarray(z["faces"], dtype=np.int32)
    normals = np.asarray(z["vertex_normals"], dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"{path}: invalid vertices shape {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"{path}: invalid faces shape {faces.shape}")
    if normals.shape != vertices.shape:
        raise RuntimeError(f"{path}: invalid normals shape {normals.shape}")
    if not np.isfinite(vertices).all() or not np.isfinite(normals).all():
        raise RuntimeError(f"{path}: non-finite geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise RuntimeError(f"{path}: face index out of range")
    return vertices, faces, normals


def basis(yaw_deg: float):
    t = math.radians(yaw_deg)
    forward = np.asarray([math.sin(t), math.cos(t), 0.0], dtype=np.float64)
    right = np.asarray([math.cos(t), -math.sin(t), 0.0], dtype=np.float64)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return forward, right, up


def asset_half_extent(vertices: np.ndarray) -> float:
    v = vertices.astype(np.float64, copy=False)
    max_extent = 0.0
    for yaw in YAW_DEG:
        _, right, up = basis(yaw)
        max_extent = max(max_extent, float(np.max(np.abs(v @ right))))
        max_extent = max(max_extent, float(np.max(np.abs(v @ up))))
    return float(max_extent * 1.08)


def extract_palette(existing_asset_dir: Path) -> np.ndarray:
    colors = set()
    for k in range(8):
        p = existing_asset_dir / "renders" / f"V{k}" / "cel_clean.png"
        if not p.is_file():
            raise RuntimeError(f"missing old palette authority: {p}")
        a = np.asarray(Image.open(p).convert("RGBA"), dtype=np.uint8)
        fg = a[..., 3] > 0
        for rgb in np.unique(a[..., :3][fg], axis=0):
            colors.add(tuple(int(x) for x in rgb))
    if len(colors) != 3:
        raise RuntimeError(f"expected exactly 3 native cel colors, got {len(colors)}: {sorted(colors)}")
    palette = np.asarray(sorted(colors, key=lambda c: (sum(c), c)), dtype=np.uint8)
    return palette


def make_clip(vertices: np.ndarray, yaw: float, half_extent: float):
    forward, right, up = basis(yaw)
    v = vertices.astype(np.float64, copy=False)
    x = (v @ right) / half_extent
    y = (v @ up) / half_extent
    # Canonical geometry is normalized well inside OpenGL clip-z. Preserve the
    # camera-forward coordinate as affine depth so smaller P·forward is nearer.
    z = v @ forward
    w = np.ones_like(z)
    pos = np.stack([x, y, z, w], axis=1).astype(np.float32)
    return pos, forward, right, up, float(z.min()), float(z.max())


def rasterize_view(ctx, vertices, faces, normals, yaw, half_extent):
    pos_np, forward, right, up, z_min, z_max = make_clip(vertices, yaw, half_extent)
    pos = torch.as_tensor(pos_np[None], dtype=torch.float32, device="cuda").contiguous()
    tri = torch.as_tensor(faces, dtype=torch.int32, device="cuda").contiguous()
    attr = torch.as_tensor(normals[None], dtype=torch.float32, device="cuda").contiguous()

    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[NATIVE, NATIVE], grad_db=False)
    nimg, _ = dr.interpolate(attr, rast, tri)

    # nvdiffrast follows OpenGL image-y convention; production files are top-left.
    rast_np = rast[0].detach().cpu().numpy()[::-1].copy()
    n_np = nimg[0].detach().cpu().numpy()[::-1].copy()

    tri_one_based = np.rint(rast_np[..., 3]).astype(np.int64)
    fg = tri_one_based > 0
    tri_id = tri_one_based[fg].astype(np.int64) - 1
    bary = rast_np[..., :2][fg].astype(np.float32)
    linear = np.flatnonzero(fg.reshape(-1)).astype(np.int64)

    nn = n_np.astype(np.float64)
    norm = np.linalg.norm(nn, axis=2, keepdims=True)
    nn = np.divide(nn, np.maximum(norm, 1e-20), out=np.zeros_like(nn), where=norm > 0)
    score = np.einsum("hwc,c->hw", nn, LIGHT)

    return {
        "fg": fg,
        "pixel_linear_index": linear,
        "triangle_id": tri_id,
        "barycentric_uv": bary,
        "score": score,
        "forward": forward,
        "right": right,
        "up": up,
        "z_min": z_min,
        "z_max": z_max,
    }


def render_rgba(raster, palette: np.ndarray):
    fg = raster["fg"]
    score = raster["score"]
    rgba = np.zeros((NATIVE, NATIVE, 4), dtype=np.uint8)
    band = np.zeros((NATIVE, NATIVE), dtype=np.uint8)
    band[(score >= DARK_THRESHOLD) & (score < LIGHT_THRESHOLD)] = 1
    band[score >= LIGHT_THRESHOLD] = 2
    rgba[..., :3][fg] = palette[band[fg]]
    rgba[..., 3][fg] = 255

    ink = rgba.copy()
    interior = ndi.binary_erosion(fg, structure=np.ones((3, 3), dtype=bool), iterations=1, border_value=0)
    outline = fg & ~interior
    ink[..., :3][outline] = np.floor(ink[..., :3][outline].astype(np.float64) * 0.1).astype(np.uint8)
    return rgba, ink


def save_png_pair(view_dir: Path, cel: np.ndarray, ink: np.ndarray):
    view_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cel, mode="RGBA").save(view_dir / "cel_clean.png")
    Image.fromarray(ink, mode="RGBA").save(view_dir / "ink_cel.png")
    Image.fromarray(cel, mode="RGBA").resize((DERIVED, DERIVED), Image.Resampling.LANCZOS).save(view_dir / "cel_clean_512.png")
    Image.fromarray(ink, mode="RGBA").resize((DERIVED, DERIVED), Image.Resampling.LANCZOS).save(view_dir / "ink_cel_512.png")


def save_raster(path: Path, r):
    np.savez_compressed(
        path,
        pixel_linear_index=r["pixel_linear_index"],
        triangle_id=r["triangle_id"],
        barycentric_uv=r["barycentric_uv"],
        resolution=np.asarray([NATIVE, NATIVE], dtype=np.int64),
        origin=np.asarray("TOP_LEFT"),
    )


def camera_object(yaw, r, half_extent):
    return {
        "contract": "realsas.level_orthographic_z_orbit.v1",
        "forward": [float(x) for x in r["forward"]],
        "half_extent": float(half_extent),
        "image_origin": "TOP_LEFT",
        "image_y_direction": "DOWN",
        "ndc_y_direction": "UP",
        "right": [float(x) for x in r["right"]],
        "screen_up": [0, 0, 1],
        "semantic_facing": "UNKNOWN",
        "up": [float(x) for x in r["up"]],
        "vertical_flip_count": 1,
        "yaw_deg": float(yaw),
        "z_max": float(r["z_max"]),
        "z_min": float(r["z_min"]),
    }


def render_asset(geometry_path: Path, existing_asset_dir: Path, out_asset_dir: Path, ctx):
    vertices, faces, normals = load_geometry(geometry_path)
    palette = extract_palette(existing_asset_dir)
    h = asset_half_extent(vertices)
    renders = out_asset_dir / "renders"
    out_asset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(geometry_path, out_asset_dir / "primary_geometry.npz")

    view_records = []
    for k, yaw in enumerate(YAW_DEG):
        vd = renders / f"V{k}"
        vd.mkdir(parents=True, exist_ok=True)
        r = rasterize_view(ctx, vertices, faces, normals, yaw, h)
        if len(r["pixel_linear_index"]) == 0:
            raise RuntimeError(f"{out_asset_dir.name} V{k}: blank native raster")
        cel, ink = render_rgba(r, palette)
        save_png_pair(vd, cel, ink)
        save_raster(vd / "raster_authority.npz", r)
        cam = camera_object(yaw, r, h)
        (vd / "camera.json").write_text(json.dumps(cam, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        view_records.append({
            "view": k,
            "yaw_deg": yaw,
            "visible_pixels": int(len(r["pixel_linear_index"])),
            "camera_sha256": sha256_file(vd / "camera.json"),
            "raster_sha256": sha256_file(vd / "raster_authority.npz"),
            "cel_sha256": sha256_file(vd / "cel_clean.png"),
            "ink_sha256": sha256_file(vd / "ink_cel.png"),
            "cel512_sha256": sha256_file(vd / "cel_clean_512.png"),
            "ink512_sha256": sha256_file(vd / "ink_cel_512.png"),
        })

    complete = {
        "schema": SCHEMA,
        "build_id": BUILD_ID,
        "canonical_asset_id": out_asset_dir.name,
        "native_resolution": NATIVE,
        "views": 8,
        "geometry_sha256": sha256_file(out_asset_dir / "primary_geometry.npz"),
        "palette_dark_mid_light": palette.tolist(),
        "half_extent": h,
        "view_records": view_records,
        "scientific_optimizer_steps": 0,
    }
    complete["content_sha256"] = canonical_json_sha(complete)
    (out_asset_dir / "RENDER_COMPLETE.json").write_text(json.dumps(complete, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return complete


def load_raster(path: Path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def compare_images(a: Path, b: Path):
    x = np.asarray(Image.open(a).convert("RGBA"), dtype=np.int16)
    y = np.asarray(Image.open(b).convert("RGBA"), dtype=np.int16)
    if x.shape != y.shape:
        return {"shape_equal": False}
    d = np.abs(x - y)
    return {
        "shape_equal": True,
        "mae": float(d.mean()),
        "max_abs": int(d.max()),
        "exact": bool(np.array_equal(x, y)),
    }


def sentinel_parity(master_root: Path, sentinel: str, scratch: Path, ctx):
    old_asset = master_root / "master" / "assets" / sentinel
    if not old_asset.is_dir():
        raise RuntimeError(f"missing sentinel master asset: {old_asset}")
    geom = old_asset / "primary_geometry.npz"
    out = scratch / sentinel
    if out.exists():
        shutil.rmtree(out)
    render_asset(geom, old_asset, out, ctx)

    per_view = []
    hard_pass = True
    for k in range(8):
        old_v = old_asset / "renders" / f"V{k}"
        new_v = out / "renders" / f"V{k}"
        ro = load_raster(old_v / "raster_authority.npz")
        rn = load_raster(new_v / "raster_authority.npz")
        support_exact = np.array_equal(ro["pixel_linear_index"], rn["pixel_linear_index"])
        tri_exact = support_exact and np.array_equal(ro["triangle_id"], rn["triangle_id"])
        bary_max = None
        if tri_exact:
            bary_max = float(np.max(np.abs(ro["barycentric_uv"].astype(np.float64) - rn["barycentric_uv"].astype(np.float64)))) if len(ro["barycentric_uv"]) else 0.0
        cel = compare_images(old_v / "cel_clean.png", new_v / "cel_clean.png")
        ink = compare_images(old_v / "ink_cel.png", new_v / "ink_cel.png")
        cel512 = compare_images(old_v / "cel_clean_512.png", new_v / "cel_clean_512.png")
        ink512 = compare_images(old_v / "ink_cel_512.png", new_v / "ink_cel_512.png")
        vp = bool(
            support_exact and tri_exact and bary_max is not None and bary_max <= 5e-5
            and cel.get("exact", False) and ink.get("exact", False)
            and cel512.get("exact", False) and ink512.get("exact", False)
        )
        hard_pass &= vp
        per_view.append({
            "view": k,
            "support_exact": support_exact,
            "triangle_id_exact": tri_exact,
            "barycentric_max_abs": bary_max,
            "cel": cel,
            "ink": ink,
            "cel512": cel512,
            "ink512": ink512,
            "pass": vp,
        })
    return {"sentinel": sentinel, "pass": hard_pass, "views": per_view}


def verify_repair_record(repair_asset_dir: Path):
    rec_path = repair_asset_dir / "repair_record.json"
    geom = repair_asset_dir / "normalized.npz"
    if not rec_path.is_file() or not geom.is_file():
        raise RuntimeError(f"missing frozen Stage-B6 repair files: {repair_asset_dir}")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    observed = sha256_file(geom)
    expected = rec.get("normalized_sha256")
    if expected != observed:
        raise RuntimeError(f"{repair_asset_dir.name}: normalized SHA mismatch {observed} != {expected}")
    if rec.get("asset") != repair_asset_dir.name:
        raise RuntimeError(f"{repair_asset_dir.name}: repair record asset mismatch")
    return rec, geom


def audit_staged_asset(asset_dir: Path, expected_geometry_sha: str):
    errors = []
    geom = asset_dir / "primary_geometry.npz"
    if not geom.is_file() or sha256_file(geom) != expected_geometry_sha:
        errors.append("geometry_sha")
    for k in range(8):
        vd = asset_dir / "renders" / f"V{k}"
        req = ["camera.json", "raster_authority.npz", "cel_clean.png", "ink_cel.png", "cel_clean_512.png", "ink_cel_512.png"]
        for name in req:
            if not (vd / name).is_file():
                errors.append(f"V{k}:{name}:missing")
        if errors:
            continue
        r = load_raster(vd / "raster_authority.npz")
        if len(r["pixel_linear_index"]) == 0:
            errors.append(f"V{k}:blank")
            continue
        cel = np.asarray(Image.open(vd / "cel_clean.png").convert("RGBA"), dtype=np.uint8)
        ink = np.asarray(Image.open(vd / "ink_cel.png").convert("RGBA"), dtype=np.uint8)
        fg = np.zeros(NATIVE * NATIVE, dtype=bool)
        fg[r["pixel_linear_index"].astype(np.int64)] = True
        fg = fg.reshape(NATIVE, NATIVE)
        if not np.array_equal(cel[..., 3] > 0, fg):
            errors.append(f"V{k}:cel_alpha_raster_support")
        if not np.array_equal(ink[..., 3], cel[..., 3]):
            errors.append(f"V{k}:ink_alpha")
        c512 = np.asarray(Image.open(vd / "cel_clean.png").convert("RGBA").resize((DERIVED, DERIVED), Image.Resampling.LANCZOS), dtype=np.uint8)
        s512 = np.asarray(Image.open(vd / "cel_clean_512.png").convert("RGBA"), dtype=np.uint8)
        if not np.array_equal(c512, s512):
            errors.append(f"V{k}:cel512_lanczos")
        i512 = np.asarray(Image.open(vd / "ink_cel.png").convert("RGBA").resize((DERIVED, DERIVED), Image.Resampling.LANCZOS), dtype=np.uint8)
        si512 = np.asarray(Image.open(vd / "ink_cel_512.png").convert("RGBA"), dtype=np.uint8)
        if not np.array_equal(i512, si512):
            errors.append(f"V{k}:ink512_lanczos")
    return {"asset": asset_dir.name, "pass": len(errors) == 0, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", required=True, help="RealSaS_MASTER_CORPUS_1024_V3 root")
    ap.add_argument("--repair-assets-root", required=True, help="Stage-B6 folder containing asset_<id>/normalized.npz")
    ap.add_argument("--staging-root", required=True)
    ap.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    ap.add_argument("--skip-sentinel", action="store_true", help="Forbidden for canonical preflight; debug only")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for canonical nvdiffrast Stage-B7 rendering")

    corpus = Path(args.corpus_root).resolve()
    repairs = Path(args.repair_assets_root).resolve()
    staging = Path(args.staging_root).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    scratch = staging / "_sentinel_scratch"
    ctx = dr.RasterizeCudaContext()

    result = {
        "schema": SCHEMA,
        "build_id": BUILD_ID,
        "nvdiffrast_expected_revision": NVDIFFRAST_EXPECTED_REV,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch_version": torch.__version__,
        "scientific_optimizer_steps": 0,
        "sentinel": None,
        "assets": [],
        "status": "RUNNING",
    }

    if args.skip_sentinel:
        raise SystemExit("--skip-sentinel is not permitted for canonical Stage-B7 preflight")

    sent = sentinel_parity(corpus, args.sentinel, scratch, ctx)
    result["sentinel"] = sent
    (staging / "SENTINEL_PARITY.json").write_text(json.dumps(sent, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not sent["pass"]:
        result["status"] = "FAIL_SENTINEL_PARITY__NO_REPAIR_ASSET_RENDERED"
        (staging / "STAGE_B7_RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("sentinel parity failed; repair staging was not opened")

    assets_out = staging / "staged_assets"
    assets_out.mkdir(parents=True, exist_ok=True)
    all_pass = True
    for asset in REPAIR_ASSETS:
        old_asset = corpus / "master" / "assets" / asset
        repair_asset = repairs / asset
        if not old_asset.is_dir():
            raise RuntimeError(f"missing old master asset {old_asset}")
        rec, geom = verify_repair_record(repair_asset)
        out = assets_out / asset
        if out.exists():
            shutil.rmtree(out)
        render_rec = render_asset(geom, old_asset, out, ctx)
        shutil.copyfile(repair_asset / "repair_record.json", out / "STAGE_B6_REPAIR_RECORD.json")
        audit = audit_staged_asset(out, rec["normalized_sha256"])
        audit["repair_record_sha256"] = sha256_file(repair_asset / "repair_record.json")
        audit["normalized_sha256"] = rec["normalized_sha256"]
        audit["render_complete_sha256"] = sha256_file(out / "RENDER_COMPLETE.json")
        result["assets"].append(audit)
        all_pass &= audit["pass"]

    result["status"] = "PASS_STAGING_ONLY__MASTER_UNCHANGED" if all_pass else "FAIL_STAGING_AUDIT__MASTER_UNCHANGED"
    result["staged_asset_count"] = len(result["assets"])
    result["pass_asset_count"] = sum(int(x["pass"]) for x in result["assets"])
    result["master_mutated"] = False
    result["content_sha256"] = canonical_json_sha(result)
    (staging / "STAGE_B7_RESULT.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "sentinel_pass": sent["pass"],
        "staged_asset_count": result["staged_asset_count"],
        "pass_asset_count": result["pass_asset_count"],
        "master_mutated": False,
    }, sort_keys=True))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
