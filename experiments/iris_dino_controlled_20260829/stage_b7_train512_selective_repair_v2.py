#!/usr/bin/env python3
"""RealSaS Stage-B7 TRAIN512 selective repair renderer V2.

ZERO-DINO-OPTIMIZER / STAGING-ONLY / MASTER-IMMUTABLE.

Authority:
- Exact historical production renderer:
  reports/post_corpus_audit/post_corpus_stage_b7_atomic_geometry_and_A_rerender_v1.py
  SHA-256 a9481bc609108b616163a6bc824bd927e93813fafc422cc06c294a58b01b9ca6
- Historical renderer nvdiffrast authority: tag v0.4.0, commit
  253ac4fcea7de5f396371124af597e6cc957bfae.

Workflow:
1. Verify source/data/runtime authority.
2. Re-render one unchanged historical sentinel using render primitives copied from
   the historical Stage-B7 production script.
3. Compare all 8 views against persisted master output.
4. Only if sentinel parity passes, render exactly the nine repaired TRAIN512
   members from frozen Stage-B6 normalized.npz into separate staging.
5. Audit staged outputs. NEVER mutate master/assets, variants, ledgers or exports.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
import nvdiffrast.torch as dr
from PIL import Image

SCHEMA = "RealSaS.StageB7Train512SelectiveRepair.v2"
BUILD_ID = "REALSAS_STAGE_B7_TRAIN512_SELECTIVE_REPAIR_V2_20260829"
HISTORICAL_STAGE_B7_SHA256 = "a9481bc609108b616163a6bc824bd927e93813fafc422cc06c294a58b01b9ca6"
NVDIFFRAST_TAG = "v0.4.0"
NVDIFFRAST_COMMIT = "253ac4fcea7de5f396371124af597e6cc957bfae"
NATIVE_RES = 1024
VIEWS = 8
DERIVE_512 = True
DEFAULT_SENTINEL = "asset_d7d4192f9dac8b146a17bc41"
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

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)

def canonical_json_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

# ---------------------------------------------------------------------------
# Historical Stage-B7 production render primitives.
# Kept semantically/code-line equivalent to the 2026-08-23 authority script.
# ---------------------------------------------------------------------------

def palette_from_id(asset_id):
    h = int(sha256_text(asset_id)[:8], 16) / 0xffffffff
    s = .45 + .25 * ((int(sha256_text(asset_id)[8:12], 16) % 1000) / 999)
    v = .78
    return np.asarray(colorsys.hsv_to_rgb(h, s, v), np.float32)

def camera_basis(yaw_deg):
    t = math.radians(yaw_deg)
    return (
        np.array([math.cos(t), -math.sin(t), 0], np.float32),
        np.array([0, 0, 1], np.float32),
        np.array([math.sin(t), math.cos(t), 0], np.float32),
    )

def compute_half_extent(V):
    mx = 1e-6
    for yaw in np.arange(8) * 45:
        r, u, _ = camera_basis(float(yaw))
        mx = max(mx, float(np.abs(V @ r).max()), float(np.abs(V @ u).max()))
    return mx * 1.08

def make_renderer():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU runtime required for canonical Stage-B7 rendering")
    device = torch.device("cuda")
    rast_ctx = dr.RasterizeCudaContext(device=device)

    def rasterize_top_left(V, Faces, N, yaw_deg, res):
        right, up, forward = camera_basis(yaw_deg)
        x = V @ right
        y = V @ up
        z = V @ forward
        he = compute_half_extent(V)
        zmin, zmax = float(z.min()), float(z.max())
        zr = max(zmax - zmin, 1e-6)
        clip = np.stack(
            [x / he, y / he, 2 * (z - zmin) / zr - 1, np.ones_like(z)], axis=1
        ).astype(np.float32)
        pos = torch.from_numpy(clip).to(device)[None]
        tri = torch.from_numpy(Faces.astype(np.int32)).to(device)
        rast, _ = dr.rasterize(rast_ctx, pos, tri, (res, res))
        rast = torch.flip(rast, dims=[1]).contiguous()
        tid = rast[0, :, :, 3].to(torch.int32) - 1
        mask = tid >= 0
        ni, _ = dr.interpolate(torch.from_numpy(N).to(device)[None], rast, tri)
        ni = Fnn.normalize(ni[0], dim=-1, eps=1e-8)
        return rast, tid, mask, ni, {
            "right": right,
            "up": up,
            "forward": forward,
            "half_extent": he,
            "z_min": zmin,
            "z_max": zmax,
        }

    def render_view(asset_id, V, Faces, N, yaw_deg, res, od):
        od.mkdir(parents=True, exist_ok=True)
        rast, tid, mask, ni, cam0 = rasterize_top_left(V, Faces, N, yaw_deg, res)
        light = Fnn.normalize(
            torch.tensor([.35, -.45, .82], device=device, dtype=torch.float32), dim=0
        )
        ndl = torch.clamp((ni * light).sum(-1), 0, 1)
        band = torch.where(ndl > .72, 1.0, torch.where(ndl > .35, .72, .48))
        base = torch.tensor(palette_from_id(asset_id), device=device)
        rgb = torch.clamp(base[None, None, :] * band[..., None], 0, 1)
        rgb = torch.where(mask[..., None], rgb, torch.zeros_like(rgb))
        m = mask.float()[None, None]
        eroded = 1 - Fnn.max_pool2d(1 - m, 3, 1, 1)
        edge = (m * (1 - eroded))[0, 0] > .5
        clean = torch.cat([rgb, mask.float()[..., None]], -1)
        ink = torch.cat(
            [torch.where(edge[..., None], rgb * .10, rgb), mask.float()[..., None]], -1
        )

        def save(t, p):
            Image.fromarray((torch.clamp(t, 0, 1) * 255).byte().cpu().numpy()).save(
                p, compress_level=6
            )

        save(clean, od / "cel_clean.png")
        save(ink, od / "ink_cel.png")

        ys, xs = torch.where(mask)
        pix = (ys * res + xs).to(torch.int32)
        tri_id = tid[ys, xs]
        uv = rast[0, ys, xs, :2].float()
        np.savez_compressed(
            od / "raster_authority.npz",
            pixel_linear_index=pix.cpu().numpy(),
            triangle_id=tri_id.cpu().numpy(),
            barycentric_uv=uv.cpu().numpy().astype(np.float32),
            resolution=np.asarray([res, res], np.int32),
            origin=np.asarray(["TOP_LEFT"]),
        )
        cam = {
            "contract": "realsas.level_orthographic_z_orbit.v1",
            "image_origin": "TOP_LEFT",
            "image_y_direction": "DOWN",
            "ndc_y_direction": "UP",
            "vertical_flip_count": 1,
            "yaw_deg": float(yaw_deg),
            "screen_up": [0, 0, 1],
            "right": cam0["right"].tolist(),
            "up": cam0["up"].tolist(),
            "forward": cam0["forward"].tolist(),
            "half_extent": float(cam0["half_extent"]),
            "z_min": cam0["z_min"],
            "z_max": cam0["z_max"],
            "semantic_facing": "UNKNOWN",
        }
        atomic_json(od / "camera.json", cam)
        if DERIVE_512:
            for n in ["cel_clean.png", "ink_cel.png"]:
                with Image.open(od / n) as im:
                    im.resize((512, 512), Image.Resampling.LANCZOS).save(
                        od / (Path(n).stem + "_512.png")
                    )

    def render_asset(asset_id, npz, rr):
        with np.load(npz, allow_pickle=False) as d:
            V = np.asarray(d["vertices"])
            F = np.asarray(d["faces"])
            N = np.asarray(d["vertex_normals"])
        for vi in range(VIEWS):
            render_view(asset_id, V, F, N, vi * 45, NATIVE_RES, rr / f"V{vi}")

    return device, render_asset

def validate_asset(stage_asset: Path):
    with np.load(stage_asset / "primary_geometry.npz", allow_pickle=False) as d:
        V = np.asarray(d["vertices"])
        F = np.asarray(d["faces"])
    total = 0
    repro = []
    for vi in range(VIEWS):
        vd = stage_asset / "renders" / f"V{vi}"
        cam = json.loads((vd / "camera.json").read_text())
        if (
            cam.get("image_origin") != "TOP_LEFT"
            or int(cam.get("vertical_flip_count", -1)) != 1
            or float(cam.get("yaw_deg")) != vi * 45
        ):
            raise RuntimeError(f"camera contract fail {vi}")
        for fn in [
            "cel_clean.png", "ink_cel.png", "cel_clean_512.png", "ink_cel_512.png",
            "raster_authority.npz", "camera.json",
        ]:
            p = vd / fn
            if not p.exists() or p.stat().st_size <= 0:
                raise RuntimeError("missing " + str(p))
        with np.load(vd / "raster_authority.npz", allow_pickle=False) as z:
            pix = np.asarray(z["pixel_linear_index"], np.int64)
            tid = np.asarray(z["triangle_id"], np.int64)
            uv = np.asarray(z["barycentric_uv"], np.float64)
        if len(pix) != len(np.unique(pix)):
            raise RuntimeError("duplicate authority pixels")
        if len(tid) and (tid.min() < 0 or tid.max() >= len(F)):
            raise RuntimeError("triangle id out of bounds")
        if not np.isfinite(uv).all():
            raise RuntimeError("nonfinite barycentric")
        alpha = np.asarray(Image.open(vd / "cel_clean.png"))[..., 3] > 0
        auth = np.zeros(NATIVE_RES * NATIVE_RES, dtype=bool)
        auth[pix] = True
        auth = auth.reshape(NATIVE_RES, NATIVE_RES)
        if not np.array_equal(alpha, auth):
            raise RuntimeError("alpha/authority parity mismatch")
        total += len(pix)
        if len(pix):
            ys = pix // NATIVE_RES
            xs = pix % NATIVE_RES
            w = np.stack([uv[:, 0], uv[:, 1], 1 - uv[:, 0] - uv[:, 1]], 1)
            P = (V[F[tid]] * w[:, :, None]).sum(1)
            right = np.asarray(cam["right"])
            up = np.asarray(cam["up"])
            he = float(cam["half_extent"])
            ndcx = (P @ right) / he
            ndcy = (P @ up) / he
            px = (ndcx + 1) * NATIVE_RES / 2 - .5
            py = (1 - ndcy) * NATIVE_RES / 2 - .5
            err = np.sqrt((px - xs) ** 2 + (py - ys) ** 2)
            repro.append(float(np.quantile(err, .95)))
    if total <= 0:
        raise RuntimeError("all8 blank after repair")
    if max(repro or [0]) > .05:
        raise RuntimeError("reprojection p95 too high: " + str(max(repro)))
    return {
        "foreground_authority_rows_total": int(total),
        "reprojection_p95_px_max": float(max(repro or [0])),
    }

def compare_images(a: Path, b: Path):
    x = np.asarray(Image.open(a).convert("RGBA"), dtype=np.uint8)
    y = np.asarray(Image.open(b).convert("RGBA"), dtype=np.uint8)
    if x.shape != y.shape:
        return {"shape_equal": False, "exact": False}
    d = np.abs(x.astype(np.int16) - y.astype(np.int16))
    return {
        "shape_equal": True,
        "exact": bool(np.array_equal(x, y)),
        "mae": float(d.mean()),
        "max_abs": int(d.max()),
    }

def load_raster(p: Path):
    with np.load(p, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}

def camera_parity(oldp: Path, newp: Path):
    a = json.loads(oldp.read_text())
    b = json.loads(newp.read_text())
    discrete = [
        "contract", "image_origin", "image_y_direction", "ndc_y_direction",
        "vertical_flip_count", "yaw_deg", "screen_up", "semantic_facing",
    ]
    disc_ok = all(a.get(k) == b.get(k) for k in discrete)
    numeric_fields = ["right", "up", "forward"]
    vec_max = 0.0
    for k in numeric_fields:
        vec_max = max(
            vec_max,
            float(np.max(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64))))
        )
    scalar_max = max(
        abs(float(a[k]) - float(b[k]))
        for k in ["half_extent", "z_min", "z_max"]
    )
    return {
        "discrete_exact": bool(disc_ok),
        "vector_max_abs": float(vec_max),
        "scalar_max_abs": float(scalar_max),
        "pass": bool(disc_ok and vec_max <= 1e-7 and scalar_max <= 1e-7),
    }

def sentinel_parity(master: Path, sentinel: str, scratch: Path, render_asset):
    old_asset = master / "master" / "assets" / sentinel
    if not old_asset.is_dir():
        raise RuntimeError(f"missing sentinel asset: {old_asset}")
    geom = old_asset / "primary_geometry.npz"
    replay = scratch / sentinel
    shutil.rmtree(replay, ignore_errors=True)
    (replay / "renders").mkdir(parents=True, exist_ok=True)
    shutil.copy2(geom, replay / "primary_geometry.npz")
    render_asset(sentinel, geom, replay / "renders")
    _ = validate_asset(replay)

    views = []
    hard = True
    for vi in range(VIEWS):
        ov = old_asset / "renders" / f"V{vi}"
        nv = replay / "renders" / f"V{vi}"
        ro, rn = load_raster(ov / "raster_authority.npz"), load_raster(nv / "raster_authority.npz")
        support = np.array_equal(ro["pixel_linear_index"], rn["pixel_linear_index"])
        tri = support and np.array_equal(ro["triangle_id"], rn["triangle_id"])
        bary_max = None
        if tri:
            bary_max = (
                float(np.max(np.abs(ro["barycentric_uv"].astype(np.float64) -
                                    rn["barycentric_uv"].astype(np.float64))))
                if len(ro["barycentric_uv"]) else 0.0
            )
        cam = camera_parity(ov / "camera.json", nv / "camera.json")
        cel = compare_images(ov / "cel_clean.png", nv / "cel_clean.png")
        ink = compare_images(ov / "ink_cel.png", nv / "ink_cel.png")
        cel512 = compare_images(ov / "cel_clean_512.png", nv / "cel_clean_512.png")
        ink512 = compare_images(ov / "ink_cel_512.png", nv / "ink_cel_512.png")
        vp = bool(
            support and tri and bary_max is not None and bary_max <= 5e-5
            and cam["pass"]
            and cel["exact"] and ink["exact"] and cel512["exact"] and ink512["exact"]
        )
        hard &= vp
        rec = {
            "view": vi,
            "support_exact": bool(support),
            "triangle_id_exact": bool(tri),
            "barycentric_max_abs": bary_max,
            "camera": cam,
            "cel": cel,
            "ink": ink,
            "cel512": cel512,
            "ink512": ink512,
            "pass": vp,
        }
        views.append(rec)
        print(
            f"[B7] sentinel V{vi}: {'PASS' if vp else 'FAIL'} "
            f"support={support} tri={tri} bary_max={bary_max} "
            f"cel={cel['exact']} ink={ink['exact']} "
            f"cel512={cel512['exact']} ink512={ink512['exact']}",
            flush=True,
        )
    return {"sentinel": sentinel, "pass": bool(hard), "views": views}

def verify_runtime_authority(master: Path):
    auth = master / "reports" / "post_corpus_audit" / "post_corpus_stage_b7_atomic_geometry_and_A_rerender_v1.py"
    if not auth.is_file():
        raise RuntimeError("historical Stage-B7 authority script missing")
    got = sha256_file(auth)
    if got != HISTORICAL_STAGE_B7_SHA256:
        raise RuntimeError(f"historical Stage-B7 SHA drift {got}")
    try:
        nvver = importlib.metadata.version("nvdiffrast")
    except Exception as e:
        raise RuntimeError("nvdiffrast package metadata unavailable") from e
    if nvver != "0.4.0":
        raise RuntimeError(f"nvdiffrast version drift expected=0.4.0 actual={nvver}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    return {
        "historical_stage_b7_sha256": got,
        "nvdiffrast_version": nvver,
        "nvdiffrast_tag": NVDIFFRAST_TAG,
        "nvdiffrast_commit": NVDIFFRAST_COMMIT,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }

def run(args):
    master = Path(args.corpus_root).resolve()
    staging = Path(args.staging_root).resolve()
    b6_path = master / "reports" / "post_corpus_audit" / "POST_CORPUS_STAGE_B6_FROZEN_REPAIR_STAGING_RESULT_V1.json"

    print("[B7] ZERO-DINO-OPTIMIZER / STAGING-ONLY", flush=True)
    print("[B7] corpus:", master, flush=True)
    print("[B7] staging:", staging, flush=True)
    staging.mkdir(parents=True, exist_ok=True)

    runtime = verify_runtime_authority(master)
    print("[B7] authority SHA: PASS", runtime["historical_stage_b7_sha256"], flush=True)
    print("[B7] nvdiffrast:", runtime["nvdiffrast_version"], NVDIFFRAST_COMMIT, flush=True)
    print("[B7] GPU:", runtime["gpu"], flush=True)

    b6 = json.loads(b6_path.read_text())
    rows = {x["asset"]: x for x in b6["repair_artifacts"]}
    if len(rows) != 56:
        raise RuntimeError(f"B6 repair authority count drift: {len(rows)}")
    missing = sorted(set(REPAIR_ASSETS) - set(rows))
    if missing:
        raise RuntimeError(f"TRAIN512 repair assets missing from B6: {missing}")

    for aid in REPAIR_ASSETS:
        row = rows[aid]
        geom = master / row["normalized_path"]
        rec = geom.parent / "repair_record.json"
        if not geom.is_file() or sha256_file(geom) != row["normalized_sha256"]:
            raise RuntimeError(f"{aid}: frozen normalized SHA mismatch")
        if not rec.is_file():
            raise RuntimeError(f"{aid}: repair record missing")
    print("[B7] repair9 authority: 9/9 PASS", flush=True)

    _, render_asset = make_renderer()

    sentinel_root = staging / "_sentinel_replay"
    sent = sentinel_parity(master, args.sentinel, sentinel_root, render_asset)
    atomic_json(staging / "SENTINEL_PARITY_V2.json", sent)

    result = {
        "schema": SCHEMA,
        "build_id": BUILD_ID,
        "runtime": runtime,
        "scientific_optimizer_steps": 0,
        "master_mutated": False,
        "sentinel": sent,
        "assets": [],
        "status": "RUNNING",
    }

    if not sent["pass"]:
        result["status"] = "FAIL_SENTINEL_PARITY__NO_REPAIR_ASSET_RENDERED"
        result["content_sha256"] = canonical_json_sha(result)
        atomic_json(staging / "STAGE_B7_RESULT_V2.json", result)
        print("[B7] FAIL_SENTINEL_PARITY__NO_REPAIR_ASSET_RENDERED", flush=True)
        raise SystemExit(3)

    print("[B7] SENTINEL 8/8 PASS — opening exact 9-asset staging route", flush=True)
    assets_out = staging / "staged_assets"
    shutil.rmtree(assets_out, ignore_errors=True)
    assets_out.mkdir(parents=True, exist_ok=True)

    all_pass = True
    for idx, aid in enumerate(REPAIR_ASSETS, 1):
        row = rows[aid]
        geom = master / row["normalized_path"]
        rec = geom.parent / "repair_record.json"
        out = assets_out / aid
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(geom, out / "primary_geometry.npz")
        render_asset(aid, geom, out / "renders")
        shutil.copy2(rec, out / "STAGE_B6_REPAIR_RECORD.json")
        metrics = validate_asset(out)
        ok = sha256_file(out / "primary_geometry.npz") == row["normalized_sha256"]
        asset_rec = {
            "asset": aid,
            "pass": bool(ok),
            "normalized_sha256": row["normalized_sha256"],
            "staged_geometry_sha256": sha256_file(out / "primary_geometry.npz"),
            "repair_record_sha256": sha256_file(out / "STAGE_B6_REPAIR_RECORD.json"),
            "render_metrics": metrics,
        }
        atomic_json(out / "STAGE_B7_STAGING_RECORD_V2.json", asset_rec)
        result["assets"].append(asset_rec)
        all_pass &= bool(ok)
        print(
            f"[B7] repair {idx}/9 {aid}: {'PASS' if ok else 'FAIL'} "
            f"rows={metrics['foreground_authority_rows_total']} "
            f"reproj_p95={metrics['reprojection_p95_px_max']:.6f}px",
            flush=True,
        )
        torch.cuda.empty_cache()

    result["staged_asset_count"] = len(result["assets"])
    result["pass_asset_count"] = sum(int(x["pass"]) for x in result["assets"])
    result["status"] = (
        "PASS_STAGING_ONLY__MASTER_UNCHANGED"
        if all_pass and result["pass_asset_count"] == 9
        else "FAIL_STAGING_AUDIT__MASTER_UNCHANGED"
    )
    result["content_sha256"] = canonical_json_sha(result)
    atomic_json(staging / "STAGE_B7_RESULT_V2.json", result)

    print("[B7] FINAL:", result["status"], flush=True)
    print("[B7] sentinel_pass:", sent["pass"], flush=True)
    print("[B7] staged_asset_count:", result["staged_asset_count"], flush=True)
    print("[B7] pass_asset_count:", result["pass_asset_count"], flush=True)
    print("[B7] master_mutated: false", flush=True)
    print("[B7] scientific_optimizer_steps: 0", flush=True)
    print("[B7] result:", staging / "STAGE_B7_RESULT_V2.json", flush=True)
    if result["status"] != "PASS_STAGING_ONLY__MASTER_UNCHANGED":
        raise SystemExit(4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus-root",
        default="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3",
    )
    ap.add_argument(
        "--staging-root",
        default="/content/drive/MyDrive/RealSaS_MASTER_CORPUS_1024_V3/reports/post_corpus_audit/B7_TRAIN512_SELECTIVE_REPAIR_STAGING_V2",
    )
    ap.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    args = ap.parse_args()
    run(args)

if __name__ == "__main__":
    main()
