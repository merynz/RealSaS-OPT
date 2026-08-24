from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from geometry import (
    choose_visible_correspondence,
    geometric_vertex_normals,
    pixel_linear_to_grid,
    reconstruct_surface,
    sha256_file,
)

VIEWS = 8
EXPECTED_STAGE_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStage.v1"
EXPECTED_STAGE_MANIFEST_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStageManifest.v1"
EXPECTED_STAGE_PROFILE = "representation_authority_geometry_only"
CACHE_META_SCHEMA = "RealSaS.IRISSinglePoseV2.CacheAsset.v2"
CACHE_MANIFEST_SCHEMA = "RealSaS.IRISSinglePoseV2.CacheManifest.v2"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path, **arrs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez(tmp, **arrs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_for(asset, tag):
    return int(hashlib.sha256(f"{asset}|{tag}".encode()).hexdigest()[:16], 16) & 0x7FFFFFFF


def sample_rows(n, take, seed):
    take = min(int(take), int(n))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=take, replace=False)) if take else np.zeros(0, np.int64)


def builder_sha256() -> dict[str, str]:
    here = Path(__file__).resolve()
    geometry = here.with_name("geometry.py")
    coords = here.with_name("coords.py")
    return {
        "prepare_representation_cache_v1.py": sha256_file(here),
        "geometry.py": sha256_file(geometry),
        "coords.py": sha256_file(coords),
    }


def cache_input_fingerprint(ar: Path, aid: str, split: str, settings: dict) -> tuple[str, dict]:
    stage_json = ar / "STAGE.json"
    if not stage_json.is_file():
        raise FileNotFoundError(stage_json)
    stage_meta = json.load(stage_json.open(encoding="utf-8"))
    if stage_meta.get("schema") != EXPECTED_STAGE_SCHEMA or stage_meta.get("stage_profile") != EXPECTED_STAGE_PROFILE:
        raise RuntimeError(f"wrong stage profile for representation cache {aid}: {stage_meta.get('schema')} {stage_meta.get('stage_profile')}")
    if stage_meta.get("images_staged") is not False:
        raise RuntimeError(f"representation cache refuses image-bearing stage {aid}")
    dependencies = {
        "stage_sha256": sha256_file(stage_json),
        "builder_sha256": builder_sha256(),
        "asset_id": aid,
        "split": split,
        "settings": settings,
    }
    return canonical_hash(dependencies), dependencies


def build_asset(
    stage_root: Path,
    record,
    out: Path,
    geom_samples=4096,
    anchors_per_view=1024,
    max_tracks=4096,
    radius_px=3,
    max_surface_error=0.003,
):
    aid = record["asset_id"]
    split = str(record["split"]).upper()
    ar = stage_root / "assets" / aid
    cp = out / "truth" / f"{aid}.npz"
    mp = out / "meta" / f"{aid}.json"
    settings = {
        "geom_samples": int(geom_samples),
        "anchors_per_view": int(anchors_per_view),
        "max_tracks": int(max_tracks),
        "radius_px": int(radius_px),
        "max_surface_error": float(max_surface_error),
        "track_xy_authority": "exact_continuous_projection_after_raster_surface_witness",
        "stage_profile": EXPECTED_STAGE_PROFILE,
        "rgb_consumed": False,
    }
    input_fingerprint, input_dependencies = cache_input_fingerprint(ar, aid, split, settings)
    if cp.exists() and mp.exists():
        meta = json.load(mp.open(encoding="utf-8"))
        if (
            meta.get("schema") == CACHE_META_SCHEMA
            and meta.get("input_fingerprint") == input_fingerprint
            and meta.get("truth_sha256") == sha256_file(cp)
            and meta.get("stage_profile") == EXPECTED_STAGE_PROFILE
        ):
            return meta

    with np.load(ar / "primary_geometry.npz", allow_pickle=False) as z:
        if set(z.files) != {"vertices", "faces"}:
            raise RuntimeError(f"IRIS firewall failed {aid}: {z.files}")
        vertices = z["vertices"].astype(np.float32)
        faces = z["faces"].astype(np.int32)
    vn = geometric_vertex_normals(vertices, faces)

    ras = []
    cams = []
    for v in range(VIEWS):
        vd = ar / "renders" / f"V{v}"
        unexpected = [p for p in vd.iterdir() if p.is_file() and p.name not in {"raster_authority.npz", "camera.json"}]
        if unexpected:
            raise RuntimeError(f"representation stage contains unexpected files {aid} V{v}: {[p.name for p in unexpected]}")
        with np.load(vd / "raster_authority.npz", allow_pickle=False) as ra:
            ras.append({k: ra[k] for k in ra.files})
        cams.append(json.load((vd / "camera.json").open(encoding="utf-8")))
        pix = ras[-1]["pixel_linear_index"]
        if len(pix) == 0 or np.any(pix[1:] <= pix[:-1]):
            raise RuntimeError(f"bad raster {aid} V{v}")
    yaws = np.asarray([float(c["yaw_deg"]) for c in cams], np.float32)
    if not np.allclose(yaws, np.arange(8, dtype=np.float32) * 45.0, atol=1e-4):
        raise RuntimeError(f"bad yaws {aid}")

    geom_xy = np.zeros((VIEWS, geom_samples, 2), np.float32)
    geom_p = np.zeros((VIEWS, geom_samples, 3), np.float32)
    geom_n = np.zeros((VIEWS, geom_samples, 3), np.float32)
    geom_mask = np.zeros((VIEWS, geom_samples), np.uint8)
    anchors = []
    for v, ra in enumerate(ras):
        n = len(ra["pixel_linear_index"])
        idx = sample_rows(n, geom_samples, seed_for(aid, f"geom{v}"))
        p, nrm = reconstruct_surface(vertices, faces, vn, ra["triangle_id"][idx], ra["barycentric_uv"][idx])
        take = len(idx)
        res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
        geom_xy[v, :take] = pixel_linear_to_grid(ra["pixel_linear_index"][idx], res)
        geom_p[v, :take] = p
        geom_n[v, :take] = nrm
        geom_mask[v, :take] = 1
        ai = sample_rows(n, anchors_per_view, seed_for(aid, f"anchor{v}"))
        ap, _ = reconstruct_surface(vertices, faces, vn, ra["triangle_id"][ai], ra["barycentric_uv"][ai])
        anchors.append(ap)

    P = np.concatenate(anchors, axis=0)
    q = np.round(P / 1e-4).astype(np.int64)
    _, uniq = np.unique(q, axis=0, return_index=True)
    P = P[np.sort(uniq)]
    T = len(P)
    track_xy = np.zeros((T, VIEWS, 2), np.float32)
    track_vis = np.zeros((T, VIEWS), np.uint8)
    track_err = np.full((T, VIEWS), np.inf, np.float32)
    track_n_view = np.zeros((T, VIEWS, 3), np.float32)
    for v in range(VIEWS):
        ok, g, err, row = choose_visible_correspondence(
            P, cams[v], ras[v], vertices, faces, vn, radius_px, max_surface_error
        )
        if ok.shape != (T,) or g.shape != (T, 2) or err.shape != (T,) or row.shape != (T,):
            raise RuntimeError(
                f"visibility correspondence shape drift {aid} V{v}: "
                f"ok={ok.shape} grid={g.shape} err={err.shape} row={row.shape} expected_T={T}"
            )
        track_vis[:, v] = ok
        track_xy[ok, v] = g[ok]
        track_err[ok, v] = err[ok]
        if np.any(ok):
            _, nn = reconstruct_surface(
                vertices, faces, vn,
                ras[v]["triangle_id"][row[ok]],
                ras[v]["barycentric_uv"][row[ok]],
            )
            track_n_view[ok, v] = nn

    support = track_vis.sum(1)
    keep = np.flatnonzero(support >= 2)
    if not len(keep):
        raise RuntimeError(f"no persistent tracks {aid}")
    if len(keep) > max_tracks:
        hh = np.asarray(
            [int(hashlib.sha256(P[i].astype(np.float32).tobytes()).hexdigest()[:16], 16) for i in keep],
            np.uint64,
        )
        order = np.lexsort((hh, -support[keep]))
        keep = keep[order[:max_tracks]]

    P = P[keep]
    track_xy = track_xy[keep]
    track_vis = track_vis[keep]
    track_err = track_err[keep]
    track_n_view = track_n_view[keep]
    support = support[keep]

    atomic_npz(
        cp,
        geom_xy=geom_xy,
        geom_p=geom_p,
        geom_n=geom_n,
        geom_mask=geom_mask,
        track_p=P.astype(np.float32),
        track_xy=track_xy,
        track_visible=track_vis,
        track_support=support.astype(np.uint8),
        track_surface_error=track_err,
        track_n_view=track_n_view,
        yaw_deg=yaws,
    )
    stage_meta = json.load((ar / "STAGE.json").open(encoding="utf-8"))
    meta = {
        "schema": CACHE_META_SCHEMA,
        "asset_id": aid,
        "split": split,
        "asset_dir": str(ar),
        "truth_path": str(cp),
        "truth_sha256": sha256_file(cp),
        "input_fingerprint": input_fingerprint,
        "input_dependencies": input_dependencies,
        "settings": settings,
        "geom_valid": int(geom_mask.sum()),
        "track_count": int(len(P)),
        "input_resolution": int(stage_meta["authority_resolution"]),
        "stage_profile": EXPECTED_STAGE_PROFILE,
        "rgb_consumed": False,
    }
    atomic_json(mp, meta)
    return meta


def main():
    ap = argparse.ArgumentParser(description="Build exact representation-authority truth cache from geometry/raster/camera only")
    ap.add_argument("--stage-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geom-samples", type=int, default=4096)
    ap.add_argument("--anchors-per-view", type=int, default=1024)
    ap.add_argument("--max-tracks", type=int, default=4096)
    ap.add_argument("--radius-px", type=int, default=3)
    ap.add_argument("--max-surface-error", type=float, default=0.003)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sm = json.load(open(a.stage_manifest, encoding="utf-8"))
    if sm.get("schema") != EXPECTED_STAGE_MANIFEST_SCHEMA or sm.get("stage_profile") != EXPECTED_STAGE_PROFILE:
        raise RuntimeError(f"wrong representation stage manifest: {sm.get('schema')} {sm.get('stage_profile')}")
    if sm.get("images_staged") is not False:
        raise RuntimeError("representation cache refuses image-bearing stage manifest")

    stage_root = Path(a.stage_manifest).parent
    out = Path(a.out)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    rows = sm["records"][: a.limit] if a.limit else sm["records"]
    result = []
    t = time.time()
    for i, r in enumerate(rows, 1):
        result.append(build_asset(
            stage_root, r, out,
            a.geom_samples, a.anchors_per_view, a.max_tracks,
            a.radius_px, a.max_surface_error,
        ))
        if i % 5 == 0 or i == len(rows):
            print(f"[repr-cache] {i}/{len(rows)} tracks_med={np.median([x['track_count'] for x in result]):.0f}", flush=True)

    manifest = {
        "schema": CACHE_MANIFEST_SCHEMA,
        "authority_resolution": 1024,
        "input_resolution": 1024,
        "stage_profile": EXPECTED_STAGE_PROFILE,
        "rgb_consumed": False,
        "record_count": len(result),
        "records": sorted(result, key=lambda x: x["asset_id"]),
        "settings": {
            "geom_samples": a.geom_samples,
            "anchors_per_view": a.anchors_per_view,
            "max_tracks": a.max_tracks,
            "radius_px": a.radius_px,
            "max_surface_error": a.max_surface_error,
            "track_truth": "geometry-only physical locus with exact continuous projection",
            "images": "not staged and not consumed by Representation Authority",
        },
        "builder_sha256": builder_sha256(),
        "elapsed_sec": time.time() - t,
    }
    atomic_json(out / "CACHE_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
