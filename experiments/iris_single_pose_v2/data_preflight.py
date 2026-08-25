from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

from stage_source import stage_asset
from prepare_cache import build_asset
from geometry import project_grid


def make_synthetic_master(root: Path):
    aid = "asset_synth"
    ar = root / "master" / "assets" / aid
    (ar / "renders").mkdir(parents=True, exist_ok=True)
    vertices = np.asarray([[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    # Deliberately include forbidden rig/mechanics-like fields in the master source.
    np.savez(
        ar / "primary_geometry.npz",
        vertices=vertices,
        faces=faces,
        parents=np.asarray([-1, 0]),
        skin=np.ones((3, 2), np.float32),
        bone_heads=np.zeros((2, 3), np.float32),
    )
    rng = np.random.default_rng(123)
    uv = []
    while len(uv) < 600:
        a, b = rng.random(2)
        if a + b <= 1:
            uv.append((a, b))
    uv = np.asarray(uv, np.float32)
    w = np.stack([uv[:, 0], uv[:, 1], 1 - uv[:, 0] - uv[:, 1]], -1)
    p = (vertices[faces[0]][None] * w[:, :, None]).sum(1)
    res = 1024
    gx = p[:, 0] / 0.5
    gy = -p[:, 1] / 0.5
    x = np.floor((gx + 1) * 0.5 * res).astype(np.int64)
    y = np.floor((gy + 1) * 0.5 * res).astype(np.int64)
    pix = y * res + x
    order = np.argsort(pix)
    pix = pix[order]
    uv = uv[order]
    _, first = np.unique(pix, return_index=True)
    pix = pix[first]
    uv = uv[first]
    for v in range(8):
        vd = ar / "renders" / f"V{v}"
        vd.mkdir(parents=True, exist_ok=True)
        np.savez(
            vd / "raster_authority.npz",
            resolution=np.asarray([res], np.int32),
            pixel_linear_index=pix,
            triangle_id=np.zeros(len(pix), np.int32),
            barycentric_uv=uv,
        )
        (vd / "camera.json").write_text(
            json.dumps({"yaw_deg": float(v * 45), "right": [1, 0, 0], "up": [0, 1, 0], "half_extent": 0.5}),
            encoding="utf-8",
        )
        for style in ("cel_clean", "ink_cel"):
            Image.new("RGBA", (1024, 1024), (255, 255, 255, 255)).save(vd / f"{style}.png")
            Image.new("RGBA", (512, 512), (255, 255, 255, 255)).save(vd / f"{style}_512.png")
    return aid


def check_continuous_track_truth(stage: Path, aid: str, truth_path: str):
    ar = stage / "assets" / aid
    cams = [json.load(open(ar / "renders" / f"V{v}" / "camera.json", encoding="utf-8")) for v in range(8)]
    with np.load(truth_path, allow_pickle=False) as z:
        p = z["track_p"].astype(np.float32)
        xy = z["track_xy"].astype(np.float32)
        vis = z["track_visible"].astype(bool)
    max_err = 0.0
    off_pixel_center_witnesses = 0
    for v, cam in enumerate(cams):
        exact = project_grid(p, cam)
        if np.any(vis[:, v]):
            err = np.linalg.norm(xy[vis[:, v], v] - exact[vis[:, v]], axis=1)
            max_err = max(max_err, float(err.max(initial=0.0)))
            pix = (exact[vis[:, v]] + 1.0) * 512.0 - 0.5
            frac = np.abs(pix - np.round(pix))
            off_pixel_center_witnesses += int(np.any(frac > 1e-4, axis=1).sum())
    if max_err > 1e-6:
        raise AssertionError(f"track_xy was quantized away from exact projection: max_grid_error={max_err}")
    if off_pixel_center_witnesses == 0:
        raise AssertionError("synthetic fixture did not exercise subpixel continuous projection")
    return {"max_grid_error": max_err, "subpixel_witnesses": off_pixel_center_witnesses}


def mutate_source_rgb(master: Path, aid: str):
    p = master / "master" / "assets" / aid / "renders" / "V0" / "cel_clean.png"
    with Image.open(p) as im:
        arr = np.asarray(im.convert("RGBA")).copy()
    arr[0, 0, 0] = 17
    Image.fromarray(arr, "RGBA").save(p)


def main():
    root = Path(tempfile.mkdtemp(prefix="irisv2_data_preflight_"))
    try:
        master = root / "masterroot"
        stage = root / "stage"
        cache = root / "cache"
        aid = make_synthetic_master(master)
        record = {"asset_id": aid, "split": "FIT", "asset_dir": str(stage / "assets" / aid)}

        stage1 = stage_asset(master, stage, aid, 256)
        with np.load(stage / "assets" / aid / "primary_geometry.npz", allow_pickle=False) as z:
            assert set(z.files) == {"vertices", "faces"}, z.files
        cache1 = build_asset(
            stage,
            record,
            cache,
            geom_samples=64,
            anchors_per_view=64,
            max_tracks=128,
            radius_px=3,
            max_surface_error=0.004,
        )
        with np.load(cache1["truth_path"], allow_pickle=False) as z:
            assert int(z["geom_mask"].sum()) > 0
            assert len(z["track_p"]) > 0
            assert int(z["track_visible"].sum()) >= 2
        continuous = check_continuous_track_truth(stage, aid, cache1["truth_path"])

        # Source byte drift must invalidate stage reuse even when shape/camera semantics are unchanged.
        mutate_source_rgb(master, aid)
        stage2 = stage_asset(master, stage, aid, 256)
        if stage1["source_fingerprint"] == stage2["source_fingerprint"]:
            raise AssertionError("source mutation failed to invalidate stage fingerprint")

        # Any stage dependency drift must invalidate truth-cache reuse as well.
        cache2 = build_asset(
            stage,
            record,
            cache,
            geom_samples=64,
            anchors_per_view=64,
            max_tracks=128,
            radius_px=3,
            max_surface_error=0.004,
        )
        if cache1["input_fingerprint"] == cache2["input_fingerprint"]:
            raise AssertionError("stage mutation failed to invalidate cache input fingerprint")

        # Truth-generation settings are part of cache identity; no silent parameter reuse.
        cache3 = build_asset(
            stage,
            record,
            cache,
            geom_samples=64,
            anchors_per_view=64,
            max_tracks=128,
            radius_px=3,
            max_surface_error=0.0035,
        )
        if cache2["input_fingerprint"] == cache3["input_fingerprint"]:
            raise AssertionError("truth settings change failed to invalidate cache fingerprint")

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "physical_firewall": stage2["physical_firewall"],
                    "track_count": cache3["track_count"],
                    "geom_valid": cache3["geom_valid"],
                    "continuous_track_truth": continuous,
                    "stage_source_invalidation": "PASS",
                    "cache_stage_invalidation": "PASS",
                    "cache_settings_invalidation": "PASS",
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
