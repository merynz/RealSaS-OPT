from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(x) for x in cmd])


def make_stage(root: Path):
    aid = "asset_partial_visibility_synth"
    stage = root / "stage"
    ar = stage / "assets" / aid
    (ar / "renders").mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(
        [[-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.0, 0.4, 0.0]],
        np.float32,
    )
    faces = np.asarray([[0, 1, 2]], np.int32)
    np.savez(ar / "primary_geometry.npz", vertices=vertices, faces=faces)

    rng = np.random.default_rng(20260824)
    uv = []
    while len(uv) < 1200:
        a, b = rng.random(2)
        if a + b <= 1.0:
            uv.append((a, b))
    uv = np.asarray(uv, np.float32)
    w = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], axis=-1)
    p = (vertices[faces[0]][None] * w[:, :, None]).sum(axis=1)

    res = 1024
    gx = p[:, 0] / 0.5
    gy = -p[:, 1] / 0.5
    x = np.floor((gx + 1.0) * 0.5 * res).astype(np.int64)
    y = np.floor((gy + 1.0) * 0.5 * res).astype(np.int64)
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
        if v == 0:
            xx = pix % res
            keep = xx < (res // 2)
            pix_v = pix[keep]
            uv_v = uv[keep]
        else:
            pix_v = pix
            uv_v = uv
        np.savez(
            vd / "raster_authority.npz",
            resolution=np.asarray([res], np.int32),
            pixel_linear_index=pix_v,
            triangle_id=np.zeros(len(pix_v), np.int32),
            barycentric_uv=uv_v,
        )
        (vd / "camera.json").write_text(
            json.dumps(
                {
                    "yaw_deg": float(v * 45),
                    "right": [1, 0, 0],
                    "up": [0, 1, 0],
                    "half_extent": 0.5,
                }
            ),
            encoding="utf-8",
        )

    (ar / "STAGE.json").write_text(
        json.dumps(
            {
                "schema": "RealSaS.IRISSinglePoseV2.RepresentationStage.v1",
                "stage_profile": "representation_authority_geometry_only",
                "asset_id": aid,
                "authority_resolution": 1024,
                "input_resolution": 1024,
                "physical_firewall": True,
                "geometry_fields": ["vertices", "faces"],
                "images_staged": False,
                "source_fingerprint": "synthetic_partial_visibility",
                "staged_sha256": {"synthetic": "regression-only"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "STAGE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "RealSaS.IRISSinglePoseV2.RepresentationStageManifest.v1",
                "stage_profile": "representation_authority_geometry_only",
                "authority_resolution": 1024,
                "input_resolution": 1024,
                "images_staged": False,
                "record_count": 1,
                "records": [
                    {
                        "asset_id": aid,
                        "split": "FIT",
                        "asset_dir": str(ar),
                        "source_fingerprint": "synthetic_partial_visibility",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return aid, stage


def main():
    here = Path(__file__).resolve().parent
    root = Path(tempfile.mkdtemp(prefix="irisv2_partial_visibility_preflight_"))
    try:
        aid, stage = make_stage(root)
        cache = root / "cache"
        run(
            [
                sys.executable,
                here / "prepare_representation_cache_v1.py",
                "--stage-manifest",
                stage / "STAGE_MANIFEST.json",
                "--out",
                cache,
                "--geom-samples",
                "128",
                "--anchors-per-view",
                "128",
                "--max-tracks",
                "512",
                "--radius-px",
                "0",
                "--max-surface-error",
                "0.004",
            ]
        )

        with np.load(cache / "truth" / f"{aid}.npz", allow_pickle=False) as z:
            vis = np.asarray(z["track_visible"], np.uint8).astype(bool)
            err = np.asarray(z["track_surface_error"], np.float32)
            assert vis.shape == err.shape, (vis.shape, err.shape)
            assert vis.shape[1] == 8
            assert vis[:, 0].any(), "V0 must retain some visible tracks"
            assert (~vis[:, 0]).any(), "V0 must contain hidden tracks to exercise partial visibility"
            assert np.isfinite(err[vis]).all(), "visible tracks require finite witness errors"
            assert np.isinf(err[~vis]).all(), "hidden tracks must preserve +inf witness error"

        cm = json.loads((cache / "CACHE_MANIFEST.json").read_text(encoding="utf-8"))
        deps = cm["builder_sha256"]
        assert set(deps) == {
            "prepare_representation_cache_v1.py",
            "geometry.py",
            "coords.py",
        }, deps

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "partial_visibility_exercised": True,
                    "track_visible_surface_error_shape_equal": True,
                    "visible_error_finite": True,
                    "hidden_error_inf": True,
                    "coords_bound_into_cache_builder_fingerprint": True,
                    "optimizer_steps": 0,
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
