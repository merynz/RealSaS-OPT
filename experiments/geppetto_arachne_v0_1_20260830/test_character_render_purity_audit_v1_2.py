from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from character_render_purity_audit_v1_2 import RenderPurityAssetInputV1, audit_asset_v1


def make_asset(root: Path, *, broken_triangle=False, raster_size=1024) -> Path:
    ad = root / "master" / "assets" / "asset_test"
    ad.mkdir(parents=True)
    vertices = np.asarray([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,1]], np.float32)
    faces = np.asarray([[0,1,4],[1,2,4],[2,3,4],[3,0,4]], np.int32)
    skin = np.ones((5, 1), np.float32)
    np.savez(ad / "primary_geometry.npz", vertices=vertices, faces=faces, skin=skin)

    H=W=int(raster_size)
    yy, xx = np.mgrid[0:H,0:W]
    mask = (xx>=int(0.3125*W))&(xx<int(0.6875*W))&(yy>=int(0.1875*H))&(yy<int(0.8125*H))
    pix = np.flatnonzero(mask.ravel()).astype(np.int64)
    tid = np.zeros(len(pix), np.int64)
    tid[(xx.ravel()[pix] >= W//2)] = 1
    if broken_triangle:
        tid[0] = 99

    rgba = np.zeros((512,512,4), np.uint8)
    rgba[...,1] = 255
    rgba[...,3] = 255
    rgba[96:416,160:352] = np.asarray([220,180,120,255], np.uint8)
    for v in range(8):
        vd=ad/"renders"/f"V{v}"; vd.mkdir(parents=True)
        np.savez(vd/"raster_authority.npz", pixel_linear_index=pix, triangle_id=tid, resolution=np.asarray([H,W],np.int32))
        Image.fromarray(rgba, "RGBA").save(vd/"cel_clean_512.png")
        (vd/"camera.json").write_text(json.dumps({"yaw_deg":v*45.0}))
    return ad


def test_native_1024_raster_with_derived_512_image_passes(tmp_path):
    ad=make_asset(tmp_path, raster_size=1024)
    row=audit_asset_v1(RenderPurityAssetInputV1("asset_test","fixture",ad,True))
    assert row["status"]=="PASS", row
    assert len(row["views"])==8
    assert row["views"][0]["raster_resolution"] == [1024, 1024]
    assert row["aggregate"]["border_touch_view_count"]==0
    assert row["views"][0]["visible_skinned_triangle_fraction_valid"] is True
    assert row["views"][0]["cel_clean_512_exists"] is True
    assert row["views"][0]["image_content_decoded"] is False


def test_derived_512_raster_fails_closed(tmp_path):
    ad=make_asset(tmp_path, raster_size=512)
    row=audit_asset_v1(RenderPurityAssetInputV1("asset_test","fixture",ad,True))
    assert row["status"]=="FAIL"
    assert "raster resolution mismatch: (512, 512) expected (1024, 1024)" in row["errors"][0]


def test_bad_triangle_fails_closed(tmp_path):
    ad=make_asset(tmp_path, broken_triangle=True, raster_size=1024)
    row=audit_asset_v1(RenderPurityAssetInputV1("asset_test","fixture",ad,False))
    assert row["status"]=="FAIL"
    assert "triangle index out of range" in row["errors"][0]
