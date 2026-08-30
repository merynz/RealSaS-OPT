from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from character_render_purity_audit_v1 import RenderPurityAssetInputV1, audit_asset_v1


def make_asset(root: Path, *, broken_triangle=False) -> Path:
    ad = root / "master" / "assets" / "asset_test"
    ad.mkdir(parents=True)
    vertices = np.asarray([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[0,0,1]], np.float32)
    faces = np.asarray([[0,1,4],[1,2,4],[2,3,4],[3,0,4]], np.int32)
    skin = np.ones((5, 1), np.float32)
    np.savez(ad / "primary_geometry.npz", vertices=vertices, faces=faces, skin=skin)
    H=W=512
    yy, xx = np.mgrid[0:H,0:W]
    mask = (xx>=160)&(xx<352)&(yy>=96)&(yy<416)
    pix = np.flatnonzero(mask.ravel()).astype(np.int64)
    tid = np.zeros(len(pix), np.int64)
    tid[(xx.ravel()[pix] >= 256)] = 1
    if broken_triangle:
        tid[0] = 99
    rgba = np.zeros((H,W,4), np.uint8)
    rgba[...,1] = 255
    rgba[...,3] = 255
    rgba[mask] = np.asarray([220,180,120,255], np.uint8)
    for v in range(8):
        vd=ad/"renders"/f"V{v}"; vd.mkdir(parents=True)
        np.savez(vd/"raster_authority.npz", pixel_linear_index=pix, triangle_id=tid, resolution=np.asarray([H,W],np.int32))
        Image.fromarray(rgba, "RGBA").save(vd/"cel_clean_512.png")
        (vd/"camera.json").write_text(json.dumps({"yaw_deg":v*45.0}))
    return ad


def test_clean_asset(tmp_path):
    ad=make_asset(tmp_path)
    row=audit_asset_v1(RenderPurityAssetInputV1("asset_test","fixture",ad,True))
    assert row["status"]=="PASS", row
    assert len(row["views"])==8
    assert row["aggregate"]["raster_image_mask_iou"]["min"]==1.0
    assert row["aggregate"]["border_touch_view_count"]==0
    assert row["views"][0]["visible_skinned_triangle_fraction_valid"] is True


def test_bad_triangle_fails_closed(tmp_path):
    ad=make_asset(tmp_path, broken_triangle=True)
    row=audit_asset_v1(RenderPurityAssetInputV1("asset_test","fixture",ad,False))
    assert row["status"]=="FAIL"
    assert "triangle index out of range" in row["errors"][0]
