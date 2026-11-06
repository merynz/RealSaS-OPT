"""Adversarial contract for original Cat UV atlas bytes and sample edit algebra."""
import json
import numpy as np
from PIL import Image
import pytest

from tools.corpus.prepare_corpus370_original_uv_atlas_edit_images_v2 import build
from tools.corpus.court_corpus370_original_cat_uv_texture_cycles_v2 import (
    score_sample_source_rgba,verify_original_atlas_variants,
)
from tools.corpus.compile_corpus370_original_source_uv_edit_plan_v1 import (
    RULE
)
import hashlib


def fake_original(tmp_path):
    src=np.zeros((512,512,4),dtype=np.uint8)
    for y in range(512):
        src[y,:,0]=y%256
        src[y,:,1]=(y*3)%256
        src[y,:,2]=(y*5)%256
        src[y,:,3]=255
    img=tmp_path/"atlas.png"
    Image.fromarray(src).save(img)
    sha=hashlib.sha256(img.read_bytes()).hexdigest()
    ops=[]
    for i,(u,v) in enumerate([(6,6),(6,7),(7,6)]):
        ops.append({
            "source_uv_edit_operation_index":i,
            "uv_tile_u":u,"uv_tile_v":v,
            "source_rgba_png_pixel_rectangle_xyxy_exclusive":[
                u*64,v*64,(u+1)*64,(v+1)*64]})
    plan={
        "schema":"RealSaS.Corpus370OriginalSourceUVAtlasEditPlan.v2",
        "status":"ORIGINAL_SOURCE_UV_OPERATION_PLAN_FROZEN__NOT_RENDER_COURTED",
        "source_uv_coordinate_origin":"GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN",
        "source_uv_to_png_y_flip_applied":False,
        "source_original_glb_sha256":"c"*64,
        "source_original_texture_sha256":sha,
        "source_texture_size_px":[512,512],
        "original_source_texture_edit_rule":RULE,
        "original_source_uv_operation_count":3,
        "selected_original_source_uv_edits":ops,
        **{n:False for n in ("graphics_owner_authorized","iris_training_authorized",
             "tessa_training_authorized","materialization_authorized","training_authorized")},
    }
    return img,plan


def test_actual_original_png_variant_byte_support_and_alpha(tmp_path):
    img,plan=fake_original(tmp_path)
    manifest=build(img,plan,"a"*64,tmp_path/"edited")
    assert len(manifest["original_atlas_arm_rows"])==8
    assert [x["original_atlas_edited_rgb_pixel_count"]
            for x in manifest["original_atlas_arm_rows"]]==[
                 0,4096,4096,4096,8192,8192,8192,12288]
    verify_original_atlas_variants(img,plan,manifest,tmp_path/"edited")
    assert (tmp_path/"edited/BASE.png").read_bytes()==img.read_bytes()


def test_original_atlas_variant_forgery_is_detected(tmp_path):
    img,plan=fake_original(tmp_path)
    manifest=build(img,plan,"a"*64,tmp_path/"edited")
    with Image.open(tmp_path/"edited/SINGLE_0.png") as f:
        arr=np.array(f)
    arr[15,15,0]=255-arr[15,15,0]
    Image.fromarray(arr).save(tmp_path/"edited/SINGLE_0.png")
    manifest["original_atlas_arm_rows"][1]["source_atlas_rgba_sha256"] = hashlib.sha256(
        (tmp_path/"edited/SINGLE_0.png").read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="SOURCE_TEXTURE_EDIT_BYTES_WRONG"):
        verify_original_atlas_variants(img,plan,manifest,tmp_path/"edited")


def make_source_samples():
    a=np.zeros((8,8,192,192,4),dtype=np.float32)
    a[...,0]=0.25
    a[...,1]=0.35
    a[...,2]=0.45
    a[...,3]=1
    for i in range(3):
        x0=i*50+10
        a[:,1+i,:,x0:x0+20,0]=.8
    for arm,sets in [(4,(0,1)),(5,(0,2)),(6,(1,2)),(7,(0,1,2))]:
        a[:,arm]=a[:,0]
        for i in sets:
            x0=i*50+10
            a[:,arm,:,x0:x0+20,0]=.8
    return a


def test_real_sample_joint_is_predicted_from_singles_not_oracle():
    a=make_source_samples()
    r=score_sample_source_rgba(a)
    assert all(v["source_per_sample_rgba_mismatch_count"]==0
               for v in r["source_heldout_pair_joint_court"])
    assert all(v["source_eight_sample_average_rgba_mismatch_count"]==0
               for v in r["source_heldout_pair_joint_court"])
    assert all(n>=32 for n in r["edited_single_source_visible_sample_count"])
    assert r["all_source_arm_alpha_max_drift"]==0


def test_unseen_joint_truth_modification_is_caught():
    a=make_source_samples()
    a[3,7,60,35,1]=.9
    r=score_sample_source_rgba(a)
    assert r["source_heldout_pair_joint_court"][3]["source_per_sample_rgba_mismatch_count"]>=1


def test_alpha_drift_is_never_ignored():
    a=make_source_samples()
    a[1,4,11,14,3]=.75
    r=score_sample_source_rgba(a)
    assert r["all_source_arm_alpha_max_drift"]>=.25


def test_invalid_sample_layout_fails_closed():
    with pytest.raises(ValueError,match="ORIGINAL_CYCLES_SOURCE_SAMPLE_RGBA_SHAPE"):
        score_sample_source_rgba(np.zeros((1,8,192,192,4),dtype=np.float32))


def test_original_atlas_overlapping_operations_are_rejected(tmp_path):
    img,plan=fake_original(tmp_path)
    plan["selected_original_source_uv_edits"][1][
        "source_rgba_png_pixel_rectangle_xyxy_exclusive"
    ]=list(plan["selected_original_source_uv_edits"][0][
        "source_rgba_png_pixel_rectangle_xyxy_exclusive"])
    with pytest.raises(ValueError,match="SOURCE_UV_TILE_RGBA_RECT_NOT_64X64"):
        build(img,plan,"a"*64,tmp_path/"edited")
