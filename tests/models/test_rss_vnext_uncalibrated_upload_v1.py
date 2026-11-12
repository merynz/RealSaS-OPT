from __future__ import annotations

from dataclasses import replace
import io
import json
from zipfile import ZipFile, ZIP_STORED

import numpy as np
import pytest
from PIL import Image

from compiler.realsas_compiler_core.rss_vnext_uncalibrated_upload_v1 import (
    RSSvNextUncalibratedTwoImageSourceV1,
    encode_uncalibrated_two_image_source_v1,
    decode_uncalibrated_two_image_source_v1,
    derive_uncalibrated_pair_conditioning_v1,
)


def _file(codec, size, *, transparent=False):
    img=Image.new(
        "RGBA" if transparent else "RGB",
        size,
        (37,119,211,0) if transparent else (37,119,211),
    )
    if transparent:
        for y in range(size[1]//4, size[1]*3//4):
            for x in range(size[0]//3,size[0]*2//3):
                img.putpixel((x,y),(222,62,41,255))
    buf=io.BytesIO()
    img.save(buf,format=codec)
    return buf.getvalue()


def _pair():
    return RSSvNextUncalibratedTwoImageSourceV1(
        original_files=(
            _file("JPEG",(121,161)),
            _file("PNG",(167,91),transparent=True),
        ),
        source_rights_claim_sha256="a"*64,
        view_claims=("OTHER","UNKNOWN"),
    )


def test_scene_background_arbitrary_pose_camera_aspect_and_mime_are_accepted():
    pair=_pair()
    pair.validate()
    manifest=pair.manifest()
    assert manifest["original_observation_count"]==2
    assert manifest["generated_view_count"]==0
    assert manifest["source_camera_state"]=="UNKNOWN_NO_ORIGINAL_CAMERA_CALIBRATION"
    assert manifest["source_pose_state"]=="UNKNOWN_NO_CANONICAL_POSE"
    assert manifest["source_rights_verified"] is False
    assert manifest["is_product_ready"] is False
    assert manifest["calibrated_rss_source_artifact"] is False
    assert manifest["source_images"][0]["format"]=="JPEG"
    assert manifest["source_images"][1]["format"]=="PNG"
    assert not manifest["source_images"][0]["source_alpha_present"]
    assert manifest["source_images"][1]["source_alpha_present"]
    assert manifest["source_images"][0]["source_foreground_segmented"] is False


def test_byte_exact_originals_survive_cas_encoding_and_decoding():
    pair=_pair()
    original=tuple(pair.original_files)
    packed=encode_uncalibrated_two_image_source_v1(pair)
    assert packed==encode_uncalibrated_two_image_source_v1(pair)
    decoded=decode_uncalibrated_two_image_source_v1(packed)
    assert decoded.original_files==original
    assert decoded.source_binding_sha256==pair.source_binding_sha256
    assert decoded.manifest()==pair.manifest()
    assert decoded.view_claims==("OTHER","UNKNOWN")


def test_derived_pair_is_letterboxed_not_cropped_or_declared_source_truth():
    pair=_pair()
    derived=derive_uncalibrated_pair_conditioning_v1(pair,target_square_size=128)
    assert derived.rgba_uint8.shape==(2,4,128,128)
    assert derived.original_footprint.shape==(2,128,128)
    assert derived.rgba_uint8.dtype==np.uint8
    assert derived.original_footprint.dtype==np.bool_
    assert derived.source_binding_sha256==pair.source_binding_sha256
    assert len(derived.conditioning_sha256)==64
    for idx,row in enumerate(derived.pixel_transform_rows):
        assert row["original_view_index"]==idx
        assert row["foreground_extracted"] is False
        assert row["camera_estimated"] is False
        assert row["pose_estimated"] is False
        assert row["resampled_width"]<=128
        assert row["resampled_height"]<=128
        assert derived.original_footprint[idx].sum()==(
            row["resampled_width"]*row["resampled_height"]
        )
    # The fully opaque scene image is kept as observed image content. No
    # alpha==255 assumption is allowed to claim character foreground.
    assert derived.rgba_uint8[0,3][derived.original_footprint[0]].min()==255
    assert np.array_equal(derived.rgba_uint8[1,3][~derived.original_footprint[1]],
                          np.zeros(np.count_nonzero(~derived.original_footprint[1]),dtype=np.uint8))
    assert pair.original_files[0].startswith(b"\xff\xd8")


def test_claimed_front_back_labels_never_become_verified_camera_geometry():
    pair=_pair()
    labelled=replace(pair,view_claims=("FRONT","BACK"))
    labelled.validate()
    assert labelled.source_binding_sha256!=pair.source_binding_sha256
    assert labelled.manifest()["source_camera_state"]=="UNKNOWN_NO_ORIGINAL_CAMERA_CALIBRATION"
    assert labelled.manifest()["calibrated_rss_source_artifact"] is False
    conditioning=derive_uncalibrated_pair_conditioning_v1(labelled,target_square_size=96)
    assert conditioning.pixel_transform_rows[0]["camera_estimated"] is False


def test_corrupted_manifests_and_bad_originals_are_rejected_without_guessing():
    pair=_pair()
    with pytest.raises(ValueError,match="TWO_IMAGE_OR_USER_CLAIM_SCHEMA"):
        replace(pair, original_files=(pair.original_files[0],)).validate()
    with pytest.raises(ValueError,match="IMAGE_DECODE_INVALID"):
        replace(pair,original_files=(pair.original_files[0], b"not a valid image")).validate()
    with pytest.raises(ValueError,match="TWO_IMAGE_OR_USER_CLAIM_SCHEMA"):
        replace(pair,view_claims=("FRONT","CAMERA_AUTO_ESTIMATED")).validate()

    packed=encode_uncalibrated_two_image_source_v1(pair)
    raw=io.BytesIO()
    with ZipFile(io.BytesIO(packed),"r") as old, ZipFile(raw,"w") as changed:
        for name in ("manifest.json","source_0","source_1"):
            data=old.read(name)
            if name=="manifest.json":
                content=json.loads(data)
                content["calibrated_rss_source_artifact"]=True
                data=json.dumps(content).encode()
            changed.writestr(name,data,compress_type=ZIP_STORED)
    with pytest.raises(ValueError,match="CAS_MANIFEST_BYTES_OR_AUTHORITY_DRIFT"):
        decode_uncalibrated_two_image_source_v1(raw.getvalue())
