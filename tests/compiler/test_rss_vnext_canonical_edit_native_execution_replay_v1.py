from __future__ import annotations

"""Actual native C++ edited puppet consequence; no source RGB ProductPASS."""

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.rss_vnext.workers.rss_vnext_canonical_edit_native_execution_replay_v1 import (
    AUTHORITY, SCHEMA,
    execute_canonical_user_edit_native_cpp_capture_v1,
    independently_replay_canonical_user_edit_native_cpp_v1,
    native_user_edit_court_hash_v1,
    reopen_native_edited_puppet_rasters_v1,
)
from tests.compiler.test_rss_vnext_edited_graphics_preview_v1 import _files
from tests.compiler.test_rss_vnext_native_preview_package_v1 import _build_player


def _capture(tmp_path):
    source,revision,edited,_,_,untouched = _files(tmp_path)
    exe=_build_player(tmp_path)
    expected=sha256(exe.read_bytes()).hexdigest()
    args={
        "source_rss_path":source,
        "rsedit_path":revision,
        "edited_preview_rss_path":edited,
        "renderer_executable":exe,
    }
    binary,court=execute_canonical_user_edit_native_cpp_capture_v1(
        **args,
        expected_renderer_executable_sha256=expected,
        frame_index=1,view_index=0,
    )
    return args,binary,court,untouched


def test_real_sha_pinned_cpp_edit_consequence_is_independently_replayed(tmp_path):
    args,binary,court,untouched=_capture(tmp_path)
    assert court.schema==SCHEMA
    assert court.authority==AUTHORITY
    assert court.source_view_index==-1
    assert court.native_edit_replay_executed is True
    assert court.edited_source_rgb_qualified is False
    assert court.edited_source_camera_observed is False
    assert court.continuous_animation_proven is False
    assert court.product_release_admission is False
    assert court.edited_mesh_lineage_hash!=court.original_mesh_lineage_hash
    assert court.visible_rgba_changed_pixel_count >= 0
    assert court.visible_owner_changed_pixel_count >= 0
    rows=reopen_native_edited_puppet_rasters_v1(
        binary,resolution=court.resolution,face_count=2_000,
    )
    assert set(rows)=={
        "original_native_rgba_u8","original_front_face_i32",
        "original_color_authority_u8","original_unknown_occlusion_u8",
        "edited_native_rgba_u8","edited_front_face_i32",
        "edited_color_authority_u8","edited_unknown_occlusion_u8",
    }
    assert set(np.unique(rows["edited_color_authority_u8"])).issubset({0,2})
    assert sha256(binary).hexdigest()==court.native_raster_bundle_sha256
    reopened=independently_replay_canonical_user_edit_native_cpp_v1(
        **args,raster_bundle=binary,claimed_court=court.to_dict(),
    )
    assert reopened==court
    assert args["source_rss_path"].read_bytes()==untouched


def test_tampered_raw_pixel_bundle_even_with_changed_digest_fails(tmp_path):
    args,binary,court,_=_capture(tmp_path)
    changed=bytearray(binary)
    # Last byte resides inside the final unknown-occlusion mask. Flip a bit.
    changed[-1]^=1
    mutated=bytes(changed)
    forged=replace(
        court,native_raster_bundle_sha256=sha256(mutated).hexdigest(),
        court_hash="",
    )
    forged=replace(forged,court_hash=native_user_edit_court_hash_v1(forged))
    with pytest.raises(QualificationError,match=(
        "BUNDLE_HASH_DRIFT|INDEPENDENT_NATIVE_RASTER_REPLAY_DRIFT"
    )):
        independently_replay_canonical_user_edit_native_cpp_v1(
            **args,raster_bundle=mutated,claimed_court=forged.to_dict(),
        )


def test_source_rgb_or_product_promotion_cannot_be_forged(tmp_path):
    args,binary,court,_=_capture(tmp_path)
    for patch in (
        {"edited_source_rgb_qualified":True},
        {"edited_source_camera_observed":True},
        {"continuous_animation_proven":True},
        {"product_release_admission":True},
        {"source_view_index":0},
    ):
        forged=replace(court,**patch,court_hash="")
        forged=replace(forged,court_hash=native_user_edit_court_hash_v1(forged))
        with pytest.raises(QualificationError,match="FALSE_AUTHORITY"):
            independently_replay_canonical_user_edit_native_cpp_v1(
                **args,raster_bundle=binary,claimed_court=forged.to_dict(),
            )


def test_wrong_renderer_pin_and_modified_source_are_refused(tmp_path):
    args,binary,court,_=_capture(tmp_path)
    with pytest.raises(QualificationError,match="BINARY_PIN_DRIFT"):
        execute_canonical_user_edit_native_cpp_capture_v1(
            **args,expected_renderer_executable_sha256="a"*64,
            frame_index=1,view_index=0,
        )
    edited_source=tmp_path/"forged-original-source.rss"
    source=bytearray(args["source_rss_path"].read_bytes())
    source[-8]^=1
    edited_source.write_bytes(source)
    with pytest.raises((QualificationError,ValueError,RuntimeError)):
        independently_replay_canonical_user_edit_native_cpp_v1(
            **{**args,"source_rss_path":edited_source},
            raster_bundle=binary,claimed_court=court.to_dict(),
        )


def test_source_view_cannot_be_selected_as_original_source_camera(tmp_path):
    args,binary,court,_=_capture(tmp_path)
    with pytest.raises(QualificationError,match="FRAME_OR_VIEW_TYPE"):
        execute_canonical_user_edit_native_cpp_capture_v1(
            **args,expected_renderer_executable_sha256=court.renderer_executable_sha256,
            frame_index=True,view_index=0,
        )
    with pytest.raises(QualificationError,match="SINGLE_CAMERA_RANGE"):
        execute_canonical_user_edit_native_cpp_capture_v1(
            **args,expected_renderer_executable_sha256=court.renderer_executable_sha256,
            frame_index=1,view_index=1,
        )
