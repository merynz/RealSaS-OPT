from __future__ import annotations

"""Actual raster bytes are independently re-decoded before Graphics claims.

Registries here are explicit test doubles; no real source truth is invented.
"""
from dataclasses import replace
import io

import numpy as np
import pytest

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    evaluate_registered_source_pose_graphics_v1,
    source_pose_graphics_court_hash_v1,
)
from compiler.rss_vnext.workers.rss_vnext_source_pose_graphics_evidence_replay_v1 import (
    SOURCE_BUNDLE_SCHEMA,NATIVE_BUNDLE_SCHEMA,
    independently_replay_registered_source_pose_graphics_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_rss_vnext_source_pose_graphics_court_v1 import _fixture


def _npz(**arrays):
    file=io.BytesIO()
    np.savez_compressed(file,**arrays)
    return file.getvalue()


def _case(*,motion=False):
    import hashlib
    raw=_fixture(motion=motion)
    source=_npz(
        target_rgba_u8=raw["target_rgba_u8"],
        target_face_owner_i32=raw["target_face_owner_i32"],
        target_known_mask=raw["target_known_mask"],
    )
    native=_npz(
        native_rgba_u8=raw["native_rgba_u8"],
        native_front_face_i32=raw["native_front_face_i32"],
        native_color_authority_u8=raw["native_color_authority_u8"],
        native_unknown_occlusion_u8=raw["native_unknown_occlusion_u8"],
    )
    src=raw["trusted_source_registry"][raw["source_witness_key"]]
    dest=raw["trusted_native_renderer_registry"][raw["renderer_witness_key"]]
    src.update({
        "raster_bundle_schema":SOURCE_BUNDLE_SCHEMA,
        "raster_bundle_sha256":hashlib.sha256(source).hexdigest(),
        "source_raster_extraction_independently_verified":True,
        "raster_artifact_not_student_prediction":True,
    })
    dest.update({
        "raster_bundle_schema":NATIVE_BUNDLE_SCHEMA,
        "raster_bundle_sha256":hashlib.sha256(native).hexdigest(),
        "cpp_native_raster_execution_independently_verified":True,
        "raster_artifact_not_student_prediction":True,
    })
    args=dict(
        source_raster_bundle=source,
        native_raster_bundle=native,
        native_face_count=raw["native_face_count"],
        live_pose_receipt=raw["live_pose_receipt"],
        source_view_index=raw["source_view_index"],
        frame_index=raw["frame_index"],
        source_target_kind=raw["source_target_kind"],
        original_source_asset_sha256=raw["original_source_asset_sha256"],
        original_source_file_sha256=raw["original_source_file_sha256"],
        source_registration_sha256=raw["source_registration_sha256"],
        native_renderer_binary_sha256=raw["native_renderer_binary_sha256"],
        source_witness_key=raw["source_witness_key"],
        renderer_witness_key=raw["renderer_witness_key"],
        frozen_thresholds=raw["thresholds"],
        trusted_source_registry=raw["trusted_source_registry"],
        trusted_native_renderer_registry=raw["trusted_native_renderer_registry"],
        **{
            k:raw[k] for k in (
                "canonical_mesh_lineage_hash","canonical_rig_lineage_hash",
                "canonical_skin_lineage_hash","compiled_graphics_hash",
                "canonical_motion_lineage_hash",
            )
        },
    )
    return raw,args


def test_replay_exact_native_cpp_sidecars_and_two_source_mask_from_raw_bytes():
    raw,args=_case()
    claimed=evaluate_registered_source_pose_graphics_v1(**raw)
    restored=independently_replay_registered_source_pose_graphics_v1(
        **args,claimed_court=claimed.to_dict(),
    )
    assert restored==claimed
    assert restored.source_registered_pixels_passed is True
    assert restored.product_release_admission is False


def test_replay_artist_postmotion_source_and_exact_xpbd_frame_bytes():
    raw,args=_case(motion=True)
    claimed=evaluate_registered_source_pose_graphics_v1(**raw)
    restored=independently_replay_registered_source_pose_graphics_v1(
        **args,claimed_court=claimed.to_dict(),
    )
    assert restored.frame_index==1
    assert restored.source_registered_pixels_passed
    assert not restored.hidden_surface_color_proven
    assert restored.product_release_admission is False


def test_cas_rehash_without_independent_source_witness_does_not_promote_pixels():
    _,args=_case()
    altered=args["source_raster_bundle"]+b"garbage"
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        independently_replay_registered_source_pose_graphics_v1(
            **{**args,"source_raster_bundle":altered},
        )
    without={**args,"trusted_source_registry":{}}
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        independently_replay_registered_source_pose_graphics_v1(**without)
    renderer=args["trusted_native_renderer_registry"][
        args["renderer_witness_key"]
    ]
    renderer["cpp_native_raster_execution_independently_verified"]=False
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        independently_replay_registered_source_pose_graphics_v1(**args)


def test_rehashed_local_court_does_not_override_independent_raw_replay():
    _,args=_case()
    court=independently_replay_registered_source_pose_graphics_v1(**args)
    forged=replace(court,known_rgb_mae=.09,court_hash="")
    forged=replace(forged,court_hash=source_pose_graphics_court_hash_v1(forged))
    with pytest.raises(QualificationError,match="CLAIMED_COURT_REPLAY_DRIFT"):
        independently_replay_registered_source_pose_graphics_v1(
            **args,claimed_court=forged.to_dict(),
        )


@pytest.mark.parametrize("field",[
    "target_face_owner_i32","target_known_mask",
    "native_front_face_i32","native_color_authority_u8",
    "native_unknown_occlusion_u8",
])
def test_tampered_required_native_or_source_image_sidecars_fail_independent_sha(field):
    raw,args=_case()
    import hashlib
    source=field.startswith("target")
    fields={
        k:raw[k] for k in (
            ("target_rgba_u8","target_face_owner_i32","target_known_mask")
            if source else
            ("native_rgba_u8","native_front_face_i32",
             "native_color_authority_u8","native_unknown_occlusion_u8")
        )
    }
    fields.pop(field)
    malformed=_npz(**fields)
    parameter="source_raster_bundle" if source else "native_raster_bundle"
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        independently_replay_registered_source_pose_graphics_v1(
            **{**args,parameter:malformed},
        )
    # Even if the test issuer dishonestly preapproved the wrong SHA,
    # archive schema rejects missing authority/owner/mask sidecars.
    registry="trusted_source_registry" if source else "trusted_native_renderer_registry"
    key="source_witness_key" if source else "renderer_witness_key"
    patched={**args[registry][args[key]],
             "raster_bundle_sha256":hashlib.sha256(malformed).hexdigest()}
    with pytest.raises(QualificationError,match="RASTER_ARCHIVE_DOMAIN"):
        independently_replay_registered_source_pose_graphics_v1(
            **{**args,parameter:malformed,registry:{args[key]:patched}},
        )


def test_self_authored_npz_cannot_claim_renderer_authority_without_external_issuer():
    _,args=_case()
    r=args["trusted_native_renderer_registry"][args["renderer_witness_key"]]
    r["raster_artifact_not_student_prediction"]=False
    with pytest.raises(QualificationError,match="RAW_INDEPENDENT_RASTER_MISSING"):
        independently_replay_registered_source_pose_graphics_v1(**args)
