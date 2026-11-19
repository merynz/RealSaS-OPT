from __future__ import annotations

"""Fail-closed source-registered post-Mechanics Graphics court regressions.

Independent registry rows are manually authored *test doubles*. No synthetic
fixture in this file constitutes actual native/source training or ProductProof.
"""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    NATIVE_ADMISSION,SOURCE_ADMISSION,SOURCE_REST,SOURCE_MOTION,
    SourcePoseGraphicsThresholdsV1,
    _array_sha,evaluate_registered_source_pose_graphics_v1,
    reopen_source_pose_graphics_court_v1,
    source_pose_graphics_court_hash_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _fixture(*,motion=False,inferred=False,unknown=False,face_mismatch=False,
             rgb_mismatch=False):
    n=16
    target=np.zeros((n,n,4),dtype=np.uint8)
    mask=np.zeros((n,n),dtype=np.bool_)
    mask[2:14,2:14]=True
    foreground=np.zeros((n,n),dtype=np.bool_)
    foreground[4:12,4:12]=True
    target[foreground]=[63,114,203,255]
    owner=np.full((n,n),-1,dtype=np.int32)
    owner[foreground]=0
    native=target.copy()
    if rgb_mismatch:
        native[foreground,0]=0
    native_owner=owner.copy()
    if face_mismatch:
        native_owner[foreground]=1
    auth=np.zeros((n,n),dtype=np.uint8)
    auth[foreground]=2 if inferred else 1
    unresolved=np.zeros((n,n),dtype=np.uint8)
    if unknown:
        unresolved[foreground]=1
    hashes={
        "canonical_mesh_lineage_hash":"a"*64,
        "canonical_rig_lineage_hash":"b"*64,
        "canonical_skin_lineage_hash":"c"*64,
        "compiled_graphics_hash":"d"*64,
        "canonical_motion_lineage_hash":"e"*64,
    }
    frames=[{
        "frame_index":0,"time_seconds":0.,
        "final_world_vertex_positions_sha256":"f"*64,
        "xpbd_surface_state_hash":None,
    }]
    idx=0
    kind=SOURCE_REST
    if motion:
        idx=1
        kind=SOURCE_MOTION
        frames.append({
            "frame_index":1,"time_seconds":1./60.,
            "final_world_vertex_positions_sha256":"0"*64,
            "xpbd_surface_state_hash":"1"*64,
        })
    receipt={
        "schema":"RealSaS.RSSvNext.LiveMechanicsNativePoseTransport.v1",
        "authority":"LIVE_MECHANICS_FINAL_SURFACE_NOT_SOURCE_OR_PRODUCT_PROOF",
        "source_color_or_visibility_qualified":False,
        "product_release_admission":False,
        "native_pose_bytes_sha256":"2"*64,
        "camera_policy_sha256":"3"*64,
        "resolution":n,
        "mesh_lineage_hash":hashes["canonical_mesh_lineage_hash"],
        "rig_lineage_hash":hashes["canonical_rig_lineage_hash"],
        "skin_lineage_hash":hashes["canonical_skin_lineage_hash"],
        "frames":frames,
    }
    policy=SourcePoseGraphicsThresholdsV1(
        max_known_rgb_mae=.015,
        max_known_alpha_mae=.015,
        min_known_silhouette_iou=.99,
        min_known_face_owner_accuracy=1.,
    )
    kwargs=dict(
        target_rgba_u8=target,target_face_owner_i32=owner,
        target_known_mask=mask,native_rgba_u8=native,
        native_front_face_i32=native_owner,
        native_color_authority_u8=auth,
        native_unknown_occlusion_u8=unresolved,
        native_face_count=2,
        original_source_asset_sha256="4"*64,
        original_source_file_sha256="5"*64,
        source_view_index=0,frame_index=idx,source_target_kind=kind,
        source_registration_sha256="6"*64,
        source_witness_key="independent-native-source-approval",
        renderer_witness_key="independent-cpp-execution-approval",
        native_renderer_binary_sha256="7"*64,
        live_pose_receipt=receipt,thresholds=policy,
        **hashes,
    )
    common={
        **hashes,
        "source_view_index":0,
        "frame_index":idx,
        "frame_time_seconds":float(frames[idx]["time_seconds"]),
        "native_pose_bytes_sha256":receipt["native_pose_bytes_sha256"],
        "registered_native_camera_sha256":receipt["camera_policy_sha256"],
    }
    source={
        **common,
        "admission_status":SOURCE_ADMISSION,
        "source_witness_key":kwargs["source_witness_key"],
        "original_source_asset_sha256":kwargs["original_source_asset_sha256"],
        "original_source_file_sha256":kwargs["original_source_file_sha256"],
        "source_registration_sha256":kwargs["source_registration_sha256"],
        "source_target_kind":kind,
        "target_rgba_sha256":_array_sha(target),
        "target_face_owner_sha256":_array_sha(owner),
        "target_known_mask_sha256":_array_sha(mask),
        "frozen_threshold_policy_sha256":policy.policy_sha256(),
        "original_two_source_observations_only":True,
        "source_camera_registration_independently_verified":True,
        "native_face_owner_transport_independently_verified":True,
        "source_known_mask_has_explicit_unknown":True,
        "generated_six_views_used_as_source_truth":False,
        "virtual_preview_camera_misrepresented_as_source":False,
        "source_issuer_role":"INDEPENDENT_SOURCE_WITNESS_AUTHORITY",
    }
    if motion:
        source["artist_verified_postmotion_rgba_and_visibility"]=True
    renderer={
        **common,
        "admission_status":NATIVE_ADMISSION,
        "renderer_witness_key":kwargs["renderer_witness_key"],
        "native_renderer_binary_sha256":kwargs["native_renderer_binary_sha256"],
        "native_rgba_sha256":_array_sha(native),
        "native_face_owner_sha256":_array_sha(native_owner),
        "native_color_authority_sha256":_array_sha(auth),
        "native_unknown_occlusion_sha256":_array_sha(unresolved),
        "final_world_vertex_positions_sha256":
            frames[idx]["final_world_vertex_positions_sha256"],
        "native_cpp_executed_for_exact_pose":True,
        "native_render_owner_map_independently_captured":True,
        "native_unknown_occlusion_sidecar_captured":True,
        "renderer_issuer_role":"INDEPENDENT_CPP_RENDER_EXECUTION",
        "source_rgb_teacher_was_renderer_output":False,
    }
    kwargs["trusted_source_registry"]={kwargs["source_witness_key"]:source}
    kwargs["trusted_native_renderer_registry"]={
        kwargs["renderer_witness_key"]:renderer,
    }
    return kwargs


def _reopen_args(kwargs):
    return {field:kwargs[field] for field in (
        "canonical_mesh_lineage_hash","canonical_rig_lineage_hash",
        "canonical_skin_lineage_hash","compiled_graphics_hash",
        "canonical_motion_lineage_hash",
    )}


def test_source_registered_original_rgba_and_face_owner_can_pass_local_only():
    kwargs=_fixture()
    proof=evaluate_registered_source_pose_graphics_v1(**kwargs)
    assert proof.source_registered_pixels_passed is True
    assert proof.known_pixel_count==144
    assert proof.known_foreground_pixel_count==64
    assert proof.known_background_pixel_count==80
    assert proof.known_rgb_mae==0.
    assert proof.known_alpha_mae==0.
    assert proof.known_silhouette_iou==1.
    assert proof.known_face_owner_accuracy==1.
    assert proof.known_unknown_occlusion_fraction==0.
    assert proof.known_inferred_color_fraction==0.
    assert proof.product_release_admission is False
    assert proof.hidden_surface_color_proven is False
    assert proof.unobserved_motion_quality_proven is False
    reopened=reopen_source_pose_graphics_court_v1(
        proof.to_dict(),**_reopen_args(kwargs),
    )
    assert reopened==proof


def test_artist_validated_postmotion_is_accepted_without_becoming_source_3d():
    kwargs=_fixture(motion=True)
    proof=evaluate_registered_source_pose_graphics_v1(**kwargs)
    assert proof.source_target_kind==SOURCE_MOTION
    assert proof.frame_index==1
    assert proof.frame_time_seconds==1./60.
    assert proof.source_registered_pixels_passed
    assert not proof.product_release_admission
    kwargs["trusted_source_registry"][kwargs["source_witness_key"]][
        "artist_verified_postmotion_rgba_and_visibility"
    ]=False
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)


@pytest.mark.parametrize("modified",[
    "rgb_mismatch","face_mismatch","inferred","unknown",
])
def test_real_native_pixel_mismatch_is_measured_not_silently_accepted(modified):
    kwargs=_fixture(**{modified:True})
    proof=evaluate_registered_source_pose_graphics_v1(**kwargs)
    assert proof.source_registered_pixels_passed is False
    assert not proof.product_release_admission
    if modified=="rgb_mismatch":
        assert proof.known_rgb_mae>.015
    elif modified=="face_mismatch":
        assert proof.known_face_owner_accuracy<1.
    elif modified=="inferred":
        assert proof.known_inferred_color_fraction>0.
    else:
        assert proof.known_unknown_occlusion_fraction>0.


@pytest.mark.parametrize("field,value",[
    ("source_camera_registration_independently_verified",False),
    ("native_face_owner_transport_independently_verified",False),
    ("generated_six_views_used_as_source_truth",True),
    ("virtual_preview_camera_misrepresented_as_source",True),
    ("source_known_mask_has_explicit_unknown",False),
])
def test_virtual_unregistered_or_student_source_is_not_accepted(field,value):
    kwargs=_fixture()
    src=kwargs["trusted_source_registry"][kwargs["source_witness_key"]]
    src[field]=value
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)


def test_same_renderer_cannot_self_certify_source_or_hide_missing_sidecars():
    kwargs=_fixture()
    kwargs["trusted_source_registry"]={}
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)
    kwargs=_fixture()
    renderer=kwargs["trusted_native_renderer_registry"][kwargs["renderer_witness_key"]]
    renderer["source_rgb_teacher_was_renderer_output"]=True
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)
    kwargs=_fixture()
    renderer=kwargs["trusted_native_renderer_registry"][kwargs["renderer_witness_key"]]
    renderer["native_unknown_occlusion_sidecar_captured"]=False
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)


def test_tampered_pixel_provenance_or_pose_camera_is_rejected():
    kwargs=_fixture()
    kwargs["live_pose_receipt"]["camera_policy_sha256"]="8"*64
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)
    kwargs=_fixture()
    rgba=kwargs["native_rgba_u8"].copy()
    rgba[5,5,0]+=1
    kwargs["native_rgba_u8"]=rgba
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)


def test_rehashed_forged_court_cannot_mint_release_or_hidden_surface_proof():
    kwargs=_fixture()
    proof=evaluate_registered_source_pose_graphics_v1(**kwargs)
    forged=replace(proof,hidden_surface_color_proven=True,court_hash="")
    forged=replace(forged,court_hash=source_pose_graphics_court_hash_v1(forged))
    with pytest.raises(QualificationError,match="REQUALIFICATION_DRIFT"):
        reopen_source_pose_graphics_court_v1(
            forged.to_dict(),**_reopen_args(kwargs),
        )
    forged=replace(proof,product_release_admission=True,court_hash="")
    forged=replace(forged,court_hash=source_pose_graphics_court_hash_v1(forged))
    with pytest.raises(QualificationError,match="REQUALIFICATION_DRIFT"):
        reopen_source_pose_graphics_court_v1(
            forged.to_dict(),**_reopen_args(kwargs),
        )


def test_unqualified_thresholds_cannot_turn_bad_pixels_into_product_pass():
    kwargs=_fixture(rgb_mismatch=True)
    p=kwargs["thresholds"]
    bad=replace(p,max_known_rgb_mae=.9)
    kwargs["thresholds"]=bad
    with pytest.raises(QualificationError,match="THRESHOLD_PRODUCT_BOUNDS"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)


def test_unknown_source_mask_is_not_transparency_and_owner_witness_must_qualify():
    kwargs=_fixture()
    target=kwargs["target_rgba_u8"].copy()
    mask=kwargs["target_known_mask"].copy()
    mask[:,:]=False
    kwargs["target_known_mask"]=mask
    with pytest.raises(QualificationError,match="KNOWN_COVERAGE_REQUIRED"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)
    kwargs=_fixture()
    owner=kwargs["target_face_owner_i32"].copy()
    owner[0,0]=0
    kwargs["target_face_owner_i32"]=owner
    with pytest.raises(QualificationError,match="WITNESS_DRIFT"):
        evaluate_registered_source_pose_graphics_v1(**kwargs)
