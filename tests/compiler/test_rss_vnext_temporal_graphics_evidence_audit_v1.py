from __future__ import annotations

"""Temporal evidence is sampled independent replay, never complete source art."""

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.rss_vnext_source_pose_graphics_court_v1 import (
    evaluate_registered_source_pose_graphics_v1,
    source_pose_graphics_court_hash_v1,
)
from compiler.realsas_compiler_core.rss_vnext_temporal_graphics_evidence_audit_v1 import (
    audit_independently_replayed_graphics_samples_v1 as audit,
)
from compiler.rss_vnext.workers.product_downstream_executor_v1 import (
    execute_proof_capability_v1,
)
from tests.compiler.test_rss_vnext_source_pose_graphics_court_v1 import _fixture
from tests.compiler.test_rss_vnext_source_pose_graphics_independent_proof_replay_v1 import (
    _record,
)
from tests.compiler.test_rss_vnext_raw_motion_proof_replay_v1 import (
    _complete_product_preflight,
)


def court(*,motion=False):
    # Synthetic court is ONLY for contract regression, NOT a real trusted host.
    return evaluate_registered_source_pose_graphics_v1(**_fixture(motion=motion))


def reseal(original, **changes):
    row=replace(original,**changes,court_hash="")
    return replace(row,court_hash=source_pose_graphics_court_hash_v1(row))


def test_empty_and_single_view_sample_never_prove_complete_clip():
    empty=audit((),canonical_motion_duration_seconds=2.)
    assert empty["independently_replayed_positive_frame_count"]==0
    assert empty["original_rest_both_views_sampled"] is False
    assert empty["complete_temporal_coverage_proven"] is False
    assert empty["product_release_admission"] is False
    single=audit([court()],canonical_motion_duration_seconds=2.)
    assert single["independently_replayed_positive_frame_count"]==1
    assert single["per_source_view"][0]["registered_original_rest_frame_count"]==1
    assert single["per_source_view"][1]["registered_original_rest_frame_count"]==0
    assert single["hidden_region_reveal_proven"] is False


def test_two_rest_views_and_motion_samples_remain_only_sampled_evidence():
    rest=court()
    second=reseal(rest,source_view_index=1)
    motion=court(motion=True)
    another=reseal(motion,frame_index=2,frame_time_seconds=.5)
    final=reseal(motion,frame_index=3,frame_time_seconds=1.)
    inventory=audit(
        [rest,second,motion,another,final],
        canonical_motion_duration_seconds=1.,
    )
    assert inventory["original_rest_both_views_sampled"] is True
    assert inventory["artist_motion_samples_present"] is True
    assert inventory["per_source_view"][0]["registered_artist_motion_sample_count"]==3
    assert inventory["per_source_view"][0]["sampled_motion_frame_indices"]==[1,2,3]
    assert inventory["per_source_view"][0]["largest_between_sample_gap_seconds"]>0
    assert inventory["complete_temporal_coverage_proven"] is False
    assert inventory["product_release_admission"] is False


def test_reject_mixed_asset_camera_renderer_and_out_of_clip():
    rest=court()
    motion=court(motion=True)
    for changed,error in (
        (reseal(motion,original_source_asset_sha256="e"*64),"MIXED_SOURCE_ASSETS"),
        (reseal(motion,registered_native_camera_sha256="d"*64),"CAMERA_OR_RENDERER_DRIFT"),
        (reseal(motion,native_renderer_binary_sha256="d"*64),"CAMERA_OR_RENDERER_DRIFT"),
        (reseal(motion,frame_time_seconds=2.),"OUTSIDE_CLIP"),
    ):
        with pytest.raises(ValueError,match=error):
            audit([rest,changed],canonical_motion_duration_seconds=1.)


def test_reject_nonmonotonic_frames_and_duplicate_addresses():
    motion=court(motion=True)
    later=reseal(motion,frame_index=2,frame_time_seconds=.2)
    earlier=reseal(motion,frame_index=3,frame_time_seconds=.1)
    with pytest.raises(ValueError,match="NONMONOTONIC_MOTION"):
        audit([motion,later,earlier],canonical_motion_duration_seconds=1.)
    with pytest.raises(ValueError,match="DUPLICATE_COURT_ADDRESS"):
        audit([motion,motion],canonical_motion_duration_seconds=1.)


def test_reject_self_rehashed_forged_product_pass():
    motion=court(motion=True)
    forged=reseal(motion,product_release_admission=True)
    with pytest.raises(ValueError,match="POSITIVE_COURT_REQUIRED"):
        audit([forged],canonical_motion_duration_seconds=1.)


def test_product_proof_injects_only_host_replayed_positive_courts(tmp_path):
    cas,request,_,_=_complete_product_preflight(tmp_path)
    replay,_,_=_record(cas,request)
    untrusted=execute_proof_capability_v1(
        request,read_object=cas.read_object,put_bytes=cas.put_bytes,
    )
    missing=untrusted["diagnostics"]["source_pose_graphics_courts"][
        "independent_temporal_sample_inventory"
    ]
    assert missing["independently_replayed_positive_frame_count"]==0
    trusted=execute_proof_capability_v1(
        request,read_object=cas.read_object,put_bytes=cas.put_bytes,
        trusted_source_native_pixel_replayer=replay,
    )
    measured=trusted["diagnostics"]["source_pose_graphics_courts"][
        "independent_temporal_sample_inventory"
    ]
    assert measured["independently_replayed_positive_frame_count"]==1
    assert measured["complete_temporal_coverage_proven"] is False
    assert measured["hidden_region_reveal_proven"] is False
    assert trusted["status"]=="FAIL"
    assert trusted["diagnostics"]["product_revision_minted"] is False
