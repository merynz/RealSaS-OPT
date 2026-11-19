from __future__ import annotations

"""Strict sampled-time inventory of independently host-replayed source courts.

This audit must only receive courts whose source/native bytes have already
been independently replayed by the host. It detects source-asset, renderer,
camera, time and frame-address drift. A few matching frames never prove
hidden regions or continuous animation.
"""

from collections import defaultdict
import math
from typing import Sequence

from .rss_vnext_source_pose_graphics_court_v1 import (
    SourcePoseGraphicsCourtV1, SOURCE_REST, SOURCE_MOTION,
    source_pose_graphics_court_hash_v1,
)

SCHEMA = "RealSaS.RSSvNext.IndependentTemporalGraphicsSampleAudit.v1"


def audit_independently_replayed_graphics_samples_v1(
    courts: Sequence[SourcePoseGraphicsCourtV1], *,
    canonical_motion_duration_seconds: float,
) -> dict:
    duration = canonical_motion_duration_seconds
    if (
        type(duration) is not float
        or not math.isfinite(duration) or duration < 0
        or not isinstance(courts, (tuple,list)) or len(courts) > 64
    ):
        raise ValueError("RSS_TEMPORAL_GRAPHICS_DURATION_OR_SAMPLE_BUDGET")
    if any(
        type(c) is not SourcePoseGraphicsCourtV1
        or c.court_hash != source_pose_graphics_court_hash_v1(c)
        or not c.source_registered_pixels_passed
        or c.product_release_admission is not False
        or c.hidden_surface_color_proven is not False
        or c.unobserved_motion_quality_proven is not False
        for c in courts
    ):
        raise ValueError("RSS_TEMPORAL_GRAPHICS_REPLAYED_POSITIVE_COURT_REQUIRED")
    if len({c.original_source_asset_sha256 for c in courts})>1:
        raise ValueError("RSS_TEMPORAL_GRAPHICS_MIXED_SOURCE_ASSETS")
    by_view=defaultdict(list)
    addressed=set()
    for c in courts:
        key=(c.source_view_index,c.frame_index,c.source_target_kind)
        if key in addressed:
            raise ValueError("RSS_TEMPORAL_GRAPHICS_DUPLICATE_COURT_ADDRESS")
        addressed.add(key)
        if (
            c.source_view_index not in (0,1)
            or not math.isfinite(c.frame_time_seconds)
            or c.frame_time_seconds < 0
            or c.source_target_kind not in (SOURCE_REST,SOURCE_MOTION)
        ):
            raise ValueError("RSS_TEMPORAL_GRAPHICS_FRAME_DOMAIN")
        if (
            c.source_target_kind==SOURCE_MOTION
            and (duration<=0 or c.frame_time_seconds>duration+1e-6)
        ):
            raise ValueError("RSS_TEMPORAL_GRAPHICS_OUTSIDE_CLIP")
        by_view[c.source_view_index].append(c)
    views=[]
    any_motion=False
    for vi in range(2):
        frames=by_view[vi]
        if frames and (
            len({c.registered_native_camera_sha256 for c in frames})!=1
            or len({c.native_renderer_binary_sha256 for c in frames})!=1
        ):
            raise ValueError("RSS_TEMPORAL_GRAPHICS_VIEW_CAMERA_OR_RENDERER_DRIFT")
        rest=[c for c in frames if c.source_target_kind==SOURCE_REST]
        motion=sorted(
            (c for c in frames if c.source_target_kind==SOURCE_MOTION),
            key=lambda c:(c.frame_time_seconds,c.frame_index),
        )
        if len(rest)>1:
            raise ValueError("RSS_TEMPORAL_GRAPHICS_DUPLICATE_REST_VIEW")
        times=[c.frame_time_seconds for c in motion]
        indices=[c.frame_index for c in motion]
        if any(
            b<=a or ib<=ia for a,b,ia,ib
            in zip(times,times[1:],indices,indices[1:])
        ):
            raise ValueError("RSS_TEMPORAL_GRAPHICS_NONMONOTONIC_MOTION")
        any_motion |= bool(motion)
        span=times[-1]-times[0] if times else 0.
        views.append({
            "source_view_index":vi,
            "registered_original_rest_frame_count":len(rest),
            "registered_artist_motion_sample_count":len(motion),
            "sampled_motion_frame_indices":indices,
            "sampled_motion_times_seconds":times,
            "sampled_motion_duration_fraction":(
                float(span/duration) if duration>0 else 0.
            ),
            "largest_between_sample_gap_seconds":max(
                (b-a for a,b in zip(times,times[1:])),default=0.,
            ),
            "independently_replayed_known_pixels":sum(
                c.known_pixel_count for c in frames
            ),
            "maximum_sampled_unknown_occlusion_fraction":max(
                (c.known_unknown_occlusion_fraction for c in frames),
                default=None,
            ),
            "registered_camera_and_renderer_consistent":bool(frames),
            "continuous_motion_visibility_proven":False,
            "hidden_surface_color_proven":False,
        })
    return {
        "schema":SCHEMA,
        "authority":"TRUSTED_SAMPLE_INVENTORY_NOT_CONTINUOUS_MOTION_PROOF",
        "independently_replayed_positive_frame_count":len(courts),
        "original_source_asset_sha256":(
            courts[0].original_source_asset_sha256 if courts else None
        ),
        "canonical_motion_duration_seconds":duration,
        "per_source_view":views,
        "original_rest_both_views_sampled":all(
            v["registered_original_rest_frame_count"]==1 for v in views
        ),
        "artist_motion_samples_present":any_motion,
        "complete_temporal_coverage_proven":False,
        "hidden_region_reveal_proven":False,
        "product_release_admission":False,
    }


__all__=["SCHEMA","audit_independently_replayed_graphics_samples_v1"]
