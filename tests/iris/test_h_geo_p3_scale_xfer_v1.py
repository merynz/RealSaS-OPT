from __future__ import annotations

import numpy as np
import torch

from tools.iris_vnext.h_geo_p3_efreq_v1 import NormalizedCameraV1
from tools.iris_vnext.h_geo_p3_scale_xfer_v1 import (
    SCALE_XFER_EPS_IRIS,
    classify_scale_transfer_v1,
    deterministic_unit_direction_v2,
    nearest_nonadjacent_candidates_v1,
    paired_surface_token_stats_v1,
)


def _prereg():
    return {
        "locked_signals": {
            "target_multisurface": {
                "pooled_tail_median_competing_margin_iris_lte": 1.0 / 128.0,
                "or_pooled_tail_fraction_with_second_sheet_within_1_over_128_gte": 0.25,
            },
            "view_input_resolution": {
                "pooled_tail_median_same_token_cell_fraction_gte": 0.5,
                "or_pooled_tail_distinct_token_view_count_lte_control_minus": 1.0,
                "or_pooled_tail_max_token_cosine_distance_lte_control_ratio": 0.70,
            },
            "carrier_transfer": {
                "p3_h_geo_to_p2_h_tail_delta_ratio_at_1_over_128_lte": 0.85,
                "and_refined128_to_bilinear128_tail_delta_ratio_at_1_over_128_lte": 1.10,
            },
        },
        "interpretation": {
            "TARGET_MULTISURFACE_LIMITED_next": "TARGET",
            "VIEW_INPUT_RESOLUTION_LIMITED_next": "VIEW",
            "CARRIER_TRANSFER_LIMITED_next": "CARRIER",
            "MIXED_next": "MIXED",
            "UNRESOLVED_next": "UNRESOLVED",
        },
    }


def _group(**kw):
    base = {
        "median_competing_surface_margin_iris": 0.1,
        "fraction_second_sheet_within_1_over_128": 0.0,
        "median_same_token_cell_fraction": 0.0,
        "median_distinct_token_cell_view_count": 6.0,
        "median_maximum_token_cosine_distance": 0.5,
        "median_stage_delta_P3_H_GEO_eps_0.00781250": 1.0,
        "median_stage_delta_P2_H_BASELINE_eps_0.00781250": 1.0,
        "median_stage_delta_P3_REFINED128_SAMPLED_PLANES_eps_0.00781250": 1.2,
        "median_stage_delta_P3_BILINEAR128_SAMPLED_PLANES_eps_0.00781250": 1.0,
    }
    base.update(kw)
    return base


def test_scale_list_is_locked_multiscale():
    assert SCALE_XFER_EPS_IRIS == (1/1024, 1/512, 1/256, 1/128, 1/64, 1/32)


def test_direction_is_deterministic_and_unit():
    a = deterministic_unit_direction_v2(3, 17)
    b = deterministic_unit_direction_v2(3, 17)
    assert np.array_equal(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-12


def test_nonadjacent_candidate_excludes_vertex_adjacent_face():
    vertices = np.asarray([
        [0,0,0], [1,0,0], [0,1,0],
        [3,0,0], [4,0,0], [3,1,0],
    ], dtype=np.float64)
    faces = np.asarray([[0,1,2], [3,4,5]], dtype=np.int64)
    tri = vertices[faces]
    normals = np.asarray([[0,0,1],[0,0,1]], dtype=np.float64)
    adjacency = [np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64)]
    out = nearest_nonadjacent_candidates_v1(np.asarray([0.2,0.2,0.1]), 0.1, 0, tri, normals, adjacency, k=2)
    assert out and out[0]["face_index"] == 1
    assert out[0]["margin_over_nearest_iris"] > 2.0


def test_frozen_token_pair_sampling_executes_with_expected_geometry():
    torch.manual_seed(7)
    tokens = torch.randn(1, 8, 1369, 384)
    camera = NormalizedCameraV1(
        origin_iris=np.asarray([0.0, 0.0, 2.0], dtype=np.float64),
        right=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        up=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        forward=np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
        half_extent_iris=1.0,
    )
    stats = paired_surface_token_stats_v1(
        tokens,
        [camera] * 8,
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([0.01, 0.0, 0.0], dtype=np.float64),
    )
    assert stats["both_in_frame_view_count"] == 8.0
    assert stats["same_token_cell_fraction"] == 1.0
    assert stats["distinct_token_cell_view_count"] == 0.0
    assert np.isfinite(stats["median_token_cosine_distance"])
    assert np.isfinite(stats["maximum_token_cosine_distance"])
    assert stats["median_projected_separation_token_cells"] > 0.0


def test_classifier_can_isolate_each_family_and_mixed():
    control = _group()
    pooled = {"tail": _group(median_competing_surface_margin_iris=0.005), "control": control}
    sig, cls, nxt = classify_scale_transfer_v1(pooled, _prereg())
    assert cls == "TARGET_MULTISURFACE_LIMITED" and sig["target_multisurface"] and nxt == "TARGET"

    pooled = {"tail": _group(median_same_token_cell_fraction=0.75), "control": control}
    sig, cls, nxt = classify_scale_transfer_v1(pooled, _prereg())
    assert cls == "VIEW_INPUT_RESOLUTION_LIMITED" and sig["view_input_resolution"] and nxt == "VIEW"

    pooled = {"tail": _group(**{
        "median_stage_delta_P3_H_GEO_eps_0.00781250": 0.7,
        "median_stage_delta_P3_REFINED128_SAMPLED_PLANES_eps_0.00781250": 1.05,
    }), "control": control}
    sig, cls, nxt = classify_scale_transfer_v1(pooled, _prereg())
    assert cls == "CARRIER_TRANSFER_LIMITED" and sig["carrier_transfer"] and nxt == "CARRIER"

    pooled = {"tail": _group(
        median_competing_surface_margin_iris=0.005,
        median_same_token_cell_fraction=0.75,
    ), "control": control}
    sig, cls, nxt = classify_scale_transfer_v1(pooled, _prereg())
    assert cls == "MIXED" and sum(sig.values()) == 2 and nxt == "MIXED"
