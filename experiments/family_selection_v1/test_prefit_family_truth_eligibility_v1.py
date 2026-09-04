from __future__ import annotations

import numpy as np

from experiments.family_selection_v1.prefit_family_truth_eligibility_v1 import (
    MIN_VISIBLE_NORMAL_POLARITY_PURITY,
    RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2,
    _Camera,
    normal_polarity_purity,
    projected_triangle_area_px2,
)


def camera() -> _Camera:
    return _Camera(
        forward=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        right=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        screen_up=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        half_extent=1.0,
        yaw_deg=0.0,
    )


def test_projected_triangle_area_policy_is_pixel_area_not_world_area():
    # With half_extent=1 at 1024 px, 1/512 world unit is exactly one pixel.
    stable = np.asarray([[[0.0, 0.0, 0.0], [2.0 / 512.0, 0.0, 0.0], [0.0, 2.0 / 512.0, 0.0]]])
    edge = np.asarray([[[0.0, 0.0, 0.0], [2.0 / 512.0, 0.0, 0.0], [0.0, 1.0 / 512.0, 0.0]]])
    stable_area = float(projected_triangle_area_px2(stable, camera())[0])
    edge_area = float(projected_triangle_area_px2(edge, camera())[0])
    assert abs(stable_area - 2.0) < 1e-12
    assert abs(edge_area - 1.0) < 1e-12
    assert stable_area > RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2
    assert edge_area <= RASTER_UNSTABLE_PROJECTED_TRIANGLE_AREA_MAX_PX2


def test_normal_polarity_purity_is_orientation_agnostic_but_inconsistency_sensitive():
    purity_pos, sign_pos = normal_polarity_purity([1.0] * 9 + [-1.0])
    purity_neg, sign_neg = normal_polarity_purity([-2.0] * 9 + [3.0])
    mixed, _ = normal_polarity_purity([1.0] * 5 + [-1.0] * 5)
    assert purity_pos == MIN_VISIBLE_NORMAL_POLARITY_PURITY
    assert purity_neg == MIN_VISIBLE_NORMAL_POLARITY_PURITY
    assert sign_pos == 1
    assert sign_neg == -1
    assert mixed == 0.5


def test_empty_normal_support_cannot_accidentally_pass_purity_gate():
    purity, polarity = normal_polarity_purity([])
    assert purity == 0.0
    assert polarity == 0
    assert purity < MIN_VISIBLE_NORMAL_POLARITY_PURITY
