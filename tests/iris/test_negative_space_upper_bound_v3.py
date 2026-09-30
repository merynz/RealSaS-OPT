import numpy as np
import torch

from models.iris.v3.negative_space_upper_bound_v3 import (
    background_distance_to_source_px,
    boundary_band_background_pool,
    cyclic_boundary_band_selection,
    distance_binned_violation_summary,
    graded_background_full_ray_barrier,
    metric_consistent_background_margin_normalized,
)


def test_metric_margin_is_one_pixel_near_boundary_and_caps_at_historical_maximum():
    d = np.asarray([1.0, 2.0, 8.0, 32.0], dtype=np.float32)
    margin = metric_consistent_background_margin_normalized(
        d,
        camera_half_extent_normalized=1.0,
        raster_width=1024,
        maximum_margin_normalized=0.04,
    )
    assert np.isclose(margin[0], 2.0 / 1024.0)
    assert np.isclose(margin[1], 4.0 / 1024.0)
    assert np.isclose(margin[2], 16.0 / 1024.0)
    assert np.isclose(margin[3], 0.04)


def test_distance_transform_is_zero_on_foreground_and_metric_outside():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    d = background_distance_to_source_px(mask)
    assert d[4, 4] == 0.0
    assert d[4, 5] == 1.0
    assert np.isclose(d[5, 5], np.sqrt(2.0))


def test_boundary_band_cycle_is_deterministic_and_exhausts_pool():
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:10, 6:10] = True
    admitted = np.ones_like(mask)
    pool, _ = boundary_band_background_pool(mask, admitted, maximum_distance_px=4.0)
    count = 7
    seen = []
    visits = int(np.ceil(pool.size / count))
    for visit in range(visits):
        idx, _ = cyclic_boundary_band_selection(
            mask,
            admitted,
            visit_index=visit,
            count=count,
            maximum_distance_px=4.0,
        )
        seen.extend(map(int, idx))
    assert set(map(int, pool)).issubset(set(seen))
    a, ad = cyclic_boundary_band_selection(mask, admitted, visit_index=3, count=count, maximum_distance_px=4.0)
    b, bd = cyclic_boundary_band_selection(mask, admitted, visit_index=3, count=count, maximum_distance_px=4.0)
    assert np.array_equal(a, b)
    assert np.array_equal(ad, bd)


def test_graded_barrier_uses_per_ray_metric_margin_and_backpropagates_all_violations():
    sdf = torch.tensor(
        [[[0.001, 0.003, 0.010], [0.010, 0.020, 0.050]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    margin = torch.tensor([[0.002, 0.040]], dtype=torch.float64)
    out = graded_background_full_ray_barrier(sdf, margin)
    out["total"].backward()
    grad = sdf.grad[0]
    assert float(grad[0, 0]) < 0.0
    assert float(grad[0, 1]) == 0.0
    assert float(grad[0, 2]) == 0.0
    assert float(grad[1, 0]) < 0.0
    assert float(grad[1, 1]) < 0.0
    assert float(grad[1, 2]) == 0.0


def test_distance_binned_summary_localizes_boundary_failures():
    d = np.asarray([1.0, 4.0, 16.0, 40.0])
    ray_min = np.asarray([-0.01, 0.01, 0.02, 0.05])
    margin = np.asarray([0.002, 0.008, 0.032, 0.04])
    out = distance_binned_violation_summary(d, ray_min, margin)
    assert out["0_2"]["zero_crossing_fraction"] == 1.0
    assert out["2_8"]["zero_crossing_fraction"] == 0.0
    assert out["8_32"]["margin_violation_fraction"] == 1.0
    assert out["gt_32"]["margin_violation_fraction"] == 0.0
