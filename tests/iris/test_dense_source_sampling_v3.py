import numpy as np
import pytest

from models.iris.v3.dense_source_sampling_v3 import (
    DenseSourceSamplingPolicyV3,
    admitted_pixel_mask_for_camera,
    build_dense_source_view_sampling_state,
    camera_pixel_ray_origins_normalized,
    sample_dense_source_view_rays,
)


def test_component_sampling_is_deterministic_and_preserves_tiny_component():
    mask = np.zeros((32, 32), dtype=bool)
    mask[5:17, 5:17] = True
    mask[24, 25] = True
    state = build_dense_source_view_sampling_state(mask, np.ones_like(mask))
    assert state.component_count == 2
    policy = DenseSourceSamplingPolicyV3()
    a = sample_dense_source_view_rays(
        state,
        fit_seed=26091909,
        training_step=17,
        view_index=3,
        policy=policy,
    )
    b = sample_dense_source_view_rays(
        state,
        fit_seed=26091909,
        training_step=17,
        view_index=3,
        policy=policy,
    )
    assert np.array_equal(a.flat_pixel_index, b.flat_pixel_index)
    assert np.array_equal(a.component_id, b.component_id)
    assert set(map(int, a.component_id[a.component_id >= 0])) == {0, 1}
    assert int(np.sum(a.component_id == 1)) >= policy.foreground_component_rays_per_component_per_view
    assert int(np.sum(a.component_id == -1)) == policy.background_rays_per_view


def test_empty_global_interior_quota_is_reallocated_to_boundary():
    mask = np.zeros((16, 16), dtype=bool)
    mask[8, 8] = True
    state = build_dense_source_view_sampling_state(mask, np.ones_like(mask))
    assert not np.any(state.interior)
    policy = DenseSourceSamplingPolicyV3()
    selection = sample_dense_source_view_rays(
        state,
        fit_seed=26091909,
        training_step=1,
        view_index=0,
        policy=policy,
    )
    expected = (
        policy.foreground_component_rays_per_component_per_view
        + policy.additional_global_foreground_boundary_rays_per_view
        + policy.additional_global_foreground_interior_rays_per_view
        + policy.background_rays_per_view
    )
    assert selection.ray_count == expected
    assert int(np.sum(selection.boundary > 0.5)) == expected - policy.background_rays_per_view


def test_source_foreground_outside_admitted_domain_fails_closed():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    admitted = np.ones_like(mask)
    admitted[2, 2] = False
    with pytest.raises(ValueError, match="SOURCE_FOREGROUND_OUTSIDE_NORMALIZATION_RAY_DOMAIN"):
        build_dense_source_view_sampling_state(mask, admitted)


def test_exact_pixel_center_camera_rays_match_historical_axis_semantics():
    origin, direction = camera_pixel_ray_origins_normalized(
        np.asarray([0], dtype=np.int64),
        width=4,
        height=4,
        camera_origin_normalized=np.asarray([0.0, 0.0, -2.0]),
        camera_right=np.asarray([1.0, 0.0, 0.0]),
        camera_screen_up=np.asarray([0.0, 1.0, 0.0]),
        camera_forward=np.asarray([0.0, 0.0, 1.0]),
        camera_half_extent_normalized=1.0,
    )
    assert np.allclose(origin[0], [-0.75, 0.75, -2.0])
    assert np.allclose(direction[0], [0.0, 0.0, 1.0])


def test_camera_admission_is_exact_normalization_cube_ray_intersection():
    admitted = admitted_pixel_mask_for_camera(
        width=32,
        height=32,
        camera_origin_normalized=np.asarray([0.0, 0.0, -2.0]),
        camera_right=np.asarray([1.0, 0.0, 0.0]),
        camera_screen_up=np.asarray([0.0, 1.0, 0.0]),
        camera_forward=np.asarray([0.0, 0.0, 1.0]),
        camera_half_extent_normalized=1.0,
    )
    assert admitted.shape == (32, 32)
    assert bool(np.all(admitted))

    wide = admitted_pixel_mask_for_camera(
        width=32,
        height=32,
        camera_origin_normalized=np.asarray([0.0, 0.0, -2.0]),
        camera_right=np.asarray([1.0, 0.0, 0.0]),
        camera_screen_up=np.asarray([0.0, 1.0, 0.0]),
        camera_forward=np.asarray([0.0, 0.0, 1.0]),
        camera_half_extent_normalized=2.0,
    )
    assert 0 < int(wide.sum()) < wide.size
