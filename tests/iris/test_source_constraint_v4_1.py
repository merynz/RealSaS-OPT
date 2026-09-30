import numpy as np

from models.iris.v4.source_constraint_v4 import build_source_constraint_view_state_v4
from models.iris.v4.source_constraint_v4_1 import (
    SourceConstraintSamplingPolicyV41,
    sample_hard_negative_refresh_candidates_by_band_v41,
    sample_hard_positive_refresh_candidates_v41,
    sample_source_constraint_view_v41,
    select_hard_negative_replay_bank_v41,
    select_hard_positive_replay_bank_v41,
    source_background_band_pools_v41,
)


def _state():
    mask = np.zeros((128, 128), dtype=bool)
    mask[48:80, 48:80] = True
    admitted = np.ones_like(mask)
    return build_source_constraint_view_state_v4(mask, admitted)


def test_v41_background_bands_are_disjoint_complete_and_nonempty():
    state = _state()
    bands = source_background_band_pools_v41(state)
    assert set(bands) == {"0_2", "2_8", "8_32", "gt_32"}
    joined = np.concatenate(list(bands.values()))
    assert len(np.unique(joined)) == len(joined)
    assert set(joined.tolist()) == set(state.all_background_flat.tolist())
    assert all(len(v) > 0 for v in bands.values())


def test_v41_hard_positive_selects_largest_margin_violation():
    idx = np.asarray([10, 11, 12, 13], dtype=np.int64)
    ray_min = np.asarray([-0.01, 0.03, -0.001, 0.02], dtype=np.float64)
    boundary = np.asarray([True, True, False, False])
    bank = select_hard_positive_replay_bank_v41(
        idx,
        ray_min,
        boundary,
        boundary_inside_margin_normalized=0.001,
        interior_inside_margin_normalized=0.002,
        bank_size=2,
    )
    assert bank.tolist() == [11, 13]


def test_v41_hard_negative_selects_lowest_sdf_inside_each_band():
    idx = np.asarray([20, 21, 22, 23], dtype=np.int64)
    sdf = np.asarray([0.1, -0.3, 0.02, -0.1], dtype=np.float64)
    bank = select_hard_negative_replay_bank_v41(idx, sdf, bank_size=2)
    assert bank.tolist() == [21, 23]


def test_v41_sampler_retains_hard_positive_and_all_four_background_bands():
    state = _state()
    policy = SourceConstraintSamplingPolicyV41()
    positive = sample_hard_positive_refresh_candidates_v41(
        state, fit_seed=26091909, refresh_index=2, view_index=0,
        count=policy.hard_positive_bank_size_per_view,
    )
    negative = sample_hard_negative_refresh_candidates_by_band_v41(
        state, fit_seed=26091909, refresh_index=2, view_index=0,
        count_per_band=policy.hard_negative_bank_size_per_band_per_view,
    )
    a = sample_source_constraint_view_v41(
        state,
        fit_seed=26091909,
        training_step=81,
        view_index=0,
        policy=policy,
        hard_positive_flat_indices=positive,
        hard_negative_flat_indices_by_band=negative,
    )
    b = sample_source_constraint_view_v41(
        state,
        fit_seed=26091909,
        training_step=81,
        view_index=0,
        policy=policy,
        hard_positive_flat_indices=positive,
        hard_negative_flat_indices_by_band=negative,
    )
    assert np.array_equal(a.flat_pixel_index, b.flat_pixel_index)
    fg = a.target_foreground > 0.5
    assert np.sum(fg & a.hard_replay) >= policy.hard_positive_replay_rays_per_view
    distance = a.distance_to_foreground_px[~fg]
    assert np.any((distance > 0) & (distance <= 2))
    assert np.any((distance > 2) & (distance <= 8))
    assert np.any((distance > 8) & (distance <= 32))
    assert np.any(distance > 32)
    assert set(a.component_id[a.component_id >= 0].tolist()) == set(range(state.component_count))
