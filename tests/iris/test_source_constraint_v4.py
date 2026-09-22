import numpy as np
import torch

from models.iris.v4.scene_first_signed_v4 import (
    SceneFirstSignedGeometryConfigV4,
    SceneFirstSignedGeometryV4,
)
from models.iris.v4.mixed_precision_query_v4 import (
    query_hard_negative_ray_min_sdf_v4,
)
from models.iris.v4.source_constraint_v4 import (
    SourceConstraintSamplingPolicyV4,
    build_source_constraint_view_state_v4,
    sample_hard_negative_refresh_candidates_v4,
    sample_source_constraint_view_v4,
    select_hard_negative_replay_bank_v4,
)


def _state():
    mask = np.zeros((128, 128), dtype=bool)
    mask[48:80, 48:80] = True
    admitted = np.ones_like(mask)
    return build_source_constraint_view_state_v4(mask, admitted)


def test_v4_sampling_retains_near_far_and_hard_replay_every_view_step():
    state = _state()
    policy = SourceConstraintSamplingPolicyV4()
    hard_bank = state.far_background_flat[: policy.hard_negative_bank_size_per_view]
    a = sample_source_constraint_view_v4(
        state,
        fit_seed=26091909,
        training_step=17,
        view_index=2,
        policy=policy,
        hard_negative_flat_indices=hard_bank,
    )
    b = sample_source_constraint_view_v4(
        state,
        fit_seed=26091909,
        training_step=17,
        view_index=2,
        policy=policy,
        hard_negative_flat_indices=hard_bank,
    )
    expected = (
        policy.foreground_component_rays_per_component_per_view * state.component_count
        + policy.foreground_boundary_rays_per_view
        + policy.foreground_interior_rays_per_view
        + policy.near_background_rays_per_view
        + policy.far_background_rays_per_view
        + policy.hard_negative_replay_rays_per_view
    )
    assert a.ray_count == expected
    assert np.array_equal(a.flat_pixel_index, b.flat_pixel_index)
    assert int(np.sum(a.hard_replay)) == policy.hard_negative_replay_rays_per_view
    background = a.target_foreground < 0.5
    assert np.all(a.distance_to_foreground_px[background] > 0.0)
    assert np.all(a.distance_to_foreground_px[~background] == 0.0)
    assert np.any((a.distance_to_foreground_px > 0) & (a.distance_to_foreground_px <= policy.near_boundary_max_px))
    assert np.any(a.distance_to_foreground_px > policy.near_boundary_max_px)


def test_v4_hard_negative_refresh_is_broad_deterministic_and_selects_lowest_sdf():
    state = _state()
    policy = SourceConstraintSamplingPolicyV4()
    a = sample_hard_negative_refresh_candidates_v4(
        state,
        fit_seed=26091909,
        refresh_index=3,
        view_index=1,
        count=policy.hard_negative_refresh_candidates_per_view,
    )
    b = sample_hard_negative_refresh_candidates_v4(
        state,
        fit_seed=26091909,
        refresh_index=3,
        view_index=1,
        count=policy.hard_negative_refresh_candidates_per_view,
    )
    assert len(a) == policy.hard_negative_refresh_candidates_per_view
    assert np.array_equal(a, b)
    sdf = np.linspace(0.5, -0.5, len(a), dtype=np.float64)
    bank = select_hard_negative_replay_bank_v4(
        a,
        sdf,
        bank_size=policy.hard_negative_bank_size_per_view,
    )
    assert len(bank) == policy.hard_negative_bank_size_per_view
    expected = set(a[np.argsort(sdf)[: policy.hard_negative_bank_size_per_view]].tolist())
    assert set(bank.tolist()) == expected


def test_v4_hard_negative_query_promotes_cached_bfloat16_planes_to_model_dtype():
    cfg = SceneFirstSignedGeometryConfigV4(
        image_token_dim=8,
        camera_feature_dim=13,
        scene_dim=24,
        camera_embed_dim=12,
        plane_size=4,
        plane_channels=6,
        field_hidden_dim=16,
        field_residual_blocks=2,
        attention_heads=4,
        scene_layers=1,
        feedforward_multiplier=2,
        dropout=0.0,
    )
    torch.manual_seed(17)
    model = SceneFirstSignedGeometryV4(cfg).eval()
    planes = torch.randn(1, 3, 6, 8, 8, dtype=torch.bfloat16)
    points = torch.tensor(
        [[
            [[-0.5, 0.0, -0.5], [-0.5, 0.0, 0.0], [-0.5, 0.0, 0.5]],
            [[0.5, 0.0, -0.5], [0.5, 0.0, 0.0], [0.5, 0.0, 0.5]],
        ]],
        dtype=torch.float32,
    )
    ray_min = query_hard_negative_ray_min_sdf_v4(
        model,
        planes,
        points,
        query_chunk=16,
    )
    assert ray_min.shape == (1, 2)
    assert ray_min.dtype == torch.float32
    assert torch.isfinite(ray_min).all()
