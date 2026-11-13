from __future__ import annotations

import math

import torch

from models.rss_vnext.realsas_virtual_camera_policy_v1 import (
    target_canonical_and_six_cameras_v1,
)


def test_canonical_source_virtual_bank_exactly_matches_six_view_training_order():
    canonical, views = target_canonical_and_six_cameras_v1(
        2, radius=2., orthographic_half_extent=1.,
    )
    assert canonical.shape == (2, 2, 13)
    assert views.shape == (2, 6, 13)
    assert torch.equal(canonical, views[:, [2, 5]])
    for index, degrees in enumerate((-90, -45, 0, 45, 90, 180)):
        radians = math.radians(degrees)
        expected_direction = torch.tensor(
            [-math.sin(radians), 0., -math.cos(radians)],
            dtype=views.dtype,
        )
        assert torch.allclose(
            views[0, index, 9:12], expected_direction,
            atol=1e-6, rtol=0.,
        )
        assert torch.allclose(views[0, index, 12], torch.tensor(1.))
    assert torch.allclose(
        canonical[:, 0, 9:12], -canonical[:, 1, 9:12],
        atol=1e-6, rtol=0.,
    )
    assert torch.allclose(views[:, :, 7], torch.ones(2, 6))


def test_virtual_camera_has_valid_orthographic_frame_without_observation_claim():
    _, bank = target_canonical_and_six_cameras_v1(1)
    axes = torch.stack((bank[:, :, 3:6], bank[:, :, 6:9], bank[:, :, 9:12]), dim=-2)
    identity = torch.eye(3)
    assert torch.allclose(axes @ axes.transpose(-1, -2), identity, atol=1e-6)
    assert bool((torch.linalg.det(axes) > .99).all())
