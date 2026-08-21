from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from descriptor_v2 import DescriptorV2Config, DescriptorV2Head
from descriptor_losses_v2 import (
    hard_negative_margin_loss,
    observation_supervised_contrastive_loss,
    same_view_dual_softmax_loss,
    sample_dense_native,
)


def test_head_shapes_and_norms():
    torch.manual_seed(0)
    head = DescriptorV2Head(DescriptorV2Config(coarse_in_dim=128, fine_in_dim=64))
    coarse = torch.randn(16, 128, 32, 32)
    fine = torch.randn(16, 64, 64, 64)
    out = head(coarse, fine)
    assert out["descriptor_coarse"].shape == (16, 64, 32, 32)
    assert out["descriptor_fine"].shape == (16, 32, 64, 64)
    assert out["descriptor_coarse_log_sigma"].shape == (16, 1, 32, 32)
    assert out["descriptor_fine_log_sigma"].shape == (16, 1, 64, 64)
    assert torch.allclose(out["descriptor_coarse"].norm(dim=1).mean(), torch.tensor(1.0), atol=1e-4)
    assert torch.allclose(out["descriptor_fine"].norm(dim=1).mean(), torch.tensor(1.0), atol=1e-4)


def _perfect_observations(B=1, V=3, N=5, D=8):
    base = F.normalize(torch.eye(N, D)[:N], dim=-1)
    zA = base.view(1, 1, N, D).repeat(B, V, 1, 1)
    zB = zA.clone()
    vis = torch.ones(B, V, N, dtype=torch.bool)
    return zA, zB, vis


def test_dual_and_margin_prefer_correct_identity():
    zA, zB, vis = _perfect_observations()
    good_dual = same_view_dual_softmax_loss(zA, zB, vis, vis)
    good_margin = hard_negative_margin_loss(zA, zB, vis, vis)
    bad = zB.roll(1, dims=2)
    bad_dual = same_view_dual_softmax_loss(zA, bad, vis, vis)
    bad_margin = hard_negative_margin_loss(zA, bad, vis, vis)
    assert good_dual < bad_dual
    assert good_margin < bad_margin


def test_observation_supcon_prefers_same_carrier_across_views_and_poses():
    zA, zB, vis = _perfect_observations(V=4)
    good = observation_supervised_contrastive_loss(zA, zB, vis, vis)
    bad = observation_supervised_contrastive_loss(zA, zB.roll(1, dims=2), vis, vis)
    assert good < bad


def test_sampling_native_coordinate_hits_corner_and_center():
    field = torch.zeros(1, 1, 1, 4, 4)
    field[0, 0, 0, 0, 0] = 1.0
    field[0, 0, 0, -1, -1] = 2.0
    xy = torch.tensor([[[[0.0, 0.0], [255.0, 255.0]]]])
    got = sample_dense_native(field, xy, image_size=256)[0, 0, :, 0]
    assert torch.allclose(got, torch.tensor([1.0, 2.0]), atol=1e-6)


if __name__ == "__main__":
    test_head_shapes_and_norms()
    test_dual_and_margin_prefer_correct_identity()
    test_observation_supcon_prefers_same_carrier_across_views_and_poses()
    test_sampling_native_coordinate_hits_corner_and_center()
    print("PASS")
