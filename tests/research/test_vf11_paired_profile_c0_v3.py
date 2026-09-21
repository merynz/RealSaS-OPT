from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

from paired_profile_c0_v3 import (  # noqa: E402
    choose_paired_anchors,
    point_to_cell_index,
    run_paired_profile,
)


class TinyField(nn.Module):
    def __init__(self, channels=2, hidden=8):
        super().__init__()
        width = 3 * channels
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.sdf_head = nn.Linear(hidden, 1)

    @staticmethod
    def sample(planes, points):
        b, _, c, h, w = planes.shape
        x, y, z = points.unbind(-1)
        grids = torch.stack(
            [
                torch.stack([x, y], -1),
                torch.stack([x, z], -1),
                torch.stack([y, z], -1),
            ],
            dim=1,
        )
        sampled = F.grid_sample(
            planes.reshape(b * 3, c, h, w),
            grids.reshape(b * 3, points.shape[1], 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return (
            sampled.squeeze(-1)
            .reshape(b, 3, c, points.shape[1])
            .permute(0, 3, 1, 2)
            .reshape(b, points.shape[1], 3 * c)
        )

    def forward(self, planes, points):
        h = self.body(self.sample(planes, points))
        return {"sdf": self.sdf_head(h).squeeze(-1)}


def test_anchor_selection_is_deterministic_and_distinct_at_depth5():
    torch.manual_seed(51)
    field = TinyField().double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    kwargs = dict(
        domain_lo=-0.875,
        domain_hi=0.875,
        anchor_depth=7,
        distinct_depth=5,
        volume_count=4,
        near_zero_count=4,
        near_zero_candidate_count=128,
        seed=1234,
    )
    a = choose_paired_anchors(field, planes, **kwargs)
    b = choose_paired_anchors(field, planes, **kwargs)
    assert a == b
    assert len(a) == 8
    keys = {
        point_to_cell_index(np.asarray(x["point"]), 5, -0.875, 0.875)
        for x in a
    }
    assert len(keys) == 8
    assert sum(x["stratum"] == "VOLUME" for x in a) == 4
    assert sum(x["stratum"] == "NEAR_ZERO" for x in a) == 4


def test_paired_profile_emits_all_depth_micro_pairs_without_contradiction():
    torch.manual_seed(53)
    field = TinyField().double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    anchors = choose_paired_anchors(
        field,
        planes,
        domain_lo=-0.875,
        domain_hi=0.875,
        anchor_depth=7,
        distinct_depth=4,
        volume_count=2,
        near_zero_count=2,
        near_zero_candidate_count=64,
        seed=4321,
    )
    out = run_paired_profile(
        field,
        planes,
        anchors=anchors,
        depths=[4, 5],
        domain_lo=-0.875,
        domain_hi=0.875,
        max_micro_depth=2,
        logger=None,
    )
    assert out["schema"] == "RealSaS.VF11C0PairedProfile.v3"
    assert out["anchor_count"] == 4
    assert out["parent_count"] == 8
    assert out["empirical_contradiction_count"] == 0
    for rec in out["records"]:
        assert set(rec["by_micro"]) == {"0", "1", "2"}
