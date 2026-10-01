from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

from profile_c1_handoff_v2 import profile_c1_handoff_v2  # noqa: E402


class TinyField(nn.Module):
    def __init__(self, channels=2, hidden=6):
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


def test_v2_handoff_profiler_preserves_primary_cohort_and_v1_pairing():
    torch.manual_seed(401)
    field = TinyField().double().eval()
    for parameter in field.parameters():
        parameter.data.zero_()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)

    c0_records = []
    v1_records = []
    for i in range(32):
        state = "PROVEN_ZERO_EXISTS" if i < 7 else ("UNKNOWN" if i < 19 else "PROVEN_EMPTY_POSITIVE")
        anchor = 32 + i
        c0_records.append({
            "anchor_id": anchor,
            "stratum": "NEAR_ZERO",
            "depth": 8,
            "cell_index_xyz": [20 + i, 40 + i, 60 + i],
            "by_micro": {"2": {"state": state}},
        })
        v1_records.append({
            "anchor_id": anchor,
            "candidate_results": [
                {"lower_margin": -10.0, "upper_margin": 10.0}
            ],
        })

    out = profile_c1_handoff_v2(
        field,
        planes,
        {"records": c0_records},
        prior_v1_profile={"records": v1_records},
        domain_lo=-0.875,
        domain_hi=0.875,
        target_depth=8,
        max_micro_depth=0,
        logger=None,
    )

    assert out["stress_all_near_zero"]["count"] == 32
    assert out["primary_handoff_zero_or_unknown"]["count"] == 19
    assert out["primary_c0_zero_exists"]["count"] == 7
    assert out["primary_c0_unknown"]["count"] == 12
    assert out["stress_all_near_zero"]["states"] == {"UNKNOWN": 32}
    assert out["empirical_contradiction_count"] == 0
    assert out["stress_all_near_zero"]["v2_to_v1_candidate0_width_ratio_median"] == 0.0
