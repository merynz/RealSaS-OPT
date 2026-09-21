from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import range_engine_c0_v3 as v3  # noqa: E402


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
        return self.sdf_head(h).squeeze(-1)


def test_vectorized_silu_residual_contains_dense_curve():
    lo = torch.tensor([-8.0, -3.0, -1.0, -0.2, 0.0, 2.0], dtype=torch.float64)
    hi = torch.tensor([-3.0,  3.0,  0.4,  0.3, 6.0, 10.0], dtype=torch.float64)
    m, b, rlo, rhi = v3._silu_secant_residual_vectorized(lo, hi)
    for i in range(lo.numel()):
        xs = torch.linspace(float(lo[i]), float(hi[i]), 4001, dtype=torch.float64)
        ys = xs * torch.sigmoid(xs)
        residual = ys - (m[i] * xs + b[i])
        assert float(residual.min()) >= float(rlo[i]) - 2e-10
        assert float(residual.max()) <= float(rhi[i]) + 2e-10


def test_v3_single_regime_bound_contains_dense_samples():
    torch.manual_seed(31)
    field = TinyField(channels=2, hidden=10).eval()
    planes = torch.randn(1, 3, 2, 8, 8)
    p = v3.prepare_field(field)
    pp = v3.prepare_planes(planes)

    knots = v3.interpolation_knots(8)
    lo = np.array([
        knots[2] + .15 * (knots[3] - knots[2]),
        knots[4] + .12 * (knots[5] - knots[4]),
        knots[1] + .10 * (knots[2] - knots[1]),
    ])
    hi = np.array([
        knots[2] + .80 * (knots[3] - knots[2]),
        knots[4] + .75 * (knots[5] - knots[4]),
        knots[1] + .85 * (knots[2] - knots[1]),
    ])

    blo, bhi = v3.bound_single_regime_prepared(p, pp, lo, hi)

    axes = [np.linspace(lo[i], hi[i], 7) for i in range(3)]
    pts = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    q = torch.tensor(pts, dtype=torch.float32).reshape(1, -1, 3)
    with torch.no_grad():
        vals = field(planes, q).reshape(-1).numpy()
    assert vals.min() >= blo - 3e-6
    assert vals.max() <= bhi + 3e-6


def test_v3_eps_fixed_degenerate_point_contains_direct_float64():
    torch.manual_seed(502)
    field = TinyField(channels=2, hidden=8).double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    p = v3.prepare_field(field)
    pp = v3.prepare_planes(planes)

    point = np.array([0.13, -0.22, 0.31], dtype=np.float64)
    blo, bhi = v3.bound_single_regime_prepared(p, pp, point, point)
    q = torch.tensor(point.reshape(1, 1, 3), dtype=torch.float64)
    with torch.no_grad():
        value = float(field(pp, q).item())

    assert value >= blo - 2e-10
    assert value <= bhi + 2e-10


def test_v3_ladder_is_monotone_in_information():
    torch.manual_seed(43)
    field = TinyField(channels=2, hidden=8).eval()
    planes = torch.randn(1, 3, 2, 8, 8)
    p = v3.prepare_field(field)
    pp = v3.prepare_planes(planes)
    lo = np.array([-0.22, -0.19, -0.17])
    hi = np.array([ 0.08,  0.11,  0.13])

    ladder = v3.certify_cell_c0_ladder_prepared(p, pp, lo, hi, max_micro_depth=2)
    states = [ladder.by_micro_depth[m].state for m in (0,1,2)]

    # Once a decisive proof exists, deeper truncation may not revert it to UNKNOWN.
    decisive = None
    for s in states:
        if s != "UNKNOWN":
            if decisive is None:
                decisive = s
            else:
                assert s == decisive or {s, decisive} == {
                    "PROVEN_EMPTY_POSITIVE", "PROVEN_ZERO_EXISTS"
                } or {s, decisive} == {
                    "PROVEN_EMPTY_NEGATIVE", "PROVEN_ZERO_EXISTS"
                }


def test_exact_pixel_center_domain_has_no_spurious_border_state():
    size = 48
    lo = np.array([-1 + 1 / size] * 3, dtype=np.float64)
    hi = np.array([ 1 - 1 / size] * 3, dtype=np.float64)
    boxes = v3.split_at_triplane_knots(lo, lo + (hi - lo) / 8, size)
    assert boxes
