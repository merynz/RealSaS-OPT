from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import range_engine_c0_v2 as r  # noqa: E402


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


def dense_points(lo, hi, n=5):
    axes = [np.linspace(float(lo[i]), float(hi[i]), n) for i in range(3)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def test_silu_secant_residual_contains_dense_curve():
    intervals = [(-8, -3), (-3, 3), (-1, .4), (-.2, .3), (0, 6), (2, 10)]
    for lo, hi in intervals:
        m, b, rlo, rhi = r._silu_secant_residual(lo, hi)
        xs = np.linspace(lo, hi, 4001)
        ys = np.array([r._silu_scalar(float(x)) for x in xs])
        residual = ys - (m * xs + b)
        assert np.min(residual) >= rlo - 2e-10
        assert np.max(residual) <= rhi + 2e-10


def test_exact_pixel_center_domain_splits_without_border_failure():
    size = 48
    lo = np.array([-1 + 1 / size] * 3, dtype=np.float64)
    hi = np.array([1 - 1 / size] * 3, dtype=np.float64)
    boxes = r.split_at_triplane_knots(lo, lo + (hi - lo) / 4, size)
    assert boxes
    for a, b in boxes:
        assert np.all(a >= -1.0)
        assert np.all(b <= 1.0)
        assert np.all(a < b)


def test_single_regime_bound_contains_dense_field_samples():
    torch.manual_seed(7)
    field = TinyField(channels=2, hidden=9).eval()
    planes = torch.randn(1, 3, 2, 8, 8)
    knots = r.interpolation_knots(8)
    lo = np.array(
        [
            knots[2] + .15 * (knots[3] - knots[2]),
            knots[3] + .10 * (knots[4] - knots[3]),
            knots[1] + .20 * (knots[2] - knots[1]),
        ],
        dtype=np.float64,
    )
    hi = np.array(
        [
            knots[2] + .75 * (knots[3] - knots[2]),
            knots[3] + .70 * (knots[4] - knots[3]),
            knots[1] + .80 * (knots[2] - knots[1]),
        ],
        dtype=np.float64,
    )
    blo, bhi = r.bound_single_regime(field, planes, lo, hi)
    pts = torch.tensor(dense_points(lo, hi, 7), dtype=torch.float32).reshape(1, -1, 3)
    with torch.no_grad():
        vals = field(planes, pts).reshape(-1).numpy()
    assert vals.min() >= blo - 2e-6
    assert vals.max() <= bhi + 2e-6


def test_dense_zero_witness_is_never_called_empty():
    torch.manual_seed(19)
    field = TinyField(channels=2, hidden=8).eval()
    planes = torch.randn(1, 3, 2, 8, 8)
    knots = r.interpolation_knots(8)
    lo = np.array([knots[3] + .25 * (knots[4] - knots[3])] * 3)
    hi = np.array([knots[3] + .75 * (knots[4] - knots[3])] * 3)
    center = torch.tensor(((lo + hi) * .5).reshape(1, 1, 3), dtype=torch.float32)
    with torch.no_grad():
        c = float(field(planes, center).item())
        field.sdf_head.bias.sub_(c)
    cert = r.certify_cell_c0(field, planes, lo, hi, max_micro_depth=2)
    assert cert.state not in {"PROVEN_EMPTY_POSITIVE", "PROVEN_EMPTY_NEGATIVE"}


def test_micro_refinement_tightens_deterministic_fixture():
    torch.manual_seed(23)
    field = TinyField(channels=2, hidden=8).eval()
    planes = torch.randn(1, 3, 2, 8, 8)
    knots = r.interpolation_knots(8)
    lo = np.array(
        [
            knots[2] + .05 * (knots[3] - knots[2]),
            knots[4] + .05 * (knots[5] - knots[4]),
            knots[3] + .05 * (knots[4] - knots[3]),
        ]
    )
    hi = np.array(
        [
            knots[2] + .95 * (knots[3] - knots[2]),
            knots[4] + .95 * (knots[5] - knots[4]),
            knots[3] + .95 * (knots[4] - knots[3]),
        ]
    )
    c0 = r.certify_cell_c0(field, planes, lo, hi, max_micro_depth=0)
    c2 = r.certify_cell_c0(field, planes, lo, hi, max_micro_depth=2)
    assert (c2.upper - c2.lower) <= (c0.upper - c0.lower) + 1e-9
