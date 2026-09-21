from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import directional_regular_c1_v1 as c1  # noqa: E402


class TinyField(nn.Module):
    def __init__(self, channels=2, hidden=9):
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


def test_c1_semantic_exact_fixtures():
    axes = [np.eye(3)[i] for i in range(3)]

    linear_grad = np.array([1.0, 2.0, 0.0])
    state = c1.classify_exact_directional_fixture(
        lambda d: (float(linear_grad @ d), float(linear_grad @ d)),
        axes,
    )
    assert state == "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"

    negative_grad = np.array([-1.0, 0.0, 0.0])
    state = c1.classify_exact_directional_fixture(
        lambda d: (float(negative_grad @ d), float(negative_grad @ d)),
        axes,
    )
    assert state == "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE"

    # f=x^2+y^2+z^2 on [-1,1]^3: every fixed direction has derivative
    # interval symmetric around zero, so no strict directional regularity.
    state = c1.classify_exact_directional_fixture(
        lambda d: (-2.0 * float(np.abs(d).sum()), 2.0 * float(np.abs(d).sum())),
        axes + [np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)],
    )
    assert state == "UNKNOWN"


def test_silu_prime_bounds_contain_dense_derivative():
    lo = torch.tensor([-9.0, -3.1, -2.5, -1.0, 0.0, 1.0], dtype=torch.float64)
    hi = torch.tensor([-2.4,  2.7,  4.0,  0.2, 3.0, 9.0], dtype=torch.float64)
    b = c1.silu_prime_bounds(lo, hi)
    for i in range(lo.numel()):
        xs = torch.linspace(float(lo[i]), float(hi[i]), 12001, dtype=torch.float64)
        s = torch.sigmoid(xs)
        deriv = s + xs * s * (1.0 - s)
        assert float(deriv.min()) >= float(b.lo[i]) - 1e-10
        assert float(deriv.max()) <= float(b.hi[i]) + 1e-10


def test_layernorm_directional_interval_contains_random_jvps():
    torch.manual_seed(101)
    field = TinyField(channels=2, hidden=7).double().eval()
    p = c1.prepare_field(field)

    center = torch.randn(6, dtype=torch.float64)
    radius = torch.rand(6, dtype=torch.float64) * 0.15 + 0.03
    x = c1._Affine(center, torch.diag(radius))

    dcenter = torch.randn(6, dtype=torch.float64)
    dradius = torch.rand(6, dtype=torch.float64) * 0.02
    dx = c1.Interval(dcenter - dradius, dcenter + dradius)

    bound = c1._layernorm_directional_interval(x, dx, p)
    layer = field.body[0]

    gen = torch.Generator().manual_seed(102)
    for _ in range(200):
        eps = torch.empty(6, dtype=torch.float64).uniform_(-1, 1, generator=gen)
        xv = (center + radius * eps).requires_grad_(True)
        de = torch.empty(6, dtype=torch.float64).uniform_(-1, 1, generator=gen)
        dv = dcenter + dradius * de

        _, jvp = torch.autograd.functional.jvp(
            lambda z: layer(z),
            xv,
            dv,
            create_graph=False,
        )
        assert bool(torch.all(jvp >= bound.lo - 2e-9))
        assert bool(torch.all(jvp <= bound.hi + 2e-9))


def test_single_regime_directional_bound_contains_dense_autograd_samples():
    torch.manual_seed(103)
    field = TinyField(channels=2, hidden=10).double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    p = c1.prepare_field(field)
    pp = c1.prepare_planes(planes)

    knots = c1.np.asarray(
        [(2.0 * k + 1.0) / 8.0 - 1.0 for k in range(8)],
        dtype=np.float64,
    )
    lo = np.array([
        knots[2] + .18 * (knots[3] - knots[2]),
        knots[3] + .12 * (knots[4] - knots[3]),
        knots[1] + .16 * (knots[2] - knots[1]),
    ])
    hi = np.array([
        knots[2] + .78 * (knots[3] - knots[2]),
        knots[3] + .82 * (knots[4] - knots[3]),
        knots[1] + .76 * (knots[2] - knots[1]),
    ])
    direction = np.array([0.7, -0.2, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    bound = c1.bound_directional_single_regime_prepared(p, pp, lo, hi, direction)

    gen = np.random.default_rng(104)
    for _ in range(300):
        pt = lo + gen.random(3) * (hi - lo)
        q = torch.tensor(pt.reshape(1, 1, 3), dtype=torch.float64, requires_grad=True)
        y = field(pp, q)["sdf"].reshape(())
        grad = torch.autograd.grad(y, q)[0].reshape(3).detach().numpy()
        val = float(grad @ direction)
        assert val >= bound.lower - 3e-8
        assert val <= bound.upper + 3e-8


def test_no_regular_state_is_minted_for_exact_zero_gradient_field():
    torch.manual_seed(105)
    field = TinyField(channels=2, hidden=8).double().eval()
    for parameter in field.parameters():
        parameter.data.zero_()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)

    # Exact constant-zero field: every directional derivative is exactly zero.
    # Strict regularity must therefore fail closed.
    lo = np.array([-0.30, -0.30, -0.30], dtype=np.float64)
    hi = np.array([ 0.30,  0.30,  0.30], dtype=np.float64)
    cert = c1.certify_directional_regular_c1(
        field, planes, lo, hi, max_micro_depth=0
    )
    assert cert.state == "UNKNOWN"
    assert cert.direction is None
