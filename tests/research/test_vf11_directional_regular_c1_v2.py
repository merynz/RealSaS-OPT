from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import directional_regular_c1_v1 as c1v1  # noqa: E402
import directional_regular_c1_v2 as c1v2  # noqa: E402


class TinyField(nn.Module):
    def __init__(self, channels=2, hidden=10):
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


def _single_regime_cell(size=8):
    knots = np.asarray(
        [(2.0 * k + 1.0) / size - 1.0 for k in range(size)],
        dtype=np.float64,
    )
    lo = np.array([
        knots[2] + .22 * (knots[3] - knots[2]),
        knots[3] + .18 * (knots[4] - knots[3]),
        knots[1] + .24 * (knots[2] - knots[1]),
    ])
    hi = np.array([
        knots[2] + .72 * (knots[3] - knots[2]),
        knots[3] + .74 * (knots[4] - knots[3]),
        knots[1] + .70 * (knots[2] - knots[1]),
    ])
    return lo, hi


def test_exact_triplane_directional_affine_contains_direct_jvp():
    torch.manual_seed(301)
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    lo, hi = _single_regime_cell()
    direction = np.array([0.71, -0.31, 0.43], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    aff = c1v2._triplane_directional_affine(planes, lo, hi, direction)
    blo, bhi = aff.bounds()

    # Compare against direct autograd of triplane sampling at many points.
    gen = np.random.default_rng(302)
    field = TinyField(channels=2, hidden=4).double().eval()
    for _ in range(250):
        pt = lo + gen.random(3) * (hi - lo)
        q = torch.tensor(pt.reshape(1, 1, 3), dtype=torch.float64, requires_grad=True)
        sampled = field.sample(planes, q).reshape(-1)
        vals = []
        for j in range(sampled.numel()):
            g = torch.autograd.grad(sampled[j], q, retain_graph=True)[0].reshape(3)
            vals.append(float(g.detach().numpy() @ direction))
        vals = torch.tensor(vals, dtype=torch.float64)
        assert bool(torch.all(vals >= blo - 2e-9))
        assert bool(torch.all(vals <= bhi + 2e-9))


def test_layernorm_correlated_tangent_affine_contains_shared_generator_jvps():
    torch.manual_seed(303)
    field = TinyField(channels=2, hidden=7).double().eval()
    p = c1v2.prepare_field(field)

    n = 6
    k = 6
    center = torch.randn(n, dtype=torch.float64)
    gx = torch.randn(n, k, dtype=torch.float64) * 0.035
    dcenter = torch.randn(n, dtype=torch.float64)
    gd = torch.randn(n, k, dtype=torch.float64) * 0.025
    x = c1v2._Affine(center, gx)
    dx = c1v2._Affine(dcenter, gd)

    bound = c1v2._layernorm_tangent_affine(x, dx, p)
    blo, bhi = bound.bounds()
    layer = field.body[0]

    gen = torch.Generator().manual_seed(304)
    for _ in range(300):
        eps = torch.empty(k, dtype=torch.float64).uniform_(-1, 1, generator=gen)
        xv = (center + gx @ eps).requires_grad_(True)
        dv = dcenter + gd @ eps
        _, jvp = torch.autograd.functional.jvp(
            lambda z: layer(z),
            xv,
            dv,
            create_graph=False,
        )
        assert bool(torch.all(jvp >= blo - 3e-8))
        assert bool(torch.all(jvp <= bhi + 3e-8))


def test_v2_single_regime_bound_contains_dense_autograd_samples():
    torch.manual_seed(305)
    field = TinyField(channels=2, hidden=10).double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    p = c1v2.prepare_field(field)
    pp = c1v2.prepare_planes(planes)
    lo, hi = _single_regime_cell()
    direction = np.array([0.7, -0.2, 0.5], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    bound = c1v2.bound_directional_single_regime_correlated_v2(
        p, pp, lo, hi, direction
    )
    assert np.isfinite(bound.lower)
    assert np.isfinite(bound.upper)
    assert bound.upper >= bound.lower

    gen = np.random.default_rng(306)
    for _ in range(400):
        pt = lo + gen.random(3) * (hi - lo)
        q = torch.tensor(pt.reshape(1, 1, 3), dtype=torch.float64, requires_grad=True)
        y = field(pp, q)["sdf"].reshape(())
        grad = torch.autograd.grad(y, q)[0].reshape(3).detach().numpy()
        val = float(grad @ direction)
        assert val >= bound.lower - 5e-8
        assert val <= bound.upper + 5e-8


def test_v2_is_strictly_tighter_than_v1_on_deterministic_fixture():
    torch.manual_seed(307)
    field = TinyField(channels=2, hidden=12).double().eval()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)
    p = c1v2.prepare_field(field)
    pp = c1v2.prepare_planes(planes)
    lo, hi = _single_regime_cell()
    direction = np.array([0.43, 0.82, -0.37], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    old = c1v1.bound_directional_single_regime_prepared(
        p, pp, lo, hi, direction
    )
    new = c1v2.bound_directional_single_regime_correlated_v2(
        p, pp, lo, hi, direction
    )
    old_width = old.upper - old.lower
    new_width = new.upper - new.lower

    assert np.isfinite(new_width)
    assert new_width < old_width, (old_width, new_width)


def test_v2_constant_zero_field_fails_closed():
    torch.manual_seed(308)
    field = TinyField(channels=2, hidden=8).double().eval()
    for parameter in field.parameters():
        parameter.data.zero_()
    planes = torch.randn(1, 3, 2, 8, 8, dtype=torch.float64)

    lo = np.array([-0.30, -0.30, -0.30], dtype=np.float64)
    hi = np.array([ 0.30,  0.30,  0.30], dtype=np.float64)
    cert = c1v2.certify_directional_regular_c1_v2(
        field, planes, lo, hi, max_micro_depth=0
    )
    assert cert.state == "UNKNOWN"
    assert cert.direction is None
