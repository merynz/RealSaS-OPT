from __future__ import annotations

"""Research-only C0 range engine for VF-11.

Purpose:
  certify only whether a cell is entirely on one side of the learned zero set,
  or prove existence of a zero by certified opposite signs.

This v2 intentionally does NOT emit PROVEN_REGULAR. C1 regularity is deferred
until C0 range tightness is demonstrated.

Mathematical structure:
  * exact piecewise bilinear triplane model inside one grid_sample interpolation regime;
  * shared affine generators for x,y,z and xy,xz,yz correlations;
  * affine LayerNorm propagation with conservative square / reciprocal-sqrt remainders;
  * correlation-preserving affine propagation through linear layers;
  * per-neuron SiLU secant residual enclosure;
  * exact splitting at triplane interpolation knots;
  * bounded recursive micro-refinement of unresolved regime cells.

Numeric status:
  formulas are conservative at the real-arithmetic level, but this research
  implementation uses ordinary IEEE-754 float64 rather than directed rounding.
  It cannot mint shipping proof authority.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class C0Certificate:
    state: str
    sign: int
    lower: float
    upper: float
    regime_box_count: int
    evaluated_box_count: int
    unresolved_leaf_count: int
    positive_leaf_count: int
    negative_leaf_count: int
    max_micro_depth: int
    max_depth_reached: int
    numerically_rigorous: bool = False


@dataclass
class _Affine:
    center: torch.Tensor
    generators: torch.Tensor

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        radius = self.generators.abs().sum(dim=-1)
        return self.center - radius, self.center + radius


def _linear(x: _Affine, layer: nn.Linear) -> _Affine:
    w = layer.weight.detach().to(dtype=torch.float64, device="cpu")
    b = (
        torch.zeros(w.shape[0], dtype=torch.float64)
        if layer.bias is None
        else layer.bias.detach().to(dtype=torch.float64, device="cpu")
    )
    return _Affine(w @ x.center + b, w @ x.generators)


def _append_component_remainder(
    x: _Affine,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
) -> _Affine:
    lo = torch.as_tensor(remainder_lo, dtype=torch.float64)
    hi = torch.as_tensor(remainder_hi, dtype=torch.float64)
    mid = (lo + hi) * 0.5
    rad = (hi - lo) * 0.5
    n = int(x.center.numel())
    old_k = int(x.generators.shape[1])
    g = torch.zeros((n, old_k + n), dtype=torch.float64)
    g[:, :old_k] = x.generators
    g[:, old_k:] = torch.diag(rad)
    return _Affine(x.center + mid, g)


def _sigmoid_scalar(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _silu_scalar(x: float) -> float:
    return x * _sigmoid_scalar(x)


def _silu_prime_scalar(x: float) -> float:
    s = _sigmoid_scalar(x)
    return s + x * s * (1.0 - s)


_SILU_DERIVATIVE_TURN = 2.3993572805154677


def _bisect_prime_equals_slope(a: float, b: float, slope: float) -> float | None:
    fa = _silu_prime_scalar(a) - slope
    fb = _silu_prime_scalar(b) - slope
    if abs(fa) <= 1e-15:
        return a
    if abs(fb) <= 1e-15:
        return b
    if fa * fb > 0.0:
        return None
    for _ in range(72):
        m = (a + b) * 0.5
        fm = _silu_prime_scalar(m) - slope
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) * 0.5


def _silu_secant_residual(lo: float, hi: float) -> tuple[float, float, float, float]:
    if hi < lo:
        raise ValueError("SILU_INTERVAL_ORDER")
    if hi - lo <= 1e-14:
        x = (lo + hi) * 0.5
        m = _silu_prime_scalar(x)
        b = _silu_scalar(x) - m * x
        return m, b, -1e-14, 1e-14

    flo, fhi = _silu_scalar(lo), _silu_scalar(hi)
    m = (fhi - flo) / (hi - lo)
    b = flo - m * lo

    cuts = [lo]
    for c in (-_SILU_DERIVATIVE_TURN, _SILU_DERIVATIVE_TURN):
        if lo < c < hi:
            cuts.append(c)
    cuts.append(hi)
    cuts.sort()

    candidates = [lo, hi]
    for a, z in zip(cuts[:-1], cuts[1:]):
        fa = _silu_prime_scalar(a) - m
        fz = _silu_prime_scalar(z) - m
        if abs(fa) <= 1e-15:
            candidates.append(a)
        if abs(fz) <= 1e-15:
            candidates.append(z)
        if fa * fz < 0.0:
            root = _bisect_prime_equals_slope(a, z, m)
            if root is not None:
                candidates.append(root)

    residuals = [_silu_scalar(x) - (m * x + b) for x in candidates]
    scale = 1.0 + max(abs(x) for x in residuals)
    pad = 2e-11 * scale
    return m, b, min(residuals) - pad, max(residuals) + pad


def _silu_affine(x: _Affine) -> _Affine:
    lo, hi = x.bounds()
    n = int(x.center.numel())
    slopes = torch.empty(n, dtype=torch.float64)
    intercepts = torch.empty(n, dtype=torch.float64)
    rem_lo = torch.empty(n, dtype=torch.float64)
    rem_hi = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        m, b, rlo, rhi = _silu_secant_residual(float(lo[i]), float(hi[i]))
        slopes[i] = m
        intercepts[i] = b
        rem_lo[i] = rlo
        rem_hi[i] = rhi
    base = _Affine(
        slopes * x.center + intercepts,
        slopes[:, None] * x.generators,
    )
    return _append_component_remainder(base, rem_lo, rem_hi)


def _layernorm_affine(x: _Affine, layer: nn.LayerNorm) -> _Affine:
    if x.center.ndim != 1:
        raise ValueError("LAYERNORM_EXPECTS_VECTOR")
    if tuple(layer.normalized_shape) != (int(x.center.numel()),):
        raise ValueError("LAYERNORM_SHAPE_UNSUPPORTED")

    n = int(x.center.numel())
    mu_c = x.center.mean()
    mu_g = x.generators.mean(dim=0)
    z_c = x.center - mu_c
    z_g = x.generators - mu_g
    z = _Affine(z_c, z_g)
    z_lo, z_hi = z.bounds()

    alpha = z_lo + z_hi
    beta = -z_lo * z_hi
    delta = (z_hi - z_lo).square() * 0.25

    var_c = (alpha * z_c + beta - delta * 0.5).mean()
    var_g = (alpha[:, None] * z_g).mean(dim=0)
    var_rem = delta.sum() / (2.0 * n)
    var_g = torch.cat([var_g, var_rem.reshape(1)])
    var = _Affine(var_c.reshape(1), var_g.reshape(1, -1))
    var_lo, var_hi = var.bounds()

    lo = float(var_lo.item()) + float(layer.eps)
    hi = float(var_hi.item()) + float(layer.eps)
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0.0 or hi < lo:
        raise ValueError("LAYERNORM_VARIANCE_BOUND_INVALID")

    if hi - lo <= 1e-15:
        slope = -0.5 * lo ** -1.5
        intercept = lo ** -0.5 - slope * lo
        rem_lo = rem_hi = 0.0
    else:
        f_lo, f_hi = lo ** -0.5, hi ** -0.5
        slope = (f_hi - f_lo) / (hi - lo)
        intercept = f_lo - slope * lo
        t_star = (-0.5 / slope) ** (2.0 / 3.0)
        t_star = min(max(t_star, lo), hi)
        residual = t_star ** -0.5 - (slope * t_star + intercept)
        rem_lo = min(0.0, residual)
        rem_hi = 0.0

    q_c = slope * var.center + intercept + (rem_lo + rem_hi) * 0.5
    q_g = slope * var.generators
    q_rad = (rem_hi - rem_lo) * 0.5
    q_g = torch.cat([q_g, torch.tensor([[q_rad]], dtype=torch.float64)], dim=1)
    q = _Affine(q_c, q_g)

    k = int(q.generators.shape[1])
    if z.generators.shape[1] < k:
        z_g_pad = torch.cat(
            [
                z.generators,
                torch.zeros((n, k - z.generators.shape[1]), dtype=torch.float64),
            ],
            dim=1,
        )
    else:
        z_g_pad = z.generators

    q_center = float(q.center.item())
    y_c = z.center * q_center
    y_g = q_center * z_g_pad + z.center[:, None] * q.generators.expand(n, -1)

    z_radius = z_g_pad.abs().sum(dim=1)
    q_radius = float(q.generators.abs().sum().item())
    product_radius = z_radius * q_radius
    y = _append_component_remainder(
        _Affine(y_c, y_g),
        -product_radius,
        product_radius,
    )

    if layer.elementwise_affine:
        gamma = layer.weight.detach().to(dtype=torch.float64, device="cpu")
        beta_ln = layer.bias.detach().to(dtype=torch.float64, device="cpu")
        return _Affine(
            y.center * gamma + beta_ln,
            y.generators * gamma[:, None],
        )
    return y


def _validate_field(field: nn.Module) -> tuple[nn.LayerNorm, nn.Linear, nn.Linear, nn.Linear]:
    body = getattr(field, "body", None)
    head = getattr(field, "sdf_head", None)
    if not isinstance(body, nn.Sequential) or len(body) != 5:
        raise ValueError("UNSUPPORTED_FIELD_BODY")
    if not (
        isinstance(body[0], nn.LayerNorm)
        and isinstance(body[1], nn.Linear)
        and isinstance(body[2], nn.SiLU)
        and isinstance(body[3], nn.Linear)
        and isinstance(body[4], nn.SiLU)
        and isinstance(head, nn.Linear)
        and head.out_features == 1
    ):
        raise ValueError("UNSUPPORTED_FIELD_ARCHITECTURE")
    return body[0], body[1], body[3], head


def _pixel_coordinate(g: float, size: int) -> float:
    return ((float(g) + 1.0) * float(size) - 1.0) * 0.5


def _plane_affine(
    plane: torch.Tensor,
    u_lo: float,
    u_hi: float,
    v_lo: float,
    v_hi: float,
    u_generator: int,
    v_generator: int,
    uv_generator: int,
) -> _Affine:
    plane = plane.detach().to(dtype=torch.float64, device="cpu")
    h, w = int(plane.shape[-2]), int(plane.shape[-1])
    if h != w:
        raise ValueError("NONSQUARE_TRIPLANE_UNSUPPORTED")

    pu0, pu1 = _pixel_coordinate(u_lo, w), _pixel_coordinate(u_hi, w)
    pv0, pv1 = _pixel_coordinate(v_lo, h), _pixel_coordinate(v_hi, h)

    ku = max(0, min(w - 2, int(math.floor((pu0 + pu1) * 0.5))))
    kv = max(0, min(h - 2, int(math.floor((pv0 + pv1) * 0.5))))
    tol = 2e-11
    if pu0 < ku - tol or pu1 > ku + 1.0 + tol:
        raise ValueError("U_NOT_IN_SINGLE_INTERPOLATION_REGIME")
    if pv0 < kv - tol or pv1 > kv + 1.0 + tol:
        raise ValueError("V_NOT_IN_SINGLE_INTERPOLATION_REGIME")

    uc = (pu0 + pu1) * 0.5 - ku
    ur = (pu1 - pu0) * 0.5
    vc = (pv0 + pv1) * 0.5 - kv
    vr = (pv1 - pv0) * 0.5

    v00 = plane[:, kv, ku]
    v10 = plane[:, kv, ku + 1]
    v01 = plane[:, kv + 1, ku]
    v11 = plane[:, kv + 1, ku + 1]

    a = v00
    b = v10 - v00
    c = v01 - v00
    d = v11 - v10 - v01 + v00

    center = a + b * uc + c * vc + d * uc * vc
    g = torch.zeros((int(plane.shape[0]), 6), dtype=torch.float64)
    g[:, u_generator] = (b + d * vc) * ur
    g[:, v_generator] = (c + d * uc) * vr
    g[:, uv_generator] = d * ur * vr
    return _Affine(center, g)


def _triplane_affine_single_regime(
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
) -> _Affine:
    p = planes.detach().to(dtype=torch.float64, device="cpu")
    if p.ndim != 5 or tuple(p.shape[:2]) != (1, 3):
        raise ValueError("PLANES_MUST_BE_1x3xCxHxW")
    xy = _plane_affine(p[0, 0], lo[0], hi[0], lo[1], hi[1], 0, 1, 3)
    xz = _plane_affine(p[0, 1], lo[0], hi[0], lo[2], hi[2], 0, 2, 4)
    yz = _plane_affine(p[0, 2], lo[1], hi[1], lo[2], hi[2], 1, 2, 5)
    return _Affine(
        torch.cat([xy.center, xz.center, yz.center]),
        torch.cat([xy.generators, xz.generators, yz.generators], dim=0),
    )


def bound_single_regime(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[float, float]:
    ln, l1, l2, head = _validate_field(field)
    x = _triplane_affine_single_regime(planes, lo, hi)
    x = _layernorm_affine(x, ln)
    x = _linear(x, l1)
    x = _silu_affine(x)
    x = _linear(x, l2)
    x = _silu_affine(x)
    x = _linear(x, head)
    lower, upper = x.bounds()
    return float(lower.item()), float(upper.item())


def interpolation_knots(size: int) -> np.ndarray:
    return np.asarray(
        [(2.0 * k + 1.0) / float(size) - 1.0 for k in range(size)],
        dtype=np.float64,
    )


def split_at_triplane_knots(
    lo: np.ndarray,
    hi: np.ndarray,
    size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    lo = np.asarray(lo, dtype=np.float64).reshape(3)
    hi = np.asarray(hi, dtype=np.float64).reshape(3)
    if np.any(lo >= hi):
        raise ValueError("INVALID_CELL")
    knots = interpolation_knots(size)
    axes: list[list[tuple[float, float]]] = []
    tol = 4e-15
    for axis in range(3):
        points = [float(lo[axis])]
        points.extend(float(k) for k in knots if lo[axis] + tol < k < hi[axis] - tol)
        points.append(float(hi[axis]))
        points = sorted(set(points))
        axes.append([(points[i], points[i + 1]) for i in range(len(points) - 1)])

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for x in axes[0]:
        for y in axes[1]:
            for z in axes[2]:
                out.append(
                    (
                        np.asarray([x[0], y[0], z[0]], dtype=np.float64),
                        np.asarray([x[1], y[1], z[1]], dtype=np.float64),
                    )
                )
    return out


def _split_octants(lo: np.ndarray, hi: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    mid = (lo + hi) * 0.5
    out = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                child_lo = np.asarray(
                    [
                        lo[0] if ix == 0 else mid[0],
                        lo[1] if iy == 0 else mid[1],
                        lo[2] if iz == 0 else mid[2],
                    ],
                    dtype=np.float64,
                )
                child_hi = np.asarray(
                    [
                        mid[0] if ix == 0 else hi[0],
                        mid[1] if iy == 0 else hi[1],
                        mid[2] if iz == 0 else hi[2],
                    ],
                    dtype=np.float64,
                )
                out.append((child_lo, child_hi))
    return out


def certify_cell_c0(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C0Certificate:
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    size = int(planes.shape[-1])
    regimes = split_at_triplane_knots(lo, hi, size)

    stack = [(a, b, 0) for a, b in regimes]
    positive = 0
    negative = 0
    unresolved = 0
    evaluated = 0
    leaf_lower = math.inf
    leaf_upper = -math.inf
    max_depth_reached = 0

    while stack:
        a, b, depth = stack.pop()
        lower, upper = bound_single_regime(field, planes, a, b)
        evaluated += 1
        max_depth_reached = max(max_depth_reached, depth)

        if lower > 0.0:
            positive += 1
            leaf_lower = min(leaf_lower, lower)
            leaf_upper = max(leaf_upper, upper)
            if negative:
                break
            continue
        if upper < 0.0:
            negative += 1
            leaf_lower = min(leaf_lower, lower)
            leaf_upper = max(leaf_upper, upper)
            if positive:
                break
            continue

        if depth < max_micro_depth:
            stack.extend((c, d, depth + 1) for c, d in _split_octants(a, b))
        else:
            unresolved += 1
            leaf_lower = min(leaf_lower, lower)
            leaf_upper = max(leaf_upper, upper)

    if positive and negative:
        state = "PROVEN_ZERO_EXISTS"
        sign = 0
    elif unresolved:
        state = "UNKNOWN"
        sign = 0
    elif positive and not negative:
        state = "PROVEN_EMPTY_POSITIVE"
        sign = 1
    elif negative and not positive:
        state = "PROVEN_EMPTY_NEGATIVE"
        sign = -1
    else:
        state = "UNKNOWN"
        sign = 0

    return C0Certificate(
        state=state,
        sign=sign,
        lower=float(leaf_lower),
        upper=float(leaf_upper),
        regime_box_count=len(regimes),
        evaluated_box_count=evaluated,
        unresolved_leaf_count=unresolved,
        positive_leaf_count=positive,
        negative_leaf_count=negative,
        max_micro_depth=max_micro_depth,
        max_depth_reached=max_depth_reached,
    )
