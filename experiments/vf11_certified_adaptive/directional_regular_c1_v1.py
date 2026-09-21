from __future__ import annotations

"""Research-only C1 directional-regularity engine for VF-11.

A C1 certificate means only this:
  there exists one fixed unit direction d such that the whole queried cell has
  <grad f(x), d> strictly positive or strictly negative.

It does NOT imply one connected component, global topology, or surface
existence. Surface existence/uniqueness needs additional sign/crossing proof.

The implementation reuses C0 affine value enclosures and propagates a separate
conservative interval enclosure for one directional derivative (a JVP).
Ordinary IEEE-754 float64 is used; results are research certificates only.
"""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import torch
from torch import nn

from range_engine_c0_v3 import (
    PreparedField,
    _Affine,
    _layernorm_affine,
    _linear,
    _pixel_coordinate,
    _plane_affine,
    _silu_affine,
    _split_octants,
    _triplane_affine_single_regime,
    prepare_field,
    prepare_planes,
    split_at_triplane_knots,
)


@dataclass(frozen=True)
class Interval:
    lo: torch.Tensor
    hi: torch.Tensor

    def validate(self) -> "Interval":
        if bool(torch.any(self.hi < self.lo)):
            raise ValueError("INVALID_INTERVAL")
        return self


@dataclass(frozen=True)
class DirectionalBound:
    lower: float
    upper: float


@dataclass(frozen=True)
class C1CandidateResult:
    direction: tuple[float, float, float]
    state: str
    lower_margin: float
    upper_margin: float
    evaluated_box_count: int
    positive_leaf_count: int
    negative_leaf_count: int
    unresolved_leaf_count: int
    max_micro_depth: int
    max_depth_reached: int
    coverage_complete: bool


@dataclass(frozen=True)
class C1Certificate:
    state: str
    direction: tuple[float, float, float] | None
    derivative_lower: float
    derivative_upper: float
    evaluated_box_count: int
    candidate_results: tuple[C1CandidateResult, ...]
    numerically_rigorous: bool = False


def _iadd(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo + b.lo, a.hi + b.hi)


def _isub(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo - b.hi, a.hi - b.lo)


def _iscale(a: Interval, s: torch.Tensor | float) -> Interval:
    s = torch.as_tensor(s, dtype=torch.float64)
    p = a.lo * s
    q = a.hi * s
    return Interval(torch.minimum(p, q), torch.maximum(p, q))


def _imul(a: Interval, b: Interval) -> Interval:
    terms = torch.stack(
        [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi],
        dim=0,
    )
    return Interval(terms.amin(dim=0), terms.amax(dim=0))


def _ilinear(x: Interval, w: torch.Tensor) -> Interval:
    c = (x.lo + x.hi) * 0.5
    r = (x.hi - x.lo) * 0.5
    oc = w @ c
    rr = w.abs() @ r
    return Interval(oc - rr, oc + rr)


def _affine_interval(x: _Affine) -> Interval:
    lo, hi = x.bounds()
    return Interval(lo, hi)


_SILU_PRIME_TURN = 2.3993572805154677


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


def silu_prime_bounds(lo: torch.Tensor, hi: torch.Tensor) -> Interval:
    lo = torch.as_tensor(lo, dtype=torch.float64)
    hi = torch.as_tensor(hi, dtype=torch.float64)
    if bool(torch.any(hi < lo)):
        raise ValueError("SILU_INTERVAL_ORDER")
    vals = [_silu_prime(lo), _silu_prime(hi)]
    for turn in (-_SILU_PRIME_TURN, _SILU_PRIME_TURN):
        t = torch.full_like(lo, turn)
        inside = (lo <= t) & (t <= hi)
        v = _silu_prime(t)
        vals.append(torch.where(inside, v, _silu_prime(lo)))
    stack = torch.stack(vals, dim=0)
    pad = 2e-12 * (1.0 + stack.abs().amax(dim=0))
    return Interval(stack.amin(dim=0) - pad, stack.amax(dim=0) + pad)


def _plane_directional_interval(
    plane: torch.Tensor,
    u_lo: float,
    u_hi: float,
    v_lo: float,
    v_hi: float,
    du_dt: float,
    dv_dt: float,
) -> Interval:
    h, w = int(plane.shape[-2]), int(plane.shape[-1])
    pu0, pu1 = _pixel_coordinate(u_lo, w), _pixel_coordinate(u_hi, w)
    pv0, pv1 = _pixel_coordinate(v_lo, h), _pixel_coordinate(v_hi, h)
    ku = max(0, min(w - 2, int(math.floor((pu0 + pu1) * 0.5))))
    kv = max(0, min(h - 2, int(math.floor((pv0 + pv1) * 0.5))))
    tol = 2e-11
    if pu0 < ku - tol or pu1 > ku + 1.0 + tol:
        raise ValueError("U_NOT_IN_SINGLE_INTERPOLATION_REGIME")
    if pv0 < kv - tol or pv1 > kv + 1.0 + tol:
        raise ValueError("V_NOT_IN_SINGLE_INTERPOLATION_REGIME")

    Ulo, Uhi = pu0 - ku, pu1 - ku
    Vlo, Vhi = pv0 - kv, pv1 - kv
    Uc, Ur = (Ulo + Uhi) * 0.5, (Uhi - Ulo) * 0.5
    Vc, Vr = (Vlo + Vhi) * 0.5, (Vhi - Vlo) * 0.5

    v00 = plane[:, kv, ku]
    v10 = plane[:, kv, ku + 1]
    v01 = plane[:, kv + 1, ku]
    v11 = plane[:, kv + 1, ku + 1]
    b = v10 - v00
    c = v01 - v00
    q = v11 - v10 - v01 + v00

    # q(U,V)=a+bU+cV+qUV, and U/V move in pixel coordinates.
    center = b * du_dt + c * dv_dt + q * (du_dt * Vc + dv_dt * Uc)
    radius = q.abs() * (abs(du_dt) * Vr + abs(dv_dt) * Ur)
    pad = 2e-12 * (1.0 + center.abs() + radius)
    return Interval(center - radius - pad, center + radius + pad)


def _triplane_directional_interval(
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> Interval:
    size = int(planes.shape[-1])
    pixel_scale = float(size) * 0.5
    dx, dy, dz = [float(v) for v in direction]
    xy = _plane_directional_interval(
        planes[0, 0], lo[0], hi[0], lo[1], hi[1],
        dx * pixel_scale, dy * pixel_scale,
    )
    xz = _plane_directional_interval(
        planes[0, 1], lo[0], hi[0], lo[2], hi[2],
        dx * pixel_scale, dz * pixel_scale,
    )
    yz = _plane_directional_interval(
        planes[0, 2], lo[1], hi[1], lo[2], hi[2],
        dy * pixel_scale, dz * pixel_scale,
    )
    return Interval(
        torch.cat([xy.lo, xz.lo, yz.lo]),
        torch.cat([xy.hi, xz.hi, yz.hi]),
    )


def _layernorm_directional_interval(
    x: _Affine,
    dx: Interval,
    p: PreparedField,
) -> Interval:
    n = int(x.center.numel())

    mu_c = x.center.mean()
    mu_g = x.generators.mean(dim=0)
    z = _Affine(x.center - mu_c, x.generators - mu_g)
    z_iv = _affine_interval(z)

    dmu = Interval(dx.lo.mean().reshape(1), dx.hi.mean().reshape(1))
    dz = _isub(dx, dmu)

    # Reproduce the C0 semantic variance enclosure, but only for the scalar
    # variance bounds needed by the derivative formula.
    z_lo, z_hi = z_iv.lo, z_iv.hi
    alpha = z_lo + z_hi
    beta = -z_lo * z_hi
    delta = (z_hi - z_lo).square() * 0.25
    var_c = (alpha * z.center + beta - delta * 0.5).mean()
    var_g = (alpha[:, None] * z.generators).mean(dim=0)
    var_rem = delta.sum() / (2.0 * n)
    var_g = torch.cat([var_g, var_rem.reshape(1)])
    var_aff = _Affine(var_c.reshape(1), var_g.reshape(1, -1))
    var_lo_t, var_hi_t = var_aff.bounds()
    var_lo = max(0.0, float(var_lo_t.item()))
    var_hi = max(0.0, float(var_hi_t.item()))

    # dvar = mean(2 z dz)
    dvar_terms = _imul(z_iv, dz)
    dvar = Interval(
        2.0 * dvar_terms.lo.mean().reshape(1),
        2.0 * dvar_terms.hi.mean().reshape(1),
    )

    t_lo = var_lo + p.ln_eps
    t_hi = var_hi + p.ln_eps
    q_iv = Interval(
        torch.tensor([t_hi ** -0.5], dtype=torch.float64),
        torch.tensor([t_lo ** -0.5], dtype=torch.float64),
    )
    coeff = Interval(
        torch.tensor([-0.5 * t_lo ** -1.5], dtype=torch.float64),
        torch.tensor([-0.5 * t_hi ** -1.5], dtype=torch.float64),
    )
    dq = _imul(coeff, dvar)

    # y = gamma * z * q + beta
    dy = _iadd(_imul(dz, q_iv), _imul(z_iv, dq))
    gamma = p.ln_gamma
    return _iscale(dy, gamma)


def bound_directional_single_regime_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> DirectionalBound:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("INVALID_DIRECTION")
    direction = direction / norm

    x = _triplane_affine_single_regime(planes, lo, hi)
    dx = _triplane_directional_interval(planes, lo, hi, direction)

    dx = _layernorm_directional_interval(x, dx, p)
    x = _layernorm_affine(x, p)

    dx = _ilinear(dx, p.w1)
    x = _linear(x, p.w1, p.b1)

    prime = silu_prime_bounds(*x.bounds())
    dx = _imul(prime, dx)
    x = _silu_affine(x)

    dx = _ilinear(dx, p.w2)
    x = _linear(x, p.w2, p.b2)

    prime = silu_prime_bounds(*x.bounds())
    dx = _imul(prime, dx)

    dx = _ilinear(dx, p.wh)
    return DirectionalBound(float(dx.lo.item()), float(dx.hi.item()))


def _candidate_directions(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    grad_min_norm: float = 1e-12,
) -> list[np.ndarray]:
    center = torch.tensor(
        ((np.asarray(lo) + np.asarray(hi)) * 0.5).reshape(1, 1, 3),
        dtype=torch.float64,
        requires_grad=True,
    )
    out = field(planes, center)["sdf"].reshape(())
    grad = torch.autograd.grad(out, center, create_graph=False)[0].reshape(3)
    g = grad.detach().cpu().numpy().astype(np.float64)
    candidates: list[np.ndarray] = []
    gn = float(np.linalg.norm(g))
    if math.isfinite(gn) and gn > grad_min_norm:
        candidates.append(g / gn)
    for axis in np.eye(3, dtype=np.float64):
        if not any(abs(float(np.dot(axis, c))) > 1.0 - 1e-12 for c in candidates):
            candidates.append(axis)
    return candidates


def _evaluate_candidate(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
    *,
    max_micro_depth: int,
) -> C1CandidateResult:
    regimes = split_at_triplane_knots(lo, hi, int(planes.shape[-1]))
    stack = [(a, b, 0) for a, b in regimes]
    positive = negative = unresolved = evaluated = 0
    global_lower = math.inf
    global_upper = -math.inf
    max_depth_reached = 0
    coverage_complete = True

    while stack:
        a, b, depth = stack.pop()
        bound = bound_directional_single_regime_prepared(p, planes, a, b, direction)
        evaluated += 1
        max_depth_reached = max(max_depth_reached, depth)
        global_lower = min(global_lower, bound.lower)
        global_upper = max(global_upper, bound.upper)

        if bound.lower > 0.0:
            positive += 1
            if negative:
                coverage_complete = False
                break
            continue
        if bound.upper < 0.0:
            negative += 1
            if positive:
                coverage_complete = False
                break
            continue

        if depth < max_micro_depth:
            stack.extend((c, d, depth + 1) for c, d in _split_octants(a, b))
        else:
            unresolved += 1

    if positive and not negative and unresolved == 0:
        state = "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
    elif negative and not positive and unresolved == 0:
        state = "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE"
    else:
        state = "UNKNOWN"

    d = tuple(float(v) for v in np.asarray(direction, dtype=np.float64))
    return C1CandidateResult(
        direction=d,
        state=state,
        lower_margin=float(global_lower),
        upper_margin=float(global_upper),
        evaluated_box_count=evaluated,
        positive_leaf_count=positive,
        negative_leaf_count=negative,
        unresolved_leaf_count=unresolved,
        max_micro_depth=max_micro_depth,
        max_depth_reached=max_depth_reached,
        coverage_complete=coverage_complete,
    )


def certify_directional_regular_c1_prepared(
    field: nn.Module,
    p: PreparedField,
    pp: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C1Certificate:
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    results: list[C1CandidateResult] = []
    total_evals = 0

    for direction in _candidate_directions(field, pp, lo, hi):
        result = _evaluate_candidate(
            p, pp, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64),
            direction, max_micro_depth=max_micro_depth,
        )
        results.append(result)
        total_evals += result.evaluated_box_count
        if result.state != "UNKNOWN":
            return C1Certificate(
                state=result.state,
                direction=result.direction,
                derivative_lower=result.lower_margin,
                derivative_upper=result.upper_margin,
                evaluated_box_count=total_evals,
                candidate_results=tuple(results),
            )

    return C1Certificate(
        state="UNKNOWN",
        direction=None,
        derivative_lower=float("nan"),
        derivative_upper=float("nan"),
        evaluated_box_count=total_evals,
        candidate_results=tuple(results),
    )


def certify_directional_regular_c1(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C1Certificate:
    return certify_directional_regular_c1_prepared(
        field,
        prepare_field(field),
        prepare_planes(planes),
        lo,
        hi,
        max_micro_depth=max_micro_depth,
    )


# Exact semantic fixture helper. The bound function itself is authoritative for
# the fixture and is deliberately independent of the production MLP.
def classify_exact_directional_fixture(
    derivative_bound_fn: Callable[[np.ndarray], tuple[float, float]],
    directions: list[np.ndarray],
) -> str:
    for d in directions:
        lo, hi = derivative_bound_fn(np.asarray(d, dtype=np.float64))
        if lo > 0.0:
            return "PROVEN_DIRECTIONAL_REGULAR_POSITIVE"
        if hi < 0.0:
            return "PROVEN_DIRECTIONAL_REGULAR_NEGATIVE"
    return "UNKNOWN"
