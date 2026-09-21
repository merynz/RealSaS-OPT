from __future__ import annotations

"""Research-only C1 V2 correlated tangent-affine engine for VF-11.

C1 V1 propagated the directional JVP as component intervals and became
practically vacuous on the frozen Knight d8 NEAR_ZERO cohort. V2 keeps the
directional tangent as an affine form for substantially longer:

* bilinear triplane directional derivatives are exact affine forms inside one
  interpolation regime and share the spatial generator basis with the primal;
* LayerNorm dvar is formed from correlated affine products z * dz;
* LayerNorm scalar factors are conservatively interval-enclosed, then injected
  as explicit independent affine uncertainty rather than collapsing dz;
* SiLU' is conservatively interval-enclosed and multiplied into the affine
  tangent with standard affine-product remainders.

The certificate semantics remain narrow: one fixed direction has a strict
whole-cell derivative sign. This is NOT a component/topology/existence proof.

Ordinary float64 only; research certificate, never shipping proof authority.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from range_engine_c0_v3 import (
    PreparedField,
    _Affine,
    _layernorm_affine,
    _linear,
    _pixel_coordinate,
    _silu_affine,
    _split_octants,
    _triplane_affine_single_regime,
    prepare_field,
    prepare_planes,
    split_at_triplane_knots,
)
from directional_regular_c1_v1 import (
    C1CandidateResult,
    C1Certificate,
    _candidate_directions,
    silu_prime_bounds,
)


@dataclass(frozen=True)
class DirectionalBoundV2:
    lower: float
    upper: float
    generator_count: int


def _pad_generators(x: _Affine, k: int) -> _Affine:
    old = int(x.generators.shape[1])
    if old == k:
        return x
    if old > k:
        raise ValueError("GENERATOR_SHRINK_FORBIDDEN")
    g = torch.cat(
        [
            x.generators,
            torch.zeros((int(x.center.numel()), k - old), dtype=torch.float64),
        ],
        dim=1,
    )
    return _Affine(x.center, g)


def _broadcast_rows(x: _Affine, n: int) -> _Affine:
    m = int(x.center.numel())
    if m == n:
        return x
    if m != 1:
        raise ValueError("AFFINE_BROADCAST_UNSUPPORTED")
    return _Affine(
        x.center.expand(n).clone(),
        x.generators.expand(n, -1).clone(),
    )


def _align(a: _Affine, b: _Affine) -> tuple[_Affine, _Affine]:
    n = max(int(a.center.numel()), int(b.center.numel()))
    a = _broadcast_rows(a, n)
    b = _broadcast_rows(b, n)
    k = max(int(a.generators.shape[1]), int(b.generators.shape[1]))
    return _pad_generators(a, k), _pad_generators(b, k)


def _affine_add(a: _Affine, b: _Affine) -> _Affine:
    a, b = _align(a, b)
    return _Affine(a.center + b.center, a.generators + b.generators)


def _affine_scale(x: _Affine, scale: torch.Tensor | float) -> _Affine:
    s = torch.as_tensor(scale, dtype=torch.float64)
    if s.ndim == 0:
        return _Affine(x.center * s, x.generators * s)
    s = s.reshape(-1)
    if int(s.numel()) != int(x.center.numel()):
        raise ValueError("AFFINE_SCALE_SHAPE")
    return _Affine(x.center * s, x.generators * s[:, None])


def _affine_mean(x: _Affine) -> _Affine:
    return _Affine(
        x.center.mean().reshape(1),
        x.generators.mean(dim=0, keepdim=True),
    )


def _affine_centered(x: _Affine) -> _Affine:
    mu = _affine_mean(x)
    return _Affine(
        x.center - mu.center,
        x.generators - mu.generators,
    )


def _append_symmetric_component_remainder(x: _Affine, radius: torch.Tensor) -> _Affine:
    radius = torch.as_tensor(radius, dtype=torch.float64).reshape(-1)
    n = int(x.center.numel())
    if int(radius.numel()) == 1 and n != 1:
        radius = radius.expand(n).clone()
    if int(radius.numel()) != n:
        raise ValueError("REMAINDER_SHAPE")
    if bool(torch.any(radius < 0.0)):
        raise ValueError("NEGATIVE_REMAINDER")
    old_k = int(x.generators.shape[1])
    g = torch.zeros((n, old_k + n), dtype=torch.float64)
    g[:, :old_k] = x.generators
    g[:, old_k:] = torch.diag(radius)
    return _Affine(x.center, g)


def _affine_product(a: _Affine, b: _Affine) -> _Affine:
    """Standard conservative affine product with independent quadratic remainder.

    If a=ac+A eps and b=bc+B eps, the linear correlated part is preserved and
    all quadratic terms are enclosed by radius(a)*radius(b).
    """
    a, b = _align(a, b)
    ac, bc = a.center, b.center
    ag, bg = a.generators, b.generators
    center = ac * bc
    generators = bc[:, None] * ag + ac[:, None] * bg
    ra = ag.abs().sum(dim=1)
    rb = bg.abs().sum(dim=1)
    base = _Affine(center, generators)
    return _append_symmetric_component_remainder(base, ra * rb)


def _affine_linear_no_bias(x: _Affine, w: torch.Tensor) -> _Affine:
    return _Affine(w @ x.center, w @ x.generators)


def _interval_factor_affine(
    lo: torch.Tensor | float,
    hi: torch.Tensor | float,
    *,
    base_generators: int,
) -> _Affine:
    lo = torch.as_tensor(lo, dtype=torch.float64).reshape(-1)
    hi = torch.as_tensor(hi, dtype=torch.float64).reshape(-1)
    if int(lo.numel()) != int(hi.numel()) or bool(torch.any(hi < lo)):
        raise ValueError("INVALID_FACTOR_INTERVAL")
    center = (lo + hi) * 0.5
    radius = (hi - lo) * 0.5
    n = int(center.numel())
    g = torch.zeros((n, base_generators + n), dtype=torch.float64)
    g[:, base_generators:] = torch.diag(radius)
    return _Affine(center, g)


def _variance_affine(z: _Affine) -> _Affine:
    """Conservative affine enclosure of mean(z^2).

    For each component on [l,u], x^2 is the secant alpha*x+beta minus a
    nonnegative residual in [0,delta]. Averaging before introducing one
    remainder generator preserves substantial cross-component correlation.
    """
    n = int(z.center.numel())
    z_lo, z_hi = z.bounds()
    alpha = z_lo + z_hi
    beta = -z_lo * z_hi
    delta = (z_hi - z_lo).square() * 0.25

    center = (alpha * z.center + beta - delta * 0.5).mean().reshape(1)
    generators = (alpha[:, None] * z.generators).mean(dim=0, keepdim=True)
    radius = (delta.sum() / (2.0 * n)).reshape(1)
    return _append_symmetric_component_remainder(
        _Affine(center, generators),
        radius,
    )


def _plane_directional_affine(
    plane: torch.Tensor,
    u_lo: float,
    u_hi: float,
    v_lo: float,
    v_hi: float,
    du_dt: float,
    dv_dt: float,
    u_generator: int,
    v_generator: int,
) -> _Affine:
    """Exact affine JVP of one bilinear plane patch in pixel coordinates."""
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

    uc = (pu0 + pu1) * 0.5 - ku
    ur = (pu1 - pu0) * 0.5
    vc = (pv0 + pv1) * 0.5 - kv
    vr = (pv1 - pv0) * 0.5

    v00 = plane[:, kv, ku]
    v10 = plane[:, kv, ku + 1]
    v01 = plane[:, kv + 1, ku]
    v11 = plane[:, kv + 1, ku + 1]

    bb = v10 - v00
    cc = v01 - v00
    q = v11 - v10 - v01 + v00

    center = bb * du_dt + cc * dv_dt + q * (du_dt * vc + dv_dt * uc)
    g = torch.zeros((int(plane.shape[0]), 6), dtype=torch.float64)
    g[:, u_generator] = q * dv_dt * ur
    g[:, v_generator] = q * du_dt * vr
    return _Affine(center, g)


def _triplane_directional_affine(
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> _Affine:
    size = int(planes.shape[-1])
    scale = float(size) * 0.5
    dx, dy, dz = [float(v) for v in direction]
    xy = _plane_directional_affine(
        planes[0, 0], lo[0], hi[0], lo[1], hi[1],
        dx * scale, dy * scale, 0, 1,
    )
    xz = _plane_directional_affine(
        planes[0, 1], lo[0], hi[0], lo[2], hi[2],
        dx * scale, dz * scale, 0, 2,
    )
    yz = _plane_directional_affine(
        planes[0, 2], lo[1], hi[1], lo[2], hi[2],
        dy * scale, dz * scale, 1, 2,
    )
    return _Affine(
        torch.cat([xy.center, xz.center, yz.center]),
        torch.cat([xy.generators, xz.generators, yz.generators], dim=0),
    )


def _layernorm_tangent_affine(
    x: _Affine,
    dx: _Affine,
    p: PreparedField,
) -> _Affine:
    """Conservative correlated-affine JVP enclosure for LayerNorm."""
    x, dx = _align(x, dx)
    z = _affine_centered(x)
    dz = _affine_centered(dx)

    var = _variance_affine(z)
    vlo_t, vhi_t = var.bounds()
    var_lo = max(0.0, float(vlo_t.item()))
    var_hi = max(var_lo, float(vhi_t.item()))
    if not (math.isfinite(var_lo) and math.isfinite(var_hi)):
        raise ValueError("LAYERNORM_VARIANCE_NONFINITE")

    # dvar = 2 * mean(z * dz), preserving shared affine generators.
    dvar = _affine_scale(_affine_mean(_affine_product(z, dz)), 2.0)

    t_lo = var_lo + p.ln_eps
    t_hi = var_hi + p.ln_eps
    if not (t_lo > 0.0 and t_hi >= t_lo):
        raise ValueError("LAYERNORM_T_INVALID")

    # q = (var+eps)^-1/2
    q_lo = t_hi ** -0.5
    q_hi = t_lo ** -0.5
    q = _interval_factor_affine(
        q_lo, q_hi, base_generators=int(dz.generators.shape[1])
    )

    # c = -0.5*(var+eps)^-3/2, so dq = c*dvar.
    c_lo = -0.5 * t_lo ** -1.5
    c_hi = -0.5 * t_hi ** -1.5
    c = _interval_factor_affine(
        c_lo, c_hi, base_generators=int(dvar.generators.shape[1])
    )
    dq = _affine_product(c, dvar)

    term1 = _affine_product(q, dz)
    term2 = _affine_product(z, dq)
    dy = _affine_add(term1, term2)
    return _affine_scale(dy, p.ln_gamma)


def _silu_tangent_affine(x: _Affine, dx: _Affine) -> _Affine:
    lo, hi = x.bounds()
    prime = silu_prime_bounds(lo, hi)
    factor = _interval_factor_affine(
        prime.lo,
        prime.hi,
        base_generators=int(dx.generators.shape[1]),
    )
    return _affine_product(factor, dx)


def bound_directional_single_regime_correlated_v2(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    direction: np.ndarray,
) -> DirectionalBoundV2:
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("INVALID_DIRECTION")
    direction = direction / norm

    x = _triplane_affine_single_regime(planes, lo, hi)
    dx = _triplane_directional_affine(planes, lo, hi, direction)

    dx = _layernorm_tangent_affine(x, dx, p)
    x = _layernorm_affine(x, p)

    dx = _affine_linear_no_bias(dx, p.w1)
    x = _linear(x, p.w1, p.b1)

    dx = _silu_tangent_affine(x, dx)
    x = _silu_affine(x)

    dx = _affine_linear_no_bias(dx, p.w2)
    x = _linear(x, p.w2, p.b2)

    dx = _silu_tangent_affine(x, dx)
    dx = _affine_linear_no_bias(dx, p.wh)

    lower, upper = dx.bounds()
    return DirectionalBoundV2(
        lower=float(lower.item()),
        upper=float(upper.item()),
        generator_count=int(dx.generators.shape[1]),
    )


def _evaluate_candidate_v2(
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
        bound = bound_directional_single_regime_correlated_v2(
            p, planes, a, b, direction
        )
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

    return C1CandidateResult(
        direction=tuple(float(v) for v in np.asarray(direction, dtype=np.float64)),
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


def certify_directional_regular_c1_v2_prepared(
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
        result = _evaluate_candidate_v2(
            p,
            pp,
            np.asarray(lo, dtype=np.float64),
            np.asarray(hi, dtype=np.float64),
            direction,
            max_micro_depth=max_micro_depth,
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


def certify_directional_regular_c1_v2(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C1Certificate:
    return certify_directional_regular_c1_v2_prepared(
        field,
        prepare_field(field),
        prepare_planes(planes),
        lo,
        hi,
        max_micro_depth=max_micro_depth,
    )
