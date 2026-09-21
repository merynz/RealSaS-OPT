from __future__ import annotations

"""Optimized, research-only C0 range engine for VF-11.

V3 preserves the C0 V2 mathematical contract while removing avoidable
implementation cost:
  * field parameters are prepared to CPU float64 once;
  * SiLU secant-residual extrema are solved in vectorized batches;
  * triplane tensors are prepared once;
  * one refinement tree yields the micro-depth 0/1/2 ladder.

No PROVEN_REGULAR state exists here. No product authority can be minted.
"""

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PreparedField:
    ln_gamma: torch.Tensor
    ln_beta: torch.Tensor
    ln_eps: float
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    wh: torch.Tensor
    bh: torch.Tensor


@dataclass(frozen=True)
class C0Summary:
    state: str
    sign: int
    lower: float
    upper: float
    unresolved_leaf_count: int
    positive_leaf_count: int
    negative_leaf_count: int
    truncation_micro_depth: int


@dataclass(frozen=True)
class C0LadderCertificate:
    by_micro_depth: Dict[int, C0Summary]
    regime_box_count: int
    actual_evaluated_box_count: int
    max_micro_depth: int
    numerically_rigorous: bool = False


@dataclass
class _Affine:
    center: torch.Tensor
    generators: torch.Tensor

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        radius = self.generators.abs().sum(dim=-1)
        return self.center - radius, self.center + radius


@dataclass
class _Node:
    depth: int
    lower: float
    upper: float
    sign: int
    children: list["_Node"] | None


def prepare_field(field: nn.Module) -> PreparedField:
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

    ln = body[0]
    n = int(np.prod(ln.normalized_shape))
    if tuple(ln.normalized_shape) != (n,):
        raise ValueError("LAYERNORM_SHAPE_UNSUPPORTED")
    gamma = (
        torch.ones(n, dtype=torch.float64)
        if not ln.elementwise_affine
        else ln.weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )
    beta = (
        torch.zeros(n, dtype=torch.float64)
        if not ln.elementwise_affine
        else ln.bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
    )

    def wb(layer: nn.Linear) -> tuple[torch.Tensor, torch.Tensor]:
        w = layer.weight.detach().to(device="cpu", dtype=torch.float64).contiguous()
        b = (
            torch.zeros(layer.out_features, dtype=torch.float64)
            if layer.bias is None
            else layer.bias.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        return w, b

    w1, b1 = wb(body[1])
    w2, b2 = wb(body[3])
    wh, bh = wb(head)
    return PreparedField(gamma, beta, float(ln.eps), w1, b1, w2, b2, wh, bh)


def prepare_planes(planes: torch.Tensor) -> torch.Tensor:
    p = planes.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if p.ndim != 5 or tuple(p.shape[:2]) != (1, 3):
        raise ValueError("PLANES_MUST_BE_1x3xCxHxW")
    if p.shape[-2] != p.shape[-1]:
        raise ValueError("NONSQUARE_TRIPLANE_UNSUPPORTED")
    return p


def _linear(x: _Affine, w: torch.Tensor, b: torch.Tensor) -> _Affine:
    return _Affine(w @ x.center + b, w @ x.generators)


def _append_component_remainder(
    x: _Affine,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
) -> _Affine:
    lo = remainder_lo.to(dtype=torch.float64)
    hi = remainder_hi.to(dtype=torch.float64)
    mid = (lo + hi) * 0.5
    rad = (hi - lo) * 0.5
    n = int(x.center.numel())
    old_k = int(x.generators.shape[1])
    g = torch.zeros((n, old_k + n), dtype=torch.float64)
    g[:, :old_k] = x.generators
    g[:, old_k:] = torch.diag(rad)
    return _Affine(x.center + mid, g)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s + x * s * (1.0 - s)


_SILU_TURN = 2.3993572805154677


def _silu_secant_residual_vectorized(
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if bool(torch.any(hi < lo)):
        raise ValueError("SILU_INTERVAL_ORDER")

    width = hi - lo
    deg = width <= 1e-14

    flo = _silu(lo)
    fhi = _silu(hi)
    safe_width = torch.where(deg, torch.ones_like(width), width)
    sec_m = (fhi - flo) / safe_width
    tangent_m = _silu_prime((lo + hi) * 0.5)
    m = torch.where(deg, tangent_m, sec_m)
    b = flo - m * lo

    rlo = torch.zeros_like(lo)
    rhi = torch.zeros_like(lo)

    # silu' is monotone on the three intervals separated by +/-_SILU_TURN.
    segment_bounds = [
        (None, -_SILU_TURN),
        (-_SILU_TURN, _SILU_TURN),
        (_SILU_TURN, None),
    ]

    for seg_lo, seg_hi in segment_bounds:
        a = lo.clone()
        z = hi.clone()
        if seg_lo is not None:
            a = torch.maximum(a, torch.full_like(a, seg_lo))
        if seg_hi is not None:
            z = torch.minimum(z, torch.full_like(z, seg_hi))
        valid = (~deg) & (a <= z)
        if not bool(torch.any(valid)):
            continue

        fa = _silu_prime(a) - m
        fz = _silu_prime(z) - m
        active = valid & (fa * fz <= 0.0)
        if not bool(torch.any(active)):
            continue

        left = a.clone()
        right = z.clone()
        fleft = fa.clone()
        fright = fz.clone()
        for _ in range(60):
            mid = (left + right) * 0.5
            fmid = _silu_prime(mid) - m
            root_left = fleft * fmid <= 0.0
            take_left = active & root_left
            take_right = active & (~root_left)
            right = torch.where(take_left, mid, right)
            fright = torch.where(take_left, fmid, fright)
            left = torch.where(take_right, mid, left)
            fleft = torch.where(take_right, fmid, fleft)

        root = (left + right) * 0.5
        residual = _silu(root) - (m * root + b)
        rlo = torch.where(active, torch.minimum(rlo, residual), rlo)
        rhi = torch.where(active, torch.maximum(rhi, residual), rhi)

    # Preserve the V2 degenerate-interval safety pad and add a tiny research pad.
    deg_pad = torch.full_like(lo, 1e-14)
    scale = 1.0 + torch.maximum(rlo.abs(), rhi.abs())
    pad = 2e-11 * scale
    rlo = torch.where(deg, -deg_pad, rlo - pad)
    rhi = torch.where(deg, deg_pad, rhi + pad)
    return m, b, rlo, rhi


def _silu_affine(x: _Affine) -> _Affine:
    lo, hi = x.bounds()
    m, b, rem_lo, rem_hi = _silu_secant_residual_vectorized(lo, hi)
    base = _Affine(m * x.center + b, m[:, None] * x.generators)
    return _append_component_remainder(base, rem_lo, rem_hi)


def _layernorm_affine(x: _Affine, p: PreparedField) -> _Affine:
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

    lo = max(0.0, float(var_lo.item())) + p.ln_eps
    hi = max(0.0, float(var_hi.item())) + p.ln_eps
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
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

    # The unary enclosure is built over t = variance + eps.  The affine
    # variable stored in `var` is variance itself, so the constant eps must be
    # included in the affine center before applying the secant/tangent line.
    # Omitting this shifts even a degenerate-point LayerNorm evaluation and can
    # make a nominal point enclosure miss the actual PyTorch field value.
    q_c = slope * (var.center + p.ln_eps) + intercept + (rem_lo + rem_hi) * 0.5
    q_g = slope * var.generators
    q_rad = (rem_hi - rem_lo) * 0.5
    q_g = torch.cat([q_g, torch.tensor([[q_rad]], dtype=torch.float64)], dim=1)
    q = _Affine(q_c, q_g)

    k = int(q.generators.shape[1])
    if z.generators.shape[1] < k:
        z_g_pad = torch.cat(
            [z.generators, torch.zeros((n, k - z.generators.shape[1]), dtype=torch.float64)],
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
    y = _append_component_remainder(_Affine(y_c, y_g), -product_radius, product_radius)
    return _Affine(
        y.center * p.ln_gamma + p.ln_beta,
        y.generators * p.ln_gamma[:, None],
    )


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

    a = v00
    bb = v10 - v00
    c = v01 - v00
    d = v11 - v10 - v01 + v00

    center = a + bb * uc + c * vc + d * uc * vc
    g = torch.zeros((int(plane.shape[0]), 6), dtype=torch.float64)
    g[:, u_generator] = (bb + d * vc) * ur
    g[:, v_generator] = (c + d * uc) * vr
    g[:, uv_generator] = d * ur * vr
    return _Affine(center, g)


def _triplane_affine_single_regime(
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
) -> _Affine:
    xy = _plane_affine(planes[0, 0], lo[0], hi[0], lo[1], hi[1], 0, 1, 3)
    xz = _plane_affine(planes[0, 1], lo[0], hi[0], lo[2], hi[2], 0, 2, 4)
    yz = _plane_affine(planes[0, 2], lo[1], hi[1], lo[2], hi[2], 1, 2, 5)
    return _Affine(
        torch.cat([xy.center, xz.center, yz.center]),
        torch.cat([xy.generators, xz.generators, yz.generators], dim=0),
    )


def bound_single_regime_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[float, float]:
    x = _triplane_affine_single_regime(planes, lo, hi)
    x = _layernorm_affine(x, p)
    x = _linear(x, p.w1, p.b1)
    x = _silu_affine(x)
    x = _linear(x, p.w2, p.b2)
    x = _silu_affine(x)
    x = _linear(x, p.wh, p.bh)
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
                    [lo[0] if ix == 0 else mid[0],
                     lo[1] if iy == 0 else mid[1],
                     lo[2] if iz == 0 else mid[2]],
                    dtype=np.float64,
                )
                child_hi = np.asarray(
                    [mid[0] if ix == 0 else hi[0],
                     mid[1] if iy == 0 else hi[1],
                     mid[2] if iz == 0 else hi[2]],
                    dtype=np.float64,
                )
                out.append((child_lo, child_hi))
    return out


def _build_node(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    depth: int,
    max_depth: int,
    counter: list[int],
) -> _Node:
    lower, upper = bound_single_regime_prepared(p, planes, lo, hi)
    counter[0] += 1
    if lower > 0.0:
        return _Node(depth, lower, upper, 1, None)
    if upper < 0.0:
        return _Node(depth, lower, upper, -1, None)
    if depth >= max_depth:
        return _Node(depth, lower, upper, 0, None)
    children = [
        _build_node(p, planes, a, b, depth + 1, max_depth, counter)
        for a, b in _split_octants(lo, hi)
    ]
    return _Node(depth, lower, upper, 0, children)


def _collect_at_depth(node: _Node, truncation: int) -> tuple[int, int, int, float, float]:
    if node.sign > 0:
        return 1, 0, 0, node.lower, node.upper
    if node.sign < 0:
        return 0, 1, 0, node.lower, node.upper
    if node.depth >= truncation or not node.children:
        return 0, 0, 1, node.lower, node.upper

    pos = neg = unk = 0
    lower = math.inf
    upper = -math.inf
    for child in node.children:
        cp, cn, cu, clo, chi = _collect_at_depth(child, truncation)
        pos += cp
        neg += cn
        unk += cu
        lower = min(lower, clo)
        upper = max(upper, chi)
    return pos, neg, unk, lower, upper


def _summary_from_counts(
    pos: int,
    neg: int,
    unk: int,
    lower: float,
    upper: float,
    micro: int,
) -> C0Summary:
    if pos and neg:
        state, sign = "PROVEN_ZERO_EXISTS", 0
    elif unk:
        state, sign = "UNKNOWN", 0
    elif pos and not neg:
        state, sign = "PROVEN_EMPTY_POSITIVE", 1
    elif neg and not pos:
        state, sign = "PROVEN_EMPTY_NEGATIVE", -1
    else:
        state, sign = "UNKNOWN", 0
    return C0Summary(state, sign, lower, upper, unk, pos, neg, micro)


def certify_cell_c0_ladder_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C0LadderCertificate:
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    regimes = split_at_triplane_knots(lo, hi, int(planes.shape[-1]))
    counter = [0]
    roots = [
        _build_node(p, planes, a, b, 0, max_micro_depth, counter)
        for a, b in regimes
    ]

    summaries: dict[int, C0Summary] = {}
    for micro in range(max_micro_depth + 1):
        pos = neg = unk = 0
        lower = math.inf
        upper = -math.inf
        for root in roots:
            rp, rn, ru, rlo, rhi = _collect_at_depth(root, micro)
            pos += rp
            neg += rn
            unk += ru
            lower = min(lower, rlo)
            upper = max(upper, rhi)
        summaries[micro] = _summary_from_counts(pos, neg, unk, lower, upper, micro)

    return C0LadderCertificate(
        by_micro_depth=summaries,
        regime_box_count=len(regimes),
        actual_evaluated_box_count=counter[0],
        max_micro_depth=max_micro_depth,
    )


def certify_cell_c0_ladder(
    field: nn.Module,
    planes: torch.Tensor,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    max_micro_depth: int = 2,
) -> C0LadderCertificate:
    return certify_cell_c0_ladder_prepared(
        prepare_field(field),
        prepare_planes(planes),
        lo,
        hi,
        max_micro_depth=max_micro_depth,
    )
