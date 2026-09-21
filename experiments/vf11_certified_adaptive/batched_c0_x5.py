from __future__ import annotations

"""X5 research-only batched C0 range engine.

This module preserves the scalar range_engine_c0_v3 mathematics for boxes
that each lie inside one triplane interpolation regime, while vectorizing the
independent-box dimension on CPU float64.

It intentionally does not change affine-generator semantics, corrected
LayerNorm epsilon handling, SiLU secant/remainder enclosure, sign/state
semantics, or unresolved-only octant refinement.
"""

from dataclasses import dataclass
import numpy as np
import torch

from range_engine_c0_v3 import (
    PreparedField,
    C0Summary,
    C0LadderCertificate,
    _silu_secant_residual_vectorized,
    _summary_from_counts,
)


def prepared_field_to_device(
    p: PreparedField,
    device: torch.device | str,
) -> PreparedField:
    d = torch.device(device)
    return PreparedField(
        p.ln_gamma.to(device=d, dtype=torch.float64).contiguous(),
        p.ln_beta.to(device=d, dtype=torch.float64).contiguous(),
        float(p.ln_eps),
        p.w1.to(device=d, dtype=torch.float64).contiguous(),
        p.b1.to(device=d, dtype=torch.float64).contiguous(),
        p.w2.to(device=d, dtype=torch.float64).contiguous(),
        p.b2.to(device=d, dtype=torch.float64).contiguous(),
        p.wh.to(device=d, dtype=torch.float64).contiguous(),
        p.bh.to(device=d, dtype=torch.float64).contiguous(),
    )


def planes_to_device(
    planes: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    return planes.detach().to(
        device=torch.device(device),
        dtype=torch.float64,
    ).contiguous()


@dataclass
class _BAffine:
    center: torch.Tensor
    generators: torch.Tensor

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        radius = self.generators.abs().sum(dim=-1)
        return self.center - radius, self.center + radius


def _blinear(x: _BAffine, w: torch.Tensor, b: torch.Tensor) -> _BAffine:
    center = torch.einsum("on,bn->bo", w, x.center) + b[None, :]
    generators = torch.einsum("on,bnk->bok", w, x.generators)
    return _BAffine(center, generators)


def _bappend_component_remainder(
    x: _BAffine,
    remainder_lo: torch.Tensor,
    remainder_hi: torch.Tensor,
) -> _BAffine:
    lo = remainder_lo.to(dtype=torch.float64)
    hi = remainder_hi.to(dtype=torch.float64)
    mid = (lo + hi) * 0.5
    rad = (hi - lo) * 0.5
    bsz, n = x.center.shape
    old_k = int(x.generators.shape[2])
    g = torch.zeros(
        (bsz, n, old_k + n),
        dtype=torch.float64,
        device=x.center.device,
    )
    g[:, :, :old_k] = x.generators
    diag = torch.arange(n, device=x.center.device)
    g[:, diag, old_k + diag] = rad
    return _BAffine(x.center + mid, g)


def _bsilu_affine(x: _BAffine) -> _BAffine:
    lo, hi = x.bounds()
    m, b, rem_lo, rem_hi = _silu_secant_residual_vectorized(lo, hi)
    base = _BAffine(
        m * x.center + b,
        m[:, :, None] * x.generators,
    )
    return _bappend_component_remainder(base, rem_lo, rem_hi)


def _blayernorm_affine(x: _BAffine, p: PreparedField) -> _BAffine:
    bsz, n = x.center.shape
    mu_c = x.center.mean(dim=1, keepdim=True)
    mu_g = x.generators.mean(dim=1, keepdim=True)
    z_c = x.center - mu_c
    z_g = x.generators - mu_g
    z = _BAffine(z_c, z_g)
    z_lo, z_hi = z.bounds()

    alpha = z_lo + z_hi
    beta = -z_lo * z_hi
    delta = (z_hi - z_lo).square() * 0.25

    var_c = (alpha * z_c + beta - delta * 0.5).mean(dim=1)
    var_g = (alpha[:, :, None] * z_g).mean(dim=1)
    var_rem = delta.sum(dim=1) / (2.0 * float(n))
    var_g = torch.cat([var_g, var_rem[:, None]], dim=1)
    var = _BAffine(var_c[:, None], var_g[:, None, :])
    var_lo, var_hi = var.bounds()
    var_lo = var_lo[:, 0]
    var_hi = var_hi[:, 0]

    lo = torch.clamp_min(var_lo, 0.0) + float(p.ln_eps)
    hi = torch.clamp_min(var_hi, 0.0) + float(p.ln_eps)
    if bool(torch.any(~torch.isfinite(lo))) or bool(torch.any(~torch.isfinite(hi))):
        raise ValueError("LAYERNORM_VARIANCE_BOUND_INVALID")
    if bool(torch.any(hi < lo)):
        raise ValueError("LAYERNORM_VARIANCE_BOUND_INVALID")

    width = hi - lo
    deg = width <= 1e-15
    safe_width = torch.where(deg, torch.ones_like(width), width)
    f_lo = lo.rsqrt()
    f_hi = hi.rsqrt()
    sec_slope = (f_hi - f_lo) / safe_width
    tangent_slope = -0.5 * lo.pow(-1.5)
    slope = torch.where(deg, tangent_slope, sec_slope)
    intercept = f_lo - slope * lo

    t_star = (-0.5 / slope).pow(2.0 / 3.0)
    t_star = torch.minimum(torch.maximum(t_star, lo), hi)
    residual = t_star.rsqrt() - (slope * t_star + intercept)
    rem_lo = torch.minimum(torch.zeros_like(residual), residual)
    rem_hi = torch.zeros_like(residual)
    rem_lo = torch.where(deg, torch.zeros_like(rem_lo), rem_lo)

    q_c = (
        slope[:, None] * (var.center + float(p.ln_eps))
        + intercept[:, None]
        + ((rem_lo + rem_hi) * 0.5)[:, None]
    )
    q_g = slope[:, None, None] * var.generators
    q_rad = ((rem_hi - rem_lo) * 0.5)[:, None, None]
    q_g = torch.cat([q_g, q_rad], dim=2)
    q = _BAffine(q_c, q_g)

    k = int(q.generators.shape[2])
    zk = int(z.generators.shape[2])
    if zk < k:
        pad = torch.zeros(
            (bsz, n, k - zk),
            dtype=torch.float64,
            device=x.center.device,
        )
        z_g_pad = torch.cat([z.generators, pad], dim=2)
    else:
        z_g_pad = z.generators

    q_center = q.center[:, 0]
    q_generators = q.generators[:, 0, :]
    y_c = z.center * q_center[:, None]
    y_g = (
        q_center[:, None, None] * z_g_pad
        + z.center[:, :, None] * q_generators[:, None, :]
    )

    z_radius = z_g_pad.abs().sum(dim=2)
    q_radius = q_generators.abs().sum(dim=1)
    product_radius = z_radius * q_radius[:, None]
    y = _bappend_component_remainder(
        _BAffine(y_c, y_g),
        -product_radius,
        product_radius,
    )
    return _BAffine(
        y.center * p.ln_gamma[None, :] + p.ln_beta[None, :],
        y.generators * p.ln_gamma[None, :, None],
    )


def _pixel_coordinate_tensor(g: torch.Tensor, size: int) -> torch.Tensor:
    return ((g + 1.0) * float(size) - 1.0) * 0.5


def _bplane_affine(
    plane: torch.Tensor,
    u_lo: torch.Tensor,
    u_hi: torch.Tensor,
    v_lo: torch.Tensor,
    v_hi: torch.Tensor,
    u_generator: int,
    v_generator: int,
    uv_generator: int,
) -> _BAffine:
    h, w = int(plane.shape[-2]), int(plane.shape[-1])
    pu0, pu1 = _pixel_coordinate_tensor(u_lo, w), _pixel_coordinate_tensor(u_hi, w)
    pv0, pv1 = _pixel_coordinate_tensor(v_lo, h), _pixel_coordinate_tensor(v_hi, h)

    ku = torch.floor((pu0 + pu1) * 0.5).to(torch.int64).clamp(0, w - 2)
    kv = torch.floor((pv0 + pv1) * 0.5).to(torch.int64).clamp(0, h - 2)
    tol = 2e-11
    if bool(torch.any(pu0 < ku.to(torch.float64) - tol)) or bool(
        torch.any(pu1 > ku.to(torch.float64) + 1.0 + tol)
    ):
        raise ValueError("U_NOT_IN_SINGLE_INTERPOLATION_REGIME")
    if bool(torch.any(pv0 < kv.to(torch.float64) - tol)) or bool(
        torch.any(pv1 > kv.to(torch.float64) + 1.0 + tol)
    ):
        raise ValueError("V_NOT_IN_SINGLE_INTERPOLATION_REGIME")

    uc = (pu0 + pu1) * 0.5 - ku.to(torch.float64)
    ur = (pu1 - pu0) * 0.5
    vc = (pv0 + pv1) * 0.5 - kv.to(torch.float64)
    vr = (pv1 - pv0) * 0.5

    v00 = plane[:, kv, ku].T.contiguous()
    v10 = plane[:, kv, ku + 1].T.contiguous()
    v01 = plane[:, kv + 1, ku].T.contiguous()
    v11 = plane[:, kv + 1, ku + 1].T.contiguous()

    a = v00
    bb = v10 - v00
    cc = v01 - v00
    d = v11 - v10 - v01 + v00

    center = (
        a
        + bb * uc[:, None]
        + cc * vc[:, None]
        + d * uc[:, None] * vc[:, None]
    )
    bsz, channels = center.shape
    g = torch.zeros(
        (bsz, channels, 6),
        dtype=torch.float64,
        device=plane.device,
    )
    g[:, :, u_generator] = (bb + d * vc[:, None]) * ur[:, None]
    g[:, :, v_generator] = (cc + d * uc[:, None]) * vr[:, None]
    g[:, :, uv_generator] = d * ur[:, None] * vr[:, None]
    return _BAffine(center, g)


def _btriplane_affine_single_regime(
    planes: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
) -> _BAffine:
    xy = _bplane_affine(
        planes[0, 0], lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], 0, 1, 3
    )
    xz = _bplane_affine(
        planes[0, 1], lo[:, 0], hi[:, 0], lo[:, 2], hi[:, 2], 0, 2, 4
    )
    yz = _bplane_affine(
        planes[0, 2], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2], 1, 2, 5
    )
    return _BAffine(
        torch.cat([xy.center, xz.center, yz.center], dim=1),
        torch.cat([xy.generators, xz.generators, yz.generators], dim=1),
    )


def bound_batch_single_regime_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    los: np.ndarray | torch.Tensor,
    his: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    device = p.w1.device
    if planes.device != device:
        raise ValueError("PREPARED_FIELD_PLANES_DEVICE_MISMATCH")
    lo = torch.as_tensor(los, dtype=torch.float64, device=device).reshape(-1, 3)
    hi = torch.as_tensor(his, dtype=torch.float64, device=device).reshape(-1, 3)
    if tuple(lo.shape) != tuple(hi.shape) or bool(torch.any(lo >= hi)):
        raise ValueError("INVALID_BATCH_BOXES")

    x = _btriplane_affine_single_regime(planes, lo, hi)
    x = _blayernorm_affine(x, p)
    x = _blinear(x, p.w1, p.b1)
    x = _bsilu_affine(x)
    x = _blinear(x, p.w2, p.b2)
    x = _bsilu_affine(x)
    x = _blinear(x, p.wh, p.bh)
    lower, upper = x.bounds()
    return (
        lower[:, 0].detach().cpu().numpy(),
        upper[:, 0].detach().cpu().numpy(),
    )


def _bound_batch_chunked_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    los: np.ndarray,
    his: np.ndarray,
    *,
    node_batch_size: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    los = np.asarray(los, dtype=np.float64).reshape(-1, 3)
    his = np.asarray(his, dtype=np.float64).reshape(-1, 3)
    n = len(los)
    if node_batch_size is None or node_batch_size <= 0 or node_batch_size >= n:
        return bound_batch_single_regime_prepared(p, planes, los, his)
    lower = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64)
    for start in range(0, n, int(node_batch_size)):
        stop = min(start + int(node_batch_size), n)
        lo_i, hi_i = bound_batch_single_regime_prepared(
            p, planes, los[start:stop], his[start:stop]
        )
        lower[start:stop] = lo_i
        upper[start:stop] = hi_i
    return lower, upper


def _split_octants_batch(los: np.ndarray, his: np.ndarray):
    los = np.asarray(los, dtype=np.float64).reshape(-1, 3)
    his = np.asarray(his, dtype=np.float64).reshape(-1, 3)
    mid = (los + his) * 0.5
    child_los = []
    child_his = []
    parent_index = []
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                bits = np.asarray([ix, iy, iz], dtype=np.int64)
                child_los.append(np.where(bits[None, :] == 0, los, mid))
                child_his.append(np.where(bits[None, :] == 0, mid, his))
                parent_index.append(np.arange(len(los), dtype=np.int64))
    clo = np.stack(child_los, axis=1).reshape(-1, 3)
    chi = np.stack(child_his, axis=1).reshape(-1, 3)
    pidx = np.stack(parent_index, axis=1).reshape(-1)
    return clo, chi, pidx


def certify_batch_single_regime_ladders_prepared(
    p: PreparedField,
    planes: torch.Tensor,
    los: np.ndarray,
    his: np.ndarray,
    *,
    max_micro_depth: int = 2,
    node_batch_size: int | None = 16,
) -> list[C0LadderCertificate]:
    if max_micro_depth < 0:
        raise ValueError("NEGATIVE_MICRO_DEPTH")
    root_los = np.asarray(los, dtype=np.float64).reshape(-1, 3)
    root_his = np.asarray(his, dtype=np.float64).reshape(-1, 3)
    if root_los.shape != root_his.shape or np.any(root_los >= root_his):
        raise ValueError("INVALID_BATCH_BOXES")
    bsz = len(root_los)
    if bsz == 0:
        return []

    frontier_lo = root_los
    frontier_hi = root_his
    frontier_root = np.arange(bsz, dtype=np.int64)

    terminal_signs: list[list[int]] = [[] for _ in range(bsz)]
    terminal_lowers: list[list[float]] = [[] for _ in range(bsz)]
    terminal_uppers: list[list[float]] = [[] for _ in range(bsz)]
    summaries: list[dict[int, C0Summary]] = [dict() for _ in range(bsz)]
    eval_counts = np.zeros(bsz, dtype=np.int64)

    for depth in range(max_micro_depth + 1):
        lower, upper = _bound_batch_chunked_prepared(
            p,
            planes,
            frontier_lo,
            frontier_hi,
            node_batch_size=node_batch_size,
        )
        signs = np.where(lower > 0.0, 1, np.where(upper < 0.0, -1, 0)).astype(np.int8)
        for rid in frontier_root:
            eval_counts[int(rid)] += 1

        by_root_indices = [np.where(frontier_root == rid)[0] for rid in range(bsz)]
        for rid in range(bsz):
            idx = by_root_indices[rid]
            pos = sum(s > 0 for s in terminal_signs[rid])
            neg = sum(s < 0 for s in terminal_signs[rid])
            unk = 0
            lows = list(terminal_lowers[rid])
            ups = list(terminal_uppers[rid])
            for j in idx:
                s = int(signs[j])
                if s > 0:
                    pos += 1
                elif s < 0:
                    neg += 1
                else:
                    unk += 1
                lows.append(float(lower[j]))
                ups.append(float(upper[j]))
            summaries[rid][depth] = _summary_from_counts(
                pos, neg, unk, min(lows), max(ups), depth
            )

        if depth >= max_micro_depth:
            break

        for j, rid_raw in enumerate(frontier_root):
            rid = int(rid_raw)
            s = int(signs[j])
            if s != 0:
                terminal_signs[rid].append(s)
                terminal_lowers[rid].append(float(lower[j]))
                terminal_uppers[rid].append(float(upper[j]))

        unknown = np.where(signs == 0)[0]
        if len(unknown) == 0:
            for future in range(depth + 1, max_micro_depth + 1):
                for rid in range(bsz):
                    prev = summaries[rid][depth]
                    summaries[rid][future] = C0Summary(
                        prev.state,
                        prev.sign,
                        prev.lower,
                        prev.upper,
                        prev.unresolved_leaf_count,
                        prev.positive_leaf_count,
                        prev.negative_leaf_count,
                        future,
                    )
            break

        u_lo = frontier_lo[unknown]
        u_hi = frontier_hi[unknown]
        u_root = frontier_root[unknown]
        clo, chi, pidx = _split_octants_batch(u_lo, u_hi)
        frontier_lo = clo
        frontier_hi = chi
        frontier_root = u_root[pidx]

    return [
        C0LadderCertificate(
            by_micro_depth=summaries[rid],
            regime_box_count=1,
            actual_evaluated_box_count=int(eval_counts[rid]),
            max_micro_depth=max_micro_depth,
        )
        for rid in range(bsz)
    ]
