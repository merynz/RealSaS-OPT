#!/usr/bin/env python3
"""Deterministic ray-aligned depth corruption for the RealSaS predicted-depth bridge.

This module is apparatus only. It does not open Proxy27 or DEV32 and does not
evaluate any downstream outcome.

Scientific invariant:
    P_hat(u,v) = P_exact(u,v) + delta_d(u,v) * F_v
where F_v is the known unit orthographic camera forward direction.

`epsilon` is the RMS along-ray displacement on each affected view after sampling
the correlated field on that view's visible raster rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np


VIEWS = 8


@dataclass(frozen=True)
class DepthCorruptionSpec:
    epsilon: float
    ell_px: float
    asymmetry: str
    namespace: str = "RealSaS.DepthToleranceBridge.v1"

    def __post_init__(self):
        if not math.isfinite(float(self.epsilon)) or float(self.epsilon) < 0:
            raise ValueError("epsilon must be finite and >= 0")
        if not math.isfinite(float(self.ell_px)) or float(self.ell_px) < 0:
            raise ValueError("ell_px must be finite and >= 0")
        if self.asymmetry not in {
            "ALL8",
            "ONE_BAD_HASHED",
            "TWO_OPPOSITE_HASHED",
            "FOUR_ALTERNATING_HASHED",
        }:
            raise ValueError(f"unsupported asymmetry: {self.asymmetry}")


def _seed64(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16) & 0x7FFF_FFFF_FFFF_FFFF


def _unit(v) -> np.ndarray:
    x = np.asarray(v, np.float64)
    if x.shape != (3,):
        raise ValueError(f"forward must be shape (3,), got {x.shape}")
    n = float(np.linalg.norm(x))
    if not math.isfinite(n) or n <= 1e-12:
        raise ValueError("forward vector must be finite and nonzero")
    return x / n


def affected_views(asset_id: str, asymmetry: str) -> tuple[int, ...]:
    """Return deterministic affected-view indices for the asymmetry condition."""
    if asymmetry == "ALL8":
        return tuple(range(VIEWS))
    base = _seed64(f"VIEW_PATTERN|{asset_id}|{asymmetry}") % VIEWS
    if asymmetry == "ONE_BAD_HASHED":
        return (int(base),)
    if asymmetry == "TWO_OPPOSITE_HASHED":
        return tuple(sorted((int(base), int((base + 4) % VIEWS))))
    if asymmetry == "FOUR_ALTERNATING_HASHED":
        parity = int(base % 2)
        return tuple(range(parity, VIEWS, 2))
    raise ValueError(asymmetry)


def _visible_xy(pixel_linear_index, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    pix = np.asarray(pixel_linear_index, np.int64).reshape(-1)
    if len(pix) and (pix.min() < 0 or pix.max() >= resolution * resolution):
        raise ValueError("pixel index out of raster bounds")
    if len(pix) and np.any(pix[1:] < pix[:-1]):
        raise ValueError("pixel_linear_index must be sorted")
    y = pix // int(resolution)
    x = pix % int(resolution)
    return x, y


def _standardize_sampled(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return x
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    if not math.isfinite(rms) or rms <= 1e-12:
        raise RuntimeError("corruption field collapsed after foreground sampling")
    return x / rms


def sampled_unit_field(
    pixel_linear_index,
    *,
    resolution: int,
    ell_px: float,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic zero-mean, unit-RMS field on visible raster rows.

    ell_px == 0:
        independent Gaussian samples on visible rows.

    ell_px > 0:
        a dense white Gaussian raster is Gaussian-low-pass filtered in Fourier
        space and then sampled on the visible rows. `ell_px` is the Gaussian
        smoothing sigma in pixel units under the continuous-frequency filter
        exp(-0.5 * (2*pi*ell)^2 * |f|^2).

    The sampled foreground values are standardized after filtering, so epsilon
    has the same RMS meaning for every correlation length and every affected view.
    """
    pix = np.asarray(pixel_linear_index, np.int64).reshape(-1)
    _visible_xy(pix, resolution)
    if len(pix) == 0:
        return np.zeros(0, np.float32)
    rng = np.random.Generator(np.random.PCG64(int(seed)))

    if float(ell_px) <= 0.0:
        return _standardize_sampled(rng.standard_normal(len(pix))).astype(np.float32)

    white = rng.standard_normal((int(resolution), int(resolution)))
    fy = np.fft.fftfreq(int(resolution))[:, None]
    fx = np.fft.rfftfreq(int(resolution))[None, :]
    filt = np.exp(
        -0.5
        * (2.0 * np.pi * float(ell_px)) ** 2
        * (fy * fy + fx * fx)
    )
    smooth = np.fft.irfft2(np.fft.rfft2(white) * filt, s=white.shape)
    x, y = _visible_xy(pix, resolution)
    sampled = smooth[y, x]
    return _standardize_sampled(sampled).astype(np.float32)


def depth_delta_for_view(
    asset_id: str,
    view: int,
    pixel_linear_index,
    *,
    resolution: int,
    spec: DepthCorruptionSpec,
) -> np.ndarray:
    """Return per-visible-row along-ray depth displacement."""
    pix = np.asarray(pixel_linear_index, np.int64).reshape(-1)
    if int(view) not in range(VIEWS):
        raise ValueError(view)
    if float(spec.epsilon) == 0.0 or int(view) not in affected_views(asset_id, spec.asymmetry):
        return np.zeros(len(pix), np.float32)

    seed = _seed64(
        f"{spec.namespace}|{asset_id}|V{int(view)}|"
        f"eps={float(spec.epsilon):.9f}|ell={float(spec.ell_px):.6f}|A={spec.asymmetry}"
    )
    unit = sampled_unit_field(
        pix,
        resolution=int(resolution),
        ell_px=float(spec.ell_px),
        seed=seed,
    )
    return (float(spec.epsilon) * unit).astype(np.float32)


def corrupt_points_along_ray(P, forward, depth_delta) -> np.ndarray:
    """Apply the only legal bridge corruption: displacement parallel to F."""
    p = np.asarray(P, np.float32)
    d = np.asarray(depth_delta, np.float32).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) != len(d):
        raise ValueError("P must be [N,3] and depth_delta [N]")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(d)):
        raise ValueError("non-finite corruption input")
    f = _unit(forward).astype(np.float32)
    return (p + d[:, None] * f[None, :]).astype(np.float32)


def ray_alignment_residual(P, P_hat, forward) -> np.ndarray:
    """Magnitude of the component of displacement perpendicular to F."""
    p = np.asarray(P, np.float64)
    q = np.asarray(P_hat, np.float64)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("P/P_hat shape mismatch")
    f = _unit(forward)
    delta = q - p
    parallel = (delta @ f)[:, None] * f[None, :]
    return np.linalg.norm(delta - parallel, axis=1)


def corruption_diagnostics(P, P_hat, forward, depth_delta) -> dict:
    d = np.asarray(depth_delta, np.float64).reshape(-1)
    perp = ray_alignment_residual(P, P_hat, forward)
    return {
        "row_count": int(len(d)),
        "depth_delta_mean": float(np.mean(d)) if len(d) else 0.0,
        "depth_delta_rms": float(np.sqrt(np.mean(d * d))) if len(d) else 0.0,
        "depth_delta_abs_p95": float(np.quantile(np.abs(d), 0.95)) if len(d) else 0.0,
        "max_ray_perpendicular_residual": float(perp.max()) if len(perp) else 0.0,
    }
