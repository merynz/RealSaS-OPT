"""Consumer-neutral, independently implemented geometric measurement kernels.

Positions and affine frames must already share a declared source space and units.
These functions measure responses; none infers an authored mechanism or contact.
"""
from __future__ import annotations

import numpy as np


def finite(value, *, ndim=None):
    out = np.asarray(value, dtype=np.float64)
    if (ndim is not None and out.ndim != ndim) or not np.isfinite(out).all():
        raise ValueError("finite array with declared rank required")
    return out


def affine_relative_frames(frame_a, frame_b, *, condition_limit=1e8):
    """A^-1 B; retain scale, shear and reflection rather than projecting to SE(3).

    Singular/ill-conditioned A returns a false mask and NaNs, never an inverse
    repaired with a pseudoinverse. Last two axes are homogeneous 4x4 matrices.
    """
    a, b = finite(frame_a), finite(frame_b)
    if a.shape != b.shape or a.shape[-2:] != (4, 4):
        raise ValueError("matched homogeneous 4x4 frames required")
    if not np.isfinite(condition_limit) or condition_limit <= 1:
        raise ValueError("condition_limit must be finite and greater than one")
    if not (np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-12, rtol=0)
            and np.allclose(b[..., 3, :], [0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("non-affine homogeneous row")
    valid = np.linalg.cond(a[..., :3, :3]) <= condition_limit
    out = np.full(a.shape, np.nan)
    # Flattening also handles an unbatched matrix without scalar indexing rules.
    af, bf, of = (x.reshape(-1, 4, 4) for x in (a, b, out))
    mask = np.asarray(valid).reshape(-1)
    of[mask] = np.linalg.solve(af[mask], bf[mask])
    return out, valid


def response_gram(response, action_weights, valid):
    """Integral of raw response inner products under an explicit discrete law.

    response: [locus, intervention, channel]. Weights are probability masses,
    including any baseline/sham entries; no per-pair renormalization on missing
    responses. A Gram entry is invalid if either endpoint misses positive mass.
    Response channels must have a declared common physical metric. Heterogeneous
    dimensions (metres/radians/etc.) require separately preregistered blocks.
    """
    r = np.asarray(response, dtype=np.float64)
    w = finite(action_weights, ndim=1)
    v = np.asarray(valid)
    if r.ndim != 3 or not r.shape[0] or not r.shape[1] or not r.shape[2]:
        raise ValueError("nonempty [locus, action, channel] response required")
    if v.dtype != np.bool_ or v.shape != r.shape[:2] or w.shape != (r.shape[1],):
        raise ValueError("mask/weight shape mismatch")
    if np.any(w < 0) or not np.isclose(w.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("explicit probability law must sum to one")
    if not np.isfinite(r[v]).all():
        raise ValueError("valid responses must be finite")
    supported = (v | (w == 0)[None, :]).all(axis=1)
    mask = supported[:, None] & supported[None, :]
    safe = np.where(v[..., None], r, 0)
    gram = np.einsum("icd,jcd,c->ij", safe, safe, w)
    return np.where(mask, gram, np.nan), mask


def triangle_metric_strain(rest, posed, *, degeneracy_tolerance):
    """Metric strain in the rest triangle's orthonormal tangent basis.

    Inputs [...,3,3]. Reports 2x2 Green strain and area ratio. This is a surface
    differential, not a fabricated full-rank volumetric deformation gradient.
    Rigid rotations/reflections preserve metric; orientation is separate truth.
    """
    x, y = finite(rest), finite(posed)
    if x.shape != y.shape or x.shape[-2:] != (3, 3):
        raise ValueError("matched [..., vertex=3, xyz=3] triangles required")
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance <= 0:
        raise ValueError("explicit positive source-unit area tolerance required")
    dx = np.stack((x[..., 1, :] - x[..., 0, :], x[..., 2, :] - x[..., 0, :]), axis=-1)
    dy = np.stack((y[..., 1, :] - y[..., 0, :], y[..., 2, :] - y[..., 0, :]), axis=-1)
    _, r = np.linalg.qr(dx, mode="reduced")
    rest_area2 = np.linalg.norm(np.cross(dx[..., 0], dx[..., 1]), axis=-1)
    posed_area2 = np.linalg.norm(np.cross(dy[..., 0], dy[..., 1]), axis=-1)
    valid = rest_area2 > degeneracy_tolerance
    inv = np.full(r.shape, np.nan)
    flat, outf, mask = r.reshape(-1, 2, 2), inv.reshape(-1, 2, 2), np.asarray(valid).reshape(-1)
    outf[mask] = np.linalg.inv(flat[mask])
    f = dy @ inv
    strain = (np.swapaxes(f, -1, -2) @ f - np.eye(2)) * .5
    ratio = np.divide(posed_area2, rest_area2, out=np.full_like(rest_area2, np.nan), where=valid)
    return strain, ratio, valid


def oriented_pair_gap(point_a, point_b, normal_a, velocity_a, velocity_b):
    """Directed normal gap and closing speed (negative means approaching).

    Normal sign must come from a qualified orientation. Thin/open surfaces may
    have no such orientation. Zero gap alone never establishes physical contact.
    """
    a, b, n, va, vb = map(finite, (point_a, point_b, normal_a, velocity_a, velocity_b))
    if not (a.shape == b.shape == n.shape == va.shape == vb.shape) or a.shape[-1:] != (3,):
        raise ValueError("matched xyz arrays required")
    if not np.allclose(np.linalg.norm(n, axis=-1), 1, atol=1e-8, rtol=0):
        raise ValueError("qualified unit normal required")
    return np.sum((b - a) * n, axis=-1), np.sum((vb - va) * n, axis=-1)


def interval_velocity(positions, timestamps, valid):
    """Forward interval velocity at actual time intervals; no edge extrapolation."""
    p, t = np.asarray(positions, dtype=np.float64), finite(timestamps, ndim=1)
    v = np.asarray(valid)
    if p.ndim != 3 or p.shape[1] != len(t) or len(t) < 2 or p.shape[2] != 3:
        raise ValueError("[locus,time,xyz] with >=2 timestamps required")
    if v.dtype != np.bool_ or v.shape != p.shape[:2] or np.any(np.diff(t) <= 0):
        raise ValueError("boolean mask and strictly increasing seconds required")
    if not np.isfinite(p[v]).all():
        raise ValueError("valid positions must be finite")
    mask = v[:, :-1] & v[:, 1:]
    vel = np.diff(p, axis=1) / np.diff(t)[None, :, None]
    return np.where(mask[..., None], vel, np.nan), mask
