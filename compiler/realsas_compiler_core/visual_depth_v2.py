"""Independent source-art raster reference with canonical depth ownership."""
from types import SimpleNamespace
import numpy as np

from .types import QualificationError
from .visual_domain_v2 import POLICY


def render_visual_depth(*, positions, depths, faces, uv, texture, resolution):
    p, z, f, uv = (np.asarray(positions, dtype=float), np.asarray(depths, dtype=float),
                   np.asarray(faces, dtype=np.int64), np.asarray(uv, dtype=float))
    tex = np.asarray(texture, dtype=np.uint8)
    if (p.ndim != 2 or p.shape[1] != 2 or z.shape != (len(p),)
            or uv.shape != p.shape or not np.isfinite(p).all()
            or not np.isfinite(z).all() or np.any(z <= 0) or not np.isfinite(uv).all()
            or f.ndim != 2 or f.shape[1] != 3 or np.any(f < 0)
            or np.any(f >= len(p)) or tex.ndim != 3 or tex.shape[2] != 4
            or resolution <= 0):
        raise QualificationError("VISUAL_CANONICAL_DEPTH_INPUT_INVALID")
    height, width = tex.shape[:2]
    p = (p + 0.5) * np.array([resolution / width, resolution / height])
    pixels, values, colors, owners = [], [], [], []

    def orient(a, b, x, y):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    def top_left(a, b):
        dy, dx = b[1] - a[1], b[0] - a[0]
        return dy < 0 or (abs(dy) <= 1e-12 and dx > 0)

    for fi, face in enumerate(f):
        a, b, c = p[face]
        area = orient(a, b, c[0], c[1])
        if abs(area) <= 1e-12:
            continue
        lo = np.maximum(0, np.floor(np.min(p[face], axis=0) - 0.5).astype(int))
        hi = np.minimum(resolution - 1, np.ceil(np.max(p[face], axis=0) - 0.5).astype(int))
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        x, y = xx.ravel() + 0.5, yy.ravel() + 0.5
        w0, w1 = orient(b, c, x, y), orient(c, a, x, y)
        w2 = orient(a, b, x, y)
        sign = 1 if area > 0 else -1
        tl = (top_left(b, c), top_left(c, a), top_left(a, b)) if area > 0 else (
            top_left(c, b), top_left(a, c), top_left(b, a))
        inside = np.ones(len(x), dtype=bool)
        for q, edge in zip((w0, w1, w2), tl):
            q = sign * q
            inside &= (q > 1e-12) | ((np.abs(q) <= 1e-12) & edge)
        if not np.any(inside):
            continue
        w0, w1 = w0[inside] / area, w1[inside] / area
        w2 = 1 - w0 - w1
        u = np.clip(w0 * uv[face[0], 0] + w1 * uv[face[1], 0] + w2 * uv[face[2], 0], 0, 1)
        v = np.clip(w0 * uv[face[0], 1] + w1 * uv[face[1], 1] + w2 * uv[face[2], 1], 0, 1)
        tx, ty = u * (width - 1), v * (height - 1)
        x0, y0 = np.floor(tx).astype(int), np.floor(ty).astype(int)
        x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
        tx, ty = (tx - x0)[:, None], (ty - y0)[:, None]
        row = ((tex[y0, x0].astype(float) / 255 * (1-tx) + tex[y0, x1].astype(float) / 255 * tx) * (1-ty)
               + (tex[y1, x0].astype(float) / 255 * (1-tx) + tex[y1, x1].astype(float) / 255 * tx) * ty)
        live = row[:, 3] > 1e-12
        pixels.append((yy.ravel()[inside] * resolution + xx.ravel()[inside])[live])
        values.append((w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]])[live])
        colors.append(row[live])
        owners.append(np.full(np.count_nonzero(live), fi, dtype=np.int32))
    n = resolution * resolution
    accum = np.zeros((n, 4), dtype=float)
    owner = np.full(n, -1, dtype=np.int32)
    ties = overflow = overlap = 0
    if pixels and sum(len(x) for x in pixels):
        pixel, depth, color, ancestry = (np.concatenate(x) for x in (pixels, values, colors, owners))
        order = np.lexsort((depth, pixel))
        pixel, depth, color, ancestry = (x[order] for x in (pixel, depth, color, ancestry))
        same = pixel[1:] == pixel[:-1]
        ties = int(np.count_nonzero(same & (np.abs(depth[1:] - depth[:-1]) <= POLICY["depth_tie_epsilon"])))
        starts = np.r_[True, ~same]
        rank = np.arange(len(pixel)) - np.maximum.accumulate(np.where(starts, np.arange(len(pixel)), 0))
        overflow = int(np.count_nonzero(rank >= POLICY["maximum_fragment_layers"]))
        overlap = int(len(np.unique(pixel[rank > 0])))
        # Front to back. Ownership is the nearest nontransparent canonical
        # fragment, independent of triangle enumeration and source alpha.
        for layer in range(int(np.max(rank)) + 1):
            take = rank == layer
            dst, src = pixel[take], color[take]
            alpha = src[:, 3]
            transmission = 1 - accum[dst, 3]
            accum[dst, :3] += src[:, :3] * alpha[:, None] * transmission[:, None]
            accum[dst, 3] += alpha * transmission
            if layer == 0:
                owner[dst] = ancestry[take]
    visible = accum[:, 3] > 1e-12
    # The deployment consumer evaluates straight RGB with one shared reciprocal.
    # Division per channel has a different last bit at half-byte boundaries.
    # Keep the exact arithmetic contract; byte parity remains fail-closed.
    inverse_alpha = 1.0 / accum[visible, 3, None]
    accum[visible, :3] *= inverse_alpha
    rgba = np.floor(np.clip(accum, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(resolution, resolution, 4)
    provenance = np.where(visible, 0, 255).astype(np.uint8).reshape(resolution, resolution)
    return SimpleNamespace(straight_rgba_u8=rgba, provenance_code=provenance,
                           owner_face_index=owner.reshape(resolution, resolution),
                           unresolved_depth_tie_count=ties, fragment_overflow_count=overflow,
                           overlap_pixel_count=overlap)
