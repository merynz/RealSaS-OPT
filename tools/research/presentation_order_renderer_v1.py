from __future__ import annotations

"""Research-only single-M fragment presentation-order court.

This mirrors the canonical single-M depth renderer, but can topologically reorder
nontransparent fragments *within one pixel* when a qualified semantic owner edge
is active. Geometry, projected depths, source/CAA material samples and alpha are
unchanged. The canonical shipping renderer is not modified.
"""

from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.single_m_caa_v1 import POLICY
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.appearance_color_v2 import (
    premultiplied_linear_to_straight_srgb_u8,
)


def stable_fragment_order(owner_ids, precedence_edges):
    """Stable topological sort of fragment indices; cycle -> original order."""
    owners = [str(x) for x in owner_ids]
    edges = set((str(a), str(b)) for a, b in precedence_edges if a != b)
    n = len(owners)
    adjacency = [set() for _ in range(n)]
    indegree = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (owners[i], owners[j]) in edges and j not in adjacency[i]:
                adjacency[i].add(j)
                indegree[j] += 1
    ready = [i for i, d in enumerate(indegree) if d == 0]
    out = []
    while ready:
        i = ready.pop(0)  # stable: original geometric order wins unconstrained ties
        out.append(i)
        for j in sorted(adjacency[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                insert = 0
                while insert < len(ready) and ready[insert] < j:
                    insert += 1
                ready.insert(insert, j)
    if len(out) != n:
        return np.arange(n, dtype=np.int64), True
    return np.asarray(out, dtype=np.int64), False


def _proof_fields(
    result,
    geometry,
    ambiguity,
    overflow,
    layer_owners,
    contributions,
    geometric_owner,
    override_mask,
    cycle_mask,
):
    result.geometry_visible = geometry
    result.final_alpha = result.straight_rgba_u8[:, :, 3] > 0
    result.exact_depth_ambiguity = ambiguity
    result.layer_overflow = overflow
    result.layer_owner_face_index = layer_owners.transpose(1, 2, 0)
    result.contributing_layer_mask = contributions.transpose(1, 2, 0)
    result.geometric_owner_face_index = geometric_owner
    # Coverage remains canonical geometry ownership, not presentation ownership.
    result.coverage_sample_owner_face_index = geometric_owner[:, :, None]
    result.coverage_sample_count = 1
    result.presentation_override_mask = override_mask
    result.presentation_cycle_mask = cycle_mask
    result.presentation_override_pixel_count = int(np.count_nonzero(override_mask))
    result.presentation_cycle_pixel_count = int(np.count_nonzero(cycle_mask))
    return result


def render_m_depth_with_presentation_order(
    *,
    positions,
    depths,
    faces,
    uv,
    texture,
    resolution,
    sample_material,
    face_owner_ids,
    precedence_edges,
):
    """Canonical fragment raster/material sampling + local semantic precedence."""
    p, z, f, uv = (
        np.asarray(positions, dtype=float),
        np.asarray(depths, dtype=float),
        np.asarray(faces, dtype=np.int64),
        np.asarray(uv, dtype=float),
    )
    tex = np.asarray(texture, dtype=np.uint8)
    owner_ids = np.asarray(face_owner_ids, dtype=object)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or z.shape != (len(p),)
        or uv.shape != p.shape
        or not np.isfinite(p).all()
        or not np.isfinite(z).all()
        or np.any(z <= 0)
        or not np.isfinite(uv).all()
        or f.ndim != 2
        or f.shape[1] != 3
        or np.any(f < 0)
        or np.any(f >= len(p))
        or tex.ndim != 3
        or tex.shape[2] != 4
        or resolution <= 0
        or owner_ids.shape != (len(f),)
    ):
        raise QualificationError("PRESENTATION_ORDER_DEPTH_INPUT_INVALID")
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
        hi = np.minimum(
            resolution - 1, np.ceil(np.max(p[face], axis=0) - 0.5).astype(int)
        )
        if np.any(lo > hi):
            continue
        yy, xx = np.mgrid[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
        x, y = xx.ravel() + 0.5, yy.ravel() + 0.5
        w0, w1 = orient(b, c, x, y), orient(c, a, x, y)
        w2 = orient(a, b, x, y)
        sign = 1 if area > 0 else -1
        tl = (
            (top_left(b, c), top_left(c, a), top_left(a, b))
            if area > 0
            else (top_left(c, b), top_left(a, c), top_left(b, a))
        )
        inside = np.ones(len(x), dtype=bool)
        for q, edge in zip((w0, w1, w2), tl):
            q = sign * q
            inside &= (q > 1e-12) | ((np.abs(q) <= 1e-12) & edge)
        if not np.any(inside):
            continue
        w0, w1 = w0[inside] / area, w1[inside] / area
        w2 = 1 - w0 - w1
        row = np.column_stack((w0, w1, w2, np.ones_like(w0)))
        pixels.append(yy.ravel()[inside] * resolution + xx.ravel()[inside])
        values.append(w0 * z[face[0]] + w1 * z[face[1]] + w2 * z[face[2]])
        colors.append(row)
        owners.append(np.full(len(w0), fi, dtype=np.int32))

    n = resolution * resolution
    accum = np.zeros((n, 4), dtype=float)
    presentation_owner = np.full(n, -1, dtype=np.int32)
    geometric_owner = np.full(n, -1, dtype=np.int32)
    output_provenance = np.full(n, 255, dtype=np.uint8)
    geometry = np.zeros(n, dtype=bool)
    ambiguity = np.zeros(n, dtype=bool)
    overflow_mask = np.zeros(n, dtype=bool)
    override_mask = np.zeros(n, dtype=bool)
    cycle_mask = np.zeros(n, dtype=bool)
    layer_owners = np.full((1, n), -1, dtype=np.int32)
    contributions = np.zeros((1, n), dtype=bool)
    ties = overflow = overlap = 0

    if pixels and sum(len(x) for x in pixels):
        pixel, depth, color, ancestry = (
            np.concatenate(x) for x in (pixels, values, colors, owners)
        )
        geometry[pixel] = True
        color = sample_material(ancestry, color[:, :3])
        fragment_provenance = np.asarray(
            getattr(sample_material, "provenance_codes", np.zeros(len(color), dtype=np.uint8))
        )
        all_order = np.lexsort((depth, pixel))
        ordered_pixel = pixel[all_order]
        first = np.r_[True, ordered_pixel[1:] != ordered_pixel[:-1]]
        output_provenance[ordered_pixel[first]] = fragment_provenance[all_order[first]]

        live = color[:, 3] > 1e-12
        fragment_provenance = fragment_provenance[live]
        pixel, depth, color, ancestry = (a[live] for a in (pixel, depth, color, ancestry))
        if len(pixel) == 0:
            result = SimpleNamespace(
                straight_rgba_u8=np.zeros((resolution, resolution, 4), np.uint8),
                provenance_code=output_provenance.reshape(resolution, resolution),
                owner_face_index=np.full((resolution, resolution), -1, np.int32),
                unresolved_depth_tie_count=0,
                fragment_overflow_count=0,
                overlap_pixel_count=0,
            )
            return _proof_fields(
                result,
                geometry.reshape(resolution, resolution),
                ambiguity.reshape(resolution, resolution),
                overflow_mask.reshape(resolution, resolution),
                layer_owners.reshape(1, resolution, resolution),
                contributions.reshape(1, resolution, resolution),
                geometric_owner.reshape(resolution, resolution),
                override_mask.reshape(resolution, resolution),
                cycle_mask.reshape(resolution, resolution),
            )

        # Canonical geometric ordering remains the proof/reference ordering.
        canonical_order = np.lexsort((depth, pixel))
        cp = pixel[canonical_order]
        cd = depth[canonical_order]
        ca = ancestry[canonical_order]
        same = cp[1:] == cp[:-1]
        ties = int(np.count_nonzero(same & (np.abs(cd[1:] - cd[:-1]) <= POLICY["depth_tie_epsilon"])))
        ambiguity[cp[1:][same & (np.abs(cd[1:] - cd[:-1]) <= POLICY["depth_tie_epsilon"])]] = True
        starts = np.r_[True, ~same]
        canonical_rank = np.arange(len(cp)) - np.maximum.accumulate(
            np.where(starts, np.arange(len(cp)), 0)
        )
        overflow = int(np.count_nonzero(canonical_rank >= POLICY["maximum_fragment_layers"]))
        overflow_mask[cp[canonical_rank >= POLICY["maximum_fragment_layers"]]] = True
        overlap = int(len(np.unique(cp[canonical_rank > 0])))
        geometric_owner[cp[starts]] = ca[starts]

        # Start from canonical ordering and minimally reorder each pixel span.
        order_chunks = []
        start = 0
        edges = tuple((str(a), str(b)) for a, b in precedence_edges)
        while start < len(canonical_order):
            end = start + 1
            while end < len(canonical_order) and cp[end] == cp[start]:
                end += 1
            local_global = canonical_order[start:end]
            local_faces = ancestry[local_global]
            local_semantic = owner_ids[local_faces]
            local_perm, cyclic = stable_fragment_order(local_semantic, edges)
            pix = int(cp[start])
            if cyclic:
                cycle_mask[pix] = True
            elif not np.array_equal(local_perm, np.arange(end - start)):
                override_mask[pix] = True
            order_chunks.append(local_global[local_perm])
            start = end
        order = np.concatenate(order_chunks) if order_chunks else canonical_order

        fragment_provenance = fragment_provenance[order]
        pixel, depth, color, ancestry = (x[order] for x in (pixel, depth, color, ancestry))
        same = pixel[1:] == pixel[:-1]
        starts = np.r_[True, ~same]
        rank = np.arange(len(pixel)) - np.maximum.accumulate(
            np.where(starts, np.arange(len(pixel)), 0)
        )
        retained = min(int(np.max(rank)) + 1, POLICY["maximum_fragment_layers"])
        layer_owners = np.full((retained, n), -1, dtype=np.int32)
        contributions = np.zeros((retained, n), dtype=bool)

        for layer in range(int(np.max(rank)) + 1):
            take = rank == layer
            dst, src = pixel[take], color[take]
            alpha = src[:, 3]
            transmission = 1 - accum[dst, 3]
            if layer < retained:
                layer_owners[layer, dst] = ancestry[take]
                contributions[layer, dst] = (alpha * transmission) > 1e-12
            accum[dst, :3] += src[:, :3] * alpha[:, None] * transmission[:, None]
            accum[dst, 3] += alpha * transmission
            if layer == 0:
                presentation_owner[dst] = ancestry[take]
                output_provenance[dst] = fragment_provenance[take]

    visible = accum[:, 3] > 1e-12
    accum[visible, :3] /= accum[visible, 3, None]
    rgba = (
        np.floor(np.clip(accum, 0, 1) * 255 + 0.5)
        .astype(np.uint8)
        .reshape(resolution, resolution, 4)
    )
    result = SimpleNamespace(
        straight_rgba_u8=rgba,
        provenance_code=output_provenance.reshape(resolution, resolution),
        owner_face_index=presentation_owner.reshape(resolution, resolution),
        unresolved_depth_tie_count=ties,
        fragment_overflow_count=overflow,
        overlap_pixel_count=overlap,
    )
    return _proof_fields(
        result,
        geometry.reshape(resolution, resolution),
        ambiguity.reshape(resolution, resolution),
        overflow_mask.reshape(resolution, resolution),
        layer_owners.reshape(-1, resolution, resolution),
        contributions.reshape(-1, resolution, resolution),
        geometric_owner.reshape(resolution, resolution),
        override_mask.reshape(resolution, resolution),
        cycle_mask.reshape(resolution, resolution),
    )
