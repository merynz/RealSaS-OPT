from __future__ import annotations

"""Blockwise skimage Lewiner marching-cubes equivalence kernel (X2).

One global sampled scalar lattice is partitioned into disjoint cube-cell blocks.
Each block receives its inclusive corner slab, runs the same skimage Lewiner MC,
and emitted vertices are welded by the unique global lattice edge on which MC
placed them.

X2 is sampled-lattice equivalence only; it is not a continuous-field certificate.
"""

from dataclasses import dataclass
from collections import Counter
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BlockMCResult:
    vertices: np.ndarray
    faces: np.ndarray
    edge_keys: tuple
    processed_block_count: int
    skipped_block_count: int


def _require_skimage():
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("scikit-image is required for block MC X2") from exc
    return measure


def _edge_key_from_global_grid_vertex(
    gxyz: np.ndarray,
    *,
    integer_tol: float = 2e-4,
):
    g = np.asarray(gxyz, dtype=np.float64).reshape(3)
    nearest = np.rint(g)
    err = np.abs(g - nearest)

    # MC vertices lie on lattice edges: two coordinates are integral, one is
    # interpolated. Exact-zero corners are excluded by the fixture contract.
    order = np.argsort(err)
    fixed_axes = order[:2]
    varying_axis = int(order[2])
    if np.any(err[fixed_axes] > integer_tol):
        raise RuntimeError(
            f"MC vertex is not on a canonical lattice edge: g={g.tolist()} err={err.tolist()}"
        )

    base = nearest.astype(np.int64)
    t = float(g[varying_axis])
    lo = int(np.floor(t))
    hi = int(np.ceil(t))
    if lo == hi:
        # A very-near-endpoint interpolation may round to an integer in float32.
        # Recover the incident edge direction from the largest fractional-error
        # axis and choose the neighboring lattice point on the side indicated by
        # the unrounded coordinate when possible.
        frac = t - float(nearest[varying_axis])
        if frac > 0.0:
            lo = int(nearest[varying_axis])
            hi = lo + 1
        elif frac < 0.0:
            hi = int(nearest[varying_axis])
            lo = hi - 1
        else:
            raise RuntimeError(
                f"MC vertex collapsed exactly to lattice corner under X2 policy: {g.tolist()}"
            )

    a = base.copy()
    b = base.copy()
    a[varying_axis] = lo
    b[varying_axis] = hi
    ta = tuple(int(v) for v in a)
    tb = tuple(int(v) for v in b)
    return (ta, tb) if ta <= tb else (tb, ta)


def _canonical_face_signature(edge_keys: tuple, faces: np.ndarray):
    rows = []
    for tri in np.asarray(faces, dtype=np.int64):
        rows.append(
            tuple(sorted((edge_keys[int(tri[0])], edge_keys[int(tri[1])], edge_keys[int(tri[2])])))
        )
    return tuple(sorted(rows))


def canonical_block_mc_signature(result: BlockMCResult):
    return tuple(sorted(result.edge_keys)), _canonical_face_signature(
        result.edge_keys, result.faces
    )


def _audit_topology(vertices: np.ndarray, faces: np.ndarray) -> dict:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return {
            "degenerate_face_count": 0,
            "duplicate_face_count": 0,
            "nonmanifold_edge_count": 0,
            "boundary_edge_count": 0,
        }

    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    canon = np.sort(f, axis=1)
    duplicate = len(canon) - len({tuple(row) for row in canon})

    incidence = Counter()
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            incidence[tuple(sorted((int(a), int(b))))] += 1

    return {
        "degenerate_face_count": int(np.sum(area2 <= 1e-14)),
        "duplicate_face_count": int(duplicate),
        "nonmanifold_edge_count": int(sum(n > 2 for n in incidence.values())),
        "boundary_edge_count": int(sum(n == 1 for n in incidence.values())),
    }


def extract_blockwise_lewiner_mc(
    values_zyx: np.ndarray,
    *,
    block_cells_xyz: tuple[int, int, int],
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.0,
) -> BlockMCResult:
    measure = _require_skimage()
    field = np.asarray(values_zyx, dtype=np.float32)
    if field.ndim != 3 or min(field.shape) < 2:
        raise ValueError("values_zyx must be [Z,Y,X] with all dimensions >= 2")
    if not np.isfinite(field).all():
        raise ValueError("non-finite scalar lattice")
    if np.any(field == float(level)):
        raise ValueError("X2 exact-zero lattice corners are forbidden")

    bx, by, bz = map(int, block_cells_xyz)
    if min(bx, by, bz) <= 0:
        raise ValueError("block cell extents must be positive")

    origin = np.asarray(origin_xyz, dtype=np.float64)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if origin.shape != (3,) or spacing.shape != (3,) or np.any(spacing <= 0.0):
        raise ValueError("invalid origin/spacing")

    rz, ry, rx = field.shape
    nx, ny, nz = rx - 1, ry - 1, rz - 1

    key_to_vid = {}
    vertices = []
    edge_keys = []
    faces = []
    processed = 0
    skipped = 0

    for z0 in range(0, nz, bz):
        z1 = min(z0 + bz, nz)
        for y0 in range(0, ny, by):
            y1 = min(y0 + by, ny)
            for x0 in range(0, nx, bx):
                x1 = min(x0 + bx, nx)
                slab = field[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
                smin, smax = float(slab.min()), float(slab.max())
                if not (smin < float(level) < smax):
                    skipped += 1
                    continue
                processed += 1

                v_zyx, local_faces, _n, _ = measure.marching_cubes(
                    slab,
                    level=float(level),
                    spacing=(1.0, 1.0, 1.0),
                    allow_degenerate=False,
                    method="lewiner",
                )
                local_gxyz = np.stack(
                    [
                        x0 + v_zyx[:, 2],
                        y0 + v_zyx[:, 1],
                        z0 + v_zyx[:, 0],
                    ],
                    axis=-1,
                ).astype(np.float64)

                remap = {}
                for i, g in enumerate(local_gxyz):
                    key = _edge_key_from_global_grid_vertex(g)
                    vid = key_to_vid.get(key)
                    world = origin + g * spacing
                    if vid is None:
                        vid = len(vertices)
                        key_to_vid[key] = vid
                        vertices.append(world)
                        edge_keys.append(key)
                    else:
                        if float(np.linalg.norm(vertices[vid] - world)) > 2e-5 * float(np.max(spacing)):
                            raise RuntimeError(
                                f"shared edge vertex position mismatch for {key}"
                            )
                    remap[int(i)] = int(vid)

                for tri in np.asarray(local_faces, dtype=np.int64):
                    faces.append(
                        tuple(remap[int(i)] for i in tri)
                    )

    if not vertices:
        return BlockMCResult(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            edge_keys=tuple(),
            processed_block_count=processed,
            skipped_block_count=skipped,
        )

    result = BlockMCResult(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64).reshape(-1, 3),
        edge_keys=tuple(edge_keys),
        processed_block_count=processed,
        skipped_block_count=skipped,
    )
    audit = _audit_topology(result.vertices, result.faces)
    if audit["degenerate_face_count"] or audit["duplicate_face_count"] or audit["nonmanifold_edge_count"]:
        raise RuntimeError(f"block MC topology alarm: {audit}")
    return result


def extract_dense_lewiner_mc(
    values_zyx: np.ndarray,
    *,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    level: float = 0.0,
) -> BlockMCResult:
    field = np.asarray(values_zyx)
    rz, ry, rx = field.shape
    return extract_blockwise_lewiner_mc(
        field,
        block_cells_xyz=(rx - 1, ry - 1, rz - 1),
        origin_xyz=origin_xyz,
        spacing_xyz=spacing_xyz,
        level=level,
    )


def topology_audit(result: BlockMCResult) -> dict:
    return _audit_topology(result.vertices, result.faces)
