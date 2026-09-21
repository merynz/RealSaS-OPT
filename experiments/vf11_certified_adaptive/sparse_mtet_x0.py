from __future__ import annotations

"""Sparse fixed-Freudenthal marching tetrahedra research kernel.

The semantic target of X0 is intentionally narrow: extracting only cubes whose
corner samples bracket zero must produce exactly the same mesh as traversing
every cube on the same canonical lattice and letting non-crossing cubes emit
nothing.

This is a deterministic topology/stitching baseline, not a product extractor.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


# Cube corners in XYZ bit order:
# 0 000, 1 100, 2 010, 3 110, 4 001, 5 101, 6 011, 7 111.
CUBE_OFFSETS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=np.int64,
)

# Freudenthal decomposition using the global 000 -> 111 body diagonal.
FREUDENTHAL_TETS = np.asarray(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class SparseMTetMesh:
    vertices: np.ndarray
    faces: np.ndarray
    source_edge_keys: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]


def _edge_key(a: np.ndarray, b: np.ndarray):
    ta = tuple(int(v) for v in a)
    tb = tuple(int(v) for v in b)
    return (ta, tb) if ta <= tb else (tb, ta)


def _interp_zero(pa: np.ndarray, pb: np.ndarray, va: float, vb: float) -> np.ndarray:
    if va == 0.0 or vb == 0.0:
        raise ValueError("X0 exact-zero corner values are forbidden")
    if (va < 0.0) == (vb < 0.0):
        raise ValueError("edge does not bracket zero")
    t = float(va / (va - vb))
    if not (0.0 < t < 1.0):
        raise RuntimeError(f"invalid interpolation parameter {t}")
    return pa + t * (pb - pa)


def _polygon_order(points: np.ndarray, grad: np.ndarray) -> list[int]:
    """Order 3/4 coplanar contour points around their centroid."""
    pts = np.asarray(points, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    gn = float(np.linalg.norm(g))
    if gn <= 1e-14:
        # This should be rare for affine tetra data; choose a stable geometric
        # normal from the contour polygon instead of silently randomizing order.
        if len(pts) < 3:
            raise RuntimeError("insufficient contour points")
        g = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        gn = float(np.linalg.norm(g))
        if gn <= 1e-14:
            raise RuntimeError("degenerate contour polygon")
    n = g / gn

    # Stable in-plane basis.
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, axis)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    c = pts.mean(axis=0)
    rel = pts - c
    ang = np.arctan2(rel @ v, rel @ u)
    order = list(np.argsort(ang))

    # Ensure polygon normal points in +gradient direction.
    q = pts[order]
    poly_n = np.zeros(3, dtype=np.float64)
    for i in range(len(q)):
        poly_n += np.cross(q[i] - c, q[(i + 1) % len(q)] - c)
    if float(np.dot(poly_n, n)) < 0.0:
        order.reverse()
    return order


def _tet_gradient(points: np.ndarray, values: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    f = np.asarray(values, dtype=np.float64)
    a = np.column_stack([p, np.ones(4, dtype=np.float64)])
    coeff, *_ = np.linalg.lstsq(a, f, rcond=None)
    return coeff[:3]


def active_cubes_from_corner_grid(values_zyx: np.ndarray) -> np.ndarray:
    """Return XYZ cube indices whose 8 corner samples strictly bracket zero."""
    f = np.asarray(values_zyx, dtype=np.float64)
    if f.ndim != 3 or min(f.shape) < 2:
        raise ValueError("values_zyx must be [Z,Y,X] with each dimension >= 2")
    if not np.isfinite(f).all():
        raise ValueError("non-finite grid")
    if np.any(f == 0.0):
        raise ValueError("X0 exact-zero corner values are forbidden")

    rz, ry, rx = f.shape
    rows = []
    for z in range(rz - 1):
        for y in range(ry - 1):
            for x in range(rx - 1):
                vals = np.asarray([
                    f[z, y, x],
                    f[z, y, x + 1],
                    f[z, y + 1, x],
                    f[z, y + 1, x + 1],
                    f[z + 1, y, x],
                    f[z + 1, y, x + 1],
                    f[z + 1, y + 1, x],
                    f[z + 1, y + 1, x + 1],
                ])
                if vals.min() < 0.0 < vals.max():
                    rows.append((x, y, z))
    return np.asarray(rows, dtype=np.int64).reshape(-1, 3)


def extract_sparse_mtet(
    values_zyx: np.ndarray,
    *,
    cube_indices_xyz: Iterable[Iterable[int]] | None = None,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> SparseMTetMesh:
    f = np.asarray(values_zyx, dtype=np.float64)
    if f.ndim != 3 or min(f.shape) < 2:
        raise ValueError("values_zyx must be [Z,Y,X]")
    if not np.isfinite(f).all():
        raise ValueError("non-finite grid")
    if np.any(f == 0.0):
        raise ValueError("X0 exact-zero corner values are forbidden")

    origin = np.asarray(origin_xyz, dtype=np.float64)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if origin.shape != (3,) or spacing.shape != (3,):
        raise ValueError("origin/spacing must be XYZ triples")
    if not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0.0):
        raise ValueError("invalid origin/spacing")

    rz, ry, rx = f.shape
    if cube_indices_xyz is None:
        cubes = np.asarray(
            [(x, y, z)
             for z in range(rz - 1)
             for y in range(ry - 1)
             for x in range(rx - 1)],
            dtype=np.int64,
        )
    else:
        cubes = np.asarray(list(cube_indices_xyz), dtype=np.int64).reshape(-1, 3)

    if len(cubes):
        if np.any(cubes < 0):
            raise ValueError("negative cube index")
        if np.any(cubes[:, 0] >= rx - 1) or np.any(cubes[:, 1] >= ry - 1) or np.any(cubes[:, 2] >= rz - 1):
            raise ValueError("cube index outside grid")

    edge_to_vid: dict[tuple, int] = {}
    vertices: list[np.ndarray] = []
    edge_keys: list[tuple] = []
    faces: list[tuple[int, int, int]] = []

    for cube in cubes:
        cxyz = np.asarray(cube, dtype=np.int64)
        gxyz = cxyz[None, :] + CUBE_OFFSETS
        vals = np.asarray([f[p[2], p[1], p[0]] for p in gxyz], dtype=np.float64)

        if not (vals.min() < 0.0 < vals.max()):
            continue

        wxyz = origin[None, :] + gxyz.astype(np.float64) * spacing[None, :]

        for tet in FREUDENTHAL_TETS:
            tv = vals[tet]
            neg = tv < 0.0
            nneg = int(np.sum(neg))
            if nneg == 0 or nneg == 4:
                continue

            local_crossings = []
            for i, j in combinations(range(4), 2):
                if neg[i] == neg[j]:
                    continue
                ci = int(tet[i])
                cj = int(tet[j])
                key = _edge_key(gxyz[ci], gxyz[cj])
                vid = edge_to_vid.get(key)
                if vid is None:
                    p = _interp_zero(
                        wxyz[ci], wxyz[cj],
                        float(vals[ci]), float(vals[cj]),
                    )
                    vid = len(vertices)
                    edge_to_vid[key] = vid
                    vertices.append(p)
                    edge_keys.append(key)
                local_crossings.append(vid)

            # A nondegenerate tetra zero slice is triangle (3) or quad (4).
            uniq = list(dict.fromkeys(local_crossings))
            if len(uniq) not in (3, 4):
                raise RuntimeError(
                    f"unexpected tetra contour vertex count {len(uniq)}"
                )

            pts = np.asarray([vertices[v] for v in uniq], dtype=np.float64)
            grad = _tet_gradient(wxyz[tet], vals[tet])
            order_local = _polygon_order(pts, grad)
            poly = [uniq[i] for i in order_local]

            if len(poly) == 3:
                faces.append(tuple(poly))
            else:
                # Fixed fan from the lexicographically smallest global vertex ID
                # after cyclic ordering. This is deterministic within the tetra.
                k = int(np.argmin(poly))
                q = poly[k:] + poly[:k]
                faces.append((q[0], q[1], q[2]))
                faces.append((q[0], q[2], q[3]))

    if not vertices:
        return SparseMTetMesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 3), dtype=np.int64),
            source_edge_keys=tuple(),
        )

    v = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    # Fail closed on degeneracy/duplicates.
    if len(tri):
        p0, p1, p2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        if np.any(area2 <= 1e-14):
            raise RuntimeError("degenerate triangle emitted")
        canon = np.sort(tri, axis=1)
        if len({tuple(r) for r in canon}) != len(canon):
            raise RuntimeError("duplicate triangle emitted")

    return SparseMTetMesh(v, tri, tuple(edge_keys))


def canonical_mesh_signature(mesh: SparseMTetMesh, decimals: int = 12):
    """Index-order-independent exact-enough signature for deterministic X0 tests."""
    verts = np.round(np.asarray(mesh.vertices, dtype=np.float64), decimals=decimals)
    vrows = sorted(tuple(map(float, row)) for row in verts)
    pos_to_new = {p: i for i, p in enumerate(vrows)}
    old_to_new = {
        i: pos_to_new[tuple(map(float, verts[i]))]
        for i in range(len(verts))
    }
    faces = sorted(
        tuple(sorted(old_to_new[int(i)] for i in face))
        for face in np.asarray(mesh.faces, dtype=np.int64)
    )
    return tuple(vrows), tuple(faces)
