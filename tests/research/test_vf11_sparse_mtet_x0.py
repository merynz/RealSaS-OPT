from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import sparse_mtet_x0 as x0  # noqa: E402


def _grid(axis, fn):
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return fn(x, y, z).astype(np.float64)


def _edge_incidence(mesh):
    cnt = Counter()
    for tri in np.asarray(mesh.faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            e = tuple(sorted((int(a), int(b))))
            cnt[e] += 1
    return cnt


def test_sparse_equals_dense_oblique_plane():
    axis = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    f = _grid(axis, lambda x, y, z: x + 0.37*y - 0.21*z - 0.1137)

    active = x0.active_cubes_from_corner_grid(f)
    sparse = x0.extract_sparse_mtet(
        f,
        cube_indices_xyz=active,
        origin_xyz=(axis[0], axis[0], axis[0]),
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    dense = x0.extract_sparse_mtet(
        f,
        cube_indices_xyz=None,
        origin_xyz=(axis[0], axis[0], axis[0]),
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    assert x0.canonical_mesh_signature(sparse) == x0.canonical_mesh_signature(dense)

    vals = (
        sparse.vertices[:, 0]
        + 0.37*sparse.vertices[:, 1]
        - 0.21*sparse.vertices[:, 2]
        - 0.1137
    )
    assert float(np.max(np.abs(vals))) < 1e-12


def test_sparse_equals_dense_sphere():
    axis = np.linspace(-1.0, 1.0, 11, dtype=np.float64)
    f = _grid(axis, lambda x, y, z: x*x + y*y + z*z - 0.53**2 + 1.7e-5)

    active = x0.active_cubes_from_corner_grid(f)
    assert 0 < len(active) < (len(axis)-1)**3

    sparse = x0.extract_sparse_mtet(
        f,
        cube_indices_xyz=active,
        origin_xyz=(axis[0],)*3,
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    dense = x0.extract_sparse_mtet(
        f,
        origin_xyz=(axis[0],)*3,
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    assert x0.canonical_mesh_signature(sparse) == x0.canonical_mesh_signature(dense)


def test_sparse_equals_dense_saddle_like_field():
    axis = np.linspace(-0.9, 0.9, 9, dtype=np.float64)
    f = _grid(axis, lambda x, y, z: z + 0.31*x*y + 0.07*x - 0.043)

    active = x0.active_cubes_from_corner_grid(f)
    sparse = x0.extract_sparse_mtet(
        f,
        cube_indices_xyz=active,
        origin_xyz=(axis[0],)*3,
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    dense = x0.extract_sparse_mtet(
        f,
        origin_xyz=(axis[0],)*3,
        spacing_xyz=(axis[1]-axis[0],)*3,
    )
    assert x0.canonical_mesh_signature(sparse) == x0.canonical_mesh_signature(dense)


def test_adjacent_cube_shared_face_does_not_create_internal_crack():
    # 3 x 2 x 2 corner lattice -> two adjacent X cubes. The plane crosses both.
    xs = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([0.0, 1.0])
    zs = np.asarray([0.0, 1.0])
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")
    f = x + 0.3*y + 0.2*z - 1.07

    mesh = x0.extract_sparse_mtet(f)
    inc = _edge_incidence(mesh)

    # Any contour edge that lies strictly inside the shared x=1 face must have
    # incidence two. Boundary-of-domain edges may have incidence one.
    shared_x = 1.0
    for (a, b), n in inc.items():
        pa, pb = mesh.vertices[a], mesh.vertices[b]
        if (
            abs(pa[0] - shared_x) < 1e-12
            and abs(pb[0] - shared_x) < 1e-12
            and np.all(pa[1:] > 1e-12)
            and np.all(pa[1:] < 1.0-1e-12)
            and np.all(pb[1:] > 1e-12)
            and np.all(pb[1:] < 1.0-1e-12)
        ):
            assert n == 2


def test_exact_zero_corner_fails_closed():
    f = np.ones((2, 2, 2), dtype=np.float64)
    f[0, 0, 0] = 0.0
    try:
        x0.extract_sparse_mtet(f)
    except ValueError as exc:
        assert "exact-zero" in str(exc)
    else:
        raise AssertionError("exact-zero corner must fail closed")
