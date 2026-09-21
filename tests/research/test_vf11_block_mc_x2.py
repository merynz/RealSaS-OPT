from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "vf11_certified_adaptive"))

import block_mc_x2 as x2  # noqa: E402


pytest.importorskip("skimage")


def _grid(nx, ny, nz, fn):
    xs = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    ys = np.linspace(-0.9, 0.9, ny, dtype=np.float64)
    zs = np.linspace(-0.8, 0.8, nz, dtype=np.float64)
    z, y, x = np.meshgrid(zs, ys, xs, indexing="ij")
    f = fn(x, y, z).astype(np.float32)
    # Fixture contract excludes exact-zero corners.
    assert not np.any(f == 0.0)
    spacing = (
        float(xs[1] - xs[0]),
        float(ys[1] - ys[0]),
        float(zs[1] - zs[0]),
    )
    origin = (float(xs[0]), float(ys[0]), float(zs[0]))
    return f, origin, spacing


def _assert_all_layouts_equal_dense(f, origin, spacing):
    dense = x2.extract_dense_lewiner_mc(
        f, origin_xyz=origin, spacing_xyz=spacing
    )
    sig = x2.canonical_block_mc_signature(dense)
    assert len(dense.faces) > 0

    for layout in ((2, 2, 2), (3, 4, 5), (5, 3, 4)):
        block = x2.extract_blockwise_lewiner_mc(
            f,
            block_cells_xyz=layout,
            origin_xyz=origin,
            spacing_xyz=spacing,
        )
        assert x2.canonical_block_mc_signature(block) == sig
        audit = x2.topology_audit(block)
        assert audit["degenerate_face_count"] == 0
        assert audit["duplicate_face_count"] == 0
        assert audit["nonmanifold_edge_count"] == 0


def test_blockwise_equals_dense_oblique_plane():
    f, origin, spacing = _grid(
        10, 9, 8,
        lambda x, y, z: x + 0.37*y - 0.21*z - 0.1137,
    )
    _assert_all_layouts_equal_dense(f, origin, spacing)


def test_blockwise_equals_dense_sphere():
    f, origin, spacing = _grid(
        12, 11, 10,
        lambda x, y, z: x*x + y*y + z*z - 0.57**2 + 1.31e-5,
    )
    _assert_all_layouts_equal_dense(f, origin, spacing)


def test_blockwise_equals_dense_saddle_like():
    f, origin, spacing = _grid(
        11, 10, 9,
        lambda x, y, z: z + 0.31*x*y + 0.07*x - 0.0437,
    )
    _assert_all_layouts_equal_dense(f, origin, spacing)


def test_blockwise_equals_dense_torus_like():
    f, origin, spacing = _grid(
        15, 14, 13,
        lambda x, y, z: (np.sqrt(x*x + y*y) - 0.53)**2 + z*z - 0.18**2 + 7.1e-6,
    )
    _assert_all_layouts_equal_dense(f, origin, spacing)


def test_exact_zero_lattice_corner_fails_closed():
    f = np.ones((3, 3, 3), dtype=np.float32)
    f[1, 1, 1] = 0.0
    with pytest.raises(ValueError, match="exact-zero"):
        x2.extract_blockwise_lewiner_mc(
            f, block_cells_xyz=(2, 2, 2)
        )
