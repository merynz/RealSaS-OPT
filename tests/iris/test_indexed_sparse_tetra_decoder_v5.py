from __future__ import annotations

import numpy as np
import pytest

from models.iris.v5.indexed_sparse_tetra_decoder_v5 import (
    SparseRegularTetraPolicyV5,
    SparseSurfaceTruncationError,
    decode_indexed_sparse_tetra_mt_v5,
    write_stage12_npz_v5,
)


def _sphere(points: np.ndarray) -> np.ndarray:
    q = np.asarray(points, dtype=np.float32)
    return np.linalg.norm(q, axis=1).astype(np.float32) - np.float32(0.55)


def _policy() -> SparseRegularTetraPolicyV5:
    return SparseRegularTetraPolicyV5(
        base_cells=8,
        fine_cells=16,
        query_chunk=4096,
        cell_chunk=64,
        refine_slab_cells=4,
        fine_gid_cell_chunk=64,
        normal_chunk=4096,
    )


def test_indexed_sparse_mt_extracts_closed_sphere_and_stage12_arrays(tmp_path):
    mesh = decode_indexed_sparse_tetra_mt_v5(
        _sphere,
        policy=_policy(),
        work_dir=tmp_path / "work",
    )
    v = mesh.vertices_normalized
    f = mesh.faces
    n = mesh.implicit_normals

    assert v.ndim == 2 and v.shape[1] == 3 and len(v) > 100
    assert f.ndim == 2 and f.shape[1] == 3 and len(f) > 100
    assert n.shape == v.shape
    assert np.isfinite(v).all()
    assert np.isfinite(n).all()
    assert np.all(f >= 0)
    assert int(f.max()) < len(v)
    assert not np.any(
        (f[:, 0] == f[:, 1])
        | (f[:, 1] == f[:, 2])
        | (f[:, 0] == f[:, 2])
    )

    radius = np.linalg.norm(v, axis=1)
    assert float(np.max(np.abs(radius - 0.55))) < 0.025
    radial = v / np.linalg.norm(v, axis=1, keepdims=True)
    alignment = np.sum(radial * n, axis=1)
    assert float(np.min(alignment)) > 0.98

    d = mesh.diagnostics
    assert d["global_shared_edge_dedup"] is True
    assert d["deformation"] is False
    assert d["whole_cell_halo"] is False
    assert d["boundary_crossing_tets"] == 0
    assert d["face_count"] == len(f)
    assert d["vertex_count"] == len(v)
    assert len(v) < 3 * len(f)

    npz = write_stage12_npz_v5(mesh, tmp_path / "stage12.npz")
    with np.load(npz, allow_pickle=False) as data:
        assert set(data.files) == {
            "vertices_normalized",
            "faces",
            "implicit_normals",
        }
        np.testing.assert_array_equal(data["vertices_normalized"], v.astype(np.float32))
        np.testing.assert_array_equal(data["faces"], f.astype(np.int32))
        np.testing.assert_array_equal(data["implicit_normals"], n.astype(np.float32))


def test_indexed_sparse_mt_is_deterministic(tmp_path):
    policy = SparseRegularTetraPolicyV5(
        base_cells=6,
        fine_cells=12,
        query_chunk=2048,
        cell_chunk=64,
        refine_slab_cells=3,
        fine_gid_cell_chunk=64,
        normal_chunk=2048,
    )
    a = decode_indexed_sparse_tetra_mt_v5(
        _sphere,
        policy=policy,
        work_dir=tmp_path / "a",
    )
    b = decode_indexed_sparse_tetra_mt_v5(
        _sphere,
        policy=policy,
        work_dir=tmp_path / "b",
    )
    np.testing.assert_array_equal(a.vertices_normalized, b.vertices_normalized)
    np.testing.assert_array_equal(a.faces, b.faces)
    np.testing.assert_array_equal(a.implicit_normals, b.implicit_normals)
    assert a.diagnostics["vertex_count"] == b.diagnostics["vertex_count"]
    assert a.diagnostics["face_count"] == b.diagnostics["face_count"]


def test_sparse_shell_fails_closed_when_zero_surface_hits_refinement_boundary(tmp_path):
    policy = _policy()

    def near_domain_boundary(points: np.ndarray) -> np.ndarray:
        q = np.asarray(points, dtype=np.float32)
        return q[:, 0] + np.float32(0.95)

    with pytest.raises(SparseSurfaceTruncationError, match="SPARSE_SHELL_TRUNCATES_ZERO_SURFACE"):
        decode_indexed_sparse_tetra_mt_v5(
            near_domain_boundary,
            policy=policy,
            work_dir=tmp_path / "boundary",
        )


def test_policy_preserves_f4_t512_t1024_geometry():
    policy = SparseRegularTetraPolicyV5()
    assert policy.base_cells == 512
    assert policy.fine_cells == 1024
    assert policy.fine_cells == 2 * policy.base_cells
    assert policy.refine_band == pytest.approx(
        np.sqrt(3.0) * (2.0 / 512.0),
        rel=0.0,
        abs=1e-15,
    )
    assert policy.normal_epsilon == pytest.approx(
        0.5 * (2.0 / 1024.0),
        rel=0.0,
        abs=1e-15,
    )
    assert policy.fail_on_boundary_crossing is True
