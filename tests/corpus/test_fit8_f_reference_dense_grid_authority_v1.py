from dataclasses import replace

import numpy as np

from tools.corpus.fit8_f_reference_dense_grid_authority_v1 import (
    CUBE_ORDERING,
    FReferenceDenseGridAuthorityV1,
    REGULAR_LATTICE,
    SCALAR_SIGN_CONVENTION,
    validate_dense_grid_bundle_arrays_v1,
)

H = "a" * 64


def _authority():
    return FReferenceDenseGridAuthorityV1(
        cell_min_ijk=(0, 0, 0),
        cell_max_exclusive_ijk=(2, 2, 2),
        lattice_origin_xyz=(-0.5, -0.5, -0.5),
        extraction_pitch=0.5,
        cube_ordering=CUBE_ORDERING,
        scalar_sign_convention=SCALAR_SIGN_CONVENTION,
        vertex_position_mode=REGULAR_LATTICE,
        bundle_npz_sha256=H,
        provenance="synthetic_guard",
        metadata={},
    )


def _arrays(authority):
    grid = authority.expected_grid()
    return {
        "x_nx3": np.asarray(grid.vertex_xyz, dtype=np.float32),
        "cube_fx8": np.asarray(grid.cube_fx8, dtype=np.int64),
    }


def test_exact_dense_grid_is_admitted():
    authority = _authority()
    validate_dense_grid_bundle_arrays_v1(
        authority,
        _arrays(authority),
        manifest_res=authority.res,
    )


def test_cube_row_permutation_is_rejected_even_if_count_matches():
    authority = _authority()
    arrays = _arrays(authority)
    arrays["cube_fx8"] = arrays["cube_fx8"][[1, 0, 2, 3, 4, 5, 6, 7]]
    try:
        validate_dense_grid_bundle_arrays_v1(authority, arrays, manifest_res=authority.res)
    except ValueError as exc:
        assert "CUBE_ORDER_DRIFT" in str(exc)
    else:
        raise AssertionError("expected cube-order drift rejection")


def test_vertex_row_permutation_or_position_drift_is_rejected():
    authority = _authority()
    arrays = _arrays(authority)
    arrays["x_nx3"] = arrays["x_nx3"].copy()
    arrays["x_nx3"][[0, 1]] = arrays["x_nx3"][[1, 0]]
    try:
        validate_dense_grid_bundle_arrays_v1(authority, arrays, manifest_res=authority.res)
    except ValueError as exc:
        assert "VERTEX_ORDER_OR_POSITION_DRIFT" in str(exc)
    else:
        raise AssertionError("expected vertex-order drift rejection")


def test_manifest_res_drift_is_rejected():
    authority = _authority()
    try:
        validate_dense_grid_bundle_arrays_v1(authority, _arrays(authority), manifest_res=(2, 2, 3))
    except ValueError as exc:
        assert "MANIFEST_RES_DRIFT" in str(exc)
    else:
        raise AssertionError("expected res drift rejection")


def test_sign_and_position_semantics_are_fail_closed():
    authority = _authority()
    for broken in (
        replace(authority, scalar_sign_convention="POSITIVE_INSIDE"),
        replace(authority, vertex_position_mode="DEFORMED_LATTICE"),
        replace(authority, cube_ordering="ARBITRARY"),
    ):
        try:
            broken.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("expected semantic drift rejection")
