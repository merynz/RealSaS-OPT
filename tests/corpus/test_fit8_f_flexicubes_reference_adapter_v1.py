from dataclasses import replace

from tools.corpus.fit8_f_flexicubes_reference_adapter_v1 import (
    CHART_TO_FLEXICUBES_EDGE,
    FlexiCubesScalarBindingV1,
    build_dense_flexicubes_grid_v1,
    chart_edge_crossing_to_flexicubes_v1,
    remap_chart_edge_values_to_flexicubes_v1,
    validate_vanilla_flexicubes_reference_admission_v1,
)


def _grid():
    return build_dense_flexicubes_grid_v1(
        cell_min_ijk=(2, 3, 4),
        cell_max_exclusive_ijk=(4, 5, 6),
        lattice_origin_xyz=(0.0, 0.0, 0.0),
        extraction_pitch=0.25,
    )


def _binding(grid):
    n = len(grid.lattice_ijks)
    return FlexiCubesScalarBindingV1(
        scalar_values=tuple(1.0 for _ in range(n)),
        orientation_valid=tuple(True for _ in range(n)),
        source_binding=tuple(f"v:{i}" for i in range(n)),
        global_sign_authority=True,
        support_clipping_required=False,
        provenance="synthetic_guard",
    )


def test_exact_chart_to_upstream_edge_permutation_and_orientation():
    assert tuple(index for index, _ in CHART_TO_FLEXICUBES_EDGE) == (
        0, 4, 2, 6, 8, 9, 11, 10, 3, 1, 7, 5
    )
    assert tuple(reversed_orientation for _, reversed_orientation in CHART_TO_FLEXICUBES_EDGE) == (
        False, False, False, False, True, True, True, True,
        False, False, False, False,
    )

    # Chart edge 4 is (0 -> 2), while upstream edge 8 is (2 -> 0).
    flex_edge, flex_t = chart_edge_crossing_to_flexicubes_v1(4, 0.25)
    assert flex_edge == 8
    assert abs(flex_t - 0.75) < 1e-12

    # Chart edge 9 and upstream edge 1 have the same orientation (1 -> 5).
    flex_edge, flex_t = chart_edge_crossing_to_flexicubes_v1(9, 0.25)
    assert flex_edge == 1
    assert abs(flex_t - 0.25) < 1e-12


def test_edge_weight_remap_uses_upstream_order_not_chart_order():
    values = tuple(float(i) for i in range(12))
    upstream = remap_chart_edge_values_to_flexicubes_v1(values)
    for chart_index, (flex_index, _) in enumerate(CHART_TO_FLEXICUBES_EDGE):
        assert upstream[flex_index] == values[chart_index]


def test_dense_grid_cube_order_matches_regular_xyz_volume_contract():
    grid = _grid()
    assert grid.res == (2, 2, 2)
    assert len(grid.lattice_ijks) == 27
    assert len(grid.cube_ijks) == 8
    assert grid.cube_ijks == (
        (2, 3, 4), (2, 3, 5), (2, 4, 4), (2, 4, 5),
        (3, 3, 4), (3, 3, 5), (3, 4, 4), (3, 4, 5),
    )

    first_cube_vertices = tuple(grid.lattice_ijks[i] for i in grid.cube_fx8[0])
    assert first_cube_vertices == (
        (2, 3, 4), (3, 3, 4), (2, 4, 4), (3, 4, 4),
        (2, 3, 5), (3, 3, 5), (2, 4, 5), (3, 4, 5),
    )


def test_vanilla_reference_rejects_open_or_support_clipped_chart_domain():
    grid = _grid()
    good = _binding(grid)
    validate_vanilla_flexicubes_reference_admission_v1(grid, good)
    assert good.vanilla_upstream_admitted

    no_global_sign = replace(good, global_sign_authority=False)
    try:
        validate_vanilla_flexicubes_reference_admission_v1(grid, no_global_sign)
    except ValueError as exc:
        assert "GLOBAL_SIGN_AUTHORITY_REQUIRED" in str(exc)
    else:
        raise AssertionError("expected global-sign admission rejection")

    needs_support_clip = replace(good, support_clipping_required=True)
    try:
        validate_vanilla_flexicubes_reference_admission_v1(grid, needs_support_clip)
    except ValueError as exc:
        assert "SUPPORT_CLIPPING_UNSUPPORTED" in str(exc)
    else:
        raise AssertionError("expected support-clipping admission rejection")

    invalid_orientation = replace(
        good,
        orientation_valid=(False,) + good.orientation_valid[1:],
    )
    try:
        validate_vanilla_flexicubes_reference_admission_v1(grid, invalid_orientation)
    except ValueError as exc:
        assert "ORIENTATION_VALIDITY_REQUIRED" in str(exc)
    else:
        raise AssertionError("expected orientation-validity admission rejection")
