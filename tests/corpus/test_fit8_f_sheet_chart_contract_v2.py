from tools.corpus.fit8_f_sheet_chart_contract_v1 import (
    AREA_SUPPORT,
    OrientedSheetContinuationV1,
    SheetPatchKeyV1,
    SheetPatchRecordV1,
)
from tools.corpus.fit8_f_sheet_chart_contract_v2 import build_sheet_chart_partition_v2


def _patch(cell, patch_id, *, component=0, open_boundary=False, nonmanifold=False):
    return SheetPatchRecordV1(
        key=SheetPatchKeyV1(cell, patch_id),
        source_component_id=component,
        support_dimension=AREA_SUPPORT,
        authored_open_boundary=open_boundary,
        authored_nonmanifold_incidence=nonmanifold,
        provenance="synthetic_v2_guard",
    )


def _cont(a, b, transport=1):
    return OrientedSheetContinuationV1(
        a=a.key,
        b=b.key,
        orientation_transport=transport,
        provenance="synthetic_v2_guard",
    )


def test_single_nonmanifold_patch_does_not_poison_entire_manifold_chart():
    a = _patch((0, 0, 0), 0)
    singular = _patch((1, 0, 0), 0, nonmanifold=True)
    b = _patch((2, 0, 0), 0)
    result = build_sheet_chart_partition_v2(
        [a, singular, b],
        [_cont(a, singular), _cont(singular, b)],
    )

    assert result.singular_patch_keys == (singular.key,)
    assert singular.key in result.topology_residual_patch_keys
    assert len(result.base_charts) == 2
    assert all(chart.base_sheet_scalar_admitted for chart in result.base_charts)
    assert all(not chart.has_authored_nonmanifold_incidence for chart in result.base_charts)
    assert len(result.residual_boundary_continuations) == 2


def test_authored_open_boundary_keeps_geometry_chart_but_requires_topology_residual():
    a = _patch((0, 0, 0), 0, open_boundary=True)
    b = _patch((1, 0, 0), 0)
    result = build_sheet_chart_partition_v2([a, b], [_cont(a, b)])

    assert len(result.base_charts) == 1
    assert result.base_charts[0].base_sheet_scalar_admitted
    assert result.base_charts[0].has_authored_open_boundary
    assert a.key in result.topology_residual_patch_keys
    assert not result.singular_patch_keys


def test_orientation_conflict_still_abstains_for_affected_base_chart():
    a = _patch((0, 0, 0), 0)
    b = _patch((1, 0, 0), 0)
    c = _patch((1, 1, 0), 0)
    d = _patch((0, 1, 0), 0)
    result = build_sheet_chart_partition_v2(
        [a, b, c, d],
        [
            _cont(a, b, 1),
            _cont(b, c, 1),
            _cont(c, d, 1),
            _cont(d, a, -1),
        ],
    )
    assert len(result.base_charts) == 1
    assert not result.base_charts[0].orientation_consistent
    assert not result.base_charts[0].base_sheet_scalar_admitted


def test_singular_to_singular_edges_remain_outside_base_scalar_graph():
    a = _patch((0, 0, 0), 0, nonmanifold=True)
    b = _patch((1, 0, 0), 0, nonmanifold=True)
    result = build_sheet_chart_partition_v2([a, b], [_cont(a, b)])
    assert not result.base_charts
    assert set(result.singular_patch_keys) == {a.key, b.key}
    assert set(result.topology_residual_patch_keys) == {a.key, b.key}
    assert not result.residual_boundary_continuations
