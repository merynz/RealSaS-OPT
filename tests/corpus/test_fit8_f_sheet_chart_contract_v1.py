from tools.corpus.fit8_f_sheet_chart_contract_v1 import (
    AREA_SUPPORT,
    OrientedSheetContinuationV1,
    SheetPatchKeyV1,
    SheetPatchRecordV1,
    build_sheet_charts_v1,
)


def _patch(cell, patch_id, component=0, *, open_boundary=False, nonmanifold=False):
    return SheetPatchRecordV1(
        key=SheetPatchKeyV1(cell, patch_id),
        source_component_id=component,
        support_dimension=AREA_SUPPORT,
        authored_open_boundary=open_boundary,
        authored_nonmanifold_incidence=nonmanifold,
        provenance="synthetic_guard",
    )


def _cont(a, b, transport=1):
    return OrientedSheetContinuationV1(
        a=a.key,
        b=b.key,
        orientation_transport=transport,
        provenance="synthetic_guard",
    )


def test_open_orientable_chart_is_admitted_with_local_sign_gauge():
    a = _patch((0, 0, 0), 0, open_boundary=True)
    b = _patch((1, 0, 0), 0, open_boundary=True)
    charts = build_sheet_charts_v1([a, b], [_cont(a, b, 1)])
    assert len(charts) == 1
    chart = charts[0]
    assert chart.orientation_consistent
    assert chart.has_authored_open_boundary
    assert chart.base_sheet_scalar_admitted
    assert chart.admission_reason == "OPEN_ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED"


def test_orientation_transport_conflict_forces_base_abstention():
    a = _patch((0, 0, 0), 0)
    b = _patch((1, 0, 0), 0)
    c = _patch((1, 1, 0), 0)
    # a->b same, b->c same, c->a flipped. a/c are not face neighbors, so use
    # a fourth cell to form a valid 4-cycle with odd parity.
    d = _patch((0, 1, 0), 0)
    continuations = [
        _cont(a, b, 1),
        _cont(b, c, 1),
        _cont(c, d, 1),
        _cont(d, a, -1),
    ]
    chart = build_sheet_charts_v1([a, b, c, d], continuations)[0]
    assert not chart.orientation_consistent
    assert not chart.base_sheet_scalar_admitted
    assert chart.admission_reason == "ORIENTATION_TRANSPORT_CONFLICT"


def test_authored_nonmanifold_incidence_requires_explicit_topology():
    a = _patch((0, 0, 0), 0, nonmanifold=True)
    chart = build_sheet_charts_v1([a], [])[0]
    assert chart.has_authored_nonmanifold_incidence
    assert not chart.base_sheet_scalar_admitted
    assert "NONMANIFOLD" in chart.admission_reason


def test_continuation_cannot_merge_distinct_source_components():
    a = _patch((0, 0, 0), 0, component=0)
    b = _patch((1, 0, 0), 0, component=1)
    try:
        build_sheet_charts_v1([a, b], [_cont(a, b)])
    except ValueError as exc:
        assert "MERGES_SOURCE_COMPONENTS" in str(exc)
    else:
        raise AssertionError("expected source component merge rejection")


def test_two_sheets_in_same_cell_remain_distinct_without_qualified_continuation():
    a = _patch((0, 0, 0), 0, component=0)
    b = _patch((0, 0, 0), 1, component=0)
    charts = build_sheet_charts_v1([a, b], [])
    assert len(charts) == 2
