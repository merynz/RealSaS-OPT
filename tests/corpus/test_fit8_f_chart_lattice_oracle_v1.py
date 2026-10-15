from dataclasses import replace

from tools.corpus.fit8_f_chart_lattice_oracle_v1 import (
    BASE_COMPATIBLE,
    RESIDUAL_REQUIRED,
    SCALAR_INCOMPATIBLE,
    ChartCellScalarTruthV1,
    ChartLatticeVertexKeyV1,
    ChartLatticeVertexSampleV1,
    FChartLatticeOracleV1,
    MultiChartLatticeBundleV1,
    evaluate_chart_lattice_oracle_v1,
)
from tools.corpus.fit8_f_sheet_chart_contract_v1 import (
    FSheetChartV1,
    SheetPatchKeyV1,
)


def _chart(chart_id=0, component=0):
    patch = SheetPatchKeyV1((0, 0, 0), 0)
    return FSheetChartV1(
        chart_id=chart_id,
        patches=(patch,),
        patch_orientation_signs=((patch, 1),),
        source_component_id=component,
        orientation_consistent=True,
        has_authored_open_boundary=False,
        has_authored_nonmanifold_incidence=False,
        base_sheet_scalar_admitted=True,
        admission_reason="ORIENTABLE_CHART_LOCAL_SIGN_ADMITTED",
    )


def _samples(chart_id=0, *, flip=False, valid=True):
    # One negative cube corner and seven positive corners gives scalar crossings
    # on the three grid edges incident to corner 0: edge indices 0, 4 and 8.
    values = [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if flip:
        values = [-value for value in values]
    ijk = (
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
        (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    )
    return tuple(
        ChartLatticeVertexSampleV1(
            key=ChartLatticeVertexKeyV1(chart_id, coord),
            oriented_scalar=value,
            unsigned_distance=abs(value),
            orientation_valid=valid,
            source_binding=f"synthetic:{chart_id}:{coord}",
            provenance="synthetic_guard",
        )
        for coord, value in zip(ijk, values)
    )


def _cell(chart_id=0, component=0, *, counts=None, reasons=()):
    if counts is None:
        counts = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    return ChartCellScalarTruthV1(
        chart_id=chart_id,
        cell_xyz=(0, 0, 0),
        patch_local_ids=(0,),
        source_component_id=component,
        teacher_edge_crossing_counts=tuple(counts),
        topology_residual_reasons=tuple(sorted(reasons)),
        provenance="synthetic_guard",
    )


def _oracle(chart_id=0, component=0, *, samples=None, cell=None):
    return FChartLatticeOracleV1(
        chart=_chart(chart_id, component),
        extraction_pitch=1.0,
        zero_epsilon=1e-9,
        vertex_samples=samples or _samples(chart_id),
        cells=(cell or _cell(chart_id, component),),
        provenance="synthetic_guard",
    )


def test_matching_chart_scalar_crossings_are_base_compatible():
    report = evaluate_chart_lattice_oracle_v1(_oracle())
    assert report.base_compatible_cells == 1
    assert report.residual_required_cells == 0
    assert report.scalar_incompatible_cells == 0
    assert report.rows[0].classification == BASE_COMPATIBLE


def test_same_chart_edge_multi_crossing_requires_residual_not_field_fabrication():
    counts = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    counts[0] = 2
    report = evaluate_chart_lattice_oracle_v1(_oracle(cell=_cell(counts=counts)))
    assert report.rows[0].classification == RESIDUAL_REQUIRED
    assert any("MULTI_CROSSING_EDGE:0:2" == reason for reason in report.rows[0].reasons)


def test_scalar_sign_mismatch_is_field_failure_not_topology_residual():
    counts = (0,) * 12
    report = evaluate_chart_lattice_oracle_v1(_oracle(cell=_cell(counts=counts)))
    assert report.rows[0].classification == SCALAR_INCOMPATIBLE
    assert any("SPURIOUS_SCALAR_CROSSING" in reason for reason in report.rows[0].reasons)


def test_global_chart_gauge_flip_preserves_zero_set_representability():
    a = evaluate_chart_lattice_oracle_v1(_oracle(samples=_samples(flip=False)))
    b = evaluate_chart_lattice_oracle_v1(_oracle(samples=_samples(flip=True)))
    assert a.rows[0].classification == b.rows[0].classification == BASE_COMPATIBLE
    assert a.scalar_crossing_count == b.scalar_crossing_count
    assert a.teacher_crossing_count == b.teacher_crossing_count


def test_invalid_orientation_samples_fail_scalar_compatibility():
    report = evaluate_chart_lattice_oracle_v1(_oracle(samples=_samples(valid=False)))
    assert report.rows[0].classification == SCALAR_INCOMPATIBLE
    assert "ORIENTATION_VALIDITY_MISSING" in report.rows[0].reasons


def test_two_charts_may_share_physical_grid_vertices_without_scalar_identity_merge():
    first = _oracle(chart_id=0, component=0)
    second = _oracle(chart_id=1, component=1)
    bundle = MultiChartLatticeBundleV1((first, second))
    bundle.validate()
    assert bundle.physical_grid_vertex_overlap_count == 8
    keys = {
        sample.key
        for oracle in bundle.oracles
        for sample in oracle.vertex_samples
    }
    assert len(keys) == 16
