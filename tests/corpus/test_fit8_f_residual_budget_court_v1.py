from tools.corpus.fit8_f_representation_capacity_court_v1 import (
    FRepresentationCapacityReportV1,
    SCHEMA_VERSION as EDGE_SCHEMA_VERSION,
)
from tools.corpus.fit8_f_residual_budget_court_v1 import evaluate_f_residual_budget_v1
from tools.corpus.fit8_f_topology_residual_court_v1 import (
    FTopologyResidualReportV1,
    SCHEMA_VERSION as FACE_SCHEMA_VERSION,
)


def _edge_report():
    return FRepresentationCapacityReportV1(
        EDGE_SCHEMA_VERSION,
        10,
        8,
        100,
        90,
        100,
        97,
        100,
        10,
        3,
        2,
        1,
        3,
        2,
    )


def _face_report(reasons=(), canonicalization=(), strands=1):
    return FTopologyResidualReportV1(
        FACE_SCHEMA_VERSION,
        1,
        strands,
        ((0, strands),),
        tuple(reasons),
        tuple(canonicalization),
    )


def test_budget_keeps_edge_face_and_canonicalization_separate():
    report = evaluate_f_residual_budget_v1(
        _edge_report(),
        [
            _face_report(strands=2),
            _face_report(("CLOSED_LOOP",), strands=3),
            _face_report((), ("POINT_DEGENERACY",), strands=1),
        ],
    )
    assert report.edge_residual_cell_fraction == 1 / 8
    assert report.edge_extra_crossing_fraction == 3 / 100
    assert report.topology_residual_face_fraction == 1 / 3
    assert report.canonicalization_face_fraction == 1 / 3
    assert report.teacher_face_strands == 6
    assert report.residual_face_strands == 3
    assert report.face_residual_strand_fraction == 1 / 2
    assert report.explicit_topology_record_fraction == (3 + 3) / (100 + 6)
