from __future__ import annotations

from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.quality import (
    FIT2_PRODUCT_MESH_QUALITY_POLICY_V1,
    coverage_gate_failures,
    mesh_raster_quality_report,
    raster_quality_gate_failures,
)
from compiler.realsas_compiler_core.types import (
    QualifiedEditableMeshIR,
    QualifiedMeshVertex,
    SurfaceSupportBinding,
)


def _mask(width=10, height=10):
    rows = [[0] * width for _ in range(height)]
    for y in range(1, 9):
        for x in range(1, 9):
            rows[y][x] = 1
    return rows


def _mesh(points, faces):
    vertices = tuple(
        QualifiedMeshVertex(
            f"V{i}",
            (float(x), float(y), 0.0),
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((f"S{i}", 1.0),)),
            f"C{i}",
            {"raster_xy": (float(x), float(y))},
        )
        for i, (x, y) in enumerate(points)
    )
    edges = sorted({tuple(sorted((face[i], face[(i + 1) % 3]))) for face in faces for i in range(3)})
    return QualifiedEditableMeshIR(
        vertices,
        tuple(faces),
        tuple(edges),
        "SURFACE",
        0,
        "CAMERA",
        {},
        "MESH",
        (),
        "OBSERVATION_DOMAIN_CDT",
    )


def test_coverage_reports_large_contiguous_hole_not_just_aggregate_recall():
    domain = ObservationRasterDomain.from_rows(_mask(), view_index=0)
    full = domain.coverage((((1.0, 1.0), (8.0, 1.0), (8.0, 8.0)), ((1.0, 1.0), (8.0, 8.0), (1.0, 8.0))))
    assert full["source_alpha_recall"] == 1.0
    assert full["alpha_iou"] == 1.0
    assert full["largest_uncovered_component_fraction"] == 0.0
    assert coverage_gate_failures(full) == ()

    half = domain.coverage((((1.0, 1.0), (4.0, 1.0), (4.0, 8.0)), ((1.0, 1.0), (4.0, 8.0), (1.0, 8.0))))
    assert half["source_alpha_recall"] < FIT2_PRODUCT_MESH_QUALITY_POLICY_V1.min_source_alpha_recall
    assert half["largest_uncovered_component_fraction"] > 0.25
    failures = coverage_gate_failures(half)
    assert "source_alpha_recall" in failures
    assert "largest_uncovered_component_fraction" in failures


def test_raster_quality_detects_duplicate_and_nonmanifold_topology():
    good = _mesh(
        ((1, 1), (8, 1), (8, 8), (1, 8)),
        (("V0", "V1", "V2"), ("V0", "V2", "V3")),
    )
    report = mesh_raster_quality_report(good)
    assert report["degenerate_faces"] == 0
    assert report["duplicate_faces"] == 0
    assert report["nonmanifold_edges"] == 0
    assert raster_quality_gate_failures(report) == ()

    bad = _mesh(
        ((1, 1), (8, 1), (8, 8), (1, 8), (4, 6)),
        (
            ("V0", "V1", "V2"),
            ("V0", "V1", "V2"),
            ("V0", "V1", "V4"),
        ),
    )
    bad_report = mesh_raster_quality_report(bad)
    assert bad_report["duplicate_faces"] == 1
    assert bad_report["nonmanifold_edges"] >= 1
    failures = raster_quality_gate_failures(bad_report)
    assert "duplicate_faces" in failures
    assert "nonmanifold_edges" in failures
