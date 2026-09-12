from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mesh.mesh_binding import mesh_candidate_lineage_hash
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    qualify_mwb2_observation_cdt_mesh,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.mesh.product_qualification import (
    FIT2_PRODUCT_MESH_QUALIFIER_ID,
    qualify_fit2_product_mwb2_observation_cdt_mesh,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _surface() -> RiggingSurfaceIR:
    coords = [(1, 1), (8, 1), (8, 8), (1, 8), (4, 4), (2, 5), (6, 5)]
    nodes = tuple(
        SurfaceNode(
            f"n{i}",
            (float(x), float(y), 0.01 * float(i)),
            (0,),
            ("synthetic",),
            (f"obs:n{i}",),
            ((0, (float(x), float(y))),),
        )
        for i, (x, y) in enumerate(coords)
    )
    relations = tuple(
        SurfaceRelation(f"e{i}", f"n{i}", f"n{i + 1}", "LOCAL_SAFE", 1.0, {})
        for i in range(len(coords) - 1)
    )
    return RiggingSurfaceIR(nodes, relations, "synthetic-surface-lineage")


def _domain() -> ObservationRasterDomain:
    rows = [[0] * 10 for _ in range(10)]
    for y in range(1, 9):
        for x in range(1, 9):
            rows[y][x] = 1
    return ObservationRasterDomain.from_rows(rows, view_index=0, source_alpha_sha256="alpha")


def _candidate():
    return build_mwb2_observation_cdt_candidate(
        _surface(),
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=_domain(),
    )


def _with_coverage(candidate, **changes):
    residual = {**candidate.residual_report, **changes}
    mutated = replace(candidate, residual_report=residual, candidate_lineage_hash="")
    return replace(mutated, candidate_lineage_hash=mesh_candidate_lineage_hash(mutated))


def test_fit2_product_qualifier_records_strict_product_admission():
    surface = _surface()
    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=_domain(),
    )
    mesh = qualify_fit2_product_mwb2_observation_cdt_mesh(surface, candidate)
    report = mesh.qualification_report
    assert report["status"] == "PASS_FIT2_PRODUCT_MESH_QUALIFICATION"
    assert report["product_mesh_qualifier"] == FIT2_PRODUCT_MESH_QUALIFIER_ID
    assert report["fit2_product_mesh_quality_pass"] is True
    assert report["exact_directional_raster_remeasurement"] is True
    assert report["fit2_product_mesh_quality"]["policy"]["min_source_alpha_recall"] == 0.94


def test_legacy_cdt_pass_at_092_recall_is_not_a_fit2_product_pass():
    surface = _surface()
    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=_domain(),
    )
    candidate = _with_coverage(
        candidate,
        source_alpha_recall=0.92,
        precision_inside_alpha=1.0,
        alpha_iou=0.92,
        largest_uncovered_component_fraction=0.01,
    )

    # This documents the exact integration hole that reopened the product gate:
    # the lower-level CDT compatibility qualifier accepts >=0.90 recall.
    legacy = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert legacy.qualification_report["status"] == "PASS_OBSERVATION_DOMAIN_CDT_QUALIFICATION"

    with pytest.raises(QualificationError, match="FIT2_PRODUCT_MESH_QUALITY_GATE_FAIL"):
        qualify_fit2_product_mwb2_observation_cdt_mesh(surface, candidate)


def test_large_connected_hole_fails_even_when_aggregate_recall_is_high():
    surface = _surface()
    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=_domain(),
    )
    candidate = _with_coverage(
        candidate,
        source_alpha_recall=0.97,
        precision_inside_alpha=1.0,
        alpha_iou=0.97,
        largest_uncovered_component_fraction=0.03,
    )
    with pytest.raises(QualificationError, match="largest_uncovered_component_fraction"):
        qualify_fit2_product_mwb2_observation_cdt_mesh(surface, candidate)
