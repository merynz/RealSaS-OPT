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


def _build(surface=None, domain=None):
    surface = surface or _surface()
    domain = domain or _domain()
    return build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=domain,
    )


def _reseal(candidate, **changes):
    mutated = replace(candidate, **changes, candidate_lineage_hash="")
    return replace(mutated, candidate_lineage_hash=mesh_candidate_lineage_hash(mutated))


def _with_reported_coverage(candidate, **changes):
    return _reseal(candidate, residual_report={**candidate.residual_report, **changes})


def _keep_only_first_face(candidate, **reported_coverage):
    assert len(candidate.faces) > 1
    residual = {**candidate.residual_report, **reported_coverage}
    return _reseal(candidate, faces=(candidate.faces[0],), residual_report=residual)


def test_fit2_product_qualifier_records_strict_product_admission():
    surface = _surface()
    domain = _domain()
    candidate = _build(surface, domain)
    mesh = qualify_fit2_product_mwb2_observation_cdt_mesh(
        surface,
        candidate,
        observation_domain=domain,
    )
    report = mesh.qualification_report
    assert report["status"] == "PASS_FIT2_PRODUCT_MESH_QUALIFICATION"
    assert report["product_mesh_qualifier"] == FIT2_PRODUCT_MESH_QUALIFIER_ID
    assert report["fit2_product_mesh_quality_pass"] is True
    assert report["exact_directional_raster_remeasurement"] is True
    assert report["coverage_recomputed_from_exact_observation_authority"] is True
    assert report["observation_mask_sha256"] == domain.mask_sha256
    assert report["fit2_product_mesh_quality"]["policy"]["min_source_alpha_recall"] == 0.94


def test_candidate_reported_coverage_cannot_fake_product_pass():
    surface = _surface()
    domain = _domain()
    candidate = _keep_only_first_face(
        _build(surface, domain),
        source_alpha_recall=0.99,
        precision_inside_alpha=1.0,
        alpha_iou=0.99,
        largest_uncovered_component_fraction=0.001,
    )

    # The historical compatibility gate trusts the sealed candidate residual and is
    # intentionally not product proof.
    legacy = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert legacy.qualification_report["status"] == "PASS_OBSERVATION_DOMAIN_CDT_QUALIFICATION"

    # Product admission re-rasterizes the exact qualified mesh against the exact
    # observation authority, so a fabricated optimistic residual cannot help.
    with pytest.raises(QualificationError, match="FIT2_PRODUCT_MESH_QUALITY_GATE_FAIL"):
        qualify_fit2_product_mwb2_observation_cdt_mesh(
            surface,
            candidate,
            observation_domain=domain,
        )


def test_legacy_092_floor_is_not_the_fit2_product_threshold():
    surface = _surface()
    domain = _domain()
    candidate = _with_reported_coverage(
        _build(surface, domain),
        source_alpha_recall=0.92,
        precision_inside_alpha=1.0,
    )
    legacy = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert legacy.qualification_report["source_alpha_recall"] == 0.92
    # The strict product path never adopts that 0.92 report; it recomputes exact
    # coverage from observation authority. This full synthetic mesh therefore passes
    # with its actual measurement rather than inheriting the compatibility floor.
    product = qualify_fit2_product_mwb2_observation_cdt_mesh(
        surface,
        candidate,
        observation_domain=domain,
    )
    assert product.qualification_report["fit2_product_mesh_quality"]["source_alpha_recall"] >= 0.94
    assert product.qualification_report["candidate_reported_coverage_diagnostics"]["source_alpha_recall"] == 0.92


def test_large_connected_hole_fails_even_when_reported_aggregate_recall_is_high():
    surface = _surface()
    domain = _domain()
    candidate = _keep_only_first_face(
        _build(surface, domain),
        source_alpha_recall=0.99,
        precision_inside_alpha=1.0,
        alpha_iou=0.99,
        largest_uncovered_component_fraction=0.001,
    )
    with pytest.raises(QualificationError, match="largest_uncovered_component_fraction"):
        qualify_fit2_product_mwb2_observation_cdt_mesh(
            surface,
            candidate,
            observation_domain=domain,
        )


def test_product_observation_hash_drift_fails_closed():
    surface = _surface()
    domain = _domain()
    candidate = _build(surface, domain)
    rows = [[0] * 10 for _ in range(10)]
    for y in range(2, 9):
        for x in range(1, 9):
            rows[y][x] = 1
    other = ObservationRasterDomain.from_rows(rows, view_index=0, source_alpha_sha256="alpha")
    with pytest.raises(QualificationError, match="OBSERVATION_MASK_HASH_MISMATCH"):
        qualify_fit2_product_mwb2_observation_cdt_mesh(
            surface,
            candidate,
            observation_domain=other,
        )
