from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from compiler.realsas_compiler_core.mesh.mwb2 import build_mwb2_candidate
from compiler.realsas_compiler_core.mesh.mwb2_cdt import (
    build_mwb2_observation_cdt_candidate,
    qualify_mwb2_observation_cdt_mesh,
)
from compiler.realsas_compiler_core.mesh.observation_domain import ObservationRasterDomain
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _surface(coords, edges, *, prefixes=None) -> RiggingSurfaceIR:
    if prefixes is None:
        ids = [f"n{i}" for i in range(len(coords))]
    else:
        ids = list(prefixes)
    nodes = tuple(
        SurfaceNode(
            ids[i],
            (float(x), float(y), 0.01 * float(i)),
            (0,),
            ("synthetic",),
            (f"obs:{ids[i]}",),
            ((0, (float(x), float(y))),),
        )
        for i, (x, y) in enumerate(coords)
    )
    relations = tuple(
        SurfaceRelation(f"e{i}", ids[a], ids[b], "LOCAL_SAFE", 1.0, {})
        for i, (a, b) in enumerate(edges)
    )
    return RiggingSurfaceIR(nodes, relations, "synthetic-surface-lineage")


def _rect_mask(width, height, rects):
    rows = [[0] * width for _ in range(height)]
    for x0, y0, x1, y1 in rects:
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                rows[y][x] = 1
    return rows


def _chain_square_surface() -> RiggingSurfaceIR:
    coords = [(1, 1), (8, 1), (8, 8), (1, 8), (4, 4), (2, 5), (6, 5)]
    edges = [(i, i + 1) for i in range(len(coords) - 1)]
    return _surface(coords, edges)


def test_observation_domain_integrity_is_fail_closed():
    domain = ObservationRasterDomain.from_rows([[0, 1], [1, 1]], view_index=0)
    domain.validate()
    tampered = replace(domain, mask_bytes=bytes([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        tampered.validate()


def test_cdt_fills_connected_domain_without_relation_triangles():
    surface = _chain_square_surface()
    domain = ObservationRasterDomain.from_rows(
        _rect_mask(10, 10, [(1, 1, 8, 8)]),
        view_index=0,
        source_alpha_sha256="synthetic-alpha",
    )

    with pytest.raises(QualificationError, match="MWB2_NO_NONDEGENERATE_SAFE_FACE"):
        build_mwb2_candidate(surface, view_index=0, camera_binding_hash="camera")

    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=domain,
    )
    assert candidate.coverage_classification == "OBSERVATION_DOMAIN_CDT"
    assert candidate.residual_report["source_alpha_recall"] == 1.0
    assert candidate.residual_report["precision_inside_alpha"] == 1.0
    assert candidate.residual_report["face_count"] >= 4
    assert candidate.solver_provenance["quality_refinement_enabled"] is False
    assert candidate.solver_provenance["cdt_promoted"] is False
    assert all(v.support_binding.mode == "IDENTITY_SURFACE_NODE" for v in candidate.vertices)

    qualified = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert qualified.qualification_report["status"] == "PASS_OBSERVATION_DOMAIN_CDT_QUALIFICATION"
    assert qualified.qualification_report["cdt_behavioral_gate_pass"] is True


def test_cdt_is_deterministic_for_identical_inputs():
    surface = _chain_square_surface()
    domain = ObservationRasterDomain.from_rows(_rect_mask(10, 10, [(1, 1, 8, 8)]), view_index=0)
    a = build_mwb2_observation_cdt_candidate(
        surface, view_index=0, camera_binding_hash="camera", observation_domain=domain
    )
    b = build_mwb2_observation_cdt_candidate(
        surface, view_index=0, camera_binding_hash="camera", observation_domain=domain
    )
    assert a.to_dict() == b.to_dict()


def test_alpha_domain_fence_rejects_concavity_crossing_faces():
    surface = _chain_square_surface()
    rows = _rect_mask(10, 10, [(1, 1, 8, 8)])
    for y in range(1, 5):
        rows[y][4] = 0
        rows[y][5] = 0
    domain = ObservationRasterDomain.from_rows(rows, view_index=0)
    candidate = build_mwb2_observation_cdt_candidate(
        surface, view_index=0, camera_binding_hash="camera", observation_domain=domain
    )
    assert candidate.residual_report["alpha_rejected_face_count"] > 0
    assert candidate.residual_report["precision_inside_alpha"] == 1.0
    assert candidate.residual_report["source_alpha_recall"] < 0.90
    with pytest.raises(QualificationError, match="ALPHA_RECALL_GATE_FAIL"):
        qualify_mwb2_observation_cdt_mesh(surface, candidate)


def test_disconnected_surface_components_never_receive_cross_component_faces():
    coords = [
        (1, 1), (4, 1), (4, 4), (1, 4),
        (7, 1), (10, 1), (10, 4), (7, 4),
    ]
    ids = ["A0", "A1", "A2", "A3", "B0", "B1", "B2", "B3"]
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]
    surface = _surface(coords, edges, prefixes=ids)
    domain = ObservationRasterDomain.from_rows(
        _rect_mask(12, 6, [(1, 1, 4, 4), (7, 1, 10, 4)]),
        view_index=0,
    )
    candidate = build_mwb2_observation_cdt_candidate(
        surface, view_index=0, camera_binding_hash="camera", observation_domain=domain
    )
    source_id = {v.candidate_vertex_id: v.metadata["source_surface_id"] for v in candidate.vertices}
    for face in candidate.faces:
        prefixes = {source_id[v][0] for v in face}
        assert len(prefixes) == 1
    assert candidate.residual_report["safe_component_count"] == 2
    assert candidate.residual_report["full_safe_component_count"] == 2
    assert candidate.residual_report["precision_inside_alpha"] == 1.0


def test_invisible_safe_carrier_does_not_split_full_surface_component():
    surface = _chain_square_surface()
    nodes = list(surface.surface_nodes)
    nodes[4] = replace(nodes[4], support_views=(), raster_bindings=())
    surface = replace(surface, surface_nodes=tuple(nodes))
    domain = ObservationRasterDomain.from_rows(
        _rect_mask(10, 10, [(1, 1, 8, 8)]),
        view_index=0,
    )

    candidate = build_mwb2_observation_cdt_candidate(
        surface,
        view_index=0,
        camera_binding_hash="camera",
        observation_domain=domain,
    )

    assert candidate.residual_report["visible_surface_node_count"] == 6
    assert candidate.residual_report["full_safe_component_count"] == 1
    assert candidate.residual_report["safe_component_count"] == 1
    assert candidate.residual_report["cdt_component_count"] == 1
    assert (
        candidate.metadata["component_partition_authority"]
        == "FULL_SAFE_SURFACE_RELATION_COMPONENTS_THEN_VISIBLE_SUBSET"
    )


def test_unbound_kernel_generated_vertex_is_rejected():
    surface = _chain_square_surface()
    domain = ObservationRasterDomain.from_rows(_rect_mask(10, 10, [(1, 1, 8, 8)]), view_index=0)
    fake = SimpleNamespace(
        success=True,
        reason="ok",
        vertices=[(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (7.123456, 7.654321)],
        triangles=[(0, 1, 2)],
        constraint_split_count=1,
    )
    with patch("compiler.realsas_compiler_core.mesh.mwb2_cdt.triangulate_production_cdt", return_value=fake):
        with pytest.raises(QualificationError, match="UNSUPPORTED_GENERATED_VERTEX"):
            build_mwb2_observation_cdt_candidate(
                surface,
                view_index=0,
                camera_binding_hash="camera",
                observation_domain=domain,
            )



def test_boundary_recovery_midpoint_is_solver_internal_only():
    surface = _chain_square_surface()
    domain = ObservationRasterDomain.from_rows(
        _rect_mask(10, 10, [(1, 1, 8, 8)]), view_index=0
    )
    fake = SimpleNamespace(
        success=True,
        reason="ok",
        vertices=[(1.0, 1.0), (8.0, 1.0), (8.0, 8.0), (1.0, 8.0), (4.5, 1.0)],
        triangles=[(0, 4, 3), (4, 2, 3), (4, 1, 2)],
        constraint_edges=[(0, 4), (4, 1), (1, 2), (2, 3), (3, 0)],
        constraint_split_count=1,
        quality_insert_count=0,
        inserted_steiner_count=0,
    )
    with patch(
        "compiler.realsas_compiler_core.mesh.mwb2_cdt.triangulate_production_cdt",
        return_value=fake,
    ):
        candidate = build_mwb2_observation_cdt_candidate(
            surface,
            view_index=0,
            camera_binding_hash="camera",
            observation_domain=domain,
        )
    assert candidate.residual_report["contracted_boundary_recovery_vertex_count"] == 1
    assert candidate.residual_report["post_contraction_triangle_count"] == 2
    assert candidate.residual_report["source_alpha_recall"] == 1.0
    assert candidate.residual_report["precision_inside_alpha"] == 1.0
    assert all(v.support_binding.mode == "IDENTITY_SURFACE_NODE" for v in candidate.vertices)
    assert all(v.metadata["generated_geometry"] is False for v in candidate.vertices)
    qualified = qualify_mwb2_observation_cdt_mesh(surface, candidate)
    assert qualified.qualification_report["cdt_behavioral_gate_pass"] is True
