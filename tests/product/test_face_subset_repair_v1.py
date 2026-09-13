from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.mesh.direct_model_skin import DIRECT_MODEL_TRANSFER_METHOD
from compiler.realsas_compiler_core.mesh.face_subset_repair import (
    qualify_exact_policy_face_subset_repair,
    validate_face_subset_repair_result,
)
from compiler.realsas_compiler_core.mesh_binding import mesh_lineage_hash, mesh_skin_lineage_hash
from compiler.realsas_compiler_core.product_external_render import (
    build_external_renderable_component,
    validate_external_renderable_component,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    QualifiedEditableMeshIR,
    QualifiedMeshSkinIR,
    QualifiedMeshSkinRow,
    QualifiedMeshVertex,
    SurfaceSupportBinding,
)
from compiler.realsas_compiler_core.v4 import build_appearance_binding
from compiler.realsas_compiler_core.v4_types import AppearanceCornerBinding
from tests.product.test_external_product_proof_and_continuity_v1 import _mechanical


def _source_mesh():
    raster = {
        "MV:0": (0.0, 0.0),
        "MV:1": (20.0, 0.0),
        "MV:2": (10.0, 20.0),
        "MV:3": (40.0, 0.0),
        "MV:4": (60.0, 0.0),
        "MV:5": (60.0, 0.05),
    }
    vertices = tuple(
        QualifiedMeshVertex(
            vertex_id,
            (float(index), 0.0, 0.0),
            SurfaceSupportBinding(
                "LOCAL_CONVEX_INTERPOLATION",
                ((f"P1:S{index}", 1.0),),
            ),
            metadata={"raster_xy": raster[vertex_id]},
        )
        for index, vertex_id in enumerate(raster)
    )
    faces = (
        ("MV:0", "MV:1", "MV:2"),
        ("MV:3", "MV:4", "MV:5"),
    )
    edges = (
        ("MV:0", "MV:1"), ("MV:1", "MV:2"), ("MV:2", "MV:0"),
        ("MV:3", "MV:4"), ("MV:4", "MV:5"), ("MV:5", "MV:3"),
    )
    mesh = QualifiedEditableMeshIR(
        vertices,
        faces,
        edges,
        "P1:RENDER:SUPPORT:S",
        0,
        "CAM:0",
        {"status": "PASS_EXTERNAL_SUPPORT"},
        "",
        support_coverage_classification="P1_FULL_SUBJECT",
    )
    return replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))


def _source_skin(mechanical, mesh):
    rows = tuple(
        QualifiedMeshSkinRow(
            vertex.canonical_mesh_vertex_id,
            (("J:ROOT", 1.0),),
            tuple(vertex.support_binding.coefficients),
            0.0,
            0.0,
        )
        for vertex in mesh.vertices
    )
    value = QualifiedMeshSkinIR(
        rows,
        mesh.surface_binding_hash,
        mechanical.skeleton.skeleton_lineage_hash,
        mechanical.skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        DIRECT_MODEL_TRANSFER_METHOD,
        {"status": "PASS_DIRECT_MODEL_EXACT_MESH_SKIN_QUALIFICATION"},
        "",
        metadata={"direct_model_query": True, "historical_weight_transfer_used": False},
    )
    return replace(value, mesh_skin_lineage_hash=mesh_skin_lineage_hash(value))


def _appearance(mesh):
    by_id = {vertex.canonical_mesh_vertex_id: vertex for vertex in mesh.vertices}
    corners = []
    for face_index, face in enumerate(mesh.faces):
        for corner_index, vertex_id in enumerate(face):
            xy = by_id[vertex_id].metadata["raster_xy"]
            corners.append(
                AppearanceCornerBinding(
                    face_index,
                    corner_index,
                    (xy[0] / 64.0, xy[1] / 64.0),
                    0,
                    xy,
                    "OBS:0",
                    "OBSERVED_LOCAL",
                )
            )
    return build_appearance_binding(
        target_view_index=0,
        mesh_binding_hash=mesh.mesh_lineage_hash,
        camera_binding_hash=mesh.camera_binding_hash,
        corner_bindings=tuple(corners),
        atlas_payload_hash="ATLAS:SOURCE",
        metadata={"artist_source": True},
    )


def _source_component():
    mechanical = _mechanical()
    mesh = _source_mesh()
    skin = _source_skin(mechanical, mesh)
    component = build_external_renderable_component(
        component_id="MAGE_FULL_SUBJECT",
        view_index=0,
        mesh=mesh,
        mesh_skin=skin,
        mechanical=mechanical,
        appearance=_appearance(mesh),
        setup_order=0,
        coverage_classification="P1_FULL_SUBJECT",
        materialization_manifest_sha256="sha256:p1-materialization",
        direct_binding_manifest_sha256="sha256:v5-direct-binding",
    )
    return mechanical, component


def _passing_coverage():
    return {
        "source_alpha_recall": 0.98,
        "precision_inside_alpha": 1.0,
        "alpha_iou": 0.98,
        "largest_uncovered_component_fraction": 0.005,
        "foreground_component_recalls": (
            {"foreground_fraction": 0.10, "recall": 0.98},
        ),
    }


def _repair(component, mechanical, **kwargs):
    return qualify_exact_policy_face_subset_repair(
        component,
        mechanical,
        remove_face_indices=(1,),
        repaired_coverage_report=_passing_coverage(),
        coverage_measurement_sha256="sha256:reraster-observation-v0",
        source_truth_ownership_sha256="sha256:source-truth-partition-v0",
        **kwargs,
    )


def test_sliver_face_subset_repair_preserves_vertices_edges_weights_and_artist_payload():
    mechanical, component = _source_component()
    result = _repair(component, mechanical)

    assert result.mesh.vertices == component.mesh.vertices
    assert result.mesh.edges == component.mesh.edges
    assert result.mesh.faces == (component.mesh.faces[0],)
    assert result.mesh_skin.rows == component.mesh_skin.rows
    assert result.mesh_skin.mesh_binding_hash == result.mesh.mesh_lineage_hash
    assert len(result.appearance.corner_bindings) == 3
    assert tuple(c.source_observation_hash for c in result.appearance.corner_bindings) == ("OBS:0",) * 3
    assert result.qualification.removed_face_indices == (1,)
    assert result.qualification.qualification_report["passed"] is True
    assert result.qualification.qualification_report["vertices_preserved_exactly"] is True
    assert result.qualification.qualification_report["edges_preserved_exactly"] is True
    assert result.qualification.qualification_report["weight_rows_preserved_exactly"] is True
    assert result.qualification.qualification_report["retriangulated"] is False
    assert result.qualification.qualification_report["new_pixels_generated"] is False


def test_good_face_cannot_be_removed_as_policy_repair():
    mechanical, component = _source_component()
    with pytest.raises(QualificationError, match="FACE_NOT_POLICY_VIOLATION:0"):
        qualify_exact_policy_face_subset_repair(
            component,
            mechanical,
            remove_face_indices=(0,),
            repaired_coverage_report=_passing_coverage(),
            coverage_measurement_sha256="sha256:reraster-observation-v0",
            source_truth_ownership_sha256="sha256:source-truth-partition-v0",
        )


def test_repaired_mesh_must_repass_full_frozen_coverage_policy():
    mechanical, component = _source_component()
    failed_coverage = {
        **_passing_coverage(),
        "source_alpha_recall": 0.90,
    }
    with pytest.raises(QualificationError, match="FROZEN_POLICY_FAILED"):
        qualify_exact_policy_face_subset_repair(
            component,
            mechanical,
            remove_face_indices=(1,),
            repaired_coverage_report=failed_coverage,
            coverage_measurement_sha256="sha256:reraster-observation-v0",
            source_truth_ownership_sha256="sha256:source-truth-partition-v0",
        )


def test_repair_rejects_placeholder_authority_and_mutation_claims():
    mechanical, component = _source_component()
    with pytest.raises(QualificationError, match="INVALID_COVERAGE_MEASUREMENT_SHA256"):
        qualify_exact_policy_face_subset_repair(
            component,
            mechanical,
            remove_face_indices=(1,),
            repaired_coverage_report=_passing_coverage(),
            coverage_measurement_sha256="pending",
            source_truth_ownership_sha256="sha256:source-truth-partition-v0",
        )

    with pytest.raises(QualificationError, match="FORBIDDEN_MUTATION:weights_mutated"):
        _repair(component, mechanical, metadata={"weights_mutated": True})


def test_repair_can_be_requalified_as_new_external_component_without_scientific_relabel():
    mechanical, component = _source_component()
    result = _repair(component, mechanical)
    validate_face_subset_repair_result(result, source_component=component)

    repaired_component = build_external_renderable_component(
        component_id=component.component_id,
        view_index=component.view_index,
        mesh=result.mesh,
        mesh_skin=result.mesh_skin,
        mechanical=mechanical,
        appearance=result.appearance,
        setup_order=component.setup_order,
        coverage_classification=component.coverage_classification,
        materialization_manifest_sha256="sha256:p1q-materialization-new",
        direct_binding_manifest_sha256="sha256:v5-direct-binding-p1q-new",
        metadata={
            "face_subset_repair_hash": result.qualification.repair_hash,
            "scientific_mechanical_surface_relabelled": False,
        },
    )
    validate_external_renderable_component(repaired_component, mechanical)
    assert repaired_component.mesh.surface_binding_hash == "P1:RENDER:SUPPORT:S"
    assert mechanical.surface.geometry_lineage_hash == "FIT1:SCIENTIFIC:S"
    assert repaired_component.metadata["scientific_mechanical_surface_relabelled"] is False
    assert repaired_component.metadata["face_subset_repair_hash"] == result.qualification.repair_hash


def test_tampered_repair_hash_fails_closed():
    mechanical, component = _source_component()
    result = _repair(component, mechanical)
    tampered = replace(result.qualification, repair_hash="sha256:tampered")
    with pytest.raises(QualificationError, match="FACE_SUBSET_REPAIR_HASH_MISMATCH"):
        validate_face_subset_repair_result(
            replace(result, qualification=tampered),
            source_component=component,
        )
