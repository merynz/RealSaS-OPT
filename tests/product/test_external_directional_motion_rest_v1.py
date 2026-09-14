from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.product_external_render import build_external_renderable_component
from compiler.realsas_compiler_core.mesh.mesh_binding import mesh_lineage_hash
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.proof.directional_motion_evaluator import (
    DirectionalMotionEvaluatorPolicyV1,
    _mesh_rest_from_surface,
)
from tests.product.test_external_render_support_v1 import _mechanical, _p1_mesh, _p1_skin, _appearance


def _component(mesh=None):
    mechanical = _mechanical()
    mesh = _p1_mesh(0) if mesh is None else mesh
    return mechanical, build_external_renderable_component(
        component_id="BODY",
        view_index=0,
        mesh=mesh,
        mesh_skin=_p1_skin(mechanical, mesh),
        mechanical=mechanical,
        appearance=_appearance(mesh),
        setup_order=0,
        coverage_classification="P1_FULL_SUBJECT",
        materialization_manifest_sha256="sha256:materialized-p1",
        direct_binding_manifest_sha256="sha256:direct-v5",
    )


def test_external_directional_rest_uses_hash_bound_raster_witness_not_scientific_surface_ids():
    mechanical, component = _component()
    rest, faces, residual = _mesh_rest_from_surface(
        component,
        mechanical.surface,
        0,
        None,
        DirectionalMotionEvaluatorPolicyV1(),
    )
    assert np.allclose(rest, np.asarray([(10.0, 10.0), (20.0, 10.0), (15.0, 20.0)]))
    assert faces == ((0, 1, 2),)
    assert residual == 0.0
    assert component.mesh.surface_binding_hash != mechanical.surface.geometry_lineage_hash


def test_external_directional_rest_requires_raster_witness():
    mechanical = _mechanical()
    mesh = _p1_mesh(0)
    vertices = list(mesh.vertices)
    vertices[0] = replace(vertices[0], metadata={})
    mesh = replace(mesh, vertices=tuple(vertices), mesh_lineage_hash="")
    mesh = replace(mesh, mesh_lineage_hash=mesh_lineage_hash(mesh))

    component = type("C", (), {})()
    component.mesh = mesh
    component.appearance = type("A", (), {"camera_binding_hash": mesh.camera_binding_hash})()
    component.metadata = {
        "external_render_support_qualification": {
            "mesh_lineage_hash": mesh.mesh_lineage_hash,
            "render_support_surface_hash": mesh.surface_binding_hash,
            "view_index": 0,
        }
    }
    with pytest.raises(QualificationError, match="EXTERNAL_RASTER_WITNESS_REQUIRED"):
        _mesh_rest_from_surface(
            component,
            mechanical.surface,
            0,
            None,
            DirectionalMotionEvaluatorPolicyV1(),
        )
