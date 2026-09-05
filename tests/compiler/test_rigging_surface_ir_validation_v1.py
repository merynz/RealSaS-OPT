from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from compiler.realsas_compiler_core.substrate.validation import (
    audit_rigging_surface_ir_v1,
    validate_rigging_surface_ir_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceRelation
from tests.compiler.test_scene_first_signed_surface_bridge_v1 import _cameras, _sphere_mesh


def _surface():
    mesh = _sphere_mesh()
    return rigging_surface_from_scene_first_zero_mesh_v1(
        mesh.vertices_normalized,
        mesh.faces,
        mesh.normals,
        _cameras(),
        normalization_center=(0.0, 0.0, 0.0),
        normalization_half_extent=1.0,
        authority_label="TEST_IRIS_SCENE_FIRST_SIGNED_V3",
        source_run_id="TEST_RUN",
        source_checkpoint_sha256="a" * 64,
        source_zero_surface_sha256="b" * 64,
        target_nodes=256,
        normal_k=32,
        visibility_depth_tolerance_norm=0.03,
    )


def test_scene_first_surface_passes_central_boundary_validator():
    surface = _surface()
    report = validate_rigging_surface_ir_v1(
        surface, require_scene_first_signed_contract=True
    )
    assert report["passed"] is True
    assert report["error_count"] == 0
    assert report["node_count"] == len(surface.surface_nodes)
    assert report["relation_count"] == len(surface.local_relations)
    assert report["raster_binding_count"] > 0
    assert report["normal_count"] == len(surface.surface_nodes)


def test_validator_rejects_support_raster_mismatch():
    surface = _surface()
    nodes = list(surface.surface_nodes)
    node = next(n for n in nodes if n.support_views)
    idx = nodes.index(node)
    nodes[idx] = replace(node, raster_bindings=())
    bad = replace(surface, surface_nodes=tuple(nodes))
    report = audit_rigging_surface_ir_v1(bad)
    assert report["passed"] is False
    assert any("RASTER_SUPPORT_VIEW_MISMATCH" in x for x in report["errors"])
    with pytest.raises(QualificationError, match="RIGGING_SURFACE_IR_AUDIT_FAIL"):
        validate_rigging_surface_ir_v1(bad)


def test_validator_rejects_unknown_relation_endpoint_and_duplicate_relation_id():
    surface = _surface()
    first = surface.local_relations[0]
    bad_rel = SurfaceRelation(
        relation_id=first.relation_id,
        a_surface_id=first.a_surface_id,
        b_surface_id="DOES_NOT_EXIST",
        relation_kind=first.relation_kind,
        score=first.score,
        metadata=first.metadata,
    )
    bad = replace(surface, local_relations=surface.local_relations + (bad_rel,))
    report = audit_rigging_surface_ir_v1(bad)
    assert report["passed"] is False
    assert "DUPLICATE_RELATION_ID" in report["errors"]
    assert any("UNKNOWN_ENDPOINT" in x for x in report["errors"])


def test_scene_first_profile_rejects_metadata_drift():
    surface = _surface()
    metadata = dict(surface.metadata)
    metadata["compact_surface_node_count"] = len(surface.surface_nodes) + 1
    bad = replace(surface, metadata=metadata)
    report = audit_rigging_surface_ir_v1(
        bad, require_scene_first_signed_contract=True
    )
    assert report["passed"] is False
    assert "SCENE_FIRST:NODE_COUNT_METADATA_DRIFT" in report["errors"]
