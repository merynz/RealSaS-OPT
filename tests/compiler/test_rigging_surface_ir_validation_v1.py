from dataclasses import replace
import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from compiler.realsas_compiler_core.substrate.validation import (
    audit_rigging_surface_ir_v1,
    validate_rigging_surface_ir_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceRelation
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3


def _sphere_mesh():
    pytest.importorskip("skimage")
    r = 30
    a = np.linspace(-1.0, 1.0, r, dtype=np.float32)
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 0.62
    return extract_zero_surface_mesh_v3(field)


def _cameras(resolution=128):
    rows = []
    for view in range(8):
        yaw = math.radians(45.0 * view)
        forward = np.asarray([-math.sin(yaw), -math.cos(yaw), 0.0], np.float64)
        origin = -4.0 * forward
        right = np.asarray([-math.cos(yaw), math.sin(yaw), 0.0], np.float64)
        rows.append({
            "view_index": view,
            "origin": origin.tolist(),
            "right": right.tolist(),
            "screen_up": [0.0, 0.0, 1.0],
            "forward": forward.tolist(),
            "half_extent": 1.05,
            "resolution": resolution,
        })
    return rows


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
