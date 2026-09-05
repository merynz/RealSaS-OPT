import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_ID,
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
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


def _assert_scene_first_surface_contract(surface):
    ids = [n.surface_id for n in surface.surface_nodes]
    assert ids
    assert len(ids) == len(set(ids))
    id_set = set(ids)

    observed = 0
    completed = 0
    per_view = np.zeros(8, dtype=np.int64)
    for node in surface.surface_nodes:
        p = np.asarray(node.P, np.float64)
        assert p.shape == (3,)
        assert np.isfinite(p).all()

        normal = np.asarray(node.derived_normal, np.float64)
        assert normal.shape == (3,)
        assert np.isfinite(normal).all()
        assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-5

        views = tuple(map(int, node.support_views))
        assert len(views) == len(set(views))
        assert all(0 <= v < 8 for v in views)
        for v in views:
            per_view[v] += 1

        binding_views = []
        for view, xy in node.raster_bindings:
            assert 0 <= int(view) < 8
            arr = np.asarray(xy, np.float64)
            assert arr.shape == (2,)
            assert np.isfinite(arr).all()
            binding_views.append(int(view))
        assert tuple(binding_views) == views

        if views:
            observed += 1
            assert "OBSERVED_SIGNED_ZERO_SURFACE" in node.validity_flags
            assert len(node.raster_bindings) == len(views)
        else:
            completed += 1
            assert "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in node.validity_flags
            assert not node.raster_bindings

        assert node.metadata["teacher_truth_used"] is False
        assert node.metadata["normal_operator"] == ZERO_SURFACE_NORMAL_OPERATOR_ID

    relation_ids = [r.relation_id for r in surface.local_relations]
    assert len(relation_ids) == len(set(relation_ids))
    degree = {sid: 0 for sid in ids}
    for relation in surface.local_relations:
        assert relation.a_surface_id in id_set
        assert relation.b_surface_id in id_set
        assert relation.a_surface_id != relation.b_surface_id
        assert relation.relation_kind == "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR"
        assert relation.metadata["teacher_truth_used"] is False
        assert relation.metadata["crosses_unknown"] is False
        assert relation.metadata["unknown_bridge"] is False
        assert math.isfinite(float(relation.metadata["world_distance"]))
        assert float(relation.metadata["world_distance"]) > 0.0
        degree[relation.a_surface_id] += 1
        degree[relation.b_surface_id] += 1

    assert min(degree.values()) > 0
    assert surface.metadata["compact_surface_node_count"] == len(surface.surface_nodes)
    assert surface.metadata["observed_node_count"] == observed
    assert surface.metadata["completed_node_count"] == completed
    assert tuple(surface.metadata["visibility_support_counts_by_view"]) == tuple(map(int, per_view.tolist()))
    assert surface.metadata["raster_coordinate_system"] == "PIXEL_CENTER_XY"
    assert surface.metadata["Nd_operator"] == ZERO_SURFACE_NORMAL_OPERATOR_ID
    assert surface.metadata["teacher_truth_used"] is False


def test_scene_first_zero_mesh_bridge_is_deterministic_and_geppetto_consumable():
    a = _surface()
    b = _surface()
    assert a.geometry_lineage_hash == b.geometry_lineage_hash
    assert 64 <= len(a.surface_nodes) <= 256
    assert len(a.local_relations) > len(a.surface_nodes)
    assert a.metadata["Nd_operator"] == ZERO_SURFACE_NORMAL_OPERATOR_ID
    assert a.metadata["teacher_truth_used"] is False
    assert a.metadata["character_gen_runtime_used"] is False
    assert a.metadata["full_3d_intermediate_allowed"] is True
    assert a.metadata["observed_node_count"] > 0
    normals = np.asarray([n.derived_normal for n in a.surface_nodes], np.float64)
    assert np.isfinite(normals).all()
    assert np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)) < 1e-5
    conditioning = GeppettoConditioningAdapterV2()([a])
    assert conditioning.features.shape[0] == 1
    assert conditioning.features.shape[2] == 24
    assert conditioning.features.shape[1] == len(a.surface_nodes)
    assert np.isfinite(conditioning.features).all()


def test_scene_first_surface_contract_is_closed_and_self_consistent():
    _assert_scene_first_surface_contract(_surface())


def test_scene_first_raster_evidence_reaches_geppetto_conditioning():
    surface = _surface()
    conditioning = GeppettoConditioningAdapterV2()([surface])
    raster_features = np.asarray(conditioning.features[0, :, 20:24], np.float64)
    assert np.isfinite(raster_features).all()
    assert np.any(np.abs(raster_features) > 1e-6)


def test_hidden_completion_nodes_are_typed_not_fabricated_as_observed():
    surface = _surface()
    completed = [n for n in surface.surface_nodes if not n.support_views]
    for node in completed:
        assert "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in node.validity_flags
        assert not node.raster_bindings
