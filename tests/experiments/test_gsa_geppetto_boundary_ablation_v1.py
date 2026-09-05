import math

import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_zero_mesh_v1,
)
from experiments.mage_scene_first_e2e_v1.gsa_geppetto_boundary_ablation_v1 import (
    conditioning_boundary_report_v1,
    gsa_graph_neighbor_index_v1,
    strip_raster_bindings_v1,
)
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2
from models.iris.v3.zero_surface_decoder_v3 import extract_zero_surface_mesh_v3


def _surface():
    pytest.importorskip("skimage")
    r = 24
    a = np.linspace(-1.0, 1.0, r, dtype=np.float32)
    z, y, x = np.meshgrid(a, a, a, indexing="ij")
    field = np.sqrt(x * x + y * y + z * z) - 0.62
    mesh = extract_zero_surface_mesh_v3(field)
    cameras = []
    for view in range(8):
        yaw = math.radians(45.0 * view)
        forward = np.asarray([-math.sin(yaw), -math.cos(yaw), 0.0], np.float64)
        origin = -4.0 * forward
        right = np.asarray([-math.cos(yaw), math.sin(yaw), 0.0], np.float64)
        cameras.append({
            "view_index": view,
            "origin": origin.tolist(),
            "right": right.tolist(),
            "screen_up": [0.0, 0.0, 1.0],
            "forward": forward.tolist(),
            "half_extent": 1.05,
            "resolution": 128,
        })
    return rigging_surface_from_scene_first_zero_mesh_v1(
        mesh.vertices_normalized,
        mesh.faces,
        mesh.normals,
        cameras,
        normalization_center=(0.0, 0.0, 0.0),
        normalization_half_extent=1.0,
        authority_label="TEST_SCENE_FIRST",
        source_run_id="TEST_RUN",
        source_checkpoint_sha256="a" * 64,
        source_zero_surface_sha256="b" * 64,
        target_nodes=192,
        normal_k=24,
        visibility_depth_tolerance_norm=0.03,
    )


def test_raster_strip_changes_only_raster_feature_columns():
    surface = _surface()
    stripped = strip_raster_bindings_v1(surface)
    full = GeppettoConditioningAdapterV2()([surface])
    compact = GeppettoConditioningAdapterV2()([stripped])
    assert full.surface_ids == compact.surface_ids
    assert np.array_equal(full.positions_normalized, compact.positions_normalized)
    report = conditioning_boundary_report_v1(surface, stripped)
    assert report["max_abs_nonraster_feature_delta"] == 0.0
    assert report["max_abs_raster_feature_delta"] > 0.0


def test_gsa_graph_16_is_cardinality_matched_and_self_first():
    surface = _surface()
    idx, telemetry = gsa_graph_neighbor_index_v1(surface, k=16)
    assert idx.shape == (len(surface.surface_nodes), 16)
    assert np.array_equal(idx[:, 0], np.arange(len(surface.surface_nodes)))
    assert telemetry["k"] == 16
    assert telemetry["self_included"] is True
    assert telemetry["undirected_relation_edge_count"] == len(surface.local_relations)
