import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_ID,
    ZERO_SURFACE_OBSERVATION_SUPPORT_ID,
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


def _all_foreground_masks(resolution=128):
    return tuple(np.ones((resolution, resolution), dtype=bool) for _ in range(8))


def _surface(*, observation_grounded=False, masks=None):
    mesh = _sphere_mesh()
    kwargs = {}
    if observation_grounded:
        kwargs.update(
            observation_alpha_masks=_all_foreground_masks() if masks is None else masks,
            observation_ids=tuple(f"OBS_V{i}" for i in range(8)),
            observation_hashes=tuple(f"{i:064x}" for i in range(8)),
            require_observation_support=True,
        )
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
        **kwargs,
    )


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
    assert a.metadata["observed_node_count"] == 0
    assert a.metadata["self_visible_node_count"] > 0
    assert a.metadata["observation_support_authority"] == "SELF_VISIBILITY_ONLY__NOT_OBSERVATION_GROUNDED"
    normals = np.asarray([n.derived_normal for n in a.surface_nodes], np.float64)
    assert np.isfinite(normals).all()
    assert np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0)) < 1e-5
    conditioning = GeppettoConditioningAdapterV2()([a])
    assert conditioning.features.shape[0] == 1
    assert conditioning.features.shape[2] == 24
    assert conditioning.features.shape[1] == len(a.surface_nodes)
    assert np.isfinite(conditioning.features).all()


def test_self_visibility_is_not_mislabeled_as_observed_without_product_masks():
    surface = _surface()
    self_visible = [n for n in surface.surface_nodes if n.support_views]
    assert self_visible
    assert all("SELF_VISIBLE_SIGNED_ZERO_SURFACE" in n.validity_flags for n in self_visible)
    assert all("OBSERVED_SIGNED_ZERO_SURFACE" not in n.validity_flags for n in self_visible)
    assert all(not n.source_observation_ids for n in self_visible)


def test_observation_grounded_support_is_typed_and_hash_bound():
    surface = _surface(observation_grounded=True)
    assert surface.metadata["observation_support_operator"] == ZERO_SURFACE_OBSERVATION_SUPPORT_ID
    assert surface.metadata["observation_support_authority"] == "EXACT_PRODUCT_ALPHA_AND_SELF_VISIBILITY"
    assert surface.metadata["observed_node_count"] > 0
    observed = [n for n in surface.surface_nodes if "OBSERVED_SIGNED_ZERO_SURFACE" in n.validity_flags]
    assert observed
    for node in observed:
        assert node.source_observation_ids
        assert len(node.source_observation_ids) == len(node.support_views)
        assert set(node.source_observation_ids).issubset({f"OBS_V{i}" for i in range(8)})
        assert node.metadata["observation_support_views"]


def test_product_alpha_can_withhold_self_visible_nodes_without_erasing_self_visibility_evidence():
    masks = list(_all_foreground_masks())
    masks[0] = np.zeros_like(masks[0])
    surface = _surface(observation_grounded=True, masks=tuple(masks))
    assert surface.metadata["visibility_support_counts_by_view"][0] > 0
    assert surface.metadata["observation_support_counts_by_view"][0] == 0
    assert all(0 not in n.support_views for n in surface.surface_nodes)
    assert any(0 in n.metadata["self_visibility_views"] for n in surface.surface_nodes)


def test_require_observation_support_fails_closed_without_masks():
    mesh = _sphere_mesh()
    with pytest.raises(Exception, match="alpha masks"):
        rigging_surface_from_scene_first_zero_mesh_v1(
            mesh.vertices_normalized,
            mesh.faces,
            mesh.normals,
            _cameras(),
            normalization_center=(0.0, 0.0, 0.0),
            normalization_half_extent=1.0,
            authority_label="TEST",
            source_run_id="TEST",
            source_checkpoint_sha256="a" * 64,
            source_zero_surface_sha256="b" * 64,
            require_observation_support=True,
        )


def test_hidden_completion_nodes_are_typed_not_fabricated_as_observed():
    surface = _surface(observation_grounded=True)
    completed = [n for n in surface.surface_nodes if not n.support_views]
    for node in completed:
        assert "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in node.validity_flags
        assert not node.raster_bindings
        assert "OBSERVED_SIGNED_ZERO_SURFACE" not in node.validity_flags
