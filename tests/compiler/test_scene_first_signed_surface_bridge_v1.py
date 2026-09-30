import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    ZERO_SURFACE_NORMAL_OPERATOR_ID,
    rigging_surface_from_scene_first_zero_mesh_v1,
    robust_zero_surface_normals_v1,
    topology_aware_zero_surface_normals_v2,
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


def test_hidden_completion_nodes_are_typed_not_fabricated_as_observed():
    surface = _surface()
    completed = [n for n in surface.surface_nodes if not n.support_views]
    for node in completed:
        assert "MODEL_COMPLETED_SIGNED_ZERO_SURFACE" in node.validity_flags
        assert not node.raster_bindings


def _two_close_orthogonal_sheets(n=13, offset=0.035):
    axis = np.linspace(-0.6, 0.6, n, dtype=np.float64)
    pts = []
    hints = []
    faces = []

    # Sheet A: XY plane, +Z.
    base_a = 0
    for y in axis:
        for x in axis:
            pts.append((x, y, 0.0))
            hints.append((0.0, 0.0, 1.0))
    for j in range(n - 1):
        for i in range(n - 1):
            a = base_a + j * n + i
            b = a + 1
            d = a + n
            cc = d + 1
            faces.extend(((a, b, cc), (a, cc, d)))

    # Sheet B: XZ plane, shifted slightly in Y, +Y. It is geometrically close
    # to A around the crossing strip but has no shared vertices or faces.
    base_b = len(pts)
    for z in axis:
        for x in axis:
            pts.append((x, float(offset), z))
            hints.append((0.0, 1.0, 0.0))
    for j in range(n - 1):
        for i in range(n - 1):
            a = base_b + j * n + i
            b = a + 1
            d = a + n
            cc = d + 1
            # Winding chosen so raw cross is -Y; V2 must orient it from hint.
            faces.extend(((a, cc, b), (a, d, cc)))

    return (
        np.asarray(pts, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        np.asarray(hints, dtype=np.float64),
    )


def _normal_angle_deg(normals, hints):
    n = np.asarray(normals, dtype=np.float64)
    h = np.asarray(hints, dtype=np.float64)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(n * h, axis=1), -1.0, 1.0)))


def test_topology_aware_v2_cannot_mix_spatially_close_disconnected_sheets():
    points, faces, hints = _two_close_orthogonal_sheets()
    v1 = robust_zero_surface_normals_v1(points, hints, k=32)
    v2 = topology_aware_zero_surface_normals_v2(points, faces, hints)

    e1 = _normal_angle_deg(v1, hints)
    e2 = _normal_angle_deg(v2, hints)

    # Euclidean kNN sees the other sheet near the crossing and tilts the fitted
    # plane. The topology-local operator is exactly constrained to incident faces.
    assert float(np.quantile(e1, 0.95)) > 5.0
    assert float(np.max(e1)) > 10.0
    assert float(np.quantile(e2, 0.95)) < 1e-5
    assert float(np.max(e2)) < 1e-5
