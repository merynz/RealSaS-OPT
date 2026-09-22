from types import SimpleNamespace

import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import (
    rasterize_triangles_half_integer_top_left,
)
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    _evaluate_candidate_source_fidelity_v1,
)


def _camera(view_index: int, resolution: int = 64):
    return qualify_camera_v3(
        {
            "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
            "view_id": f"V{view_index}",
            "view_index": view_index,
            "origin": [0.0, 0.0, -2.0],
            "right": [1.0, 0.0, 0.0],
            "screen_up": [0.0, 1.0, 0.0],
            "forward": [0.0, 0.0, 1.0],
            "half_extent": 2.0,
            "resolution": resolution,
        },
        view_id=f"V{view_index}",
        view_index=view_index,
    )


def _candidate(dx: float = 0.0):
    return SimpleNamespace(
        vertices=(
            SimpleNamespace(candidate_vertex_id="a", P=(-0.6 + dx, -0.6, 0.0)),
            SimpleNamespace(candidate_vertex_id="b", P=(0.6 + dx, -0.6, 0.0)),
            SimpleNamespace(candidate_vertex_id="c", P=(-0.6 + dx, 0.6, 0.0)),
        ),
        faces=(("a", "b", "c"),),
    )


def _fixture():
    cameras = tuple(_camera(i) for i in range(8))
    camera_set = SimpleNamespace(cameras=cameras, camera_set_hash="camera-set")
    observation = SimpleNamespace(
        views=tuple(
            SimpleNamespace(view_index=i, width=64, height=64) for i in range(8)
        ),
        observation_set_hash="observation-set",
    )
    geometry = SimpleNamespace(
        camera_set_binding_hash="camera-set",
        observation_set_binding_hash="observation-set",
        policy={
            "min_recall": 0.999,
            "min_precision": 0.999,
            "max_largest_coherent_hole_fraction": 0.00025,
            "max_interior_uncovered_fraction": 0.0005,
            "min_component_recall": 0.999,
            "component_min_foreground_fraction": 0.001,
            "max_silhouette_edge_p95_px": 0.5,
        },
    )
    base = _candidate()
    points = np.asarray([vertex.P for vertex in base.vertices], dtype=np.float64)
    source = {}
    for camera in cameras:
        projected = project_points_xyz_v3(points, camera)
        triangle = tuple((float(row[0]), float(row[1])) for row in projected)
        source[int(camera.view_index)] = rasterize_triangles_half_integer_top_left(
            (triangle,),
            width=64,
            height=64,
        )
    return base, geometry, camera_set, observation, source


def test_actual_candidate_exact_reprojection_passes_stage13_policy():
    candidate, geometry, cameras, observation, source = _fixture()
    passed, rows = _evaluate_candidate_source_fidelity_v1(
        candidate=candidate,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source,
    )
    assert passed is True
    assert len(rows) == 8
    assert all(row["passed"] for row in rows)
    assert all(abs(row["silhouette_recall"] - 1.0) < 1e-12 for row in rows)
    assert all(abs(row["silhouette_precision"] - 1.0) < 1e-12 for row in rows)
    assert all(abs(row["component_recall"] - 1.0) < 1e-12 for row in rows)
    assert all(abs(row["silhouette_edge_p95_px"]) < 1e-12 for row in rows)


def test_actual_candidate_geometry_drift_fails_instead_of_inheriting_stage13():
    _candidate0, geometry, cameras, observation, source = _fixture()
    shifted = _candidate(dx=0.8)
    passed, rows = _evaluate_candidate_source_fidelity_v1(
        candidate=shifted,
        geometry=geometry,
        cameras=cameras,
        observation=observation,
        source_foreground=source,
    )
    assert passed is False
    assert any(not row["passed"] for row in rows)
    assert any(
        row["silhouette_recall"] < geometry.policy["min_recall"]
        or row["silhouette_precision"] < geometry.policy["min_precision"]
        or row["silhouette_edge_p95_px"]
        > geometry.policy["max_silhouette_edge_p95_px"]
        for row in rows
    )
