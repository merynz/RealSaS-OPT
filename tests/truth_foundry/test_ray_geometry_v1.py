import numpy as np

from compiler.realsas_compiler_core.camera_geometry_v2 import CameraProjectionV3
from models.iris.truth.ray_geometry_v1 import (
    measure_pixel_ray_geometry_v1,
    regular_pixel_grid_v1,
)


def _camera(resolution=100):
    return CameraProjectionV3(
        view_id="V0",
        view_index=0,
        origin=(0.0, 0.0, -10.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, -1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=resolution,
    )


def test_complete_stack_contains_all_interior_hits_in_depth_order():
    vertices = np.asarray(
        [
            [-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.0, 0.8, 1.0],
            [-0.8, -0.8, 2.0], [0.8, -0.8, 2.0], [0.0, 0.8, 2.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    (ray,) = measure_pixel_ray_geometry_v1(vertices, faces, _camera(), [(50, 50)])
    assert ray.complete is True
    assert ray.measurement_valid is True
    assert [hit.face_index for hit in ray.hits] == [0, 1]
    assert ray.hits[0].depth < ray.hits[1].depth
    assert ray.tie_groups == ()


def test_depth_crossing_is_preserved_locally_instead_of_globalized():
    # Same projected triangles. Face 1 crosses face 0 in depth from left to right.
    vertices = np.asarray(
        [
            [-0.9, -0.8, 2.0], [0.9, -0.8, 2.0], [0.0, 0.9, 2.0],
            [-0.9, -0.8, 1.0], [0.9, -0.8, 3.0], [0.0, 0.9, 2.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    left, right = measure_pixel_ray_geometry_v1(vertices, faces, _camera(), [(30, 60), (70, 60)])
    assert [hit.face_index for hit in left.hits] == [1, 0]
    assert [hit.face_index for hit in right.hits] == [0, 1]


def test_depth_tie_is_explicit_and_never_broken_by_face_id():
    vertices = np.asarray(
        [
            [-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.0, 0.8, 1.0],
            [-0.8, -0.8, 1.0], [0.8, -0.8, 1.0], [0.0, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    (ray,) = measure_pixel_ray_geometry_v1(
        vertices,
        faces,
        _camera(),
        [(50, 50)],
        depth_tie_epsilon_source=1.0e-9,
    )
    assert ray.reason == "COMPLETE_WITH_DEPTH_TIE"
    assert ray.tie_groups == ((0, 1),)


def test_projected_triangle_boundary_fails_closed():
    vertices = np.asarray(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    # Pixel center (50.5,49.5) lies on the projected diagonal x+y=100.
    (ray,) = measure_pixel_ray_geometry_v1(vertices, faces, _camera(), [(50, 49)])
    assert ray.complete is False
    assert ray.measurement_valid is False
    assert ray.reason == "PROJECTED_TRIANGLE_BOUNDARY_AMBIGUOUS"


def test_regular_grid_is_deterministic_unique_and_in_bounds():
    first = regular_pixel_grid_v1(1024, grid_size=32)
    second = regular_pixel_grid_v1(1024, grid_size=32)
    assert first == second
    assert len(first) == 32 * 32
    assert len(set(first)) == len(first)
    assert min(min(pixel) for pixel in first) >= 0
    assert max(max(pixel) for pixel in first) < 1024
