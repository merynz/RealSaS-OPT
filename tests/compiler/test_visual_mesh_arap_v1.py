import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import rasterize_triangles_half_integer_top_left
from compiler.realsas_compiler_core.camera_geometry_v2 import CameraProjectionV3

from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_source_visual_points_to_projected_surface_v1,
    build_visual_mesh_from_region_labels_v1,
    partition_source_mask_by_face_components_v1,
    partition_source_mask_by_safe_face_adjacency_v1,
    safe_face_deformation_components_v1,
    bind_points_barycentric,
    build_visual_mesh_from_mask,
    source_texel_xy_to_raster_xy,
)


def test_visual_mesh_arap_preserves_simple_mask_and_moves_handle():
    mask = np.zeros((96, 96), dtype=bool)
    mask[16:80, 24:72] = True
    mesh = build_visual_mesh_from_mask(mask, target_edge_px=12)
    assert len(mesh.positions) > 10
    assert len(mesh.faces) > 10
    bindings = bind_points_barycentric(mesh, np.asarray([[48.0, 48.0]]))
    arap = Arap2D(mesh, bindings)
    out, qa = arap.solve(np.asarray([[54.0, 48.0]]), iterations=3)
    assert out.shape == mesh.positions.shape
    assert np.isfinite(out).all()
    assert qa.max_handle_residual_px < 2.0
    assert qa.flipped_triangles == 0


def _visual_coverage(mesh):
    raster_positions = source_texel_xy_to_raster_xy(mesh.positions)
    triangles = tuple(
        tuple(
            tuple(map(float, raster_positions[int(vertex_index)]))
            for vertex_index in face
        )
        for face in np.asarray(mesh.faces, dtype=np.int64)
    )
    raw = rasterize_triangles_half_integer_top_left(
        triangles,
        width=int(mesh.width),
        height=int(mesh.height),
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(mesh.height, mesh.width).astype(bool)


def test_visual_mesh_cdt_matches_disconnected_mask_and_hole_exactly():
    mask = np.zeros((48, 64), dtype=bool)
    mask[4:26, 5:31] = True
    mask[10:18, 12:21] = False
    mask[30:43, 39:58] = True
    mesh = build_visual_mesh_from_mask(mask, target_edge_px=8)
    predicted = _visual_coverage(mesh)
    assert np.array_equal(predicted, mask)
    expected_uv = np.column_stack(
        (
            mesh.positions[:, 0] / (mesh.width - 1),
            mesh.positions[:, 1] / (mesh.height - 1),
        )
    )
    assert np.max(np.abs(mesh.uv - expected_uv)) <= 1.0e-12


def test_visual_texel_to_raster_transform_is_exact_half_pixel_contract():
    points = np.asarray(
        [[0.0, 0.0], [3.0, 7.0], [-0.5, 12.5]],
        dtype=np.float64,
    )
    raster = source_texel_xy_to_raster_xy(points)
    assert np.array_equal(
        raster,
        points + np.asarray([0.5, 0.5], dtype=np.float64),
    )


def test_continuous_visual_mechanical_binding_handles_half_pixel_boundaries():
    camera = CameraProjectionV3(
        view_id="V0",
        view_index=0,
        origin=(0.0, 0.0, -2.0),
        right=(1.0, 0.0, 0.0),
        screen_up=(0.0, 1.0, 0.0),
        forward=(0.0, 0.0, 1.0),
        half_extent=1.0,
        resolution=16,
    )

    def world_from_raster(x, y, z):
        return (
            float(x) / 8.0 - 1.0,
            1.0 - float(y) / 8.0,
            float(z),
        )

    # Face 0 is farther away but has the same projected triangle as face 1.
    back = [
        world_from_raster(2.0, 2.0, 1.0),
        world_from_raster(10.0, 2.0, 1.0),
        world_from_raster(2.0, 10.0, 1.0),
    ]
    front = [
        world_from_raster(2.0, 2.0, 0.0),
        world_from_raster(10.0, 2.0, 0.0),
        world_from_raster(2.0, 10.0, 0.0),
    ]
    xyz = np.asarray(back + front, dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)

    # Source coordinates are raster coordinates minus the exact +0.5 contract.
    points = np.asarray(
        [
            [1.5, 1.5],  # exact silhouette vertex
            [5.5, 1.5],  # exact silhouette edge
            [3.5, 3.5],  # interior
            [12.0, 12.0],  # outside
        ],
        dtype=np.float64,
    )
    result = bind_source_visual_points_to_projected_surface_v1(
        points_source_xy=points,
        mechanical_positions_xyz=xyz,
        mechanical_faces=faces,
        camera=camera,
    )
    assert result["owner_face_index"].tolist() == [1, 1, 1, -1]
    assert result["valid"].tolist() == [True, True, True, False]
    assert np.isfinite(result["barycentric"][:3]).all()
    assert np.all(result["barycentric"][:3] >= -1.0e-9)
    assert np.allclose(result["barycentric"][:3].sum(axis=1), 1.0, atol=1.0e-12)
    assert np.allclose(result["depth"][:3], 2.0, atol=1.0e-12)


def test_safe_face_components_split_across_unsafe_bridge():
    faces = np.asarray(
        [
            [0, 1, 2],
            [1, 3, 2],
            [1, 4, 3],
        ],
        dtype=np.int64,
    )
    labels, sizes = safe_face_deformation_components_v1(
        faces,
        unsafe_face_indices=(1,),
    )
    assert labels.tolist() == [0, -1, 1]
    assert sizes.tolist() == [1, 1]


def test_deformation_region_partition_fills_unsafe_band_without_cross_region_faces():
    mask = np.zeros((32, 48), dtype=bool)
    mask[4:28, 4:44] = True
    # Face 0 owns the left, face 1 is unsafe/unavailable in the central band,
    # face 2 owns the right.
    owner = np.full(mask.shape, -1, dtype=np.int64)
    owner[4:28, 4:21] = 0
    owner[4:28, 21:27] = 1
    owner[4:28, 27:44] = 2
    face_components = np.asarray([10, -1, 20], dtype=np.int32)

    final, seeds, rows = partition_source_mask_by_face_components_v1(
        mask,
        owner,
        face_components,
        minimum_seed_pixels=16,
    )
    assert np.all(final[mask] >= 0)
    assert set(final[mask].tolist()) == {10, 20}
    assert set(seeds[mask].tolist()) == {-1, 10, 20}
    assert sum(row["pixel_count"] for row in rows) == int(mask.sum())

    region_mesh = build_visual_mesh_from_region_labels_v1(
        mask,
        final,
        target_edge_px=8,
    )
    combined = region_mesh.mesh
    predicted = _visual_coverage(combined)
    assert np.array_equal(predicted, mask)
    assert set(region_mesh.face_region_id.tolist()) == {10, 20}
    for face_index, face in enumerate(np.asarray(combined.faces, dtype=np.int64)):
        rid = int(region_mesh.face_region_id[face_index])
        assert np.all(region_mesh.vertex_region_id[face] == rid)


def test_safe_face_adjacency_charts_cut_screen_occlusion_even_inside_one_global_component():
    mask = np.zeros((10, 14), dtype=bool)
    mask[2:8, 2:12] = True
    # Two mechanical islands are globally part of one conceptual body, but these
    # raster neighbors do not share a mechanical edge. The chart contract must
    # not weld them merely because they touch in screen space.
    faces = np.asarray(
        [
            [0, 1, 2],
            [1, 3, 2],  # adjacent to face 0
            [4, 5, 6],
            [5, 7, 6],  # adjacent to face 2
        ],
        dtype=np.int64,
    )
    owner = np.full(mask.shape, -1, dtype=np.int64)
    owner[2:8, 2:5] = 0
    owner[2:8, 5:7] = 1
    owner[2:8, 7:10] = 2
    owner[2:8, 10:12] = 3

    final, seed, rows = partition_source_mask_by_safe_face_adjacency_v1(
        mask,
        owner,
        faces,
        unsafe_face_indices=(),
        minimum_seed_pixels=4,
    )
    assert np.all(final[mask] >= 0)
    assert len(set(final[mask].tolist())) == 2
    assert len(rows) == 2

    region_mesh = build_visual_mesh_from_region_labels_v1(
        mask,
        final,
        target_edge_px=4,
    )
    assert np.array_equal(_visual_coverage(region_mesh.mesh), mask)
    for face_index, face in enumerate(np.asarray(region_mesh.mesh.faces, dtype=np.int64)):
        rid = int(region_mesh.face_region_id[face_index])
        assert np.all(region_mesh.vertex_region_id[face] == rid)
