import numpy as np

from compiler.realsas_compiler_core.mesh.product_coverage_v1 import rasterize_triangles_half_integer_top_left

from compiler.realsas_compiler_core.visual_mesh_arap_v1 import (
    Arap2D,
    bind_points_barycentric,
    build_visual_mesh_from_mask,
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
    triangles = tuple(
        tuple(
            (
                float(mesh.positions[int(vertex_index), 0]) + 0.5,
                float(mesh.positions[int(vertex_index), 1]) + 0.5,
            )
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
