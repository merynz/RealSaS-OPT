import numpy as np

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
