import numpy as np

from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    rigging_surface_from_scene_first_compact_v1,
)
from models.geppetto.v2.geppetto_conditioning_v2 import GeppettoConditioningAdapterV2


def _surface():
    points = np.asarray([
        [-0.4, -0.4, 0.0], [0.4, -0.4, 0.0], [0.4, 0.4, 0.0], [-0.4, 0.4, 0.0],
        [-0.4, -0.4, 0.4], [0.4, -0.4, 0.4], [0.4, 0.4, 0.4], [-0.4, 0.4, 0.4],
    ], dtype=np.float64)
    normals = np.tile(np.asarray([[0.0, -1.0, 0.0]]), (len(points), 1))
    support = np.ones((len(points), 8), dtype=np.uint8)
    raster = np.zeros((len(points), 8, 2), dtype=np.float64)
    for i, p in enumerate(points):
        raster[i, :, 0] = (p[0] + 1.0) * 512.0 - 0.5
        raster[i, :, 1] = (1.0 - p[2]) * 512.0 - 0.5
    relations = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    return rigging_surface_from_scene_first_compact_v1(
        points, normals, support, raster, relations,
        authority_label="TEST_SCENE_FIRST",
        source_run_id="TEST_RUN",
        source_checkpoint_sha256="a" * 64,
        source_zero_surface_sha256="b" * 64,
        resolution=1024,
    )


def test_bridge_is_deterministic_and_geppetto_consumable():
    a = _surface(); b = _surface()
    assert a.geometry_lineage_hash == b.geometry_lineage_hash
    assert a.metadata["Nd_operator_sha256"]
    assert a.metadata["teacher_truth_used"] is False
    assert a.metadata["normal_field_gradient_used"] is False
    assert len(a.local_relations) == 12
    batch = GeppettoConditioningAdapterV2()((a,))
    assert batch.features.shape == (1, 8, 24)
    assert np.isfinite(batch.features).all()
