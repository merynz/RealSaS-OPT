from __future__ import annotations

import unittest
import numpy as np

from tools.iris_vnext.build_fit8_surface_raster_truth_v1 import _downsample_mask, _raster_depth_normal


class SurfaceRasterTruthTests(unittest.TestCase):
    def test_mask_downsample_is_conservative_max_pool(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1, 1] = True
        mask[6, 7] = True
        out = _downsample_mask(mask, 4)
        self.assertEqual(out.shape, (4, 4))
        self.assertTrue(out[0, 0])
        self.assertTrue(out[3, 3])
        self.assertEqual(int(out.sum()), 2)

    def test_simple_front_facing_triangle_depth_and_normal(self):
        vertices = np.asarray([[-0.5,-0.5,0.0],[0.5,-0.5,0.0],[0.0,0.5,0.0]], dtype=np.float64)
        faces = np.asarray([[0,1,2]], dtype=np.int64)
        camera = {
            "origin":[0.0,0.0,-2.0], "right":[1.0,0.0,0.0], "screen_up":[0.0,1.0,0.0],
            "forward":[0.0,0.0,1.0], "half_extent":1.0, "clip_start":0.01, "clip_end":10.0,
        }
        depth, normal, valid = _raster_depth_normal(vertices, faces, camera, resolution=32, common_half_extent=1.0)
        self.assertTrue(valid.any())
        self.assertTrue(np.allclose(depth[valid], 2.0, atol=1e-6))
        self.assertTrue(np.allclose(np.abs(normal[valid, 2]), 1.0, atol=1e-6))
        self.assertTrue(np.allclose(normal[valid, :2], 0.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
