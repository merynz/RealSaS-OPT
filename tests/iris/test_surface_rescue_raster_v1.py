from __future__ import annotations

import unittest

import torch

from models.iris.vnext.surface_rescue_raster_v1 import binary_precision_recall_v1, project_iris_to_pytorch3d_ndc_v1


class SurfaceRescueRasterTests(unittest.TestCase):
    def test_realsas_camera_maps_screen_right_to_negative_pytorch3d_x(self):
        vertices = torch.tensor([[0.5, 0.25, 0.0], [-0.5, -0.25, 0.0]], dtype=torch.float32)
        ndc, world = project_iris_to_pytorch3d_ndc_v1(
            vertices,
            common_center_world=torch.zeros(3),
            common_half_extent=2.0,
            camera_origin_world=torch.tensor([0.0, 0.0, -4.0]),
            camera_right_world=torch.tensor([1.0, 0.0, 0.0]),
            camera_screen_up_world=torch.tensor([0.0, 1.0, 0.0]),
            camera_forward_world=torch.tensor([0.0, 0.0, 1.0]),
            camera_half_extent=2.0,
        )
        self.assertTrue(torch.allclose(world, vertices * 2.0))
        self.assertTrue(torch.allclose(ndc[:, 0], torch.tensor([-0.5, 0.5])))
        self.assertTrue(torch.allclose(ndc[:, 1], torch.tensor([0.25, -0.25])))
        self.assertTrue(torch.allclose(ndc[:, 2], torch.tensor([2.0, 2.0])))

    def test_binary_precision_recall_is_explicit(self):
        predicted = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)
        truth = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        result = binary_precision_recall_v1(predicted, truth)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
