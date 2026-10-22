import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import (
    GraphicsRelationHeadV3,
    GraphicsRelationHeadV3Config,
    RayStackVisibilityHeadV3,
    relation_camera_geometry_v3,
)


class GraphicsRelationHeadV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.iris = IRISVNextConfigV1()
        self.relation = GraphicsRelationHeadV3Config(
            feature_width=96,
            geometry_width=48,
            camera_width=48,
            pair_width=128,
            residual_blocks=2,
        )
        self.head = GraphicsRelationHeadV3(self.iris, self.relation)

    @staticmethod
    def camera(batch_shape):
        value = torch.zeros(*batch_shape, 13)
        value[..., 0:3] = torch.tensor([0.0, 0.0, -2.0])
        value[..., 3:6] = torch.tensor([1.0, 0.0, 0.0])
        value[..., 6:9] = torch.tensor([0.0, 1.0, 0.0])
        value[..., 9:12] = torch.tensor([0.0, 0.0, 1.0])
        value[..., 12] = 1.0
        return value

    def test_explicit_camera_geometry_recovers_depth_and_same_ray(self):
        point_a = torch.tensor([[[0.25, 0.50, 0.00]]])
        point_b = torch.tensor([[[0.25, 0.50, 0.50]]])
        camera = self.camera((1, 1))
        geometry = relation_camera_geometry_v3(point_a, point_b, camera)
        row = dict(zip(geometry.feature_names, geometry.values[0, 0].tolist()))
        self.assertAlmostEqual(row["depth_a"], 2.0, places=6)
        self.assertAlmostEqual(row["depth_b"], 2.5, places=6)
        self.assertAlmostEqual(row["delta_depth"], -0.5, places=6)
        self.assertAlmostEqual(row["u_a"], 0.25, places=6)
        self.assertAlmostEqual(row["v_a"], 0.50, places=6)
        self.assertAlmostEqual(row["u_b"], 0.25, places=6)
        self.assertAlmostEqual(row["v_b"], 0.50, places=6)
        self.assertLess(row["same_ray_residual"], 1.0e-5)

    def test_pair_api_has_no_visibility_or_neural_confidence(self):
        shape = (2, 3)
        feature_a = torch.randn(*shape, self.iris.local_feature_dim)
        feature_b = torch.randn(*shape, self.iris.local_feature_dim)
        point_a = torch.randn(*shape, 3) * 0.25
        point_b = torch.randn(*shape, 3) * 0.25
        camera = self.camera(shape)
        value = self.head.relation_probabilities(
            feature_a,
            feature_b,
            point_a,
            point_b,
            camera,
        )
        self.assertEqual(
            set(value),
            {
                "same_drawable_affinity",
                "cut_probability",
                "attachment_affinity",
                "front_probability",
                "occlusion_probability",
                "projected_alpha_boundary_probability",
                "projected_appearance_boundary_probability",
            },
        )
        self.assertNotIn("confidence", value)
        self.assertNotIn("visibility_probability_a", value)
        self.assertNotIn("visibility_probability_b", value)

    def test_ray_visibility_consumes_complete_stack(self):
        head = RayStackVisibilityHeadV3(
            self.iris,
            width=64,
            heads=8,
            layers=2,
            mlp_ratio=2,
        )
        hit_h = torch.randn(2, 4, self.iris.local_feature_dim)
        hit_xyz = torch.randn(2, 4, 3) * 0.25
        hit_depth = torch.tensor([[1.0, 2.0, 3.0, 0.0], [0.5, 1.5, 2.5, 3.5]])
        valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
        camera = self.camera((2,))
        p = head(
            hit_h=hit_h,
            hit_xyz=hit_xyz,
            hit_depth=hit_depth,
            camera_features=camera,
            valid_mask=valid,
        )
        self.assertEqual(p.shape, (2, 4))
        self.assertTrue(torch.all((p[valid] >= 0.0) & (p[valid] <= 1.0)))
        self.assertEqual(float(p[0, 3]), 0.0)


if __name__ == "__main__":
    unittest.main()
