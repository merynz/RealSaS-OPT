import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v1 import GraphicsRelationHeadV1
from models.iris.vnext.heads_v2 import (
    GraphicsRelationHeadV2,
    migrate_graphics_relation_head_v1_to_v2,
)


class GraphicsRelationHeadV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.cfg = IRISVNextConfigV1()
        self.v1 = GraphicsRelationHeadV1(self.cfg)
        self.v2, self.receipt = migrate_graphics_relation_head_v1_to_v2(self.v1)
        self.a = torch.randn(2, 5, self.cfg.local_feature_dim)
        self.b = torch.randn(2, 5, self.cfg.local_feature_dim)
        self.camera = torch.randn(2, 5, self.cfg.camera_feature_dim)

    def test_migration_preserves_existing_static_outputs(self):
        old = self.v1.static_relations(self.a, self.b)
        new = self.v2.static_relations(self.a, self.b)
        for name in (
            "same_drawable_affinity",
            "cut_probability",
            "attachment_affinity",
            "confidence",
        ):
            self.assertTrue(torch.allclose(old[name], new[name], atol=1e-7, rtol=1e-6), name)
        self.assertNotIn("alpha_boundary_probability", new)
        self.assertNotIn("appearance_boundary_probability", new)

    def test_migration_preserves_existing_view_outputs(self):
        old = self.v1.view_relations(self.a, self.b, self.camera)
        new = self.v2.view_relations(self.a, self.b, self.camera)
        for name in (
            "front_probability",
            "occlusion_probability",
            "visibility_probability_a",
            "visibility_probability_b",
            "confidence",
        ):
            self.assertTrue(torch.allclose(old[name], new[name], atol=1e-7, rtol=1e-6), name)

    def test_new_projected_boundaries_start_from_neutral_prior(self):
        new = self.v2.view_relations(self.a, self.b, self.camera)
        for name in (
            "projected_alpha_boundary_probability",
            "projected_appearance_boundary_probability",
        ):
            self.assertTrue(torch.equal(new[name], torch.full_like(new[name], 0.5)))

    def test_combined_probability_map_matches_graphics_truth_v2_names(self):
        value = self.v2.relation_probabilities(self.a, self.b, self.camera)
        self.assertEqual(
            set(value),
            {
                "same_drawable_affinity",
                "cut_probability",
                "attachment_affinity",
                "front_probability",
                "occlusion_probability",
                "visibility_probability_a",
                "visibility_probability_b",
                "projected_alpha_boundary_probability",
                "projected_appearance_boundary_probability",
            },
        )

    def test_receipt_does_not_claim_historical_mutation(self):
        self.assertFalse(self.receipt.historical_v1_mutated)
        self.assertEqual(self.receipt.new_boundary_initial_probability, 0.5)


if __name__ == "__main__":
    unittest.main()
