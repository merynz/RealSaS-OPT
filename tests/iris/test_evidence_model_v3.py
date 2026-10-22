import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.training_v3 import (
    graphics_relation_loss_v3,
    mechanical_demand_loss_v3,
)


class IRISVNextEvidenceModelV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.cfg = IRISVNextConfigV1(
            scene_dim=64,
            camera_embed_dim=32,
            plane_size=4,
            plane_channels=16,
            field_hidden_dim=64,
            field_residual_blocks=2,
            local_feature_dim=64,
            attention_heads=8,
            scene_layers=2,
            topology_width=64,
            topology_heads=8,
            topology_decoder_layers=1,
            topology_memory_tokens=8,
            topology_refinement_layers=1,
            topology_source_width=32,
            extraction_training_resolution=8,
            surface_witness_count=16,
        )
        relation = GraphicsRelationHeadV3Config(
            feature_width=64,
            geometry_width=32,
            camera_width=32,
            pair_width=96,
            residual_blocks=2,
        )
        self.model = IRISVNextEvidenceModelV3(self.cfg, relation)

    def test_topology_is_factored_out_and_uncalibrated_uncertainty_not_public(self):
        self.assertFalse(self.model.topology_decoder_embedded)
        self.assertFalse(hasattr(self.model, "structure"))
        planes = torch.randn(2, 3, self.cfg.plane_channels, 8, 8)
        points = torch.randn(2, 6, 3).clamp(-0.9, 0.9)
        local = self.model.query_local(planes, points)
        self.assertEqual(set(local), {"H", "field", "extraction_delta"})
        self.assertNotIn("log_uncertainty", local)
        evidence = self.model.query_evidence(planes, points)
        self.assertNotIn("uncertainty", evidence["mechanical_demand"])
        self.assertIn("contact_persistence_probability", evidence["mechanical_demand"])

    def test_parameter_groups_are_specialist_separated(self):
        groups = self.model.parameter_groups()
        self.assertEqual(set(groups), {"scene_core", "mechanical", "appearance", "relation"})
        ids = []
        for parameters in groups.values():
            ids.extend(id(p) for p in parameters)
        self.assertEqual(len(ids), len(set(ids)))

    def test_mechanical_loss_uses_contact_only_where_valid(self):
        h = torch.randn(2, 5, self.cfg.local_feature_dim)
        predicted = self.model.mechanical_demand(h)
        truth = {
            "stretch_major": predicted["stretch_major"].detach().clone(),
            "stretch_minor": predicted["stretch_minor"].detach().clone(),
            "tangent_direction_2theta": predicted["tangent_direction_2theta"].detach().clone(),
            "bend_demand": predicted["bend_demand"].detach().clone(),
            "rigidity_probability": predicted["rigidity_probability"].detach().clone(),
            "contact_persistence_probability": torch.ones(2, 5),
        }
        none_valid = mechanical_demand_loss_v3(
            predicted,
            truth,
            contact_valid_mask=torch.zeros(2, 5, dtype=torch.bool),
        )
        some_valid = mechanical_demand_loss_v3(
            predicted,
            truth,
            contact_valid_mask=torch.ones(2, 5, dtype=torch.bool),
        )
        self.assertEqual(float(none_valid["contact_supervision_present"]), 0.0)
        self.assertEqual(float(some_valid["contact_supervision_present"]), 1.0)
        self.assertGreaterEqual(float(some_valid["contact_persistence"]), 0.0)

    def test_relation_loss_rejects_old_pair_visibility_and_confidence(self):
        shape = (2, 3)
        predicted = {
            "same_drawable_affinity": torch.full(shape, 0.5),
            "cut_probability": torch.full(shape, 0.5),
            "attachment_affinity": torch.full(shape, 0.5),
            "front_probability": torch.full(shape, 0.5),
            "occlusion_probability": torch.full(shape, 0.5),
            "projected_alpha_boundary_probability": torch.full(shape, 0.5),
            "projected_appearance_boundary_probability": torch.full(shape, 0.5),
        }
        truth = {name: torch.zeros(shape) for name in predicted}
        validity = {name: torch.ones(shape, dtype=torch.bool) for name in predicted}
        result = graphics_relation_loss_v3(predicted, truth, validity)
        self.assertTrue(torch.isfinite(result["total"]))

        bad = {**predicted, "confidence": torch.full(shape, 0.5)}
        with self.assertRaisesRegex(ValueError, "IRIS_RELATION_V3_FORBIDDEN_PAIR_OUTPUT_PRESENT"):
            graphics_relation_loss_v3(bad, truth, validity)


if __name__ == "__main__":
    unittest.main()
