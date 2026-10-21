import inspect
import unittest

import torch

from models.iris.vnext.contracts_v1 import IRISVNextConfigV1
from models.iris.vnext.heads_v3 import GraphicsRelationHeadV3Config
from models.iris.vnext.model_v3 import IRISVNextEvidenceModelV3
from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from models.iris.vnext.presentation_prediction_v1 import (
    PREDICTED_PRESENTATION_PROVENANCE_V1,
    concat_role_free_presentation_evidence_v1,
)


class PredictedPresentationV1Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(177)
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
        self.model = IRISVNextEvidenceModelV3(
            self.cfg,
            relation,
            ray_visibility_width=64,
            ray_visibility_heads=8,
            ray_visibility_layers=2,
            ray_visibility_mlp_ratio=2,
        )
        self.scene = torch.randn(1, 3, self.cfg.plane_channels, 8, 8)

    @staticmethod
    def camera(batch=1):
        value = torch.zeros(batch, 13)
        value[..., 0:3] = torch.tensor([0.0, 0.0, -2.0])
        value[..., 3:6] = torch.tensor([1.0, 0.0, 0.0])
        value[..., 6:9] = torch.tensor([0.0, 1.0, 0.0])
        value[..., 9:12] = torch.tensor([0.0, 0.0, 1.0])
        value[..., 12] = 1.0
        return value

    def test_provenance_forbids_teacher_and_materialized_truth_in_forward(self):
        self.assertFalse(
            PREDICTED_PRESENTATION_PROVENANCE_V1[
                "materialized_v2_truth_in_inference_forward"
            ]
        )
        self.assertFalse(
            PREDICTED_PRESENTATION_PROVENANCE_V1[
                "teacher_topology_in_inference_forward"
            ]
        )
        self.assertFalse(
            PREDICTED_PRESENTATION_PROVENANCE_V1[
                "teacher_rig_or_skin_in_inference_forward"
            ]
        )
        self.assertFalse(
            PREDICTED_PRESENTATION_PROVENANCE_V1[
                "named_semantic_taxonomy_in_inference_forward"
            ]
        )
        signatures = " ".join(
            str(inspect.signature(getattr(self.model, name)))
            for name in (
                "predict_surface_presentation",
                "predict_pair_presentation",
                "predict_ray_presentation",
            )
        ).lower()
        for forbidden in ("teacher", "manifest", "payload", "semantic", "rig", "skin"):
            self.assertNotIn(forbidden, signatures)

    def test_pair_and_ray_predictions_form_typed_tessa_memory(self):
        xyz_a = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.1]]])
        xyz_b = torch.tensor([[[0.0, 0.0, 0.2], [0.1, 0.0, 0.3]]])
        pair_camera = self.camera()[:, None, :].expand(-1, 2, -1)
        valid = torch.ones(1, 2, dtype=torch.bool)
        pair = self.model.predict_pair_presentation(
            scene_planes=self.scene,
            xyz_a=xyz_a,
            xyz_b=xyz_b,
            camera_features=pair_camera,
            token_valid_mask=valid,
        )

        hit_xyz = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.35]]]
        )
        hit_depth = torch.tensor([[2.0, 2.2, 2.35]])
        hit_valid = torch.ones(1, 3, dtype=torch.bool)
        ray = self.model.predict_ray_presentation(
            scene_planes=self.scene,
            hit_xyz=hit_xyz,
            hit_depth=hit_depth,
            camera_features=self.camera(),
            hit_valid_mask=hit_valid,
        )

        evidence = concat_role_free_presentation_evidence_v1(pair, ray)
        self.assertEqual(evidence.values.shape[1], 5)
        front_valid = PRESENTATION_FEATURE_NAMES_V1.index("front_valid")
        global_valid = PRESENTATION_FEATURE_NAMES_V1.index("global_visibility_valid")
        self.assertTrue(bool((pair.values[..., front_valid] == 1.0).all()))
        self.assertTrue(bool((ray.values[..., global_valid] == 1.0).all()))
        self.assertTrue(bool(evidence.valid_mask.all()))

    def test_surface_prediction_keeps_observed_rgba_distinct_from_prediction(self):
        xyz = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        source_rgba = torch.tensor(
            [[[0.2, 0.3, 0.4, 1.0], [0.7, 0.6, 0.5, 0.8]]]
        )
        surface = self.model.predict_surface_presentation(
            scene_planes=self.scene,
            xyz=xyz,
            camera_features=self.camera(),
            token_valid_mask=valid,
            source_rgba=source_rgba,
            source_rgba_valid=valid,
        )
        source_r = PRESENTATION_FEATURE_NAMES_V1.index("source_r")
        source_rgba_valid = PRESENTATION_FEATURE_NAMES_V1.index("source_rgba_valid")
        self.assertTrue(torch.allclose(surface.values[..., source_r], source_rgba[..., 0]))
        self.assertTrue(bool((surface.values[..., source_rgba_valid] == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
