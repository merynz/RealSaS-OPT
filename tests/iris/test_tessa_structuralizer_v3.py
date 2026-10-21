import unittest

import torch

from models.iris.vnext.presentation_evidence_v1 import PRESENTATION_FEATURE_NAMES_V1
from models.tessa.v3 import TESSAStructuralizerConfigV3, TESSAStructuralizerV3


class TESSAStructuralizerV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.presentation_dim = len(PRESENTATION_FEATURE_NAMES_V1)
        self.cfg = TESSAStructuralizerConfigV3(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=4,
            surface_memory_tokens=4,
            presentation_memory_tokens=2,
            conditioning_token_budget=10,
            memory_source_width=32,
            memory_refinement_layers=1,
            local_attention_window=16,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        self.model = TESSAStructuralizerV3(
            scene_input_dim=12,
            surface_input_dim=20,
            presentation_input_dim=self.presentation_dim,
            config=self.cfg,
        )

    def test_reference_config_preserves_1024_total_memory_budget(self):
        reference = TESSAStructuralizerConfigV3()
        reference.validate()
        self.assertEqual(reference.total_conditioning_tokens, 1024)
        self.assertEqual(reference.vocab_size, 1032)

    def test_tri_stream_memory_has_exact_matched_budget(self):
        memory = self.model.encode_memory(
            scene_source=torch.randn(2, 5, 12),
            surface_source=torch.randn(2, 6, 20),
            presentation_source=torch.randn(2, 3, self.presentation_dim),
            presentation_valid_mask=torch.tensor([[True, True, False], [True, True, True]]),
        )
        self.assertEqual(memory.shape, (2, 10, 64))

    def test_null_and_informative_presentation_use_same_architecture_shape(self):
        scene = torch.randn(1, 5, 12)
        surface = torch.randn(1, 6, 20)
        null = torch.zeros(1, 1, self.presentation_dim)
        info = torch.randn(1, 4, self.presentation_dim)
        a = self.model.encode_memory(
            scene_source=scene,
            surface_source=surface,
            presentation_source=null,
            presentation_valid_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        b = self.model.encode_memory(
            scene_source=scene,
            surface_source=surface,
            presentation_source=info,
            presentation_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        )
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[1], self.cfg.conditioning_token_budget)

    def test_decoder_vocabulary_and_objective_remain_v2_compatible(self):
        input_ids = torch.tensor([[self.cfg.BOS, self.cfg.COMPONENT_BEGIN, self.cfg.CHART_BEGIN, self.cfg.FACE_BREAK]])
        labels = torch.tensor([[self.cfg.COMPONENT_BEGIN, self.cfg.CHART_BEGIN, self.cfg.FACE_BREAK, 7]])
        out = self.model(
            scene_source=torch.randn(1, 3, 12),
            surface_source=torch.randn(1, 4, 20),
            presentation_source=torch.zeros(1, 1, self.presentation_dim),
            presentation_valid_mask=torch.ones(1, 1, dtype=torch.bool),
            input_ids=input_ids,
            labels=labels,
        )
        self.assertEqual(out.logits.shape, (1, 4, self.cfg.vocab_size))
        self.assertEqual(out.conditioning_memory.shape, (1, 10, 64))
        self.assertIsNotNone(out.objective)
        self.assertTrue(torch.isfinite(out.loss))


if __name__ == "__main__":
    unittest.main()
