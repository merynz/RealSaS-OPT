import unittest

import torch

from models.tessa.v2 import (
    TESSAStructuralizerConfigV2,
    TESSAStructuralizerV2,
    topology_sequence_objective_v2,
)


class TESSAStructuralizerV2Tests(unittest.TestCase):
    def test_reference_capacity_defaults_are_not_diagnostic_scale(self):
        cfg = TESSAStructuralizerConfigV2()
        cfg.validate()
        self.assertEqual(cfg.d_model, 1024)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.decoder_layers, 24)
        self.assertEqual(cfg.feedforward_dim, 4096)
        self.assertEqual(cfg.coordinate_bins, 1024)
        self.assertEqual(cfg.scene_memory_tokens, 512)
        self.assertEqual(cfg.surface_memory_tokens, 512)
        self.assertEqual(cfg.total_conditioning_tokens, 1024)
        self.assertGreaterEqual(cfg.local_attention_window, 4096)
        self.assertEqual(cfg.max_faces, 32768)
        self.assertEqual(cfg.max_vertices, 32768)

    def test_coordinate_metric_distinguishes_near_and_far_wrong_bins(self):
        cfg = TESSAStructuralizerConfigV2(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=4,
            surface_memory_tokens=4,
            memory_source_width=32,
            memory_refinement_layers=0,
            local_attention_window=16,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        label = torch.tensor([[32]], dtype=torch.long)

        near = torch.full((1, 1, cfg.vocab_size), -12.0)
        near[0, 0, 33] = 12.0
        far = torch.full((1, 1, cfg.vocab_size), -12.0)
        far[0, 0, 110] = 12.0

        near_result = topology_sequence_objective_v2(near, label, config=cfg)
        far_result = topology_sequence_objective_v2(far, label, config=cfg)
        # Hard CE is effectively the same wrong-token penalty, while the metric
        # term knows one geometric miss is one bin and the other is very far.
        self.assertAlmostEqual(
            float(near_result.token_cross_entropy),
            float(far_result.token_cross_entropy),
            places=5,
        )
        self.assertLess(
            float(near_result.coordinate_metric),
            float(far_result.coordinate_metric),
        )
        self.assertLess(
            float(near_result.coordinate_expected_abs_error_bins),
            float(far_result.coordinate_expected_abs_error_bins),
        )

    def test_dual_stream_memory_preserves_typed_token_budgets(self):
        cfg = TESSAStructuralizerConfigV2(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=5,
            surface_memory_tokens=7,
            memory_source_width=32,
            memory_refinement_layers=1,
            local_attention_window=16,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        model = TESSAStructuralizerV2(
            scene_input_dim=11,
            surface_input_dim=13,
            config=cfg,
        )
        scene = torch.randn(2, 9, 11)
        surface = torch.randn(2, 6, 13)
        memory = model.encode_memory(scene_source=scene, surface_source=surface)
        self.assertEqual(memory.shape, (2, 12, 64))

        input_ids = torch.tensor(
            [[cfg.BOS, cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN, cfg.FACE_BREAK],
             [cfg.BOS, cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN, cfg.FACE_BREAK]],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [[cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN, cfg.FACE_BREAK, 5],
             [cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN, cfg.FACE_BREAK, 7]],
            dtype=torch.long,
        )
        out = model(
            scene_source=scene,
            surface_source=surface,
            input_ids=input_ids,
            labels=labels,
        )
        self.assertEqual(out.logits.shape, (2, 4, cfg.vocab_size))
        self.assertIsNotNone(out.loss)
        self.assertTrue(torch.isfinite(out.loss))


if __name__ == "__main__":
    unittest.main()
