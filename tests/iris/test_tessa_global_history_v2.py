import unittest

import torch

from models.tessa.v2 import (
    HierarchicalCausalSelfAttentionV2,
    TESSAGlobalHistoryPolicyV2,
    TESSAStructuralizerConfigV2,
)


class TESSAGlobalHistoryV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.cfg = TESSAStructuralizerConfigV2(
            d_model=32,
            n_heads=4,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=4,
            surface_memory_tokens=4,
            memory_source_width=16,
            memory_refinement_layers=0,
            local_attention_window=4,
            query_chunk_size=2,
            gradient_checkpointing=False,
        )
        self.policy = TESSAGlobalHistoryPolicyV2(
            block_size=2,
            max_summary_tokens=16,
        )
        self.attn = HierarchicalCausalSelfAttentionV2(self.cfg, self.policy).eval()

    def test_future_tokens_cannot_change_past_outputs(self):
        x = torch.randn(1, 8, self.cfg.d_model)
        baseline = self.attn(x)
        changed = x.clone()
        changed[:, 6:] += 100.0
        mutated = self.attn(changed)
        self.assertTrue(torch.allclose(baseline[:, :6], mutated[:, :6], atol=1e-6, rtol=1e-6))

    def test_old_tokens_beyond_local_window_reach_tail_through_global_history(self):
        x = torch.randn(1, 8, self.cfg.d_model)
        baseline = self.attn(x)
        changed = x.clone()
        # Token 0 is outside the exact local window for the final query. It is
        # nevertheless part of a completed global-history summary.
        changed[:, 0] += 3.0
        mutated = self.attn(changed)
        delta = torch.max(torch.abs(baseline[:, -1] - mutated[:, -1])).item()
        self.assertGreater(delta, 1.0e-8)

    def test_global_history_capacity_fails_closed(self):
        policy = TESSAGlobalHistoryPolicyV2(block_size=2, max_summary_tokens=1)
        attn = HierarchicalCausalSelfAttentionV2(self.cfg, policy)
        x = torch.randn(1, 10, self.cfg.d_model)
        with self.assertRaisesRegex(RuntimeError, "TESSA_V2_GLOBAL_HISTORY_CAPACITY_EXCEEDED"):
            attn(x)


if __name__ == "__main__":
    unittest.main()
