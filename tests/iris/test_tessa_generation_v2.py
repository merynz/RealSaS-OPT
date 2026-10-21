import unittest
from types import SimpleNamespace

import numpy as np
import torch

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from models.tessa.v2 import (
    TESSAGenerationPolicyV2,
    TESSAGrammarStateV2,
    TESSAStructuralizerConfigV2,
    generate_tessa_candidate_v2,
)


class _ScriptedTopologyModel:
    """Tiny deterministic decoder used only to prove free-running control flow."""

    def __init__(self, config, target_tokens):
        self.config = config
        self.target_tokens = tuple(map(int, target_tokens))
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def decode_tokens(self, *, memory, input_ids, **_kwargs):
        next_index = int(input_ids.shape[1])
        target = (
            self.target_tokens[next_index]
            if next_index < len(self.target_tokens)
            else self.config.EOS
        )
        logits = torch.full(
            (1, input_ids.shape[1], self.config.vocab_size),
            -40.0,
            device=input_ids.device,
        )
        logits[0, -1, target] = 40.0
        return SimpleNamespace(logits=logits)


class TESSAGenerationV2Tests(unittest.TestCase):
    def setUp(self):
        self.cfg = TESSAStructuralizerConfigV2(
            d_model=32,
            n_heads=4,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=32,
            surface_memory_tokens=32,
            memory_source_width=16,
            memory_refinement_layers=0,
            local_attention_window=128,
            query_chunk_size=8,
            gradient_checkpointing=False,
        )
        self.codec_cfg = TESSAConfigV1(
            d_model=32,
            n_heads=4,
            surface_layers=0,
            decoder_layers=1,
            coordinate_bins=128,
            surface_latent_count=64,
            local_attention_window=128,
            query_chunk_size=8,
            max_faces=self.cfg.max_faces,
            max_vertices=self.cfg.max_vertices,
        )
        vertices = np.asarray(
            [
                [-0.25, -0.25, -0.25],
                [0.25, -0.25, -0.25],
                [0.00, 0.25, -0.25],
                [0.00, 0.00, 0.25],
            ],
            dtype=np.float64,
        )
        faces = np.asarray(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
            dtype=np.int64,
        )
        self.sequence = encode_adjacent_teacher_mesh_v1(
            vertices,
            faces,
            component_id="tetra",
            chart_id="global",
            cfg=self.codec_cfg,
        ).token_ids

    def test_streaming_grammar_accepts_strict_teacher_sequence(self):
        state = TESSAGrammarStateV2.from_prefix(self.sequence, config=self.cfg)
        self.assertTrue(state.complete)
        self.assertEqual(state.face_count, 4)
        self.assertEqual(state.component_count, 1)

    def test_streaming_grammar_rejects_empty_chart(self):
        state = TESSAGrammarStateV2.from_prefix(
            [self.cfg.BOS, self.cfg.COMPONENT_BEGIN, self.cfg.CHART_BEGIN],
            config=self.cfg,
        )
        mask = state.allowed_token_mask(config=self.cfg)
        self.assertFalse(bool(mask[self.cfg.CHART_END]))
        with self.assertRaisesRegex(ValueError, "TESSA_V2_GENERATION_GRAMMAR_VIOLATION"):
            state.advance(self.cfg.CHART_END, config=self.cfg)

    def test_scripted_free_running_generation_reaches_eos_and_court(self):
        model = _ScriptedTopologyModel(self.cfg, self.sequence)
        memory = torch.randn(1, self.cfg.total_conditioning_tokens, self.cfg.d_model)
        result = generate_tessa_candidate_v2(
            model,
            memory=memory,
            policy=TESSAGenerationPolicyV2(
                maximum_new_tokens=len(self.sequence) + 4,
                sampling=False,
            ),
        )
        self.assertTrue(result.completed)
        self.assertEqual(result.stop_reason, "EOS")
        self.assertEqual(result.token_ids, tuple(self.sequence))
        self.assertTrue(result.topology_court.grammar_valid)
        self.assertEqual(result.topology_court.face_count, 4)
        self.assertEqual(result.topology_court.boundary_edge_count, 0)
        self.assertFalse(result.topology_court.scientific_pass_claimed)
        self.assertFalse(result.topology_court.training_authorized)
        self.assertTrue(model.training)  # original mode restored


if __name__ == "__main__":
    unittest.main()
