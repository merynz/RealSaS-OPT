import unittest

import numpy as np

from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from models.tessa.v2 import TESSAStructuralizerConfigV2, evaluate_decoded_topology_v2


class TESSADecodedTopologyCourtV2Tests(unittest.TestCase):
    def setUp(self):
        # The V2 court intentionally reuses the strict V1-compatible grammar
        # decoder. Keep this unit fixture above V1's historical scaling-policy
        # minima rather than weakening survivor contracts for a tiny tetra test.
        self.cfg = TESSAStructuralizerConfigV2(
            d_model=64,
            n_heads=8,
            decoder_layers=1,
            coordinate_bins=128,
            scene_memory_tokens=32,
            surface_memory_tokens=32,
            memory_source_width=32,
            memory_refinement_layers=0,
            local_attention_window=128,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )
        self.codec_cfg = TESSAConfigV1(
            d_model=64,
            n_heads=8,
            surface_layers=0,
            decoder_layers=1,
            coordinate_bins=128,
            surface_latent_count=64,
            local_attention_window=128,
            query_chunk_size=4,
            max_faces=self.cfg.max_faces,
            max_vertices=self.cfg.max_vertices,
        )

    def test_valid_closed_tetra_reports_topology_not_training_pass(self):
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
        sequence = encode_adjacent_teacher_mesh_v1(
            vertices,
            faces,
            component_id="tetra",
            chart_id="global",
            cfg=self.codec_cfg,
        )
        court = evaluate_decoded_topology_v2(sequence.token_ids, config=self.cfg)
        self.assertTrue(court.grammar_valid)
        self.assertEqual(court.face_count, 4)
        self.assertEqual(court.geometric_component_count, 1)
        self.assertEqual(court.boundary_edge_count, 0)
        self.assertEqual(court.nonmanifold_edge_count, 0)
        self.assertEqual(court.zero_area_face_count, 0)
        self.assertTrue(court.finite_geometry)
        self.assertTrue(court.within_scale_policy)
        self.assertFalse(court.scientific_pass_claimed)
        self.assertFalse(court.training_authorized)

    def test_invalid_free_running_sequence_fails_closed(self):
        court = evaluate_decoded_topology_v2(
            [self.cfg.BOS, self.cfg.EOS],
            config=self.cfg,
        )
        self.assertFalse(court.grammar_valid)
        self.assertIsNotNone(court.decode_failure)
        self.assertFalse(court.scientific_pass_claimed)
        self.assertFalse(court.training_authorized)


if __name__ == "__main__":
    unittest.main()
