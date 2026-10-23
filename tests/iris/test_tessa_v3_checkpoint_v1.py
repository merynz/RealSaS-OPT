import tempfile
import unittest
from pathlib import Path

import torch

from models.tessa.v3.checkpoint_v1 import (
    CHECKPOINT_SCHEMA_V1,
    load_checkpoint_v1,
    write_checkpoint_v1,
)
from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3
from models.tessa.v3.model_v3 import TESSAStructuralizerV3


class TESSAV3CheckpointV1Tests(unittest.TestCase):
    @staticmethod
    def config():
        return TESSAStructuralizerConfigV3(
            d_model=32,
            n_heads=4,
            decoder_layers=1,
            mlp_ratio=2,
            dropout=0.0,
            coordinate_bins=128,
            max_faces=16,
            max_vertices=32,
            scene_memory_tokens=2,
            surface_memory_tokens=2,
            presentation_memory_tokens=1,
            conditioning_token_budget=5,
            memory_source_width=16,
            memory_refinement_layers=0,
            local_attention_window=8,
            query_chunk_size=4,
            gradient_checkpointing=False,
        )

    @classmethod
    def model(cls):
        torch.manual_seed(731)
        model = TESSAStructuralizerV3(
            scene_input_dim=7,
            surface_input_dim=9,
            presentation_input_dim=11,
            config=cls.config(),
        )
        model.eval()
        model.requires_grad_(False)
        return model

    def test_roundtrip_preserves_exact_state_and_freezes(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_checkpoint_v1(Path(tmp) / "tessa.pt", model)
            loaded = load_checkpoint_v1(path)
        self.assertEqual(loaded.architecture_id, TESSAStructuralizerV3.architecture_id)
        self.assertEqual(loaded.memory.scene.input_dim, 7)
        self.assertEqual(loaded.memory.surface.input_dim, 9)
        self.assertEqual(loaded.memory.presentation.input_dim, 11)
        self.assertFalse(loaded.training)
        self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
        expected = model.state_dict()
        actual = loaded.state_dict()
        self.assertEqual(set(expected), set(actual))
        for key in expected:
            self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def test_export_rejects_trainable_model(self):
        model = TESSAStructuralizerV3(
            scene_input_dim=7,
            surface_input_dim=9,
            presentation_input_dim=11,
            config=self.config(),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "MODEL_NOT_FROZEN"):
                write_checkpoint_v1(Path(tmp) / "bad.pt", model)

    def test_unknown_training_baggage_and_state_drift_fail_closed(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            good = write_checkpoint_v1(Path(tmp) / "good.pt", model)
            payload = torch.load(good, map_location="cpu", weights_only=True)
            self.assertEqual(payload["schema"], CHECKPOINT_SCHEMA_V1)

            baggage = dict(payload)
            baggage["optimizer_state"] = {"step": 1}
            baggage_path = Path(tmp) / "baggage.pt"
            torch.save(baggage, baggage_path)
            with self.assertRaisesRegex(RuntimeError, "PAYLOAD_INVALID"):
                load_checkpoint_v1(baggage_path)

            drift = dict(payload)
            drift_state = dict(drift["model_state_dict"])
            drift_state.pop(next(iter(drift_state)))
            drift["model_state_dict"] = drift_state
            drift_path = Path(tmp) / "drift.pt"
            torch.save(drift, drift_path)
            with self.assertRaisesRegex(RuntimeError, "STATE_DRIFT"):
                load_checkpoint_v1(drift_path)

    def test_architecture_and_input_dim_drift_fail_closed(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            good = write_checkpoint_v1(Path(tmp) / "good.pt", model)
            payload = torch.load(good, map_location="cpu", weights_only=True)

            architecture = dict(payload)
            architecture["architecture_id"] = "wrong"
            path = Path(tmp) / "architecture.pt"
            torch.save(architecture, path)
            with self.assertRaisesRegex(RuntimeError, "ARCHITECTURE_DRIFT"):
                load_checkpoint_v1(path)

            dims = dict(payload)
            dims["scene_input_dim"] = 0
            path = Path(tmp) / "dims.pt"
            torch.save(dims, path)
            with self.assertRaisesRegex(RuntimeError, "SCENE_DIM_INVALID"):
                load_checkpoint_v1(path)


if __name__ == "__main__":
    unittest.main()
