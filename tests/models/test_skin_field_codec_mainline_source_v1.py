from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "geppetto_arachne_r6_20260901"
TARGET = ROOT / "models" / "skin_field_codec" / "v1"


class SkinFieldCodecMainlineSourceV1(unittest.TestCase):
    def test_codec_and_checkpoint_are_byte_identical(self) -> None:
        for name in ("skin_field_codec_v1.py", "skin_field_codec_checkpoint_v1.py"):
            self.assertEqual((SOURCE / name).read_bytes(), (TARGET / name).read_bytes(), name)

    def test_config_was_semantically_split_from_mixed_candidate_config(self) -> None:
        config = (TARGET / "config_v1.py").read_text(encoding="utf-8")
        shim = (TARGET / "candidate_config_v1.py").read_text(encoding="utf-8")
        self.assertIn("class SkinFieldCodecConfigV1", config)
        self.assertIn('architecture_id: str = "RealSaS.SkinFieldCodec.ContinuousJointField.v1"', config)
        self.assertNotIn("GeppettoCandidateConfigV1", config)
        self.assertNotIn("ArachneCandidateConfigV1", config)
        self.assertIn("from .config_v1 import SKIN_FIELD_CODEC_V1, SkinFieldCodecConfigV1", shim)

    def test_default_config_hash_is_preserved(self) -> None:
        from models.skin_field_codec.v1.config_v1 import SKIN_FIELD_CODEC_V1
        self.assertEqual(
            SKIN_FIELD_CODEC_V1.config_hash,
            "24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715",
        )


if __name__ == "__main__":
    unittest.main()
