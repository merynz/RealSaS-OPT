from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "geppetto_arachne_r6_20260901"
TARGET = ROOT / "models" / "arachne" / "v2"
PROMOTED = (
    "arachne_candidate_v2.py",
    "conditioning_v2.py",
    "conditioning_v1.py",
    "arachne_geometry_v2.py",
)


class ArachneMainlineSourceV1(unittest.TestCase):
    def test_current_v2_source_is_byte_identical(self) -> None:
        for name in PROMOTED:
            self.assertEqual((SOURCE / name).read_bytes(), (TARGET / name).read_bytes(), name)

    def test_dependency_bridges_target_current_model_homes(self) -> None:
        codec_bridge = (TARGET / "skin_field_codec_v1.py").read_text(encoding="utf-8")
        geppetto_bridge = (TARGET / "geppetto_conditioning_v2.py").read_text(encoding="utf-8")
        self.assertIn("models.skin_field_codec.v1.skin_field_codec_v1", codec_bridge)
        self.assertIn("models.geppetto.v2.geppetto_conditioning_v2", geppetto_bridge)
        self.assertNotIn("experiments.", codec_bridge)
        self.assertNotIn("experiments.", geppetto_bridge)

    def test_v1_candidate_not_promoted_as_current(self) -> None:
        self.assertTrue((SOURCE / "arachne_candidate_v1.py").is_file())
        self.assertFalse((TARGET / "arachne_candidate_v1.py").exists())

    def test_current_architecture_symbol_present(self) -> None:
        source = (TARGET / "arachne_candidate_v2.py").read_text(encoding="utf-8")
        self.assertIn("class ArachneCandidateV2(nn.Module):", source)
        self.assertIn("RealSaS.ArachneCandidate.SegmentAwareJointField.v2", source)


if __name__ == "__main__":
    unittest.main()
