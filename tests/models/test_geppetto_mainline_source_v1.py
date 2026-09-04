from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "geppetto_arachne_r6_20260901"
TARGET = ROOT / "models" / "geppetto" / "v2"
PROMOTED = (
    "geppetto_candidate_v2.py",
    "geppetto_conditioning_v2.py",
    "geppetto_checkpoint_v2.py",
    "geppetto_loss_v2.py",
)


class GeppettoMainlineSourceV1(unittest.TestCase):
    def test_promoted_v2_source_is_byte_identical(self) -> None:
        for name in PROMOTED:
            self.assertEqual((SOURCE / name).read_bytes(), (TARGET / name).read_bytes(), name)

    def test_v1_candidate_not_promoted_as_current_v2(self) -> None:
        self.assertTrue((SOURCE / "geppetto_candidate_v1.py").is_file())
        self.assertFalse((TARGET / "geppetto_candidate_v1.py").exists())

    def test_current_contract_symbols_present(self) -> None:
        candidate = (TARGET / "geppetto_candidate_v2.py").read_text(encoding="utf-8")
        conditioning = (TARGET / "geppetto_conditioning_v2.py").read_text(encoding="utf-8")
        loss = (TARGET / "geppetto_loss_v2.py").read_text(encoding="utf-8")
        self.assertIn("class GeppettoCandidateV2(nn.Module):", candidate)
        self.assertIn("LatentAutoregressiveSetProposal.v3", candidate)
        self.assertIn("FEATURE_CONTRACT_V2", conditioning)
        self.assertIn("assert len(FEATURE_CONTRACT_V2)==24", conditioning)
        self.assertIn("_canonical_integer_tie_matrix", loss)
        self.assertIn("_coincident_topology_mappings", loss)


if __name__ == "__main__":
    unittest.main()
