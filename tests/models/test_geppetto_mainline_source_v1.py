from __future__ import annotations

from pathlib import Path
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[2]
OLD_SOURCE = ROOT / "experiments" / "geppetto_arachne_r6_20260901"
OLD_TARGET = ROOT / "models" / "geppetto" / "v2"
FROZEN_SOURCE = ROOT / "experiments" / "geppetto_reference_strength_fullstack_v1"
CURRENT = ROOT / "models" / "geppetto" / "reference_strength_v1"

FROZEN_SOURCE_COMMIT = "f7be46f0a97df62a793ebf91b22297c894854f39"
FROZEN_CANDIDATE_GIT_BLOB = "e9b626815c63a96ac1d390580e1aa19f8bf1cfb2"
FROZEN_TENSORIZATION_GIT_BLOB = "647fdcb98c305c3f7d3afd05dd61449999721034"
FROZEN_NO_SLOT_GIT_BLOB = "770909a9c2992ada0f3018badb2a42632a124b3b"
CHECKPOINT_SHA256 = "b75f991564b64cfcec9b50b006544380ee482362a8439775bb505002349cbc30"
QUALIFIED_SKELETON_SHA256 = "48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992"
ARCHITECTURE_ID = (
    "RealSaS.Geppetto.ReferenceStrength.DirectSurfaceCausalDiffusion."
    "DeterministicViewDirection.v1"
)


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


class GeppettoMainlineSourceV1(unittest.TestCase):
    def test_current_frozen_candidate_is_byte_preserved(self) -> None:
        source = (FROZEN_SOURCE / "geppetto_reference_strength_candidate_v1.py").read_bytes()
        target = (CURRENT / "geppetto_reference_strength_candidate_v1.py").read_bytes()
        self.assertEqual(source, target)
        self.assertEqual(git_blob_sha1(target), FROZEN_CANDIDATE_GIT_BLOB)

    def test_current_lossless_tensorization_is_byte_preserved(self) -> None:
        source = (FROZEN_SOURCE / "rigging_surface_tensorization_v1.py").read_bytes()
        target = (CURRENT / "rigging_surface_tensorization_v1.py").read_bytes()
        self.assertEqual(source, target)
        self.assertEqual(git_blob_sha1(target), FROZEN_TENSORIZATION_GIT_BLOB)

    def test_no_slot_algorithm_and_current_import_home_are_preserved(self) -> None:
        frozen = (FROZEN_SOURCE / "geppetto_reference_strength_no_learned_slot_v1.py").read_bytes()
        self.assertEqual(git_blob_sha1(frozen), FROZEN_NO_SLOT_GIT_BLOB)
        current = (CURRENT / "geppetto_reference_strength_no_learned_slot_v1.py").read_text(encoding="utf-8")
        self.assertIn("from .geppetto_reference_strength_candidate_v1 import", current)
        self.assertIn(ARCHITECTURE_ID, current)
        self.assertIn(FROZEN_SOURCE_COMMIT, current)
        self.assertNotIn("nn.Parameter", current)

    def test_package_rebinds_frozen_historical_tensorization_name_to_models_home(self) -> None:
        init = (CURRENT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("sys.modules[_HISTORICAL_TENSORIZATION_MODULE] = _promoted_tensorization", init)
        self.assertIn("from . import rigging_surface_tensorization_v1", init)

    def test_checkpoint_authority_is_hash_bound(self) -> None:
        auth = (CURRENT / "checkpoint_authority_v1.py").read_text(encoding="utf-8")
        self.assertIn(CHECKPOINT_SHA256, auth)
        self.assertIn(QUALIFIED_SKELETON_SHA256, auth)
        self.assertIn(FROZEN_SOURCE_COMMIT, auth)
        self.assertIn("GENERALIZATION_CLAIM = False", auth)

    def test_prior_v2_is_preserved_as_provenance_not_deleted(self) -> None:
        promoted = (
            "geppetto_candidate_v2.py",
            "geppetto_conditioning_v2.py",
            "geppetto_checkpoint_v2.py",
            "geppetto_loss_v2.py",
        )
        for name in promoted:
            self.assertTrue((OLD_TARGET / name).is_file(), name)
            self.assertTrue((OLD_SOURCE / name).is_file(), name)

    def test_readme_declares_reference_strength_as_current_fit1_source(self) -> None:
        root_readme = (ROOT / "models" / "geppetto" / "README.md").read_text(encoding="utf-8")
        self.assertIn("models/geppetto/reference_strength_v1/", root_readme)
        self.assertIn("FIT1", root_readme)
        self.assertIn("Arachne", root_readme)


if __name__ == "__main__":
    unittest.main()
