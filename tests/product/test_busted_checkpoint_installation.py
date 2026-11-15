from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from product.busted_site.checkpoint_installation import (
    FAMILIES, SCHEMA, inspect_installation,
)
from product.busted_site.server import BustedApplication


class CheckpointInstallationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "models.json"
        self.specs = {}
        for family in FAMILIES:
            payload = ("checkpoint:" + family).encode()
            (self.root / (family + ".pt")).write_bytes(payload)
            self.specs[family] = {
                "path": family + ".pt",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "architecture_id": "RealSaS." + family + ".ProductModel.v1",
            }
        self._write()

    def _write(self):
        self.manifest.write_text(json.dumps({"schema": SCHEMA, "models": self.specs}))

    def inspect(self):
        return inspect_installation(self.manifest, self.root)

    def test_all_digests_verified_but_product_authority_remains_closed(self):
        outcome = self.inspect()
        self.assertTrue(outcome["checkpoints_present"])
        self.assertTrue(outcome["manifest_valid"])
        self.assertFalse(outcome["compile_available"])
        self.assertFalse(outcome["product_authority"])
        self.assertIn("PRODUCT_MODELRELEASE_COMPILER_PROOF_GATE_NOT_CLOSED", outcome["blockers"])
        for slot in outcome["slots"].values():
            self.assertEqual(slot["status"], "DIGEST_VERIFIED")

    def test_mutated_checkpoint_is_rejected(self):
        (self.root / "IRIS.pt").write_bytes(b"changed")
        outcome = self.inspect()
        self.assertEqual(outcome["slots"]["IRIS"]["status"], "DIGEST_MISMATCH")
        self.assertFalse(outcome["checkpoints_present"])

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            destination = Path(outside) / "model.pt"
            destination.write_bytes(b"outside")
            (self.root / "AXIS.pt").unlink()
            (self.root / "AXIS.pt").symlink_to(destination)
            outcome = self.inspect()
        self.assertEqual(outcome["slots"]["AXIS"]["status"], "PATH_INVALID")

    def test_duplicate_slot_paths_rejected(self):
        self.specs["MIRA"]["path"] = "AXIS.pt"
        self._write()
        self.assertEqual(self.inspect()["slots"]["MIRA"]["status"], "PATH_INVALID")

    def test_unknown_slot_and_invalid_schema_fail_closed(self):
        del self.specs["MIRA"]
        self._write()
        outcome = self.inspect()
        self.assertFalse(outcome["manifest_valid"])
        self.assertFalse(outcome["checkpoints_present"])

    def test_path_traversal_and_family_mismatch(self):
        self.specs["TESSA"]["path"] = "../untrusted.pt"
        self.specs["IRIS"]["architecture_id"] = "RealSaS.MIRA.ProductModel.v1"
        self._write()
        outcome = self.inspect()
        self.assertEqual(outcome["slots"]["TESSA"]["status"], "CONTRACT_INVALID")
        self.assertEqual(outcome["slots"]["IRIS"]["status"], "CONTRACT_INVALID")

    def test_api_does_not_disclose_private_checkpoint_paths(self):
        env = {
            "BUSTED_CHECKPOINT_ROOT": str(self.root),
            "BUSTED_CHECKPOINT_MANIFEST": str(self.manifest),
        }
        with patch.dict(os.environ, env):
            app = BustedApplication.default()
            status, body, mime = app.dispatch("GET", "/api/checkpoints/status")
            self.assertEqual(status, 200)
            self.assertEqual(mime, "application/json")
            self.assertNotIn(str(self.root), json.dumps(body))
            self.assertTrue(body["checkpoints_present"])
            project_status, project, _ = app.dispatch("GET", "/api/project")
            self.assertEqual(project_status, 200)
            self.assertFalse(project["compile_available"])
            self.assertEqual(project["checkpoint_inspection"], "/api/checkpoints/status")


if __name__ == "__main__":
    unittest.main()
