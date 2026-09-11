from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from product.living_compile.common import LivingCompileError
from product.living_compile.runtime import runtime_clips, runtime_frame
from product.living_compile.scene import build_scene
from tests.product.test_living_compile_v4 import fixture


class LivingCompileStaticCandidateTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name) / "bundle"
        self.root.mkdir()
        fixture(self.root)
        (self.root / "proof" / "product_proof_bundle_ir.json").unlink()
        proof_bakes = self.root / "proof" / "motion_bakes"
        if proof_bakes.is_dir():
            for path in proof_bakes.glob("*.json"):
                path.unlink()

    def tearDown(self):
        self.t.cleanup()

    def test_missing_proof_keeps_static_real_artifacts_visible_and_runtime_withheld(self):
        scene = build_scene(self.root)
        self.assertFalse(scene["proof"]["available"])
        self.assertEqual(scene["proof"]["overall_status"], "UNAVAILABLE")
        self.assertFalse(scene["proof"]["passed"])
        self.assertEqual(scene["proof"]["recommended_claim"], "RIGGING_CORE_INSPECTION_ONLY__PRODUCT_UNQUALIFIED")
        self.assertIn("PRODUCT_PROOF_UNAVAILABLE", scene["proof"]["blockers"])
        self.assertTrue(scene["puppet"]["meshes"])
        self.assertTrue(scene["puppet"]["rig"]["controls"])
        self.assertFalse(scene["index"]["artifact_readiness"]["proof"])
        self.assertFalse(scene["runtime"]["preview_available"])
        self.assertEqual(scene["authority"]["product_proof_authority"], "ABSENT__STATIC_INSPECTION_ONLY")

        runtime = runtime_clips(self.root)
        self.assertEqual(runtime["proof_status"], "UNAVAILABLE")
        self.assertEqual(runtime["clips"], [])
        self.assertFalse(runtime["preview_available"])
        self.assertEqual(runtime["authority"], "RUNTIME_PREVIEW_WITHHELD_UNTIL_CURRENT_PASS_PROOF")
        with self.assertRaises(LivingCompileError):
            runtime_frame(self.root, "IDLE", 0.5)


if __name__ == "__main__":
    unittest.main()
