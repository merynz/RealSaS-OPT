from __future__ import annotations

"""Public Living Compile export guards; no ProductRevision or RSS PASS claimed."""

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

from product.living_compile.authoring import save_user_layer
from product.living_compile.common import LivingCompileError
from product.living_compile.server import LivingCompileApplication
from tests.product.test_living_compile_v4 import fixture


class LivingCompileExportGuardV1(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "current-puppet"
        self.root.mkdir()
        fixture(self.root)
        self.native = self.root / "native" / "realsas_runtime.rss"
        self.native.parent.mkdir()
        # Path presence only. This suite tests unsafe export REJECTION;
        # the dummy bytes are not, and cannot claim to be, qualified RSS.
        self.native.write_bytes(b"NOT_A_QUALIFIED_RUNTIME")
        self.app = LivingCompileApplication.default()

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        return self.app.dispatch("POST", "/api/export", {"bundle": str(self.root)})

    def test_edit_protects_original_native_bytes(self):
        source = self.root / "puppet" / "canonical_puppet_graph_v3.json"
        original = source.read_bytes()
        saved = save_user_layer(self.root, {
            "view_id": "V7",
            "edits": [{
                "control_id": "ROOT", "rotation_degrees": 24,
                "translation_xy": [3, 0],
            }],
        })
        self.assertTrue(Path(saved["path"]).is_file())
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_USER_EDITS_REQUIRE_COMPILER_REQUALIFICATION"):
            self.export()
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(self.native.read_bytes(), b"NOT_A_QUALIFIED_RUNTIME")

    def test_weight_edit_on_other_view_blocks_export(self):
        saved = save_user_layer(self.root, {
            "view_id": "V3",
            "weight_edits": [{
                "mesh_id": "V3:BODY", "vertex_index": 0,
                "control_id": "ROOT", "weight_delta": 0.2,
            }],
        })
        self.assertTrue(saved["requires_dynamic_revalidation"])
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_USER_EDITS_REQUIRE_COMPILER_REQUALIFICATION"):
            self.export()

    def test_nonpassing_product_proof_blocks_export(self):
        path = self.root / "proof" / "product_proof_bundle_ir.json"
        for status in ("FAIL", "UNKNOWN", None):
            proof = json.loads(path.read_text())
            proof["overall_status"] = status
            path.write_text(json.dumps(proof))
            with self.subTest(status=status):
                with self.assertRaisesRegex(LivingCompileError, "EXPORT_PRODUCT_PROOF_PASS_REQUIRED"):
                    self.export()

    def test_proof_bound_to_other_product_blocks_export(self):
        path = self.root / "proof" / "product_proof_bundle_ir.json"
        proof = json.loads(path.read_text())
        proof["source_product_state_hash"] = "x" * 64
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(LivingCompileError, "proof/product lineage mismatch"):
            self.export()

    def test_malformed_user_layer_cannot_bypass_requalification(self):
        layer = self.root.parent / (self.root.name + ".user") / "layers" / "view_00.json"
        layer.parent.mkdir(parents=True)
        layer.write_text('{"requires_compiler_requalification":false}')
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_USER_EDITS_REQUIRE_COMPILER_REQUALIFICATION"):
            self.export()

    def blob(self, relative):
        return self.app.dispatch(
            "GET",
            "/api/blob?bundle=" + quote(str(self.root), safe="")
            + "&path=" + quote(relative, safe=""),
        )

    def test_raw_rss_blob_cannot_bypass_pending_edit_export_gate(self):
        save_user_layer(self.root, {
            "view_id": "V0",
            "edits": [{"control_id": "ROOT", "translation_xy": [1, 0]}],
        })
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_USER_EDITS_REQUIRE_COMPILER_REQUALIFICATION"):
            self.blob("native/realsas_runtime.rss")

    def test_raw_rss_blob_cannot_bypass_failing_proof(self):
        path = self.root / "proof" / "product_proof_bundle_ir.json"
        proof = json.loads(path.read_text())
        proof["overall_status"] = "FAIL"
        path.write_text(json.dumps(proof))
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_PRODUCT_PROOF_PASS_REQUIRED"):
            self.blob("native/realsas_runtime.rss")

    def test_undeclared_rss_blob_is_not_downloadable(self):
        other = self.root / "native" / "unqualified.rss"
        other.write_bytes(b"UNKNOWN_PACKAGE")
        with self.assertRaisesRegex(LivingCompileError, "EXPORT_UNDECLARED_RSS_BLOB_FORBIDDEN"):
            self.blob("native/unqualified.rss")

    def test_non_rss_art_assets_remain_accessible_while_edit_pending(self):
        save_user_layer(self.root, {
            "view_id": "V0",
            "edits": [{"control_id": "ROOT", "translation_xy": [1, 0]}],
        })
        status, raw, _ = self.blob("puppet/canonical_puppet_graph_v3.json")
        self.assertEqual(status, 200)
        self.assertIn(b"CanonicalPuppetGraph.v3", raw)

    def test_unedited_but_missing_native_package_still_blocks_export(self):
        self.native.unlink()
        with self.assertRaisesRegex(LivingCompileError, "proof-gated native"):
            self.export()


if __name__ == "__main__":
    unittest.main()
