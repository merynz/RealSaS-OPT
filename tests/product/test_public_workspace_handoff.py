from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from product.living_compile.server import LivingCompileApplication


class PublicWorkspaceHandoffTests(unittest.TestCase):
    def test_proof_gated_workspace_bundle_materializes_without_internal_ids(self):
        compile_id = "cmp_" + "a" * 32
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("bundle_manifest.json", json.dumps({"schema_version": "test"}))
                archive.writestr(
                    "puppet/canonical_puppet_graph_v3.json",
                    json.dumps({"schema_version": "RealSaS.CanonicalPuppetGraph.v3"}),
                )
                archive.writestr(
                    "proof/product_proof_bundle_ir.json",
                    json.dumps({"schema_version": "RealSaS.ProductProofBundleIR.v1", "overall_status": "PASS"}),
                )

            app = LivingCompileApplication.default()
            env = {
                "REALSAS_PRODUCT_GATEWAY_URL": "https://product-gateway.example",
                "REALSAS_PRODUCT_GATEWAY_TOKEN": "service-only-token",
                "REALSAS_PRODUCT_COMPILE_ENABLED": "1",
                "REALSAS_PRODUCT_BUNDLE_ROOT": str(root),
            }
            with patch.dict("os.environ", env, clear=False), patch.object(
                app,
                "_gateway_bytes",
                return_value=(200, payload.getvalue()),
            ) as gateway:
                status, handoff, content_type = app.dispatch(
                    "GET",
                    f"/api/compile/workspace?id={compile_id}",
                )

            self.assertEqual(status, 200)
            self.assertEqual(content_type, "application/json")
            self.assertEqual(handoff["compile_id"], compile_id)
            self.assertEqual(handoff["authority"], "PROOF_GATED_WORKSPACE_BUNDLE")
            self.assertTrue(handoff["workspace_url"].startswith("/workspace?bundle="))
            self.assertNotIn("engine_release_id", handoff)
            self.assertNotIn("product_revision_id", handoff)
            self.assertNotIn("artifact_id", handoff)
            self.assertTrue((root / handoff["bundle"] / "bundle_manifest.json").is_file())
            self.assertTrue((root / handoff["bundle"] / "proof/product_proof_bundle_ir.json").is_file())
            gateway.assert_called_once()


if __name__ == "__main__":
    unittest.main()
