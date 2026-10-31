from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import call, patch

from product.living_compile.common import LivingCompileError
from product.living_compile.server import LivingCompileApplication


class StudioPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.compile_id = "cmp_" + "a" * 32
        self.bundle = f"public/{self.compile_id}/" + "b" * 64
        self.project_id = "stp_" + "c" * 32
        self.env = {
            "REALSAS_PRODUCT_GATEWAY_URL": "https://product-gateway.example",
            "REALSAS_PRODUCT_GATEWAY_TOKEN": "service-only-token",
            "REALSAS_PRODUCT_COMPILE_ENABLED": "1",
            "REALSAS_PRODUCT_BUNDLE_ROOT": self.temp.name,
        }
        self.app = LivingCompileApplication.default()

    def tearDown(self):
        self.temp.cleanup()

    def test_studio_session_exposes_only_public_project_head(self):
        project = {
            "project_id": self.project_id,
            "compile_id": self.compile_id,
            "lock_version": 4,
            "product_revision_id": "internal-revision",
        }
        head = {
            "project_id": self.project_id,
            "compile_id": self.compile_id,
            "revision_number": 4,
            "lock_version": 4,
            "authoring_payload": {"schema_version": "RealSaS.UserPuppetEditLayer.v3"},
            "authoring_revision_id": "internal-authoring-revision",
        }
        with patch.dict(os.environ, self.env, clear=False), patch.object(
            self.app,
            "_gateway_json",
            side_effect=[(200, project), (200, head)],
        ) as gateway:
            status, payload, content_type = self.app.dispatch(
                "GET", f"/api/studio/session?bundle={self.bundle}"
            )
        self.assertEqual(status, 200)
        self.assertEqual(content_type, "application/json")
        self.assertEqual(payload["schema_version"], "RealSaS.StudioSession.v1")
        self.assertEqual(payload["project"]["project_id"], self.project_id)
        self.assertEqual(payload["head"]["lock_version"], 4)
        self.assertNotIn("product_revision_id", payload["project"])
        self.assertNotIn("authoring_revision_id", payload["head"])
        gateway.assert_has_calls([
            call(
                "POST",
                f"/v1/product/compiles/{self.compile_id}/studio-project",
                {"created_by": "product-shell"},
            ),
            call("GET", f"/v1/product/studio/projects/{self.project_id}/head"),
        ])

    def test_persist_uses_client_held_lock_and_verifies_compile_identity(self):
        layer = {"schema_version": "RealSaS.UserPuppetEditLayer.v3"}
        project = {"project_id": self.project_id, "compile_id": self.compile_id}
        revision = {
            "project_id": self.project_id,
            "revision_number": 6,
            "lock_version": 9,
            "requires_requalification": True,
        }
        with patch.dict(os.environ, self.env, clear=False), patch.object(
            self.app,
            "_gateway_json",
            side_effect=[(200, project), (201, revision)],
        ) as gateway:
            out = self.app._persist_studio_layer(
                self.compile_id,
                layer,
                project_id=self.project_id,
                expected_lock_version=8,
            )
        self.assertEqual(out["lock_version"], 9)
        gateway.assert_has_calls([
            call("GET", f"/v1/product/studio/projects/{self.project_id}"),
            call(
                "POST",
                f"/v1/product/studio/projects/{self.project_id}/revisions",
                {
                    "expected_lock_version": 8,
                    "authoring_payload": layer,
                    "created_by": "product-shell",
                },
            ),
        ])

    def test_persist_rejects_project_from_another_compile(self):
        project = {"project_id": self.project_id, "compile_id": "cmp_" + "d" * 32}
        with patch.dict(os.environ, self.env, clear=False), patch.object(
            self.app, "_gateway_json", return_value=(200, project)
        ):
            with self.assertRaisesRegex(LivingCompileError, "does not belong"):
                self.app._persist_studio_layer(
                    self.compile_id,
                    {"schema_version": "RealSaS.UserPuppetEditLayer.v3"},
                    project_id=self.project_id,
                    expected_lock_version=2,
                )

    def test_history_is_bounded_before_gateway_forward(self):
        with patch.dict(os.environ, self.env, clear=False), patch.object(
            self.app,
            "_gateway_json",
            return_value=(200, {"revisions": []}),
        ) as gateway:
            status, payload, _ = self.app.dispatch(
                "GET", f"/api/studio/history?id={self.project_id}&limit=999999"
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"revisions": []})
        gateway.assert_called_once_with(
            "GET", f"/v1/product/studio/projects/{self.project_id}/revisions?limit=200"
        )


if __name__ == "__main__":
    unittest.main()
