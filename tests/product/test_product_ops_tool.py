from __future__ import annotations

import unittest
from unittest.mock import patch

from tools import product_ops_v1


class ProductOpsToolTests(unittest.TestCase):
    def snapshot(self, *, ready: bool = True):
        return {
            "schema_version": "RealSaS.ProductOpsSnapshot.v1",
            "ready": ready,
            "source_code_sha": "a" * 40,
            "preflight": {
                "ready": ready,
                "engine_release_id": "release-1",
                "canonical_main_source_verified": ready,
                "models": [
                    {"family": "IRIS_TESSA", "ready": ready},
                    {"family": "AXIS_MIRA", "ready": ready},
                ],
                "blockers": [] if ready else ["MODEL_PRODUCTION_CHANNEL_UNBOUND:IRIS_TESSA"],
            },
            "compile_sessions": [
                {"state": "PASS", "count": 4},
                {"state": "RUNNING", "count": 1},
            ],
            "workspace_materializations": [{"state": "PASS", "count": 4}],
            "studio": {
                "active_projects": 3,
                "authoring_revisions": 12,
                "heads_requiring_requalification": 2,
            },
        }

    def test_format_snapshot_surfaces_product_operational_state(self):
        text = product_ops_v1.format_snapshot(self.snapshot())
        self.assertIn("Product Doctor: READY", text)
        self.assertIn("IRIS_TESSA=READY", text)
        self.assertIn("PASS=4", text)
        self.assertIn("RUNNING=1", text)
        self.assertIn("requalification_heads=2", text)
        self.assertIn("Blockers: none", text)

    def test_fetch_snapshot_requires_expected_schema_and_operator_auth(self):
        responses = [
            {"status": "ok"},
            self.snapshot(),
        ]
        with patch.object(product_ops_v1, "_request_json", side_effect=responses) as request:
            snapshot = product_ops_v1.fetch_snapshot(
                "https://platform.example/", "operator-token", timeout=3.0
            )
        self.assertTrue(snapshot["ready"])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args[0], "https://platform.example/health/live")
        self.assertIsNone(request.call_args_list[0].kwargs.get("token"))
        self.assertEqual(
            request.call_args_list[1].args[0],
            "https://platform.example/v1/operator/product/ops-snapshot",
        )
        self.assertEqual(request.call_args_list[1].kwargs["token"], "operator-token")

    def test_main_can_fail_closed_when_release_is_not_ready(self):
        with patch.object(product_ops_v1, "fetch_snapshot", return_value=self.snapshot(ready=False)):
            code = product_ops_v1.main(
                [
                    "--platform-url",
                    "https://platform.example",
                    "--operator-token",
                    "operator-token",
                    "--fail-not-ready",
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
