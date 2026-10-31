from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.product_deploy_preflight_v1 import (
    build_report,
    evaluate_local_environment,
    parse_env_file,
)


GOOD_ENV = {
    "REALSAS_SOURCE_CODE_SHA": "a" * 40,
    "REALSAS_DATABASE_URL": "postgres://realsas:secret@db.internal:5432/realsas?sslmode=require",
    "REALSAS_TEMPORAL_ADDRESS": "temporal.internal:7233",
    "REALSAS_TEMPORAL_NAMESPACE": "default",
    "REALSAS_PRODUCT_GATEWAY_TOKEN": "g" * 48,
    "REALSAS_OPERATOR_TOKEN": "o" * 48,
    "REALSAS_ENGINE_IMAGE": "registry.internal/realsas-engine:sha-deadbeef",
    "REALSAS_PRODUCT_COMPILE_ENABLED": "0",
    "REALSAS_SOURCE_MATERIALIZER_COMMAND": "",
    "REALSAS_WORKSPACE_MATERIALIZER_COMMAND": "",
}


PLATFORM_READY = {
    "schema_version": "RealSaS.ProductDeploymentPreflight.v1",
    "ready": True,
    "blockers": [],
    "models": [
        {"family": family, "ready": True}
        for family in ("AXIS", "IRIS", "MIRA", "TESSA")
    ],
}


class ProductDeployPreflightTest(unittest.TestCase):
    @patch("tools.product_deploy_preflight_v1.importlib.util.find_spec")
    def test_exact8_can_be_ready_while_public_frontback_stays_fail_closed(self, find_spec):
        find_spec.return_value = object()
        local = evaluate_local_environment(GOOD_ENV)
        self.assertTrue(local["exact8_local_ready"])
        self.assertFalse(local["public_frontback_local_ready"])
        self.assertTrue(local["workspace"]["uses_builtin_default"])
        report = build_report(local=local, platform=PLATFORM_READY)
        self.assertTrue(report["exact8_product_ready"])
        self.assertFalse(report["public_frontback_ready"])
        self.assertIn("PUBLIC_FRONTBACK_SOURCE_MATERIALIZER_UNBOUND", report["blockers"])
        self.assertIn("PUBLIC_FRONTBACK_COMPILE_FLAG_DISABLED", report["blockers"])

    @patch("tools.product_deploy_preflight_v1.importlib.util.find_spec")
    def test_public_lane_requires_explicit_source_materializer_and_feature_flag(self, find_spec):
        find_spec.return_value = object()
        env = {
            **GOOD_ENV,
            "REALSAS_SOURCE_MATERIALIZER_COMMAND": "python -m realsas.product_input.frontback_v2",
            "REALSAS_PRODUCT_COMPILE_ENABLED": "1",
        }
        local = evaluate_local_environment(env)
        report = build_report(local=local, platform=PLATFORM_READY)
        self.assertTrue(local["public_frontback_local_ready"])
        self.assertTrue(report["exact8_product_ready"])
        self.assertTrue(report["public_frontback_ready"])
        self.assertNotIn("PUBLIC_FRONTBACK_SOURCE_MATERIALIZER_UNBOUND", report["blockers"])
        self.assertNotIn("PUBLIC_FRONTBACK_COMPILE_FLAG_DISABLED", report["blockers"])

    @patch("tools.product_deploy_preflight_v1.importlib.util.find_spec")
    def test_placeholder_release_identity_and_engine_fail_closed(self, find_spec):
        find_spec.return_value = object()
        env = {
            **GOOD_ENV,
            "REALSAS_SOURCE_CODE_SHA": "0" * 40,
            "REALSAS_ENGINE_IMAGE": "realsas-engine:unbound",
        }
        local = evaluate_local_environment(env)
        self.assertFalse(local["exact8_local_ready"])
        self.assertIn("SOURCE_CODE_SHA_INVALID_OR_PLACEHOLDER", local["blockers"])
        self.assertIn("ENGINE_IMAGE_UNBOUND", local["blockers"])

    @patch("tools.product_deploy_preflight_v1.importlib.util.find_spec")
    def test_platform_blocker_prevents_exact8_ready(self, find_spec):
        find_spec.return_value = object()
        local = evaluate_local_environment(GOOD_ENV)
        platform = {**PLATFORM_READY, "ready": False, "blockers": ["MODEL_PRODUCTION_CHANNEL_UNBOUND:IRIS"]}
        report = build_report(local=local, platform=platform)
        self.assertFalse(report["exact8_product_ready"])
        self.assertIn("MODEL_PRODUCTION_CHANNEL_UNBOUND:IRIS", report["blockers"])

    def test_env_parser_accepts_export_and_quotes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text(
                "# comment\nexport A=one\nB=\"two words\"\nC='three'\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_env_file(path), {"A": "one", "B": "two words", "C": "three"})


if __name__ == "__main__":
    unittest.main()
