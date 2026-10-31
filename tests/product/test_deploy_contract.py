from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy" / "product"


class ProductDeployContractTests(unittest.TestCase):
    def test_public_surface_is_loopback_only_and_platform_is_not_published(self):
        compose = (DEPLOY / "compose.yaml").read_text(encoding="utf-8")
        self.assertIn('"127.0.0.1:${REALSAS_WEB_PORT:-8765}:8765"', compose)
        self.assertIn('expose: ["8080"]', compose)
        self.assertNotIn('"8080:8080"', compose)
        self.assertNotIn('"0.0.0.0:${REALSAS_WEB_PORT', compose)

    def test_compile_is_disabled_by_default_and_learned_execution_is_explicit(self):
        compose = (DEPLOY / "compose.yaml").read_text(encoding="utf-8")
        env_example = (DEPLOY / ".env.example").read_text(encoding="utf-8")
        self.assertIn("REALSAS_PRODUCT_COMPILE_ENABLED: ${REALSAS_PRODUCT_COMPILE_ENABLED:-0}", compose)
        self.assertIn("profiles: [\"workers\"]", compose)
        self.assertIn("profiles: [\"engine\"]", compose)
        self.assertIn("REALSAS_PRODUCT_COMPILE_ENABLED=0", env_example)
        self.assertIn("REALSAS_ENGINE_IMAGE=realsas-engine:unbound", env_example)

    def test_platform_image_binds_exact_source_identity(self):
        dockerfile = (DEPLOY / "Dockerfile.platform").read_text(encoding="utf-8")
        self.assertIn("ARG REALSAS_SOURCE_CODE_SHA", dockerfile)
        self.assertIn("test \"${#REALSAS_SOURCE_CODE_SHA}\" -eq 40", dockerfile)
        self.assertIn("-X main.sourceCodeSHA=${REALSAS_SOURCE_CODE_SHA}", dockerfile)
        self.assertIn("USER realsas", dockerfile)

    def test_product_image_runs_unprivileged_and_keeps_workspace_persistent(self):
        dockerfile = (DEPLOY / "Dockerfile.product").read_text(encoding="utf-8")
        self.assertIn("USER realsas", dockerfile)
        self.assertIn("REALSAS_PRODUCT_BUNDLE_ROOT=/var/lib/realsas/bundles", dockerfile)
        self.assertIn('VOLUME ["/var/lib/realsas/bundles"]', dockerfile)

    def test_example_configuration_contains_no_live_domain_or_enabled_compile(self):
        env_example = (DEPLOY / ".env.example").read_text(encoding="utf-8")
        self.assertNotIn("parilte.com", env_example)
        self.assertNotIn("REALSAS_PRODUCT_COMPILE_ENABLED=1", env_example)
        self.assertIn("replace-with-random-product-gateway-token", env_example)
        self.assertIn("replace-with-random-operator-token", env_example)


if __name__ == "__main__":
    unittest.main()
