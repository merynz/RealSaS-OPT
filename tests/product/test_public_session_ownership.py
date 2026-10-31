from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from product.living_compile.common import LivingCompileError
from product.living_compile import public_server


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, _limit: int) -> bytes:
        return b'{"status":"ok"}'


class PublicSessionOwnershipTests(unittest.TestCase):
    def test_cookie_secret_is_high_entropy_and_platform_receives_only_digest(self):
        secret = "ab" * 32
        digest = public_server.owner_digest(secret)
        self.assertEqual(len(digest), 64)
        self.assertNotEqual(secret, digest)

        app = public_server.PublicLivingCompileApplication.default()
        env = {
            "REALSAS_PRODUCT_GATEWAY_URL": "https://platform.example",
            "REALSAS_PRODUCT_GATEWAY_TOKEN": "service-token",
        }
        token = public_server._current_owner_sha256.set(digest)
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _Response()

        try:
            with patch.dict(os.environ, env, clear=False), patch(
                "urllib.request.urlopen", side_effect=fake_urlopen
            ):
                status, raw = app._gateway_request(
                    "GET",
                    "/v1/product/compiles/cmp_" + "c" * 32,
                    accept="application/json",
                    max_bytes=1024,
                )
        finally:
            public_server._current_owner_sha256.reset(token)

        self.assertEqual(status, 200)
        self.assertEqual(raw, b'{"status":"ok"}')
        headers = {key.lower(): value for key, value in captured["request"].header_items()}
        self.assertEqual(headers["x-realsas-product-owner"], digest)
        self.assertEqual(headers["authorization"], "Bearer service-token")
        self.assertNotIn(secret, "\n".join(headers.values()))

    def test_gateway_fails_closed_without_owner_context(self):
        app = public_server.PublicLivingCompileApplication.default()
        env = {
            "REALSAS_PRODUCT_GATEWAY_URL": "https://platform.example",
            "REALSAS_PRODUCT_GATEWAY_TOKEN": "service-token",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaisesRegex(LivingCompileError, "owner context is not bound"):
                app._gateway_request(
                    "GET",
                    "/v1/product/compiles/cmp_" + "c" * 32,
                    accept="application/json",
                    max_bytes=1024,
                )

    def test_cookie_parser_reuses_valid_secret_and_rotates_invalid_cookie(self):
        secret = "12" * 32
        parsed, fresh = public_server.parse_owner_cookie(
            f"other=x; {public_server.PRODUCT_SESSION_COOKIE}={secret}"
        )
        self.assertEqual(parsed, secret)
        self.assertFalse(fresh)

        parsed, fresh = public_server.parse_owner_cookie(
            f"{public_server.PRODUCT_SESSION_COOKIE}=not-valid"
        )
        self.assertTrue(fresh)
        self.assertRegex(parsed, r"^[0-9a-f]{64}$")
        self.assertNotEqual(parsed, "not-valid")

    def test_cookie_header_is_httponly_same_site_and_secure_when_requested(self):
        secret = "34" * 32
        local = public_server.session_cookie_header(secret, secure=False)
        self.assertIn("HttpOnly", local)
        self.assertIn("SameSite=Lax", local)
        self.assertNotIn("; Secure", local)

        production = public_server.session_cookie_header(secret, secure=True)
        self.assertIn("HttpOnly", production)
        self.assertIn("SameSite=Lax", production)
        self.assertIn("; Secure", production)


if __name__ == "__main__":
    unittest.main()
