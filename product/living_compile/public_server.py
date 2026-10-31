from __future__ import annotations

import argparse
import contextvars
import hashlib
import http.cookies
import json
import os
import re
import secrets
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .common import LivingCompileError
from .server import (
    PRODUCT_GATEWAY_TIMEOUT_SECONDS,
    PUBLIC_HTTP_ENV,
    LivingCompileApplication,
    ProductGatewayHTTPError,
    _is_loopback_bind,
)

PRODUCT_SESSION_COOKIE = "realsas_product_session"
PRODUCT_COOKIE_SECURE_ENV = "REALSAS_PRODUCT_COOKIE_SECURE"
PRODUCT_SESSION_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
_OWNER_SECRET_RE = re.compile(r"^[0-9a-f]{64}$")
_current_owner_sha256: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "realsas_product_owner_sha256", default=None
)


def new_owner_secret() -> str:
    return secrets.token_hex(32)


def owner_digest(secret: str) -> str:
    if not _OWNER_SECRET_RE.fullmatch(secret):
        raise LivingCompileError("invalid product session secret")
    return hashlib.sha256(bytes.fromhex(secret)).hexdigest()


def parse_owner_cookie(raw_cookie: str | None) -> tuple[str, bool]:
    if raw_cookie:
        jar = http.cookies.SimpleCookie()
        try:
            jar.load(raw_cookie)
        except http.cookies.CookieError:
            jar = http.cookies.SimpleCookie()
        morsel = jar.get(PRODUCT_SESSION_COOKIE)
        if morsel is not None:
            candidate = str(morsel.value or "").strip().lower()
            if _OWNER_SECRET_RE.fullmatch(candidate):
                return candidate, False
    return new_owner_secret(), True


def session_cookie_header(secret: str, *, secure: bool) -> str:
    if not _OWNER_SECRET_RE.fullmatch(secret):
        raise LivingCompileError("invalid product session secret")
    parts = [
        f"{PRODUCT_SESSION_COOKIE}={secret}",
        "Path=/",
        f"Max-Age={PRODUCT_SESSION_MAX_AGE_SECONDS}",
        "HttpOnly",
        "SameSite=Lax",
    ]
    if secure:
        parts.append("Secure")
    return "; ".join(parts)


class PublicLivingCompileApplication(LivingCompileApplication):
    """Production HTTP variant that scopes every Platform product call.

    The browser keeps a random HttpOnly bearer secret. Only its SHA-256 digest
    crosses the product-shell -> Platform service boundary. Platform persists
    that digest on the public compile and enforces it on compile/workspace/Studio
    reads and writes. The service bearer token remains independent and server-side.
    """

    def _gateway_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        accept: str,
        content_type: str | None = None,
        max_bytes: int,
    ) -> tuple[int, bytes]:
        base = self._gateway_base()
        if base is None:
            raise LivingCompileError("public product gateway is not configured")
        token = self._gateway_token()
        if not token:
            raise LivingCompileError("public product gateway service token is not configured")
        owner = _current_owner_sha256.get()
        if not owner or not re.fullmatch(r"[0-9a-f]{64}", owner):
            raise LivingCompileError("public product owner context is not bound")

        headers = {
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-RealSaS-Product-Owner": owner,
        }
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(base + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=PRODUCT_GATEWAY_TIMEOUT_SECONDS) as response:
                status = int(response.status)
                raw = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raw = exc.read(1024 * 1024)
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                payload = {}
            message = payload.get("message") if isinstance(payload, dict) else None
            error = payload.get("error") if isinstance(payload, dict) else None
            raise ProductGatewayHTTPError(
                int(exc.code), message or error or f"product gateway rejected request with HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LivingCompileError(f"product gateway unavailable: {exc.reason}") from exc
        if len(raw) > max_bytes:
            raise LivingCompileError("product gateway response exceeds allowed size")
        return status, raw


def make_public_handler(app: PublicLivingCompileApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RealSaS/1"

        def _cookie_secure(self) -> bool:
            if os.environ.get(PRODUCT_COOKIE_SECURE_ENV, "").strip() == "1":
                return True
            forwarded = self.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip().lower()
            return forwarded == "https"

        def _handle(self) -> None:
            secret, fresh_cookie = parse_owner_cookie(self.headers.get("Cookie"))
            context_token = _current_owner_sha256.set(owner_digest(secret))
            try:
                try:
                    body = None
                    if self.command in {"POST", "PUT", "PATCH"}:
                        length = int(self.headers.get("Content-Length") or 0)
                        if length > 8 * 1024 * 1024:
                            raise LivingCompileError("request body exceeds 8 MiB")
                        raw = self.rfile.read(length) if length else b""
                        body = json.loads(raw.decode("utf-8")) if raw else {}
                    status, payload, content_type = app.dispatch(self.command, self.path, body)
                except ProductGatewayHTTPError as exc:
                    status, payload, content_type = (
                        exc.status,
                        {"error": type(exc).__name__, "message": str(exc)},
                        "application/json",
                    )
                except LivingCompileError as exc:
                    status, payload, content_type = (
                        400,
                        {"error": type(exc).__name__, "message": str(exc)},
                        "application/json",
                    )
                except Exception as exc:
                    status, payload, content_type = (
                        500,
                        {"error": type(exc).__name__, "message": str(exc)},
                        "application/json",
                    )

                data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header(
                    "Content-Type",
                    content_type
                    + (
                        "; charset=utf-8"
                        if content_type.startswith("text/") or content_type == "application/json"
                        else ""
                    ),
                )
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                )
                if fresh_cookie:
                    self.send_header(
                        "Set-Cookie",
                        session_cookie_header(secret, secure=self._cookie_secure()),
                    )
                self.end_headers()
                self.wfile.write(data)
            finally:
                _current_owner_sha256.reset(context_token)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[realsas] {self.address_string()} - {fmt % args}")

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if not _is_loopback_bind(host) and os.environ.get(PUBLIC_HTTP_ENV, "").strip() != "1":
        raise LivingCompileError(
            f"refusing non-loopback bind without {PUBLIC_HTTP_ENV}=1; put RealSaS behind an authenticated TLS reverse proxy"
        )
    app = PublicLivingCompileApplication.default()
    server = ThreadingHTTPServer((host, int(port)), make_public_handler(app))
    print(f"RealSaS: http://{host}:{port}")
    print("Public product ownership is bound to an HttpOnly browser session; Platform receives only its SHA-256 digest.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSaS owner-scoped public product shell")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
