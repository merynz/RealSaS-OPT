from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .authoring import save_user_layer
from .common import Json, LivingCompileError, bundle_path, finite, product_bundle_root, within
from .runtime import runtime_clips, runtime_frame
from .scene import build_scene


PUBLIC_HTTP_ENV = "REALSAS_PUBLIC_HTTP"


@dataclass
class LivingCompileApplication:
    static_root: Path

    @classmethod
    def default(cls) -> "LivingCompileApplication":
        return cls(Path(__file__).resolve().parent / "static")

    def static_bytes(self, request_path: str) -> tuple[bytes, str]:
        if request_path == "/":
            relative = "index.html"
        else:
            relative = posixpath.normpath(request_path.lstrip("/"))
            if relative.startswith("../"):
                raise LivingCompileError("unsafe static path")
        path = within(self.static_root, relative)
        if not path.is_file():
            raise LivingCompileError(f"static asset not found: {relative}")
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def project_capabilities(self) -> Json:
        bundle_root = product_bundle_root()
        return {
            "schema_version": "RealSaS.ProductCapabilities.v2",
            "product_name": "RealSaS",
            "mode": "OPEN_EXISTING_COMPILED_CHARACTER",
            "service_ready": True,
            "compile_available": False,
            "compile_blockers": [
                "PRODUCTION_MODEL_RELEASES_PENDING",
                "PRODUCT_PROOF_CAPABILITY_BINDING_PENDING",
                "PRODUCTREVISION_CAPABILITY_FINALIZATION_PENDING",
            ],
            "product_roles": [
                "INPUT",
                "SHAPE",
                "MESH",
                "RIG",
                "SKIN",
                "MECHANICS",
                "GRAPHICS",
                "MOTION",
                "PROOF",
            ],
            "bundle_access": {
                "configured_root": bundle_root is not None,
                "reference_kind": "ROOT_RELATIVE_BUNDLE_ID" if bundle_root is not None else "LEGACY_LOCAL_PATH",
            },
            "latency_targets": {
                "preview_seconds_max": 30,
                "market_bar_full_compile_seconds_max": 180,
                "canonical_goal_full_compile_seconds_max": 60,
            },
            "authority": {
                "existing_bundle_inspection": True,
                "user_edits_require_requalification": True,
                "runtime_preview_requires_proof_pass": True,
                "compile_from_ui": False,
                "heuristic_model_fallback_allowed": False,
            },
        }

    def dispatch(self, method: str, target: str, body: Json | None = None) -> tuple[int, Json | bytes, str]:
        parsed = urllib.parse.urlparse(target)
        query = urllib.parse.parse_qs(parsed.query)
        if method == "GET" and parsed.path == "/healthz":
            return 200, {"status": "ok", "service": "realsas-product-shell"}, "application/json"
        if method == "GET" and parsed.path == "/readyz":
            return 200, self.project_capabilities(), "application/json"
        if method == "GET" and (parsed.path == "/" or parsed.path.startswith("/static/")):
            data, content_type = self.static_bytes("/" if parsed.path == "/" else parsed.path[len("/static"):])
            return 200, data, content_type
        if method == "GET" and parsed.path == "/api/living/scene":
            return 200, build_scene(bundle_path((query.get("bundle") or [None])[0])), "application/json"
        if method == "GET" and parsed.path == "/api/runtime/clips":
            return 200, runtime_clips(bundle_path((query.get("bundle") or [None])[0])), "application/json"
        if method == "GET" and parsed.path == "/api/runtime/frame":
            root = bundle_path((query.get("bundle") or [None])[0])
            clip = (query.get("clip") or [""])[0]
            t = finite((query.get("time") or [0.0])[0], field="time")
            return 200, runtime_frame(root, clip, t), "application/json"
        if method == "GET" and parsed.path == "/api/blob":
            root = bundle_path((query.get("bundle") or [None])[0])
            rel = (query.get("path") or [""])[0]
            path = within(root, rel)
            if not path.is_file():
                raise LivingCompileError(f"bundle asset not found: {rel}")
            return 200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if method == "POST" and parsed.path == "/api/living/edit-layer":
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            return 200, save_user_layer(bundle_path(body.get("bundle")), body), "application/json"
        if method == "POST" and parsed.path == "/api/export":
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            root = bundle_path(body.get("bundle"))
            native = root / "native" / "realsas_runtime.rss"
            if not native.is_file():
                raise LivingCompileError("proof-gated native .rss package is not present in this output root")
            return 200, {
                "schema_version": "RealSaS.LivingCompileExportRef.v1",
                "package_path": str(native),
                "authority": "EXISTING_PROOF_GATED_NATIVE_RUNTIME",
            }, "application/json"
        if method == "GET" and parsed.path == "/api/project":
            return 200, self.project_capabilities(), "application/json"
        if method == "POST" and parsed.path in {"/api/input/upload", "/api/project", "/api/compile/start", "/api/compile"}:
            raise LivingCompileError(
                "production compile orchestration is intentionally fail-closed until PRODUCTION ModelReleases, product.proof and ProductRevision capability finalization are bound"
            )
        if method == "GET" and parsed.path == "/api/compile/status":
            raise LivingCompileError("no product-shell compile job exists yet; production capability orchestration is not bound")
        raise LivingCompileError(f"unsupported route: {method} {parsed.path}")


def make_handler(app: LivingCompileApplication):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RealSaS/1"

        def _handle(self) -> None:
            try:
                body = None
                if self.command in {"POST", "PUT", "PATCH"}:
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > 8 * 1024 * 1024:
                        raise LivingCompileError("request body exceeds 8 MiB")
                    raw = self.rfile.read(length) if length else b""
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                status, payload, content_type = app.dispatch(self.command, self.path, body)
            except LivingCompileError as exc:
                status, payload, content_type = 400, {"error": type(exc).__name__, "message": str(exc)}, "application/json"
            except Exception as exc:
                status, payload, content_type = 500, {"error": type(exc).__name__, "message": str(exc)}, "application/json"
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") or content_type == "application/json" else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        do_GET = _handle
        do_POST = _handle

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[realsas] {self.address_string()} - {fmt % args}")

    return Handler


def _is_loopback_bind(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    if not _is_loopback_bind(host) and os.environ.get(PUBLIC_HTTP_ENV, "").strip() != "1":
        raise LivingCompileError(
            f"refusing non-loopback bind without {PUBLIC_HTTP_ENV}=1; put RealSaS behind an authenticated TLS reverse proxy"
        )
    app = LivingCompileApplication.default()
    server = ThreadingHTTPServer((host, int(port)), make_handler(app))
    print(f"RealSaS: http://{host}:{port}")
    print("Product shell is ready. Compile remains fail-closed until production model/proof/revision bindings are complete.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSaS product/editor shell")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
