from __future__ import annotations

"""Single main-owned BUSTED Awaken/Studio HTTP entrypoint.

The static site and the current production-data API share a server implementation.
Historical Living Compile handlers remain the implementation of existing
proof-gated workspace reads and edits; no second animation/Compiler solver is
introduced here.

PUBLIC MODE is deliberately fail-closed: the server accepts only opaque IDs
inside its configured workspaces root. Compile remains OFFLINE until the
RSS.vNext canonical ProductRevision gate actually becomes executable.
"""
import argparse
from hashlib import sha256
from dataclasses import dataclass
import os
from pathlib import Path
import posixpath
import re
from zipfile import BadZipFile, ZipFile
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from product.living_compile.common import (
    Json, LivingCompileError, bundle_path, load_product, load_proof, within,
)
from product.living_compile.server import LivingCompileApplication, make_handler

_WORKSPACE_ID = re.compile(r"[A-Za-z0-9_-]{16,80}\Z")
_PAGES = frozenset((
    "/", "/index.html", "/studio.html", "/motion.html",
    "/export.html", "/runtime-config.js", "/favicon.svg",
    "/busted-mark.svg", "/robots.txt",
))
_WORKSPACE_READ = frozenset((
    "/api/living/scene", "/api/runtime/clips",
    "/api/runtime/frame", "/api/blob",
))
_WORKSPACE_WRITE = frozenset(("/api/living/edit-layer", "/api/export"))

_NATIVE_RSS_RELATIVE = "native/realsas_runtime.rss"
_MAX_NATIVE_RSS_BYTES = 64 * 1024 * 1024
_MAX_NATIVE_UNPACKED_BYTES = 512 * 1024 * 1024


def _qualified_native_rss_v1(value: str) -> tuple[bytes, str]:
    """Read the existing packaged runtime only after checking its V4 Proof.

    This does not upgrade legacy Proof to RSS.vNext ProductRevision authority;
    it only gates delivery of an already-packaged, supported V4 runtime.
    """
    root = bundle_path(value)
    product = load_product(root)
    proof = load_proof(root, product)
    if proof.get("overall_status") != "PASS":
        raise LivingCompileError("BUSTED_NATIVE_EXPORT_PROOF_PASS_REQUIRED")
    native = within(root, _NATIVE_RSS_RELATIVE)
    if not native.is_file():
        raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_MISSING")
    size = native.stat().st_size
    if not 1 <= size <= _MAX_NATIVE_RSS_BYTES:
        raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_SIZE_INVALID")
    try:
        with ZipFile(native, "r") as archive:
            entries = archive.infolist()
            if (
                not entries or "manifest.txt" not in {x.filename for x in entries}
                or sum(x.file_size for x in entries) > _MAX_NATIVE_UNPACKED_BYTES
                or any(x.flag_bits & 1 for x in entries)
            ):
                raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_ARCHIVE_INVALID")
            if archive.testzip() is not None:
                raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_CRC_INVALID")
    except (BadZipFile, OSError, ValueError) as exc:
        raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_DECODE_INVALID") from exc
    raw = native.read_bytes()
    if len(raw) != size:
        raise LivingCompileError("BUSTED_NATIVE_EXPORT_RSS_CHANGED_DURING_READ")
    return raw, sha256(raw).hexdigest()



@dataclass
class BustedApplication(LivingCompileApplication):
    """One static and API server; public bundle tokens never become raw paths."""

    workspace_root: Path | None = None
    public_origin: str = ""
    public_mode: bool = False

    @classmethod
    def default(cls) -> "BustedApplication":
        public_mode = os.environ.get("BUSTED_PUBLIC_MODE", "0") == "1"
        root_raw = os.environ.get("BUSTED_WORKSPACE_ROOT", "").strip()
        public_origin = os.environ.get("BUSTED_PUBLIC_ORIGIN", "").strip()
        workspace_root = Path(root_raw).resolve() if root_raw else None
        if public_mode:
            if workspace_root is None or not workspace_root.is_dir():
                raise LivingCompileError("BUSTED_PUBLIC_WORKSPACE_ROOT_REQUIRED")
            if not public_origin.startswith("https://"):
                raise LivingCompileError("BUSTED_PUBLIC_HTTPS_ORIGIN_REQUIRED")
            origin = urlsplit(public_origin)
            if (
                not origin.hostname or origin.path not in ("", "/")
                or origin.query or origin.fragment
                or origin.username or origin.password
            ):
                raise LivingCompileError("BUSTED_PUBLIC_ORIGIN_INVALID")
        return cls(Path(__file__).resolve().parent, workspace_root, public_origin, public_mode)

    def _bound_bundle(self, value: str | None) -> str:
        if not isinstance(value, str) or not _WORKSPACE_ID.fullmatch(value):
            raise LivingCompileError("BUSTED_PUBLIC_WORKSPACE_ID_INVALID")
        if self.workspace_root is None:
            raise LivingCompileError("BUSTED_PUBLIC_WORKSPACE_ROOT_REQUIRED")
        root = self.workspace_root.resolve()
        target = (root / value).resolve()
        if target.parent != root or not target.is_dir():
            raise LivingCompileError("BUSTED_PUBLIC_WORKSPACE_NOT_FOUND")
        return str(target)

    def _public_scope(
        self, method: str, target: str, body: Json | None
    ) -> tuple[str, Json | None, str | None]:
        parsed = urlsplit(target)
        route = parsed.path
        is_workspace = (
            (method == "GET" and route in _WORKSPACE_READ)
            or (method == "POST" and route in _WORKSPACE_WRITE)
        )
        if not is_workspace:
            return target, body, None
        if method == "GET":
            query = parse_qs(parsed.query, keep_blank_values=True)
            provided = query.get("bundle") or []
            if len(provided) != 1:
                raise LivingCompileError("BUSTED_PUBLIC_WORKSPACE_TOKEN_REQUIRED")
            token = provided[0]
            query["bundle"] = [self._bound_bundle(token)]
            sanitized = urlunsplit((
                parsed.scheme, parsed.netloc, parsed.path,
                urlencode(query, doseq=True), parsed.fragment,
            ))
            return sanitized, body, token
        if not isinstance(body, dict):
            raise LivingCompileError("BUSTED_PUBLIC_JSON_BODY_REQUIRED")
        token = body.get("bundle")
        resolved = self._bound_bundle(token)
        return target, {**body, "bundle": resolved}, token

    def dispatch(
        self, method: str, target: str, body: Json | None = None
    ) -> tuple[int, Json | bytes, str]:
        parsed = urlsplit(target)
        # Source of the 91-file frontend is main: no preview branch is required
        # for the product server. This is the same BUSTED UI deployed on Pages.
        if method == "GET" and (
            parsed.path in _PAGES or parsed.path.startswith("/static/")
        ):
            if parsed.path == "/":
                relative = "/index.html"
            else:
                relative = posixpath.normpath(parsed.path)
            data, content_type = self.static_bytes(relative)
            return 200, data, content_type

        if parsed.path == "/api/project" and method == "GET":
            status, payload, mime = super().dispatch(method, target, body)
            if not isinstance(payload, dict):
                raise LivingCompileError("BUSTED_PRODUCT_CAPABILITY_SCHEMA_DRIFT")
            # An untrained or unqualified Compiler is NEVER reported READY.
            return status, {
                **payload,
                "product_name": "BUSTED",
                "public_compile_gateway": False,
                "compile_available": False,
                "compile_blockers": [
                    "RSS_VNEXT_PRODUCTREVISION_INFERENCE_TO_COMPILER_BRIDGE_OPEN",
                ],
                "website_owner": "MAIN_PRODUCT_BUSTED_SITE",
                "input_policy": "TWO_OBSERVED_VIEWS_FRONT_BACK",
            }, mime

        rewritten, body, token = (
            self._public_scope(method, target, body)
            if self.public_mode else (target, body, None)
        )
        if method == "POST" and parsed.path == "/api/export":
            if not isinstance(body, dict):
                raise LivingCompileError("BUSTED_NATIVE_EXPORT_BODY_REQUIRED")
            raw, digest = _qualified_native_rss_v1(str(body.get("bundle") or ""))
            download_identity = token if token is not None else str(body["bundle"])
            download_url = "/api/blob?" + urlencode({
                "bundle": download_identity,
                "path": _NATIVE_RSS_RELATIVE,
            })
            return 200, {
                "schema_version": "RealSaS.BustedQualifiedNativeDownload.v1",
                "authority": "EXISTING_PROOF_GATED_NATIVE_RUNTIME",
                "format": "rss",
                "download_url": download_url,
                "content_sha256": digest,
                "size_bytes": len(raw),
                "edited_studio_state_included": False,
            }, "application/json"
        if method == "GET" and parsed.path == "/api/blob":
            query = parse_qs(urlsplit(rewritten).query, keep_blank_values=True)
            roots, paths = query.get("bundle") or [], query.get("path") or []
            if len(roots) == 1 and len(paths) == 1:
                root = bundle_path(roots[0])
                requested = within(root, paths[0])
                native = within(root, _NATIVE_RSS_RELATIVE)
                if requested == native:
                    raw, _ = _qualified_native_rss_v1(str(root))
                    return 200, raw, "application/octet-stream"
        status, payload, mime = super().dispatch(method, rewritten, body)
        if token is not None and isinstance(payload, dict):
            payload = dict(payload)
            if "bundle_root" in payload:
                payload["bundle_root"] = token
            if "path" in payload and str(payload["path"]).startswith(str(self.workspace_root)):
                # The user's library path is an internal address. It must not
                # be disclosed as a public download path.
                payload.pop("path")
        return status, payload, mime


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    app = BustedApplication.default()
    if host not in ("127.0.0.1", "::1", "localhost") and not app.public_mode:
        raise LivingCompileError("BUSTED_NON_LOOPBACK_REQUIRES_PUBLIC_MODE")
    server = ThreadingHTTPServer((host, int(port)), make_handler(app))
    print(f"BUSTED main product server: http://{host}:{port}")
    print("RSS.vNext public compile remains fail-closed until qualified ProductRevision wiring passes")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BUSTED main product frontend and API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
