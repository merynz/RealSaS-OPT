"""Loopback source viewer. Cached views never authorize execution or reuse."""

from __future__ import annotations

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from threading import Lock

from tools.architecture_contracts import compare_maps

SUFFIXES = {
    ".py",
    ".go",
    ".mod",
    ".sum",
    ".cpp",
    ".cc",
    ".c",
    ".hpp",
    ".h",
    ".cu",
    ".cuh",
    ".html",
    ".yml",
    ".yaml",
    ".md",
    ".toml",
}


def checkout_token(root: Path, extra_paths=()) -> str:
    """Content-based invalidation includes new/deleted sources, HEAD and dirtiness."""
    digest = hashlib.sha256()
    for args in [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain", "--untracked-files=all"),
    ]:
        digest.update(subprocess.check_output(["git", *args], cwd=root))
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    extra = set(extra_paths)
    for rel in sorted((set(paths) | extra) - {""}):
        path = root / rel
        if (
            path.suffix not in SUFFIXES
            and rel != "canonical/MAINLINE_EXECUTION_PLAN_V2.json"
            and rel not in extra
        ):
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"DELETED")
    return digest.hexdigest()


class LiveMap:
    def __init__(self, root: Path, generate=None):
        self.root = root
        self.generate = generate or self._generate
        self.lock = Lock()
        self.token = None
        self.last_good = None
        self.response = None

    def _generate(self):
        # Fresh process reloads the generator, adapter AST, Python closure and Go
        # registry after edits; no long-lived imported plan becomes authority.
        result = subprocess.run(
            [sys.executable, str(self.root / "tools/realsas_architecture.py"), "map"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            detail = result.stderr[-4000:]
            if not detail and result.stdout:
                try:
                    audit = json.loads(result.stdout).get("audit", {})
                    detail = json.dumps(
                        {
                            k: audit.get(k)
                            for k in ("missing_critical_files", "missing_module_paths")
                        }
                    )
                except ValueError:
                    pass
            raise RuntimeError(detail or f"map exited {result.returncode}")
        return json.loads(result.stdout)

    def read(self):
        with self.lock:
            extra = [
                r["path"]
                for r in (self.last_good or {})
                .get("source", {})
                .get("fingerprinted_files", [])
            ]
            token = checkout_token(self.root, extra)
            if token == self.token:
                return self.response
            try:
                current = self.generate()
                if checkout_token(self.root, extra) != token:
                    raise RuntimeError(
                        "Checkout changed during generation; refresh to retry."
                    )
                changes = (
                    compare_maps(self.last_good, current) if self.last_good else None
                )
                self.last_good = current
                response = {
                    "status": "CURRENT",
                    "map": current,
                    "changes": changes,
                    "error": None,
                }
            except Exception as exc:
                response = {
                    "status": "STALE" if self.last_good else "UNAVAILABLE",
                    "map": self.last_good,
                    "changes": None,
                    "error": str(exc),
                }
                # Do not cache failures: transient generation errors retry on refresh.
                return response
            current_paths = [
                r["path"]
                for r in current.get("source", {}).get("fingerprinted_files", [])
            ]
            self.token, self.response = (
                checkout_token(self.root, current_paths),
                response,
            )
            return response


def handler_for(view: LiveMap):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                body = (
                    (view.root / "tools/architecture_map.html")
                    .read_text()
                    .replace("__ARCHITECTURE_DATA__", "null")
                    .encode()
                )
                mime = "text/html; charset=utf-8"
            elif self.path == "/api/map":
                body = json.dumps(view.read()).encode()
                mime = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def serve(root: Path, port: int):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(LiveMap(root)))
    print(
        f"Live architecture: http://127.0.0.1:{server.server_port} (source inspection only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
