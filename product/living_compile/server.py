from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .authoring import save_user_layer
from .common import Json, LivingCompileError, bundle_path, finite, product_bundle_root, within
from .runtime import runtime_clips, runtime_frame
from .scene import build_scene
from .workspace_cache import materialize_workspace_bundle


PUBLIC_HTTP_ENV = "REALSAS_PUBLIC_HTTP"
PRODUCT_GATEWAY_URL_ENV = "REALSAS_PRODUCT_GATEWAY_URL"
PRODUCT_GATEWAY_TOKEN_ENV = "REALSAS_PRODUCT_GATEWAY_TOKEN"
PRODUCT_COMPILE_ENABLED_ENV = "REALSAS_PRODUCT_COMPILE_ENABLED"
PRODUCT_GATEWAY_TIMEOUT_SECONDS = 12
MAX_WORKSPACE_BUNDLE_DOWNLOAD_BYTES = 256 * 1024 * 1024
_PUBLIC_BUNDLE = re.compile(r"^public/(cmp_[0-9a-f]{32})/[0-9a-f]{64}$")
_PUBLIC_STUDIO_PROJECT = re.compile(r"^stp_[0-9a-f]{32}$")
_PUBLIC_STUDIO_REQUALIFICATION = re.compile(r"^srq_[0-9a-f]{32}$")

_INTERNAL_KEYS = {
    "artifact_id",
    "artifact_ids",
    "attempt_id",
    "authoring_revision_id",
    "command_id",
    "engine_release_id",
    "execution_goal_id",
    "input_artifact_ids",
    "model_release_id",
    "model_release_ids",
    "product_revision_id",
    "session_id",
    "source_compile_session_id",
    "source_product_revision_id",
}


class ProductGatewayHTTPError(LivingCompileError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)


@dataclass
class LivingCompileApplication:
    static_root: Path

    @classmethod
    def default(cls) -> "LivingCompileApplication":
        return cls(Path(__file__).resolve().parent / "static")

    def static_bytes(self, request_path: str) -> tuple[bytes, str]:
        if request_path == "/":
            relative = "index.html"
        elif request_path in {"/workspace", "/workspace/", "/studio", "/studio/"}:
            relative = "workspace.html"
        else:
            relative = posixpath.normpath(request_path.lstrip("/"))
            if relative.startswith("../"):
                raise LivingCompileError("unsafe static path")
        path = within(self.static_root, relative)
        if not path.is_file():
            raise LivingCompileError(f"static asset not found: {relative}")
        return path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    def _gateway_base(self) -> str | None:
        raw = os.environ.get(PRODUCT_GATEWAY_URL_ENV, "").strip().rstrip("/")
        if not raw:
            return None
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise LivingCompileError(f"{PRODUCT_GATEWAY_URL_ENV} must be an absolute http(s) URL")
        return raw

    def _gateway_token(self) -> str:
        return os.environ.get(PRODUCT_GATEWAY_TOKEN_ENV, "").strip()

    def _production_compile_enabled(self) -> bool:
        return (
            self._gateway_base() is not None
            and bool(self._gateway_token())
            and os.environ.get(PRODUCT_COMPILE_ENABLED_ENV, "").strip() == "1"
            and product_bundle_root() is not None
        )

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
        headers = {"Accept": accept, "Authorization": f"Bearer {token}"}
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

    def _gateway_json(self, method: str, path: str, body: Json | None = None) -> tuple[int, Json]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        status, raw = self._gateway_request(
            method,
            path,
            body=data,
            accept="application/json",
            content_type="application/json" if data is not None else None,
            max_bytes=8 * 1024 * 1024,
        )
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:
            raise LivingCompileError("product gateway returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise LivingCompileError("product gateway returned a non-object response")
        return status, payload

    def _gateway_bytes(self, method: str, path: str, *, accept: str, max_bytes: int) -> tuple[int, bytes]:
        return self._gateway_request(method, path, accept=accept, max_bytes=max_bytes)

    def _public_payload(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: self._public_payload(item)
                for key, item in value.items()
                if str(key).lower() not in _INTERNAL_KEYS
            }
        if isinstance(value, list):
            return [self._public_payload(item) for item in value]
        return value

    def _compile_id_for_bundle(self, bundle: Any) -> str | None:
        value = str(bundle or "").strip().replace("\\", "/").strip("/")
        match = _PUBLIC_BUNDLE.fullmatch(value)
        return match.group(1) if match else None

    def _ensure_studio_session(self, compile_id: str) -> Json:
        if not self._production_compile_enabled():
            raise LivingCompileError("Studio persistence requires the production product gateway")
        safe_compile = urllib.parse.quote(compile_id, safe="-_.~")
        _, project = self._gateway_json(
            "POST",
            f"/v1/product/compiles/{safe_compile}/studio-project",
            {"created_by": "product-shell"},
        )
        project_id = str(project.get("project_id") or "").strip()
        if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
            raise LivingCompileError("product gateway returned invalid Studio project id")
        safe_project = urllib.parse.quote(project_id, safe="-_.~")
        _, head = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}/head")
        if str(head.get("project_id") or "") != project_id or str(head.get("compile_id") or "") != compile_id:
            raise LivingCompileError("Studio project/head identity mismatch")
        return {
            "schema_version": "RealSaS.StudioSession.v1",
            "project": self._public_payload(project),
            "head": self._public_payload(head),
        }

    def _persist_studio_layer(
        self,
        compile_id: str,
        layer: Json,
        *,
        project_id: str | None = None,
        expected_lock_version: int | None = None,
    ) -> Json:
        if not self._production_compile_enabled():
            raise LivingCompileError("Studio persistence requires the production product gateway")
        if project_id is None:
            session = self._ensure_studio_session(compile_id)
            project = session["project"]
            project_id = str(project.get("project_id") or "").strip()
            expected_lock_version = int(session["head"].get("lock_version", 0))
        else:
            project_id = str(project_id).strip()
            if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
                raise LivingCompileError("valid public Studio project id required")
            if expected_lock_version is None or int(expected_lock_version) < 0:
                raise LivingCompileError("Studio session lock version required")
            safe_project = urllib.parse.quote(project_id, safe="-_.~")
            _, project = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}")
            if str(project.get("compile_id") or "") != compile_id:
                raise LivingCompileError("Studio project does not belong to this compiled character")
        if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
            raise LivingCompileError("product gateway returned invalid Studio project id")
        safe_project = urllib.parse.quote(project_id, safe="-_.~")
        _, revision = self._gateway_json(
            "POST",
            f"/v1/product/studio/projects/{safe_project}/revisions",
            {
                "expected_lock_version": int(expected_lock_version or 0),
                "authoring_payload": layer,
                "created_by": "product-shell",
            },
        )
        return self._public_payload(revision)

    def _restore_studio_revision(
        self,
        compile_id: str,
        project_id: str,
        expected_lock_version: int,
        revision_number: int,
    ) -> Json:
        if not self._production_compile_enabled():
            raise LivingCompileError("Studio persistence requires the production product gateway")
        project_id = str(project_id).strip()
        if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
            raise LivingCompileError("valid public Studio project id required")
        if int(expected_lock_version) < 0 or int(revision_number) <= 0:
            raise LivingCompileError("valid Studio restore lock/revision required")
        safe_project = urllib.parse.quote(project_id, safe="-_.~")
        _, project = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}")
        if str(project.get("compile_id") or "") != compile_id:
            raise LivingCompileError("Studio project does not belong to this compiled character")
        _, revision = self._gateway_json(
            "POST",
            f"/v1/product/studio/projects/{safe_project}/restore",
            {
                "expected_lock_version": int(expected_lock_version),
                "revision_number": int(revision_number),
                "created_by": "product-shell",
            },
        )
        _, head = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}/head")
        if str(head.get("project_id") or "") != project_id or str(head.get("compile_id") or "") != compile_id:
            raise LivingCompileError("Studio restored head identity mismatch")
        return {
            "schema_version": "RealSaS.StudioRestore.v1",
            "revision": self._public_payload(revision),
            "head": self._public_payload(head),
        }

    def _request_studio_requalification(
        self,
        compile_id: str,
        project_id: str,
        expected_lock_version: int,
        revision_number: int,
    ) -> Json:
        if not self._production_compile_enabled():
            raise LivingCompileError("Studio requalification requires the production product gateway")
        project_id = str(project_id).strip()
        if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
            raise LivingCompileError("valid public Studio project id required")
        if int(expected_lock_version) < 0 or int(revision_number) <= 0:
            raise LivingCompileError("valid saved Studio revision required")
        safe_project = urllib.parse.quote(project_id, safe="-_.~")
        _, project = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}")
        if str(project.get("compile_id") or "") != compile_id:
            raise LivingCompileError("Studio project does not belong to this compiled character")
        _, payload = self._gateway_json(
            "POST",
            f"/v1/product/studio/projects/{safe_project}/requalifications",
            {
                "expected_lock_version": int(expected_lock_version),
                "revision_number": int(revision_number),
                "created_by": "product-shell",
            },
        )
        requalification_id = str(payload.get("requalification_id") or "").strip()
        if not _PUBLIC_STUDIO_REQUALIFICATION.fullmatch(requalification_id):
            raise LivingCompileError("product gateway returned invalid Studio requalification id")
        if str(payload.get("project_id") or "") != project_id:
            raise LivingCompileError("Studio requalification project identity mismatch")
        return self._public_payload(payload)

    def _studio_requalification_status(self, project_id: str, requalification_id: str) -> Json:
        if not self._production_compile_enabled():
            raise LivingCompileError("Studio requalification requires the production product gateway")
        project_id = str(project_id).strip()
        requalification_id = str(requalification_id).strip()
        if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
            raise LivingCompileError("valid public Studio project id required")
        if not _PUBLIC_STUDIO_REQUALIFICATION.fullmatch(requalification_id):
            raise LivingCompileError("valid public Studio requalification id required")
        safe_project = urllib.parse.quote(project_id, safe="-_.~")
        safe_requalification = urllib.parse.quote(requalification_id, safe="-_.~")
        _, payload = self._gateway_json(
            "GET",
            f"/v1/product/studio/projects/{safe_project}/requalifications/{safe_requalification}",
        )
        if str(payload.get("requalification_id") or "") != requalification_id or str(payload.get("project_id") or "") != project_id:
            raise LivingCompileError("Studio requalification status identity mismatch")
        return self._public_payload(payload)

    def project_capabilities(self) -> Json:
        bundle_root = product_bundle_root()
        gateway_configured = self._gateway_base() is not None
        gateway_token_configured = bool(self._gateway_token())
        compile_requested = os.environ.get(PRODUCT_COMPILE_ENABLED_ENV, "").strip() == "1"
        compile_available = self._production_compile_enabled()
        studio_requalification_available = compile_available
        blockers: list[str] = []
        if not gateway_configured:
            blockers.append("PUBLIC_PRODUCT_GATEWAY_NOT_CONFIGURED")
        if not gateway_token_configured:
            blockers.append("PUBLIC_PRODUCT_GATEWAY_TOKEN_NOT_CONFIGURED")
        if not compile_requested:
            blockers.append("PRODUCT_COMPILE_NOT_ENABLED")
        if bundle_root is None:
            blockers.append("PRODUCT_BUNDLE_ROOT_NOT_CONFIGURED")
        return {
            "schema_version": "RealSaS.ProductCapabilities.v3",
            "product_name": "RealSaS",
            "mode": "AWAKEN_AND_EDIT_COMPILED_CHARACTER",
            "service_ready": True,
            "compile_available": compile_available,
            "public_compile_gateway": gateway_configured and gateway_token_configured,
            "studio_persistence": gateway_configured and gateway_token_configured,
            "studio_requalification_executor": studio_requalification_available,
            "studio_requalification": {
                "available": studio_requalification_available,
                "supported_slice": "ANIMATION_LINEAR_V1",
                "supported_edit_domains": ["animation"],
                "max_animation_clips": 1,
                "supported_curves": ["linear"],
                "result_authority": "NEW_PRODUCT_REVISION_AFTER_PROOF_PASS",
                "result_workspace": "NEW_CLEAN_PROOF_GATED_WORKSPACE",
                "base_product_revision_mutated": False,
            },
            "compile_blockers": blockers,
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
                "compile_from_ui": compile_available,
                "browser_receives_platform_internal_ids": False,
                "browser_receives_gateway_service_token": False,
                "heuristic_model_fallback_allowed": False,
                "presentation_preview_has_product_authority": False,
                "studio_edits_mutate_product_revision": False,
                "studio_requalification_requires_saved_revision": True,
                "studio_requalification_requires_proof_pass": True,
            },
        }

    def dispatch(self, method: str, target: str, body: Json | None = None) -> tuple[int, Json | bytes, str]:
        parsed = urllib.parse.urlparse(target)
        query = urllib.parse.parse_qs(parsed.query)
        if method == "GET" and parsed.path == "/healthz":
            return 200, {"status": "ok", "service": "realsas-product-shell"}, "application/json"
        if method == "GET" and parsed.path == "/readyz":
            return 200, self.project_capabilities(), "application/json"
        if method == "GET" and (
            parsed.path in {"/", "/workspace", "/workspace/", "/studio", "/studio/"}
            or parsed.path.startswith("/static/")
        ):
            path = (
                parsed.path
                if parsed.path in {"/", "/workspace", "/workspace/", "/studio", "/studio/"}
                else parsed.path[len("/static"):]
            )
            data, content_type = self.static_bytes(path)
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
        if method == "GET" and parsed.path == "/api/studio/session":
            if not self._production_compile_enabled():
                raise LivingCompileError("Studio persistence is fail-closed")
            bundle_ref = (query.get("bundle") or [""])[0]
            compile_id = self._compile_id_for_bundle(bundle_ref)
            if compile_id is None:
                raise LivingCompileError("Studio session requires a public proof-gated bundle")
            return 200, self._ensure_studio_session(compile_id), "application/json"
        if method == "GET" and parsed.path == "/api/studio/history":
            if not self._production_compile_enabled():
                raise LivingCompileError("Studio persistence is fail-closed")
            project_id = (query.get("id") or [""])[0].strip()
            if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
                raise LivingCompileError("valid public Studio project id required")
            raw_limit = (query.get("limit") or ["50"])[0]
            try:
                limit = max(1, min(200, int(raw_limit)))
            except (TypeError, ValueError) as exc:
                raise LivingCompileError("valid Studio revision history limit required") from exc
            safe_project = urllib.parse.quote(project_id, safe="-_.~")
            status, payload = self._gateway_json(
                "GET", f"/v1/product/studio/projects/{safe_project}/revisions?limit={limit}"
            )
            return status, self._public_payload(payload), "application/json"
        if method == "POST" and parsed.path == "/api/studio/restore":
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            bundle_ref = body.get("bundle")
            compile_id = self._compile_id_for_bundle(bundle_ref)
            if compile_id is None:
                raise LivingCompileError("Studio restore requires a public proof-gated bundle")
            project_id = str(body.get("project_id") or "").strip()
            try:
                expected_lock_version = int(body.get("expected_lock_version"))
                revision_number = int(body.get("revision_number"))
            except (TypeError, ValueError) as exc:
                raise LivingCompileError("valid Studio restore lock/revision required") from exc
            return 200, self._restore_studio_revision(
                compile_id,
                project_id,
                expected_lock_version,
                revision_number,
            ), "application/json"
        if method == "POST" and parsed.path == "/api/studio/requalify":
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            bundle_ref = body.get("bundle")
            compile_id = self._compile_id_for_bundle(bundle_ref)
            if compile_id is None:
                raise LivingCompileError("Studio requalification requires a public proof-gated bundle")
            project_id = str(body.get("project_id") or "").strip()
            try:
                expected_lock_version = int(body.get("expected_lock_version"))
                revision_number = int(body.get("revision_number"))
            except (TypeError, ValueError) as exc:
                raise LivingCompileError("valid saved Studio revision required") from exc
            payload = self._request_studio_requalification(
                compile_id,
                project_id,
                expected_lock_version,
                revision_number,
            )
            return 202, payload, "application/json"
        if method == "GET" and parsed.path == "/api/studio/requalification":
            project_id = (query.get("project") or [""])[0].strip()
            requalification_id = (query.get("id") or [""])[0].strip()
            return 200, self._studio_requalification_status(project_id, requalification_id), "application/json"
        if method == "POST" and parsed.path == "/api/living/edit-layer":
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            bundle_ref = body.get("bundle")
            saved = save_user_layer(bundle_path(bundle_ref), body)
            saved["persistence"] = "LOCAL_AUTHORING_CACHE"
            compile_id = self._compile_id_for_bundle(bundle_ref)
            if compile_id is not None and self._production_compile_enabled():
                layer_path = Path(str(saved.get("path") or ""))
                if not layer_path.is_file():
                    raise LivingCompileError("validated Studio authoring layer was not materialized")
                layer = json.loads(layer_path.read_text(encoding="utf-8"))
                if not isinstance(layer, dict):
                    raise LivingCompileError("validated Studio authoring layer must be an object")
                project_id = body.get("studio_project_id")
                raw_lock = body.get("expected_lock_version")
                lock_version = None
                if project_id is not None:
                    try:
                        lock_version = int(raw_lock)
                    except (TypeError, ValueError) as exc:
                        raise LivingCompileError("Studio session lock version required") from exc
                saved["studio_revision"] = self._persist_studio_layer(
                    compile_id,
                    layer,
                    project_id=str(project_id) if project_id is not None else None,
                    expected_lock_version=lock_version,
                )
                saved["persistence"] = "PLATFORM_STUDIO_REVISION"
            return 200, saved, "application/json"
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
        if method == "GET" and parsed.path == "/api/studio/project":
            if not self._production_compile_enabled():
                raise LivingCompileError("Studio persistence is fail-closed")
            project_id = (query.get("id") or [""])[0].strip()
            if not _PUBLIC_STUDIO_PROJECT.fullmatch(project_id):
                raise LivingCompileError("valid public Studio project id required")
            safe_project = urllib.parse.quote(project_id, safe="-_.~")
            status, payload = self._gateway_json("GET", f"/v1/product/studio/projects/{safe_project}")
            return status, self._public_payload(payload), "application/json"
        if method == "POST" and parsed.path == "/api/compile/start":
            if not self._production_compile_enabled():
                raise LivingCompileError(
                    "production compile is fail-closed; configure the public product gateway, service token, workspace bundle root, and explicitly enable product compile"
                )
            if not isinstance(body, dict):
                raise LivingCompileError("JSON request body required")
            status, payload = self._gateway_json("POST", "/v1/product/compiles", body)
            return status, self._public_payload(payload), "application/json"
        if method == "GET" and parsed.path == "/api/compile/status":
            if not self._production_compile_enabled():
                raise LivingCompileError("production compile is fail-closed")
            compile_id = (query.get("id") or [""])[0].strip()
            if not compile_id or len(compile_id) > 160:
                raise LivingCompileError("valid public compile id required")
            safe_id = urllib.parse.quote(compile_id, safe="-_.~")
            status, payload = self._gateway_json("GET", f"/v1/product/compiles/{safe_id}")
            return status, self._public_payload(payload), "application/json"
        if method == "GET" and parsed.path == "/api/compile/workspace":
            if not self._production_compile_enabled():
                raise LivingCompileError("production compile is fail-closed")
            compile_id = (query.get("id") or [""])[0].strip()
            if not compile_id or len(compile_id) > 96:
                raise LivingCompileError("valid public compile id required")
            safe_id = urllib.parse.quote(compile_id, safe="-_.~")
            _, payload = self._gateway_bytes(
                "GET",
                f"/v1/product/compiles/{safe_id}/workspace-bundle",
                accept="application/zip",
                max_bytes=MAX_WORKSPACE_BUNDLE_DOWNLOAD_BYTES,
            )
            bundle = materialize_workspace_bundle(compile_id, payload)
            return 200, {
                "schema_version": "RealSaS.PublicWorkspaceHandoff.v1",
                "compile_id": compile_id,
                "bundle": bundle,
                "workspace_url": "/studio?bundle=" + urllib.parse.quote(bundle, safe=""),
                "authority": "PROOF_GATED_WORKSPACE_BUNDLE",
            }, "application/json"
        if method == "POST" and parsed.path in {"/api/input/upload", "/api/project", "/api/compile"}:
            raise LivingCompileError("unsupported legacy product-shell write route")
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
            except ProductGatewayHTTPError as exc:
                status, payload, content_type = exc.status, {"error": type(exc).__name__, "message": str(exc)}, "application/json"
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
    print("Awaken surface is ready. Production compile remains fail-closed unless gateway, workspace storage, and product authority are configured.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSaS Awaken and editable character workspace")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
