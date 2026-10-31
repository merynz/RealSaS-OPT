from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLATFORM_URL_ENV = "REALSAS_PLATFORM_URL"
OPERATOR_TOKEN_ENV = "REALSAS_OPERATOR_TOKEN"
WORKER_NAME_ENV = "REALSAS_PRODUCT_WORKSPACE_WORKER_NAME"
MATERIALIZER_COMMAND_ENV = "REALSAS_WORKSPACE_MATERIALIZER_COMMAND"
POLL_SECONDS_ENV = "REALSAS_PRODUCT_WORKSPACE_POLL_SECONDS"
LEASE_SECONDS_ENV = "REALSAS_PRODUCT_WORKSPACE_LEASE_SECONDS"
MAX_ATTEMPTS_ENV = "REALSAS_PRODUCT_WORKSPACE_MAX_ATTEMPTS"

REQUIRED_ROLES = {
    "mesh:canonical",
    "rig:canonical",
    "skin:canonical",
    "mechanics:canonical",
    "graphics:canonical",
    "motion:canonical",
    "proof:product_pass",
}
REQUIRED_BUNDLE_FILES = {
    "bundle_manifest.json",
    "puppet/canonical_puppet_graph_v3.json",
    "proof/product_proof_bundle_ir.json",
}


class WorkspaceWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    platform_url: str
    operator_token: str
    worker_name: str
    materializer_command: tuple[str, ...]
    poll_seconds: float
    lease_seconds: int
    max_attempts: int

    @classmethod
    def from_env(cls) -> "Config":
        platform_url = os.environ.get(PLATFORM_URL_ENV, "http://127.0.0.1:8080").strip().rstrip("/")
        parsed = urllib.parse.urlparse(platform_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise WorkspaceWorkerError(f"{PLATFORM_URL_ENV} must be an absolute http(s) URL")
        token = os.environ.get(OPERATOR_TOKEN_ENV, "").strip()
        if not token:
            raise WorkspaceWorkerError(f"{OPERATOR_TOKEN_ENV} is required")
        worker_name = os.environ.get(WORKER_NAME_ENV, "product-workspace-worker").strip()
        if not worker_name:
            raise WorkspaceWorkerError(f"{WORKER_NAME_ENV} must not be empty")
        command_raw = os.environ.get(MATERIALIZER_COMMAND_ENV, "").strip()
        if not command_raw:
            raise WorkspaceWorkerError(
                f"{MATERIALIZER_COMMAND_ENV} is required; workspace authority is never synthesized by the web shell"
            )
        command = tuple(shlex.split(command_raw))
        poll_seconds = float(os.environ.get(POLL_SECONDS_ENV, "2.0"))
        lease_seconds = int(os.environ.get(LEASE_SECONDS_ENV, "900"))
        max_attempts = int(os.environ.get(MAX_ATTEMPTS_ENV, "3"))
        if not command:
            raise WorkspaceWorkerError(f"{MATERIALIZER_COMMAND_ENV} produced an empty command")
        if poll_seconds < 0.2 or poll_seconds > 60:
            raise WorkspaceWorkerError("poll seconds must be in [0.2, 60]")
        if lease_seconds < 60 or lease_seconds > 7200:
            raise WorkspaceWorkerError("lease seconds must be in [60, 7200]")
        if max_attempts < 1 or max_attempts > 20:
            raise WorkspaceWorkerError("max attempts must be in [1, 20]")
        return cls(
            platform_url=platform_url,
            operator_token=token,
            worker_name=worker_name,
            materializer_command=command,
            poll_seconds=poll_seconds,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )


class PlatformClient:
    def __init__(self, config: Config):
        self.config = config

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        raw_body: bytes | None = None,
        headers: dict[str, str] | None = None,
        allow_no_content: bool = False,
    ) -> tuple[int, bytes, dict[str, str]]:
        if json_body is not None and raw_body is not None:
            raise WorkspaceWorkerError("request cannot carry both json_body and raw_body")
        data = raw_body
        request_headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.operator_token}",
        }
        if json_body is not None:
            data = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            self.config.platform_url + path,
            data=data,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=max(30, self.config.lease_seconds)) as response:
                status = int(response.status)
                payload = response.read(270 * 1024 * 1024)
                response_headers = {str(k).lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            payload = exc.read(2 * 1024 * 1024)
            try:
                parsed = json.loads(payload.decode("utf-8")) if payload else {}
            except Exception:
                parsed = {}
            detail = parsed.get("error") if isinstance(parsed, dict) else None
            raise WorkspaceWorkerError(detail or f"Platform HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise WorkspaceWorkerError(f"Platform unavailable: {exc.reason}") from exc
        if status == 204 and allow_no_content:
            return status, b"", response_headers
        if status < 200 or status >= 300:
            raise WorkspaceWorkerError(f"unexpected Platform HTTP {status}")
        return status, payload, response_headers

    def claim(self) -> dict[str, Any] | None:
        status, payload, _ = self._request(
            "POST",
            "/v1/operator/product/workspace/claim",
            json_body={"owner": self.config.worker_name, "lease_seconds": self.config.lease_seconds},
            allow_no_content=True,
        )
        if status == 204:
            return None
        try:
            value = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise WorkspaceWorkerError("workspace claim response is not valid JSON") from exc
        if not isinstance(value, dict):
            raise WorkspaceWorkerError("workspace claim response must be an object")
        return value

    def download_artifact(self, compile_id: str, lease_token: str, role: str) -> tuple[bytes, str, str]:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        safe_role = urllib.parse.quote(role, safe="")
        _, payload, headers = self._request(
            "GET",
            f"/v1/operator/product/compiles/{safe_id}/workspace-source/{safe_role}",
            headers={"X-RealSaS-Lease-Token": lease_token},
        )
        return (
            payload,
            headers.get("x-realsas-artifact-type", "unknown"),
            headers.get("x-realsas-schema-version", "unknown"),
        )

    def complete(self, compile_id: str, lease_token: str, payload: bytes) -> None:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        self._request(
            "POST",
            f"/v1/operator/product/compiles/{safe_id}/workspace-bundle",
            raw_body=payload,
            headers={
                "Content-Type": "application/zip",
                "X-RealSaS-Lease-Token": lease_token,
                "X-RealSaS-Worker": self.config.worker_name,
            },
        )

    def fail(self, compile_id: str, lease_token: str, error_code: str, retry: bool) -> None:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        self._request(
            "POST",
            f"/v1/operator/product/compiles/{safe_id}/workspace-fail",
            json_body={
                "lease_token": lease_token,
                "error_code": error_code[:200],
                "retry": bool(retry),
            },
        )


def _role_filename(role: str) -> str:
    return role.replace(":", "__").replace("/", "_") + ".bin"


def _validate_bundle(path: Path) -> None:
    if not path.is_file():
        raise WorkspaceWorkerError("WORKSPACE_BUNDLE_OUTPUT_MISSING")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = {name.replace("\\", "/") for name in archive.namelist()}
            bad = [name for name in names if name.startswith("/") or "../" in name]
            if bad:
                raise WorkspaceWorkerError("WORKSPACE_BUNDLE_PATH_INVALID")
            missing = sorted(REQUIRED_BUNDLE_FILES - names)
            if missing:
                raise WorkspaceWorkerError("WORKSPACE_BUNDLE_REQUIRED_FILE_MISSING:" + ",".join(missing))
            for info in archive.infolist():
                if info.file_size > 256 * 1024 * 1024:
                    raise WorkspaceWorkerError("WORKSPACE_BUNDLE_ENTRY_TOO_LARGE")
                if info.CRC == 0 and info.file_size > 0:
                    archive.read(info.filename)
    except zipfile.BadZipFile as exc:
        raise WorkspaceWorkerError("WORKSPACE_BUNDLE_ZIP_INVALID") from exc


def _write_request(
    root: Path,
    compile_id: str,
    lease: dict[str, Any],
    artifacts: list[dict[str, str]],
) -> tuple[Path, Path]:
    output = root / "workspace_bundle.zip"
    request = root / "workspace_materialization_request.json"
    payload = {
        "schema_version": "RealSaS.ProductWorkspaceMaterializationRequest.v1",
        "compile_id": compile_id,
        "product_revision_id": str(lease.get("product_revision_id") or ""),
        "attempt": int(lease.get("attempt") or 1),
        "artifacts": artifacts,
        "required_roles": sorted(REQUIRED_ROLES),
        "required_output": {
            "schema": "RealSaS.ProductWorkspaceBundleZip.v1",
            "path": str(output),
            "required_files": sorted(REQUIRED_BUNDLE_FILES),
        },
        "authority": {
            "source_product_revision_is_immutable": True,
            "workspace_is_derived_not_product_authority": True,
            "proof_must_be_preserved": True,
        },
    }
    request.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return request, output


def materialize_once(config: Config, client: PlatformClient | None = None) -> bool:
    client = client or PlatformClient(config)
    lease = client.claim()
    if lease is None:
        return False
    compile_id = str(lease.get("compile_id") or "").strip()
    lease_token = str(lease.get("lease_token") or "").strip()
    attempt = int(lease.get("attempt") or 1)
    rows = lease.get("artifacts")
    if not compile_id or not lease_token or not isinstance(rows, list):
        raise WorkspaceWorkerError("invalid workspace lease payload")
    roles = {str(row.get("role") or "") for row in rows if isinstance(row, dict)}
    if roles != REQUIRED_ROLES:
        raise WorkspaceWorkerError("WORKSPACE_CANONICAL_ROLE_SET_MISMATCH")

    try:
        with tempfile.TemporaryDirectory(prefix="realsas-workspace-") as temp_dir:
            root = Path(temp_dir)
            downloaded: list[dict[str, str]] = []
            for row in rows:
                role = str(row["role"])
                body, artifact_type, schema_version = client.download_artifact(compile_id, lease_token, role)
                path = root / _role_filename(role)
                path.write_bytes(body)
                downloaded.append({
                    "role": role,
                    "path": str(path),
                    "artifact_type": artifact_type,
                    "schema_version": schema_version,
                })
            request_path, output_path = _write_request(root, compile_id, lease, downloaded)
            command = [*config.materializer_command, "--request", str(request_path)]
            proc = subprocess.run(
                command,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=max(60, config.lease_seconds - 15),
                check=False,
                env={**os.environ, "REALSAS_WORKSPACE_MATERIALIZATION_REQUEST": str(request_path)},
            )
            if proc.returncode != 0:
                raise WorkspaceWorkerError(f"WORKSPACE_MATERIALIZER_EXIT_{proc.returncode}")
            _validate_bundle(output_path)
            client.complete(compile_id, lease_token, output_path.read_bytes())
            return True
    except Exception as exc:
        code = str(exc).strip().upper().replace(" ", "_")[:200] or type(exc).__name__.upper()
        retry = attempt < config.max_attempts
        try:
            client.fail(compile_id, lease_token, code, retry)
        except Exception as report_exc:
            raise WorkspaceWorkerError(f"{exc}; failure reporting also failed: {report_exc}") from exc
        if retry:
            return True
        raise


def run(config: Config, *, once: bool = False) -> int:
    client = PlatformClient(config)
    while True:
        did_work = materialize_once(config, client)
        if once:
            return 0
        if not did_work:
            time.sleep(config.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RealSaS ProductRevision workspace materializer")
    parser.add_argument("--once", action="store_true", help="claim at most one workspace job and exit")
    args = parser.parse_args(argv)
    return run(Config.from_env(), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
