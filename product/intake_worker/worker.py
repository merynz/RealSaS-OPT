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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
)


PLATFORM_URL_ENV = "REALSAS_PLATFORM_URL"
OPERATOR_TOKEN_ENV = "REALSAS_OPERATOR_TOKEN"
WORKER_NAME_ENV = "REALSAS_PRODUCT_INTAKE_WORKER_NAME"
MATERIALIZER_COMMAND_ENV = "REALSAS_SOURCE_MATERIALIZER_COMMAND"
POLL_SECONDS_ENV = "REALSAS_PRODUCT_INTAKE_POLL_SECONDS"
LEASE_SECONDS_ENV = "REALSAS_PRODUCT_INTAKE_LEASE_SECONDS"
MAX_ATTEMPTS_ENV = "REALSAS_PRODUCT_INTAKE_MAX_ATTEMPTS"


class IntakeWorkerError(RuntimeError):
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
            raise IntakeWorkerError(f"{PLATFORM_URL_ENV} must be an absolute http(s) URL")
        token = os.environ.get(OPERATOR_TOKEN_ENV, "").strip()
        if not token:
            raise IntakeWorkerError(f"{OPERATOR_TOKEN_ENV} is required")
        worker_name = os.environ.get(WORKER_NAME_ENV, "product-intake-worker").strip()
        if not worker_name:
            raise IntakeWorkerError(f"{WORKER_NAME_ENV} must not be empty")
        command_raw = os.environ.get(MATERIALIZER_COMMAND_ENV, "").strip()
        if not command_raw:
            raise IntakeWorkerError(
                f"{MATERIALIZER_COMMAND_ENV} is required; the worker never fabricates missing views"
            )
        command = tuple(shlex.split(command_raw))
        if not command:
            raise IntakeWorkerError(f"{MATERIALIZER_COMMAND_ENV} produced an empty command")
        poll_seconds = float(os.environ.get(POLL_SECONDS_ENV, "2.0"))
        lease_seconds = int(os.environ.get(LEASE_SECONDS_ENV, "900"))
        max_attempts = int(os.environ.get(MAX_ATTEMPTS_ENV, "3"))
        if poll_seconds < 0.2 or poll_seconds > 60:
            raise IntakeWorkerError("poll seconds must be in [0.2, 60]")
        if lease_seconds < 60 or lease_seconds > 7200:
            raise IntakeWorkerError("lease seconds must be in [60, 7200]")
        if max_attempts < 1 or max_attempts > 20:
            raise IntakeWorkerError("max attempts must be in [1, 20]")
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
            raise IntakeWorkerError("request cannot carry both json_body and raw_body")
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
                payload = response.read(70 * 1024 * 1024)
                response_headers = {k: v for k, v in response.headers.items()}
        except urllib.error.HTTPError as exc:
            payload = exc.read(2 * 1024 * 1024)
            try:
                parsed = json.loads(payload.decode("utf-8")) if payload else {}
            except Exception:
                parsed = {}
            detail = parsed.get("error") if isinstance(parsed, dict) else None
            raise IntakeWorkerError(detail or f"Platform HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise IntakeWorkerError(f"Platform unavailable: {exc.reason}") from exc
        if status == 204 and allow_no_content:
            return status, b"", response_headers
        if status < 200 or status >= 300:
            raise IntakeWorkerError(f"unexpected Platform HTTP {status}")
        return status, payload, response_headers

    def claim(self) -> dict[str, Any] | None:
        status, payload, _ = self._request(
            "POST",
            "/v1/operator/product/intake/claim",
            json_body={
                "owner": self.config.worker_name,
                "lease_seconds": self.config.lease_seconds,
            },
            allow_no_content=True,
        )
        if status == 204:
            return None
        try:
            value = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise IntakeWorkerError("claim response is not valid JSON") from exc
        if not isinstance(value, dict):
            raise IntakeWorkerError("claim response must be an object")
        return value

    def download_view(self, compile_id: str, lease_token: str, slot: str) -> tuple[bytes, str, str]:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        safe_slot = urllib.parse.quote(slot, safe="")
        _, payload, headers = self._request(
            "GET",
            f"/v1/operator/product/compiles/{safe_id}/intake-view/{safe_slot}",
            headers={"X-RealSaS-Lease-Token": lease_token},
        )
        return (
            payload,
            headers.get("Content-Type", "application/octet-stream"),
            headers.get("X-RealSaS-Original-Name", f"{slot}.bin"),
        )

    def complete(self, compile_id: str, lease_token: str, payload: bytes) -> dict[str, Any]:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        _, raw, _ = self._request(
            "POST",
            f"/v1/operator/product/compiles/{safe_id}/source-observation-bytes",
            raw_body=payload,
            headers={
                "Content-Type": "application/octet-stream",
                "X-RealSaS-Lease-Token": lease_token,
                "X-RealSaS-Worker": self.config.worker_name,
            },
        )
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise IntakeWorkerError("completion response is not valid JSON") from exc
        if not isinstance(value, dict):
            raise IntakeWorkerError("completion response must be an object")
        return value

    def fail(self, compile_id: str, lease_token: str, error_code: str, retry: bool) -> None:
        safe_id = urllib.parse.quote(compile_id, safe="-_.~")
        self._request(
            "POST",
            f"/v1/operator/product/compiles/{safe_id}/intake-fail",
            json_body={
                "lease_token": lease_token,
                "error_code": error_code[:200],
                "retry": bool(retry),
            },
        )


def _suffix_for_mime(mime_type: str) -> str:
    return {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
    }.get(mime_type.lower(), ".bin")


def _write_request(
    root: Path,
    compile_id: str,
    lease: dict[str, Any],
    downloaded: list[dict[str, str]],
) -> tuple[Path, Path]:
    output_path = root / "source_observation_v1.npz"
    request_path = root / "materialization_request.json"
    request = {
        "schema_version": "RealSaS.PublicSourceMaterializationRequest.v1",
        "compile_id": compile_id,
        "attempt": int(lease.get("attempt") or 1),
        "views": downloaded,
        "required_output": {
            "schema": "RealSaS.SourceObservation.v1",
            "path": str(output_path),
            "camera_policy": "FIXED_EIGHT_VIEW_NO_PER_VIEW_RECENTER_NO_PER_VIEW_POSE_CHANGE",
        },
        "authority": {
            "raw_views_are_external_evidence_only": True,
            "missing_views_must_not_be_duplicated_or_declared_observed": True,
            "materializer_must_emit_a_codec_valid_source_observation": True,
        },
    }
    request_path.write_text(json.dumps(request, indent=2, sort_keys=True), encoding="utf-8")
    return request_path, output_path


def materialize_once(config: Config, client: PlatformClient | None = None) -> bool:
    client = client or PlatformClient(config)
    lease = client.claim()
    if lease is None:
        return False
    compile_id = str(lease.get("compile_id") or "").strip()
    lease_token = str(lease.get("lease_token") or "").strip()
    views = lease.get("views")
    attempt = int(lease.get("attempt") or 1)
    if not compile_id or not lease_token or not isinstance(views, list) or not views:
        raise IntakeWorkerError("invalid intake lease payload")

    try:
        with tempfile.TemporaryDirectory(prefix="realsas-intake-") as temp_dir:
            root = Path(temp_dir)
            downloaded: list[dict[str, str]] = []
            for item in views:
                if not isinstance(item, dict):
                    raise IntakeWorkerError("lease view entry must be an object")
                slot = str(item.get("slot") or "").strip()
                payload, mime_type, original_name = client.download_view(compile_id, lease_token, slot)
                path = root / f"{slot}{_suffix_for_mime(mime_type)}"
                path.write_bytes(payload)
                downloaded.append({
                    "slot": slot,
                    "path": str(path),
                    "mime_type": mime_type,
                    "original_name": original_name,
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
                env={**os.environ, "REALSAS_MATERIALIZATION_REQUEST": str(request_path)},
            )
            if proc.returncode != 0:
                raise IntakeWorkerError(f"MATERIALIZER_EXIT_{proc.returncode}")
            if not output_path.is_file():
                raise IntakeWorkerError("MATERIALIZER_OUTPUT_MISSING")
            output = output_path.read_bytes()
            decode_source_observation_arrays_v1(output)
            client.complete(compile_id, lease_token, output)
            return True
    except Exception as exc:
        code = str(exc).strip().upper().replace(" ", "_")[:200] or type(exc).__name__.upper()
        retry = attempt < config.max_attempts
        try:
            client.fail(compile_id, lease_token, code, retry)
        except Exception as report_exc:
            raise IntakeWorkerError(f"{exc}; failure reporting also failed: {report_exc}") from exc
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
    parser = argparse.ArgumentParser(description="RealSaS public source materialization worker")
    parser.add_argument("--once", action="store_true", help="claim at most one intake job and exit")
    args = parser.parse_args(argv)
    return run(Config.from_env(), once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
