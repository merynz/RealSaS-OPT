#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed RealSaS product deployment preflight.

The report separates the immediately shippable controlled exact-8 lane from the
public front/back lane. Exact-8 uses the operator SourceObservation ingress and
therefore does not pretend the front/back source materializer exists. Public
front/back readiness additionally requires that materializer and the public
compile feature flag.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import urllib.error
import urllib.parse
import urllib.request


SCHEMA = "RealSaS.ProductDeployPreflight.v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_MARKERS = (
    "replace-me",
    "replace-with",
    "unbound",
    "example",
    "changeme",
)


class ProductDeployPreflightError(RuntimeError):
    pass


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ProductDeployPreflightError(f"ENV_LINE_INVALID:{line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ProductDeployPreflightError(f"ENV_KEY_EMPTY:{line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def merged_environment(env_file: Path | None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    values = parse_env_file(env_file) if env_file else {}
    for key, value in dict(environ or os.environ).items():
        if str(value).strip() != "":
            values[str(key)] = str(value)
    return values


def _placeholder(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return not lowered or any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> bool:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})
    return bool(passed)


def evaluate_local_environment(env: Mapping[str, str]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    source_sha = str(env.get("REALSAS_SOURCE_CODE_SHA") or "").strip()
    source_ok = bool(SHA40.fullmatch(source_sha)) and source_sha != "0" * 40
    if not _check(checks, "source_code_sha", source_ok, source_sha or "missing"):
        blockers.append("SOURCE_CODE_SHA_INVALID_OR_PLACEHOLDER")

    database = str(env.get("REALSAS_DATABASE_URL") or "").strip()
    database_ok = database.startswith(("postgres://", "postgresql://")) and not _placeholder(database)
    if not _check(checks, "database_url", database_ok, "configured" if database_ok else "missing-or-placeholder"):
        blockers.append("DATABASE_URL_NOT_PRODUCTION_CONFIGURED")

    temporal = str(env.get("REALSAS_TEMPORAL_ADDRESS") or "").strip()
    temporal_ok = bool(temporal) and not _placeholder(temporal)
    if not _check(checks, "temporal_address", temporal_ok, temporal or "missing"):
        blockers.append("TEMPORAL_ADDRESS_NOT_CONFIGURED")

    gateway = str(env.get("REALSAS_PRODUCT_GATEWAY_TOKEN") or "").strip()
    operator = str(env.get("REALSAS_OPERATOR_TOKEN") or "").strip()
    gateway_ok = len(gateway) >= 32 and not _placeholder(gateway)
    operator_ok = len(operator) >= 32 and not _placeholder(operator)
    tokens_distinct = gateway_ok and operator_ok and gateway != operator
    if not _check(checks, "product_gateway_token", gateway_ok, "configured" if gateway_ok else "weak-missing-or-placeholder"):
        blockers.append("PRODUCT_GATEWAY_TOKEN_NOT_READY")
    if not _check(checks, "operator_token", operator_ok, "configured" if operator_ok else "weak-missing-or-placeholder"):
        blockers.append("OPERATOR_TOKEN_NOT_READY")
    if not _check(checks, "service_tokens_distinct", tokens_distinct, "distinct" if tokens_distinct else "must-be-distinct"):
        blockers.append("SERVICE_TOKENS_NOT_DISTINCT")

    engine_image = str(env.get("REALSAS_ENGINE_IMAGE") or "").strip()
    engine_ok = bool(engine_image) and not _placeholder(engine_image)
    if not _check(checks, "engine_image", engine_ok, engine_image or "missing"):
        blockers.append("ENGINE_IMAGE_UNBOUND")

    workspace_command = str(env.get("REALSAS_WORKSPACE_MATERIALIZER_COMMAND") or "").strip()
    # Compose intentionally supplies the built-in deterministic materializer when
    # this variable is empty. Empty therefore means canonical default, not unbound.
    workspace_ok = not workspace_command or not _placeholder(workspace_command)
    workspace_detail = workspace_command or "built-in:python -m product.workspace_worker.materialize_workspace_v1"
    if not _check(checks, "workspace_materializer", workspace_ok, workspace_detail):
        blockers.append("WORKSPACE_MATERIALIZER_INVALID")

    numpy_ready = importlib.util.find_spec("numpy") is not None
    pillow_ready = importlib.util.find_spec("PIL") is not None
    exact8_tool_deps = numpy_ready and pillow_ready
    _check(
        checks,
        "exact8_launcher_dependencies",
        exact8_tool_deps,
        f"numpy={numpy_ready},pillow={pillow_ready}",
    )
    if not exact8_tool_deps:
        blockers.append("EXACT8_LAUNCHER_DEPENDENCIES_MISSING")

    source_materializer = str(env.get("REALSAS_SOURCE_MATERIALIZER_COMMAND") or "").strip()
    source_materializer_ready = bool(source_materializer) and not _placeholder(source_materializer)
    public_enabled = str(env.get("REALSAS_PRODUCT_COMPILE_ENABLED") or "0").strip() == "1"

    exact8_local_ready = (
        source_ok
        and database_ok
        and temporal_ok
        and gateway_ok
        and operator_ok
        and tokens_distinct
        and engine_ok
        and workspace_ok
        and exact8_tool_deps
    )
    public_local_ready = exact8_local_ready and source_materializer_ready and public_enabled

    return {
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "source_code_sha": source_sha,
        "operator_token": operator,
        "exact8_local_ready": exact8_local_ready,
        "public_frontback_local_ready": public_local_ready,
        "public_frontback": {
            "source_materializer_ready": source_materializer_ready,
            "source_materializer_command": source_materializer,
            "public_compile_enabled": public_enabled,
        },
        "workspace": {
            "materializer_ready": workspace_ok,
            "materializer_command": workspace_detail,
            "uses_builtin_default": not bool(workspace_command),
        },
    }


def _http_json(
    api_root: str,
    path: str,
    *,
    token: str | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    root = api_root.strip().rstrip("/")
    parsed = urllib.parse.urlparse(root)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ProductDeployPreflightError("PLATFORM_URL_INVALID")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(root + path, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1024 * 1024).decode("utf-8", errors="replace")
        raise ProductDeployPreflightError(f"PLATFORM_HTTP_{exc.code}:{detail}") from exc
    except urllib.error.URLError as exc:
        raise ProductDeployPreflightError(f"PLATFORM_UNAVAILABLE:{exc.reason}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProductDeployPreflightError("PLATFORM_RESPONSE_JSON_INVALID") from exc
    if not isinstance(value, dict):
        raise ProductDeployPreflightError("PLATFORM_RESPONSE_OBJECT_REQUIRED")
    return value


def platform_preflight(api_root: str, operator_token: str) -> dict[str, Any]:
    health = _http_json(api_root, "/health/live")
    if health.get("status") != "ok":
        raise ProductDeployPreflightError("PLATFORM_HEALTH_NOT_OK")
    state = _http_json(api_root, "/v1/operator/product/preflight", token=operator_token)
    if state.get("schema_version") != "RealSaS.ProductDeploymentPreflight.v1":
        raise ProductDeployPreflightError("PLATFORM_PREFLIGHT_SCHEMA_DRIFT")
    return state


def build_report(
    *,
    local: Mapping[str, Any],
    platform: Mapping[str, Any] | None,
) -> dict[str, Any]:
    platform_ready = bool(platform and platform.get("ready") is True)
    exact8_ready = bool(local.get("exact8_local_ready")) and platform_ready
    public_ready = bool(local.get("public_frontback_local_ready")) and platform_ready
    blockers = list(local.get("blockers") or [])
    if platform is None:
        blockers.append("PLATFORM_PREFLIGHT_NOT_RUN")
    else:
        blockers.extend(str(value) for value in platform.get("blockers") or [])
    if not (local.get("public_frontback") or {}).get("source_materializer_ready"):
        blockers.append("PUBLIC_FRONTBACK_SOURCE_MATERIALIZER_UNBOUND")
    if not (local.get("public_frontback") or {}).get("public_compile_enabled"):
        blockers.append("PUBLIC_FRONTBACK_COMPILE_FLAG_DISABLED")
    return {
        "schema_version": SCHEMA,
        "exact8_product_ready": exact8_ready,
        "public_frontback_ready": public_ready,
        "local": {key: value for key, value in local.items() if key != "operator_token"},
        "platform": platform,
        "blockers": sorted(set(blockers)),
        "interpretation": {
            "exact8": "shipping product DAG can be exercised with exact controlled SourceObservation evidence",
            "public_frontback": "public 1-2 view intake is enabled only when its source materializer is explicitly bound",
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default="deploy/product/.env")
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--lane", choices=("exact8", "public", "all"), default="all")
    parser.add_argument("--skip-platform", action="store_true", help="Local configuration audit only; never reports a lane ready")
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env_file = Path(args.env_file).expanduser().resolve()
    if not env_file.is_file():
        raise ProductDeployPreflightError(f"ENV_FILE_MISSING:{env_file}")
    env = merged_environment(env_file)
    local = evaluate_local_environment(env)
    platform = None if args.skip_platform else platform_preflight(args.api_root, str(local.get("operator_token") or ""))
    report = build_report(local=local, platform=platform)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out:
        output = Path(args.json_out).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.lane == "exact8":
        return 0 if report["exact8_product_ready"] else 2
    if args.lane == "public":
        return 0 if report["public_frontback_ready"] else 2
    return 0 if report["exact8_product_ready"] and report["public_frontback_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
