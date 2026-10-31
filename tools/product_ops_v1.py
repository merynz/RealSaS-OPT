#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_PLATFORM_URL = "http://127.0.0.1:8080"
SNAPSHOT_PATH = "/v1/operator/product/ops-snapshot"
HEALTH_PATH = "/health/live"


class ProductOpsError(RuntimeError):
    pass


def _request_json(url: str, *, token: str | None = None, timeout: float = 8.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read(1024 * 1024)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}
        message = payload.get("error") if isinstance(payload, dict) else None
        raise ProductOpsError(message or f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise ProductOpsError(f"platform unavailable: {exc.reason}") from exc
    if status < 200 or status >= 300:
        raise ProductOpsError(f"HTTP {status} from {url}")
    if len(raw) > 8 * 1024 * 1024:
        raise ProductOpsError("platform response exceeded 8 MiB")
    try:
        payload = json.loads(raw.decode("utf-8")) if raw else {}
    except Exception as exc:
        raise ProductOpsError("platform returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ProductOpsError("platform returned non-object JSON")
    return payload


def fetch_snapshot(platform_url: str, operator_token: str, *, timeout: float = 8.0) -> dict[str, Any]:
    base = platform_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        raise ProductOpsError("platform URL must be absolute http(s)")
    token = operator_token.strip()
    if not token:
        raise ProductOpsError("operator token is required")
    health = _request_json(base + HEALTH_PATH, timeout=timeout)
    if health.get("status") != "ok":
        raise ProductOpsError("platform liveness check did not return status=ok")
    snapshot = _request_json(base + SNAPSHOT_PATH, token=token, timeout=timeout)
    if snapshot.get("schema_version") != "RealSaS.ProductOpsSnapshot.v1":
        raise ProductOpsError("unexpected product ops snapshot schema")
    return snapshot


def _counts(rows: Any) -> str:
    if not isinstance(rows, list) or not rows:
        return "none"
    parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        state = str(row.get("state") or "UNKNOWN")
        count = int(row.get("count") or 0)
        parts.append(f"{state}={count}")
    return ", ".join(parts) if parts else "none"


def format_snapshot(snapshot: dict[str, Any]) -> str:
    preflight = snapshot.get("preflight") if isinstance(snapshot.get("preflight"), dict) else {}
    studio = snapshot.get("studio") if isinstance(snapshot.get("studio"), dict) else {}
    models = preflight.get("models") if isinstance(preflight.get("models"), list) else []
    blockers = preflight.get("blockers") if isinstance(preflight.get("blockers"), list) else []
    release_id = str(preflight.get("engine_release_id") or "UNBOUND")
    source_sha = str(snapshot.get("source_code_sha") or preflight.get("source_code_sha") or "UNKNOWN")

    lines = [
        f"RealSaS Product Doctor: {'READY' if snapshot.get('ready') else 'BLOCKED'}",
        f"Source SHA: {source_sha}",
        f"EngineRelease: {release_id}",
        f"Canonical main: {'PASS' if preflight.get('canonical_main_source_verified') else 'BLOCK'}",
        "Models: " + (
            ", ".join(
                f"{row.get('family', 'UNKNOWN')}={'READY' if row.get('ready') else 'BLOCK'}"
                for row in models
                if isinstance(row, dict)
            )
            or "none"
        ),
        f"Compile sessions: {_counts(snapshot.get('compile_sessions'))}",
        f"Workspace materializations: {_counts(snapshot.get('workspace_materializations'))}",
        "Studio: "
        f"projects={int(studio.get('active_projects') or 0)}, "
        f"revisions={int(studio.get('authoring_revisions') or 0)}, "
        f"requalification_heads={int(studio.get('heads_requiring_requalification') or 0)}",
    ]
    if blockers:
        lines.append("Blockers:")
        lines.extend(f"  - {item}" for item in blockers)
    else:
        lines.append("Blockers: none")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the production-focused RealSaS operator snapshot without mutating product authority."
    )
    parser.add_argument(
        "--platform-url",
        default=os.environ.get("REALSAS_PLATFORM_URL", DEFAULT_PLATFORM_URL),
        help="Platform API root (default: REALSAS_PLATFORM_URL or http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--operator-token",
        default=os.environ.get("REALSAS_OPERATOR_TOKEN", ""),
        help="Operator bearer token (default: REALSAS_OPERATOR_TOKEN)",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--fail-not-ready",
        action="store_true",
        help="Exit 2 when the product release/model preflight is not ready.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        snapshot = fetch_snapshot(args.platform_url, args.operator_token, timeout=args.timeout)
    except ProductOpsError as exc:
        print(f"product doctor error: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(format_snapshot(snapshot))
    if args.fail_not_ready and not snapshot.get("ready"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
