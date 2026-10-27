#!/usr/bin/env python3
"""Read-only typed artifact inventory for one Platform subject.

The query set is source-controlled in the trigger document. This inspector only
uses Platform GET endpoints and writes a receipt; it never imports artifacts,
executes a capability, mutates model state, mints qualifications, or promotes a
product revision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TRIGGER_SCHEMA = "RealSaS.PlatformSubjectArtifactInventoryTrigger.v1"
RECEIPT_SCHEMA = "RealSaS.PlatformSubjectArtifactInventory.v1"
OPERATION = "PLATFORM_SUBJECT_ARTIFACT_INVENTORY"


def _get_json(api_root: str, path: str) -> Any:
    request = Request(api_root.rstrip("/") + path, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_HTTP_STATUS:{path}:{response.status}")
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"PLATFORM_ARTIFACT_INVENTORY_HTTP_{exc.code}:{path}:{detail}"
        ) from exc


def load_trigger(path: Path) -> dict[str, Any]:
    trigger = json.loads(path.read_text(encoding="utf-8"))
    if trigger.get("schema") != TRIGGER_SCHEMA:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_TRIGGER_SCHEMA")
    if trigger.get("operation") != OPERATION:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_TRIGGER_OPERATION")
    if trigger.get("enabled") is not True:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_TRIGGER_DISABLED")
    subject_slug = str(trigger.get("subject_slug") or "").strip()
    if not subject_slug:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_SUBJECT_REQUIRED")
    queries = trigger.get("queries")
    if not isinstance(queries, list) or not queries:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_QUERIES_REQUIRED")
    names: set[str] = set()
    for row in queries:
        if not isinstance(row, dict):
            raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_QUERY_OBJECT_REQUIRED")
        name = str(row.get("name") or "").strip()
        artifact_type = str(row.get("artifact_type") or "").strip()
        schema_version = str(row.get("schema_version") or "").strip()
        limit = row.get("limit", 25)
        if not name or not artifact_type or not schema_version:
            raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_QUERY_INCOMPLETE")
        if name in names:
            raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_QUERY_DUPLICATE:{name}")
        names.add(name)
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_QUERY_LIMIT:{name}")
    return trigger


def _query_artifacts(
    api_root: str,
    subject_id: str,
    artifact_type: str,
    schema_version: str,
    limit: int,
) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "artifact_type": artifact_type,
            "schema_version": schema_version,
            "limit": limit,
        }
    )
    rows = _get_json(api_root, f"/v1/subjects/{subject_id}/artifacts?{query}")
    if not isinstance(rows, list):
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_LIST_INVALID")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_ROW_INVALID")
        artifact_id = str(row.get("artifact_id") or "")
        if not artifact_id or artifact_id in seen:
            continue
        seen.add(artifact_id)
        metadata = _get_json(api_root, f"/v1/artifacts/{artifact_id}")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_METADATA_INVALID:{artifact_id}")
        if str(metadata.get("artifact_type") or "") != artifact_type:
            raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_TYPE_DRIFT:{artifact_id}")
        if str(metadata.get("schema_version") or "") != schema_version:
            raise RuntimeError(f"PLATFORM_ARTIFACT_INVENTORY_SCHEMA_DRIFT:{artifact_id}")
        out.append(
            {
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "schema_version": schema_version,
                "semantic_sha256": str(metadata.get("semantic_sha256") or ""),
                "content_sha256": str(metadata.get("content_sha256") or ""),
                "size_bytes": int(metadata.get("size_bytes") or 0),
                "producer_contract": str(metadata.get("producer_contract") or ""),
                "implementation_sha256": str(metadata.get("implementation_sha256") or ""),
                "policy_sha256": str(metadata.get("policy_sha256") or ""),
                "source_kind": str(row.get("source_kind") or ""),
                "source_id": str(row.get("source_id") or ""),
                "source_role": str(row.get("role") or ""),
                "source_origin": str(row.get("origin") or ""),
            }
        )
    return out


def inspect(api_root: str, trigger: dict[str, Any], expected_main_sha: str) -> dict[str, Any]:
    subject_slug = str(trigger["subject_slug"])
    subject = _get_json(api_root, f"/v1/subjects/by-slug/{subject_slug}")
    if not isinstance(subject, dict) or subject.get("schema") != "RealSaS.SubjectIdentity.v1":
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_SUBJECT_IDENTITY_SCHEMA")
    subject_id = str(subject.get("subject_id") or "")
    if not subject_id or str(subject.get("slug") or "") != subject_slug:
        raise RuntimeError("PLATFORM_ARTIFACT_INVENTORY_SUBJECT_IDENTITY_DRIFT")

    results: dict[str, Any] = {}
    for query in trigger["queries"]:
        name = str(query["name"])
        artifact_type = str(query["artifact_type"])
        schema_version = str(query["schema_version"])
        limit = int(query.get("limit", 25))
        artifacts = _query_artifacts(
            api_root, subject_id, artifact_type, schema_version, limit
        )
        results[name] = {
            "artifact_type": artifact_type,
            "schema_version": schema_version,
            "count": len(artifacts),
            "query_limit": limit,
            "artifacts": artifacts,
        }

    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_PLATFORM_SUBJECT_ARTIFACT_INVENTORY",
        "source_code_sha": expected_main_sha.strip().lower(),
        "request_revision": int(trigger.get("request_revision") or 0),
        "subject": {
            "subject_id": subject_id,
            "slug": subject_slug,
            "display_name": str(subject.get("display_name") or ""),
        },
        "queries": results,
        "platform_mutation_performed": False,
        "artifact_import_performed": False,
        "model_inference_executed": False,
        "compiler_capability_executed": False,
        "qualification_minted": False,
        "model_promotion_performed": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--trigger", type=Path, required=True)
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trigger = load_trigger(args.trigger)
    receipt = inspect(args.api_root, trigger, args.expected_main_sha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {name: int(row["count"]) for name, row in receipt["queries"].items()}
    print(json.dumps({"status": receipt["status"], "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
