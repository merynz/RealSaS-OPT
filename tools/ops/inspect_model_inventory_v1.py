#!/usr/bin/env python3
"""Read-only Platform model release/channel inventory for product closure diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

FAMILIES = ("IRIS", "TESSA", "AXIS", "MIRA")


def get_json(api_root: str, path: str) -> Any:
    request = Request(api_root.rstrip("/") + path, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"MODEL_INVENTORY_HTTP_STATUS:{path}:{response.status}")
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"MODEL_INVENTORY_HTTP_{exc.code}:{path}:{detail}") from exc


def inspect(api_root: str, expected_main_sha: str) -> dict[str, Any]:
    releases = get_json(api_root, "/v1/model-releases")
    channels = get_json(api_root, "/v1/model-channels")
    if not isinstance(releases, list) or not isinstance(channels, list):
        raise RuntimeError("MODEL_INVENTORY_LIST_RESPONSE_REQUIRED")

    family_rows: dict[str, Any] = {}
    for family in FAMILIES:
        family_releases = []
        family_channels = []
        for row in releases:
            if str(row.get("family") or "") != family:
                continue
            release_id = str(row.get("id") or "")
            bound = sorted(
                str(channel.get("channel") or "")
                for channel in channels
                if str(channel.get("family") or "") == family
                and str(channel.get("model_release_id") or "") == release_id
            )
            family_releases.append({
                "id": release_id,
                "name": str(row.get("name") or ""),
                "architecture_id": str(row.get("architecture_id") or ""),
                "checkpoint_artifact_id": str(row.get("checkpoint_artifact_id") or ""),
                "checkpoint_content_sha256": str(row.get("checkpoint_content_sha256") or ""),
                "lifecycle_state": str(row.get("lifecycle_state") or ""),
                "evaluations": list(row.get("evaluations") or []),
                "bound_channels": bound,
            })
        for channel in channels:
            if str(channel.get("family") or "") == family:
                family_channels.append({
                    "channel": str(channel.get("channel") or ""),
                    "model_release_id": str(channel.get("model_release_id") or ""),
                })
        family_rows[family] = {
            "release_count": len(family_releases),
            "channel_count": len(family_channels),
            "releases": family_releases,
            "channels": family_channels,
        }

    return {
        "schema": "RealSaS.ReadOnlyModelInventory.v1",
        "status": "PASS_READ_ONLY_INVENTORY",
        "source_code_sha": expected_main_sha,
        "families": family_rows,
        "total_release_count": len(releases),
        "total_channel_count": len(channels),
        "platform_mutation_performed": False,
        "model_registration_performed": False,
        "model_evaluation_performed": False,
        "model_promotion_performed": False,
        "training_submitted": False,
        "product_revision_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="http://127.0.0.1:8080")
    parser.add_argument("--expected-main-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.api_root, args.expected_main_sha.strip().lower())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
