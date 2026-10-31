"""Rebind a Post139 CandidateManifest to the final SourceInventory payload.

The original source-inventory producer wrote ``inventory_sha256`` before adding
registry metadata to SOURCE_INVENTORY.  This additive repair is intended for
recovered workspaces so their manifest points at the exact final inventory
without rewriting any raw bytes or audit evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "RealSaS.Post139.CandidateManifest.v1"
INVENTORY_SCHEMA = "RealSaS.Post139.SourceInventory.v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    inventory_path = Path(args.inventory)
    manifest_path = Path(args.manifest)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise SystemExit("POST139_REBIND_INVENTORY_SCHEMA")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise SystemExit("POST139_REBIND_MANIFEST_SCHEMA")
    if inventory.get("source_id") != manifest.get("source_id"):
        raise SystemExit("POST139_REBIND_SOURCE_ID_MISMATCH")
    manifest["inventory_sha256"] = digest(inventory)
    manifest["inventory_binding"] = "FINAL_SOURCE_INVENTORY_CANONICAL_JSON_SHA256"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("POST139_CANDIDATE_MANIFEST_REBOUND", manifest["source_id"], manifest["inventory_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
