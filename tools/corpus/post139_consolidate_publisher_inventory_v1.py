"""Consolidate every publisher tree and acquired-byte ledger into one inventory.

Publisher presence is evidence, not training admission.  This file intentionally
keeps props, attachments, redundant representations, textures and static files
visible even when the frozen training roster excludes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "RealSaS.Post139.SourceRegistry.v1"
SCHEMA = "RealSaS.Post139.ExactPublisherInventory.v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry = read_json(Path(args.registry))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise SystemExit("POST139_PUBLISHER_INVENTORY_REGISTRY_SCHEMA_MISMATCH")
    workspace = Path(args.workspace).resolve()
    source_rows = []
    missing = []
    total_publisher_files = 0
    total_acquired_files = 0
    publisher_extensions: Counter[str] = Counter()
    acquired_extensions: Counter[str] = Counter()

    for source in registry["sources"]:
        source_id = source["id"]
        if source["role"] == "QUARANTINE":
            source_rows.append({
                "source_id": source_id,
                "source_family": source["source_family"],
                "pack": source["pack"],
                "role": source["role"],
                "status": "QUARANTINED_NOT_ACQUIRED",
                "provenance": source["provenance"],
                "license": source.get("license"),
            })
            continue
        root = workspace / source_id
        tree_path = root / "PUBLISHER_TREE.json"
        ledger_path = root / "inventory" / "RAW_FILE_LEDGER.json"
        inventory_path = root / "inventory" / "SOURCE_INVENTORY.json"
        receipt_path = root / "ACQUISITION_RECEIPT.json"
        absent = [path.name for path in (tree_path, ledger_path, inventory_path, receipt_path) if not path.is_file()]
        if absent:
            missing.append({"source_id": source_id, "missing": absent})
            source_rows.append({
                "source_id": source_id,
                "source_family": source["source_family"],
                "pack": source["pack"],
                "role": source["role"],
                "status": "MISSING_EVIDENCE",
                "missing": absent,
            })
            continue
        tree = read_json(tree_path)
        ledger = read_json(ledger_path)
        inventory = read_json(inventory_path)
        receipt = read_json(receipt_path)
        total_publisher_files += int(tree.get("file_count", 0))
        total_acquired_files += int(ledger.get("file_count", 0))
        publisher_extensions.update(tree.get("extension_counts", {}))
        acquired_extensions.update(ledger.get("extension_counts", {}))
        source_rows.append({
            "source_id": source_id,
            "source_family": source["source_family"],
            "pack": source["pack"],
            "role": source["role"],
            "status": "COMPLETE",
            "provenance": source["provenance"],
            "license": source.get("license"),
            "acquisition": receipt,
            "publisher_file_count": tree.get("file_count"),
            "publisher_extension_counts": tree.get("extension_counts", {}),
            "publisher_tree_digest": tree.get("tree_digest"),
            "publisher_files": tree.get("files", []),
            "publisher_representation_group_count": inventory.get("publisher_representation_group_count", inventory.get("representation_group_count")),
            "publisher_representation_groups": inventory.get("publisher_representation_groups", inventory.get("representation_groups", [])),
            "acquired_file_count": ledger.get("file_count"),
            "acquired_total_bytes": ledger.get("total_bytes"),
            "acquired_extension_counts": ledger.get("extension_counts", {}),
            "acquired_ledger_sha256": ledger.get("ledger_sha256"),
            "acquired_files": ledger.get("files", []),
            "audit_candidate_count": inventory.get("audit_candidate_count"),
            "canonical_representation_groups": inventory.get("representation_groups", []),
        })

    payload = {
        "schema": SCHEMA,
        "status": "COMPLETE" if not missing else "INCOMPLETE",
        "registry_source_count": len(registry["sources"]),
        "non_quarantine_source_count": sum(row["role"] != "QUARANTINE" for row in registry["sources"]),
        "complete_source_count": sum(row.get("status") == "COMPLETE" for row in source_rows),
        "missing_source_count": len(missing),
        "missing": missing,
        "publisher_file_count": total_publisher_files,
        "acquired_file_count": total_acquired_files,
        "publisher_extension_counts": dict(sorted(publisher_extensions.items())),
        "acquired_extension_counts": dict(sorted(acquired_extensions.items())),
        "sources": source_rows,
    }
    payload["inventory_digest"] = digest(source_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("POST139_EXACT_PUBLISHER_INVENTORY", payload["status"], total_publisher_files, total_acquired_files, len(missing))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
