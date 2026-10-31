"""Bind an exact source inventory to registry provenance for structural audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REGISTRY_SCHEMA = "RealSaS.Post139.SourceRegistry.v1"
INVENTORY_SCHEMA = "RealSaS.Post139.SourceInventory.v1"
MANIFEST_SCHEMA = "RealSaS.Post139.CandidateManifest.v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise SystemExit("POST139_REGISTRY_SCHEMA_MISMATCH")
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise SystemExit("POST139_INVENTORY_SCHEMA_MISMATCH")
    rows = [row for row in registry["sources"] if row["id"] == args.source_id]
    if len(rows) != 1:
        raise SystemExit(f"POST139_SOURCE_ID_NOT_UNIQUE:{args.source_id}:{len(rows)}")
    source = rows[0]
    if inventory.get("source_id") != args.source_id:
        raise SystemExit("POST139_INVENTORY_SOURCE_ID_MISMATCH")

    candidates = []
    used = set()
    for group in inventory.get("representation_groups", []):
        audit_source = group.get("audit_source")
        if not audit_source:
            continue
        base = args.source_id + "--" + group["representation_key"].replace("/", "--")
        subject_id = base
        suffix = 2
        while subject_id in used:
            subject_id = f"{base}--{suffix}"
            suffix += 1
        used.add(subject_id)
        candidates.append({
            "subject_id": subject_id,
            "path": audit_source["path"],
            "source_sha256": audit_source["sha256"],
            "representation_key": group["representation_key"],
            "representation_count": len(group["representations"]),
        })

    payload = {
        "schema": MANIFEST_SCHEMA,
        "source_id": args.source_id,
        "source_family": source["source_family"],
        "pack": source["pack"],
        "provenance": source["provenance"],
        "license": source.get("license"),
        "role": source["role"],
        "candidate_policy": source.get("candidate_policy"),
        "materialization_eligible": bool(source.get("materialization_eligible")),
        "inventory_raw_ledger_sha256": inventory["raw_ledger_sha256"],
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("POST139_AUDIT_MANIFEST_PASS", args.source_id, len(candidates))


if __name__ == "__main__":
    main()
