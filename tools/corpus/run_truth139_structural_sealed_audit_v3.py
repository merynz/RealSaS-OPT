#!/usr/bin/env python3
"""Run the sealed Truth139 structural/capability audit without blocking on archival web-page bytes.

The October-4 corpus manifest/ledger/seal and every GLTF/GLB primary plus local external dependency
remain exact-byte authority. Historical source-page snapshots are provenance evidence, not geometry/
rig/material capability bytes. Their binding is preserved and availability is reported explicitly,
but a missing archived HTML page may not suppress structural truth discovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus import run_truth139_sealed_raw_audit_v1 as seal_v1
from tools.corpus.audit_truth139_gltf_capabilities_v2 import audit as capability_audit_v2
from tools.corpus.run_truth139_sealed_raw_audit_v2 import _source_family_tokens

SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"
PASS = "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"


def run(raw_root: Path, fit8_selection: Path) -> dict:
    raw_root = raw_root.resolve()
    manifest, ledger, _seal = seal_v1.verify_seal(raw_root)
    base = capability_audit_v2(
        raw_root,
        fit8_selection=fit8_selection,
        expected_corpus_root_sha256=seal_v1.EXPECTED_ROOT_SHA256,
    )
    if base.get("schema") != "RealSaS.Truth139RawCapabilityAudit.v2":
        raise ValueError("TRUTH139_V3_CAPABILITY_AUDIT_SCHEMA")

    packs = {
        str(row["id"]): row
        for row in manifest.get("results") or ()
        if row.get("provider") == "quaternius"
    }
    enriched = []
    provenance_by_pack: dict[str, dict] = {}
    for row in base.get("subjects") or ():
        value = dict(row)
        source = dict(value["source"])
        relative = str(source["relative_path"])
        expected = ledger.get(relative)
        if expected is None or expected != source["sha256"]:
            raise ValueError("TRUTH139_V3_PRIMARY_SOURCE_LEDGER_DRIFT:" + relative)
        path = (raw_root / relative).resolve()
        if not path.is_file() or seal_v1.sha256(path) != expected:
            raise ValueError("TRUTH139_V3_PRIMARY_SOURCE_BYTES_DRIFT:" + relative)

        dependencies = seal_v1._external_dependencies(path, raw_root, ledger)
        source["dependencies"] = dependencies
        source["dependency_count"] = len(dependencies)

        pack_id = str(source.get("pack_id") or "")
        pack_row = packs.get(pack_id)
        if pack_row is None:
            raise ValueError("TRUTH139_V3_SOURCE_PACK_NOT_IN_SEALED_MANIFEST:" + relative)
        source["official_url"] = str(pack_row.get("official_url") or "")
        source["drive_folder_id"] = str(pack_row.get("drive_folder_id") or "")
        snapshot = {
            "path": str(pack_row.get("source_snapshot") or ""),
            "sha256": str(pack_row.get("source_snapshot_sha256") or ""),
        }
        if not snapshot["path"] or len(snapshot["sha256"]) != 64 or ledger.get(snapshot["path"]) != snapshot["sha256"]:
            raise ValueError("TRUTH139_V3_SOURCE_PAGE_BINDING_INVALID:" + pack_id)
        snapshot_path = (raw_root / snapshot["path"]).resolve()
        available = snapshot_path.is_file() and seal_v1.sha256(snapshot_path) == snapshot["sha256"]
        source["source_page_snapshot"] = snapshot
        source["historical_source_page_snapshot_byte_available"] = bool(available)
        source["provenance_status"] = (
            "HISTORICAL_SNAPSHOT_EXACT_BYTE_AVAILABLE"
            if available
            else "HISTORICAL_SNAPSHOT_BINDING_PRESERVED__BYTE_UNAVAILABLE__STRUCTURAL_AUDIT_NOT_BLOCKED"
        )
        provenance_by_pack.setdefault(
            pack_id,
            {
                "pack_id": pack_id,
                "official_url": source["official_url"],
                "historical_snapshot": snapshot,
                "historical_snapshot_byte_available": bool(available),
            },
        )

        lineage = dict(value.get("lineage") or {})
        lineage["source_family_signature"] = _source_family_tokens(value)
        value["lineage"] = lineage
        source["source_family_signature"] = list(lineage["source_family_signature"])
        value["source"] = source
        enriched.append(value)

    passed = (
        base.get("status") == "PASS_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
        and base.get("historical_cardinality_reproduced") is True
    )
    provenance_rows = [provenance_by_pack[key] for key in sorted(provenance_by_pack)]
    missing = [row["pack_id"] for row in provenance_rows if not row["historical_snapshot_byte_available"]]
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": PASS if passed else "FAIL_OR_INCOMPLETE_SEALED_RAW_AUDIT_V2__DO_NOT_BUILD_APPARATUS",
        "audit_mode": "STRUCTURAL_CAPABILITY_AUTHORITY__ARCHIVAL_WEB_PROVENANCE_REPORTED_SEPARATELY",
        "corpus": {
            "root_path": str(raw_root),
            "root_sha256": seal_v1.EXPECTED_ROOT_SHA256,
            "manifest_sha256": seal_v1.EXPECTED_MANIFEST_SHA256,
            "ledger_sha256": seal_v1.EXPECTED_LEDGER_SHA256,
            "seal_json_sha256": seal_v1.EXPECTED_SEAL_JSON_SHA256,
            "ledger_entry_count": len(ledger),
            "quaternius_gltf_glb_ledger_count": seal_v1.EXPECTED_GLTF_GLBS,
            "seal_verified": True,
        },
        "base_capability_audit_schema": base["schema"],
        "base_capability_audit_identity_sha256": base["audit_identity_sha256"],
        "v2_repairs": {
            **dict(base["v2_repairs"]),
            "source_family_lineage": "CREATOR_PACK_NORMALIZED_ASSET_STEM__NOT_PACK_LEVEL_SPLIT",
        },
        "provenance": {
            "historical_snapshot_binding_preserved": True,
            "historical_snapshot_bytes_required_for_structural_capability_pass": False,
            "pack_count": len(provenance_rows),
            "historical_snapshot_byte_available_pack_count": len(provenance_rows) - len(missing),
            "historical_snapshot_byte_missing_pack_count": len(missing),
            "historical_snapshot_byte_missing_pack_ids": missing,
            "packs": provenance_rows,
        },
        "counts": dict(base["counts"]),
        "subjects": enriched,
        "duplicate_groups": list(base.get("duplicate_groups") or ()),
        "file_failures": list(base.get("file_failures") or ()),
        "split_authorized": False,
        "training_authorized": False,
        "next_gate": "BUILD_HISTORICAL_REPRODUCTION_AND_CAPABILITY_AWARE_CORPUS_MANIFEST",
    }
    payload["audit_identity_sha256"] = seal_v1.value_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--fit8-selection", default="canonical/FIT8_DIVERSITY_SELECTION_V1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(Path(args.raw_root), Path(args.fit8_selection).resolve())
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "output": str(output),
        "status": result["status"],
        "counts": result["counts"],
        "provenance": {k: result["provenance"][k] for k in (
            "historical_snapshot_byte_available_pack_count",
            "historical_snapshot_byte_missing_pack_count",
            "historical_snapshot_byte_missing_pack_ids",
        )},
        "audit_identity_sha256": result["audit_identity_sha256"],
    }, sort_keys=True), flush=True)
    if result["status"] != PASS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
