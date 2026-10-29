#!/usr/bin/env python3
"""Run the hardened Truth139 V2 capability audit against the exact October-4 seal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus import run_truth139_sealed_raw_audit_v1 as seal_v1
from tools.corpus.audit_truth139_gltf_capabilities_v2 import audit as capability_audit_v2

SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"


def _source_family_tokens(value: dict) -> list[str]:
    rows = list(value.get("duplicate_source_files") or ()) or [dict(value.get("source") or {})]
    tokens = set()
    for source in rows:
        creator = str(source.get("creator") or "").lower()
        pack = str(source.get("pack_id") or "").lower()
        stem = str(source.get("normalized_asset_stem") or "").lower()
        if not creator or not pack or not stem:
            raise ValueError("TRUTH139_V2_SOURCE_FAMILY_FIELDS_MISSING")
        tokens.add("SRCF:" + seal_v1.value_hash({"creator": creator, "pack_id": pack, "normalized_asset_stem": stem}))
    return sorted(tokens)


def run(raw_root: Path, fit8_selection: Path) -> dict:
    raw_root = raw_root.resolve()
    manifest, ledger, _seal = seal_v1.verify_seal(raw_root)
    base = capability_audit_v2(
        raw_root,
        fit8_selection=fit8_selection,
        expected_corpus_root_sha256=seal_v1.EXPECTED_ROOT_SHA256,
    )
    if base.get("schema") != "RealSaS.Truth139RawCapabilityAudit.v2":
        raise ValueError("TRUTH139_V2_CAPABILITY_AUDIT_SCHEMA")

    pack = {
        str(row["id"]): row
        for row in manifest.get("results") or ()
        if row.get("provider") == "quaternius"
    }
    enriched = []
    for row in base.get("subjects") or ():
        value = dict(row)
        source = dict(value["source"])
        relative = str(source["relative_path"])
        expected = ledger.get(relative)
        if expected is None or expected != source["sha256"]:
            raise ValueError("TRUTH139_V2_PRIMARY_SOURCE_LEDGER_DRIFT:" + relative)
        path = (raw_root / relative).resolve()
        if not path.is_file() or seal_v1.sha256(path) != expected:
            raise ValueError("TRUTH139_V2_PRIMARY_SOURCE_BYTES_DRIFT:" + relative)
        source["dependencies"] = seal_v1._external_dependencies(path, raw_root, ledger)
        source["dependency_count"] = len(source["dependencies"])
        pack_row = pack.get(str(source.get("pack_id") or ""))
        if pack_row is None:
            raise ValueError("TRUTH139_V2_SOURCE_PACK_NOT_IN_SEALED_MANIFEST:" + relative)
        source["official_url"] = str(pack_row.get("official_url") or "")
        source["drive_folder_id"] = str(pack_row.get("drive_folder_id") or "")
        source["source_page_snapshot"] = {
            "path": str(pack_row.get("source_snapshot") or ""),
            "sha256": str(pack_row.get("source_snapshot_sha256") or ""),
        }
        snapshot = source["source_page_snapshot"]
        snapshot_path = (raw_root / snapshot["path"]).resolve()
        if (
            not snapshot["path"]
            or ledger.get(snapshot["path"]) != snapshot["sha256"]
            or not snapshot_path.is_file()
            or seal_v1.sha256(snapshot_path) != snapshot["sha256"]
        ):
            raise ValueError("TRUTH139_V2_SOURCE_PAGE_SNAPSHOT_DRIFT:" + relative)
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
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": (
            "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
            if passed
            else "FAIL_OR_INCOMPLETE_SEALED_RAW_AUDIT_V2__DO_NOT_BUILD_APPARATUS"
        ),
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
        "counts": dict(base["counts"]),
        "subjects": enriched,
        "duplicate_groups": list(base.get("duplicate_groups") or ()),
        "file_failures": list(base.get("file_failures") or ()),
        "split_authorized": False,
        "training_authorized": False,
        "next_gate": "BUILD_TRUTH139_APPARATUS_MANIFEST_AND_COMMON_OBSERVATIONS",
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
    print(json.dumps({"output": str(output), "status": result["status"], "counts": result["counts"], "audit_identity_sha256": result["audit_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
