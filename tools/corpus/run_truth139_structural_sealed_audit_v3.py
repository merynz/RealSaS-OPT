#!/usr/bin/env python3
"""Run the sealed Truth139 structural/capability audit without blocking on archival web-page bytes.

The October-4 corpus manifest/ledger/seal and every GLTF/GLB primary plus local external dependency
remain exact-byte authority. Historical source-page snapshots are provenance evidence, not geometry/
rig/material capability bytes. Their binding is preserved and availability is reported explicitly,
but a missing archived HTML page may not suppress structural truth discovery.

Historical Flat/Textured presentation variants are collapsed by a path-derived family key directly
from the 143 rigged/animated source-file rows. This is intentionally separate from exact geometry
identity: exact-geometry equivalence is useful leakage evidence, but it is not the documented
October-4 subject identity rule and therefore may not silently reduce the historical 139 cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

from tools.corpus import run_truth139_sealed_raw_audit_v1 as seal_v1
from tools.corpus.audit_truth139_gltf_capabilities_v2 import (
    _capability_rank,
    _group_development_taint,
    _union_lineage,
    audit as capability_audit_v2,
)
from tools.corpus.run_truth139_sealed_raw_audit_v2 import _source_family_tokens

SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"
PASS = "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
PRESENTATION_VARIANT_SEGMENTS = {
    "flat",
    "flat colors",
    "flat colours",
    "textured",
}


def _historical_variant_key(row: dict) -> str:
    """Return a source-family key that strips only known presentation-style path segments.

    Category folders such as Big/Blob/Flying remain in the key, so same-named assets in genuinely
    different families are not collapsed. Only style-only directories used by the historical
    Flat/Textured duplicate family disappear.
    """
    source = dict(row.get("source") or {})
    relative = PurePosixPath(str(source.get("relative_path") or ""))
    parts = list(relative.parts)
    try:
        q = parts.index("quaternius")
    except ValueError:
        q = -1
    creator = str(source.get("creator") or "").lower()
    pack = str(source.get("pack_id") or "").lower()
    tail = parts[q + 2 :] if q >= 0 and q + 2 <= len(parts) else parts
    normalized_tail = [part.lower() for part in tail if part.lower() not in PRESENTATION_VARIANT_SEGMENTS]
    if not creator or not pack or not normalized_tail:
        raise ValueError("TRUTH139_V3_HISTORICAL_VARIANT_KEY_FIELDS_MISSING")
    return seal_v1.value_hash({"creator": creator, "pack_id": pack, "path": normalized_tail})


def _collapse_historical_presentation_variants(file_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in file_rows:
        grouped.setdefault(_historical_variant_key(row), []).append(row)

    collapsed: list[dict] = []
    groups: list[dict] = []
    for family_key, family in sorted(grouped.items()):
        ordered = sorted(family, key=_capability_rank)
        representative = dict(ordered[0])
        lineage = _union_lineage(family)
        taint = _group_development_taint(family)
        source_rows: list[dict] = []
        seen_paths: set[str] = set()
        for member in family:
            candidates = list(member.get("duplicate_source_files") or ()) or [dict(member.get("source") or {})]
            for source in candidates:
                relative = str(source.get("relative_path") or "")
                if relative and relative not in seen_paths:
                    source_rows.append(dict(source))
                    seen_paths.add(relative)
        source_rows.sort(key=lambda row: str(row.get("relative_path") or ""))

        full = bool((representative.get("capabilities") or {}).get("FULL_TRUTH139_RAW_ELIGIBLE"))
        blockers = [
            str(value)
            for member in family
            for value in (member.get("raw_eligibility_blockers") or ())
            if str(value) != "DEVELOPMENT_OR_REGRESSION_EXCLUSION"
        ]
        if taint:
            blockers.append("DUPLICATE_FAMILY_DEVELOPMENT_OR_REGRESSION_EXCLUSION")

        historical_identity = seal_v1.value_hash({"historical_variant_family": family_key})
        value = {
            **representative,
            "subject_id": "truth139_" + historical_identity[:20],
            "subject_identity_signature": historical_identity,
            "exact_geometry_identity_signature": str(representative.get("subject_identity_signature") or ""),
            "historical_variant_family_signature": family_key,
            "lineage": lineage,
            "duplicate_source_files": source_rows,
            "duplicate_source_file_count": len(source_rows),
            "duplicate_family_development_taint": taint,
            "development_exclusion": (
                "DUPLICATE_FAMILY_DEVELOPMENT_TAINT:" + ",".join(taint)
                if taint
                else None
            ),
            "raw_eligibility_blockers": sorted(set(blockers)),
            "representative_selection_policy": "MAX_TRUTH_CAPABILITY_THEN_TEXTURED_NONFLAT_THEN_PATH",
            "raw_eligible_for_truth139": bool(full and not taint),
            "admitted_for_truth139": False,
            "training_authorized": False,
        }
        collapsed.append(value)
        if len(source_rows) > 1:
            groups.append(
                {
                    "subject_id": value["subject_id"],
                    "identity": historical_identity,
                    "historical_variant_family_signature": family_key,
                    "representative": str((representative.get("source") or {}).get("relative_path") or ""),
                    "source_files": [str(row.get("relative_path") or "") for row in source_rows],
                    "development_taint": taint,
                    "lineage_union": lineage,
                    "collapse_reason": "HISTORICAL_PRESENTATION_VARIANT_FAMILY",
                }
            )
    return collapsed, groups


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

    historical_file_rows = list(base.get("rigged_animated_file_rows") or ())
    if len(historical_file_rows) != 143:
        raise ValueError(f"TRUTH139_V3_HISTORICAL_FILE_ROW_CARDINALITY:{len(historical_file_rows)}")

    packs = {
        str(row["id"]): row
        for row in manifest.get("results") or ()
        if row.get("provider") == "quaternius"
    }
    enriched = []
    provenance_by_pack: dict[str, dict] = {}
    for row in historical_file_rows:
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

        value["source"] = source
        enriched.append(value)

    subjects, duplicate_groups = _collapse_historical_presentation_variants(enriched)
    for value in subjects:
        lineage = dict(value.get("lineage") or {})
        lineage["source_family_signature"] = _source_family_tokens(value)
        value["lineage"] = lineage
        source = dict(value["source"])
        source["source_family_signature"] = list(lineage["source_family_signature"])
        value["source"] = source

    base_counts = dict(base["counts"])
    counts = {
        **base_counts,
        "deduplicated_rigged_animated_subjects": len(subjects),
        "duplicate_files_collapsed": len(historical_file_rows) - len(subjects),
        "development_or_regression_excluded_subjects": sum(bool(row.get("development_exclusion")) for row in subjects),
        "raw_full_family_eligible_subjects_before_development_exclusion": sum(
            bool((row.get("capabilities") or {}).get("FULL_TRUTH139_RAW_ELIGIBLE")) for row in subjects
        ),
        "raw_full_family_eligible_subjects_after_development_exclusion": sum(
            bool(row.get("raw_eligible_for_truth139")) for row in subjects
        ),
    }
    structural_core_passed = bool(
        counts.get("gltf_glb_files_scanned") == seal_v1.EXPECTED_GLTF_GLBS
        and counts.get("files_audited") == seal_v1.EXPECTED_GLTF_GLBS
        and counts.get("file_audit_failures") == 0
        and counts.get("rigged_animated_files") == 143
    )
    historical_cardinality_reproduced = bool(
        counts.get("deduplicated_rigged_animated_subjects") == 139
        and counts.get("duplicate_files_collapsed") == 4
    )

    provenance_rows = [provenance_by_pack[key] for key in sorted(provenance_by_pack)]
    missing = [row["pack_id"] for row in provenance_rows if not row["historical_snapshot_byte_available"]]
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": PASS if structural_core_passed else "FAIL_STRUCTURAL_ASSET_AUDIT__DO_NOT_BUILD_APPARATUS",
        "audit_mode": "STRUCTURAL_CAPABILITY_AUTHORITY__HISTORICAL_SUBJECT_PROJECTION_FROM_143_FILE_ROWS__ARCHIVAL_WEB_PROVENANCE_SEPARATE",
        "structural_core_passed": structural_core_passed,
        "historical_cardinality_reproduced": historical_cardinality_reproduced,
        "historical_reproduction_status": (
            "CARDINALITY_REPRODUCED__FULL_COURT_PENDING"
            if historical_cardinality_reproduced
            else "CARDINALITY_MISMATCH__NONBLOCKING_STRUCTURAL_FINDING"
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
        "base_capability_audit_status": str(base.get("status") or ""),
        "v2_repairs": {
            **dict(base["v2_repairs"]),
            "source_family_lineage": "CREATOR_PACK_NORMALIZED_ASSET_STEM__NOT_PACK_LEVEL_SPLIT",
            "historical_presentation_variant_dedup": "143_FILE_ROWS_THEN_PATH_FAMILY_AFTER_STRIPPING_STYLE_ONLY_FLAT_TEXTURED_SEGMENTS",
            "exact_geometry_identity_is_not_historical_subject_identity": True,
            "historical_cardinality_is_separate_court": True,
            "archival_web_snapshot_bytes_are_nonblocking_provenance": True,
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
        "counts": counts,
        "subjects": subjects,
        "duplicate_groups": duplicate_groups,
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
        "historical_reproduction_status": result["historical_reproduction_status"],
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
