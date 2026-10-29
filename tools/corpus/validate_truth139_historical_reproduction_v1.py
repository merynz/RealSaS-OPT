#!/usr/bin/env python3
"""Cross-check a sealed Truth139 V2 audit against the October-4 historical census.

This court is deliberately independent of the 395->143->139 total-cardinality check inside the
scanner.  It verifies that the same 139 identities are distributed across the same source packs
and still exhibit the historical skeleton-lineage cardinality.  A coincidentally-correct total
cannot pass if deduplication/grouping moved subjects between families.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.Truth139HistoricalReproductionCourt.v1"
EXPECTED_PACK_COUNTS = {
    "quaternius-ultimate-monsters": 50,
    "quaternius-cute-animated-monsters": 21,
    "quaternius-ultimate-animated-animals": 12,
    "quaternius-ultimate-space-kit": 12,
    "quaternius-ultimate-modular-men": 11,
    "quaternius-ultimate-modular-women": 10,
    "quaternius-platformer-kit": 10,
    "quaternius-rpg-characters": 6,
    "quaternius-animated-mech": 4,
    "quaternius-toon-shooter-kit": 3,
}
EXPECTED_SUBJECT_COUNT = 139
EXPECTED_EXACT_SKELETON_SIGNATURE_COUNT = 33


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate(audit: dict) -> dict:
    if audit.get("schema") != "RealSaS.Truth139SealedRawAudit.v2":
        raise ValueError("TRUTH139_REPRO_AUDIT_SCHEMA")
    if audit.get("status") != "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING":
        raise ValueError("TRUTH139_REPRO_AUDIT_NOT_PASS")
    subjects = list(audit.get("subjects") or ())
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise ValueError("TRUTH139_REPRO_SUBJECT_COUNT")

    packs = Counter(str((row.get("source") or {}).get("pack_id") or "") for row in subjects)
    actual_pack_counts = dict(sorted(packs.items()))
    pack_match = actual_pack_counts == dict(sorted(EXPECTED_PACK_COUNTS.items()))

    skeleton_tokens = {
        str(token)
        for row in subjects
        for token in ((row.get("lineage") or {}).get("skeleton_hierarchy_signature") or ())
    }
    skeleton_match = len(skeleton_tokens) == EXPECTED_EXACT_SKELETON_SIGNATURE_COUNT

    duplicate_groups = list(audit.get("duplicate_groups") or ())
    duplicate_source_files_collapsed = sum(max(0, len(row.get("source_files") or ()) - 1) for row in duplicate_groups)
    collapse_match = duplicate_source_files_collapsed == 4

    raw = [dict(row.get("raw_structure") or {}) for row in subjects]
    derived = {
        "deduplicated_animation_clip_count": sum(int(row.get("animation_count", 0)) for row in raw),
        "deduplicated_skinned_vertex_count": sum(int(row.get("skinned_vertex_count", 0)) for row in raw),
        "deduplicated_zero_weight_vertex_count": sum(int(row.get("zero_weight_vertex_count", 0)) for row in raw),
        "maximum_influences_per_vertex": max((int(row.get("maximum_influences_per_vertex", 0)) for row in raw), default=0),
        "maximum_weight_sum_error": max((float(row.get("maximum_weight_sum_error", 0.0)) for row in raw), default=0.0),
    }

    passed = pack_match and skeleton_match and collapse_match
    payload = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_HISTORICAL_139_REPRODUCTION" if passed else "FAIL_HISTORICAL_139_REPRODUCTION",
        "source_audit_identity_sha256": str(audit.get("audit_identity_sha256") or ""),
        "expected_subject_count": EXPECTED_SUBJECT_COUNT,
        "actual_subject_count": len(subjects),
        "expected_pack_counts": dict(sorted(EXPECTED_PACK_COUNTS.items())),
        "actual_pack_counts": actual_pack_counts,
        "pack_distribution_exact": pack_match,
        "expected_exact_skeleton_signature_count": EXPECTED_EXACT_SKELETON_SIGNATURE_COUNT,
        "actual_exact_skeleton_signature_count": len(skeleton_tokens),
        "skeleton_signature_cardinality_exact": skeleton_match,
        "expected_duplicate_source_files_collapsed": 4,
        "actual_duplicate_source_files_collapsed": duplicate_source_files_collapsed,
        "duplicate_collapse_exact": collapse_match,
        "deduplicated_derived_metrics_non_authoritative": derived,
        "split_authorized": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["court_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).resolve().read_text())
    result = validate(audit)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({"status": result["status"], "court_identity_sha256": result["court_identity_sha256"], "actual_pack_counts": result["actual_pack_counts"]}, sort_keys=True), flush=True)
    if result["status"] != "PASS_HISTORICAL_139_REPRODUCTION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
