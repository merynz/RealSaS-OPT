#!/usr/bin/env python3
"""Truth139 raw capability audit V2.

V2 preserves the byte/geometry scanner from V1 but fixes three corpus-level
semantics before any decisive 139 audit is allowed:

* TESSA raw representability is gated on total asset vertices/faces, matching
  the canonical sequence codec scale policy rather than per primitive limits.
* Flat/Textured (or otherwise exact-identity) duplicate families carry the
  union of every variant's leakage lineage and development taint.
* The representative source is chosen by truth capability first, then the
  historical textured/non-flat preference, never by filename preference alone.

This remains a raw-byte pre-admission audit. Blender import/render, Platform
admission, split selection, training and qualification are still forbidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.tessa.v1.model_v1 import TESSAConfigV1
from tools.corpus import audit_truth139_gltf_capabilities_v1 as v1

AUDIT_SCHEMA = "RealSaS.Truth139RawCapabilityAudit.v2"
SUBJECT_SCHEMA = "RealSaS.Truth139RawSubjectCapability.v2"


def _total_tessa_raw_precheck(row: dict) -> tuple[bool, dict]:
    cfg = TESSAConfigV1()
    structure = row.get("raw_structure") or {}
    total_vertices = int(structure.get("triangle_position_count_total", structure.get("vertex_count", 0)))
    total_faces = int(structure.get("triangle_face_count_total", structure.get("face_count", 0)))
    valid = total_vertices > 0 and total_faces > 0 and total_vertices <= cfg.max_vertices and total_faces <= cfg.max_faces
    return valid, {
        "policy": "TOTAL_ASSET_RAW_PRECHECK__FINAL_CODEC_COURT_AFTER_ADMITTED_IMPORT",
        "total_vertex_count": total_vertices,
        "total_face_count": total_faces,
        "max_vertices": int(cfg.max_vertices),
        "max_faces": int(cfg.max_faces),
    }


def _capability_rank(row: dict) -> tuple:
    caps = row.get("capabilities") or {}
    full = int(bool(caps.get("FULL_TRUTH139_RAW_ELIGIBLE")))
    family_count = sum(int(bool(caps.get(name))) for name in (
        "OBSERVATION_FSTAR", "SURFACE_NORMAL_TRUTH", "STAGE47_MECHANICS",
        "STAGE48_APPEARANCE_PRESENTATION", "TESSA_TOPOLOGY_TEACHER"))
    path = str((row.get("source") or {}).get("relative_path") or "").lower()
    flat_penalty = int("/flat/" in path or "flat" in Path(path).stem.lower())
    textured_penalty = 0 if "textured" in path else 1
    return (-full, -family_count, flat_penalty, textured_penalty, path)


def _union_lineage(group: list[dict]) -> dict:
    fields = (
        "source_sha256", "source_identity", "skeleton_hierarchy_signature",
        "topology_lineage_signature", "material_lineage_signature",
    )
    return {
        field: sorted({str(token) for row in group for token in ((row.get("lineage") or {}).get(field) or ())})
        for field in fields
    }


def _group_development_taint(group: list[dict]) -> list[str]:
    return sorted({str(row["development_exclusion"]) for row in group if row.get("development_exclusion")})


def _audit_file_v2(path: Path, raw_root: Path, development_shas: set[str], development_aliases: set[tuple[str, str, str]]) -> dict:
    row = v1._audit_file(path, raw_root, development_shas, development_aliases)
    structure = dict(row.get("raw_structure") or {})
    structure["triangle_position_count_total"] = int(structure.get("vertex_count", 0))
    structure["triangle_face_count_total"] = int(structure.get("face_count", 0))
    row["raw_structure"] = structure
    tessa_ok, tessa_court = _total_tessa_raw_precheck(row)
    row["capabilities"] = dict(row["capabilities"])
    row["capabilities"]["TESSA_TOPOLOGY_TEACHER"] = bool(tessa_ok)
    row["capabilities"]["FULL_TRUTH139_RAW_ELIGIBLE"] = bool(
        row["capabilities"].get("OBSERVATION_FSTAR")
        and row["capabilities"].get("SURFACE_NORMAL_TRUTH")
        and row["capabilities"].get("STAGE47_MECHANICS")
        and row["capabilities"].get("STAGE48_APPEARANCE_PRESENTATION")
        and tessa_ok
    )
    row["tessa_raw_representability"] = tessa_court
    blockers = [b for b in list(row.get("raw_eligibility_blockers") or ()) if b != "TESSA_TOPOLOGY_TEACHER"]
    if not tessa_ok:
        blockers.append("TESSA_TOPOLOGY_TOTAL_ASSET_SCALE")
    row["raw_eligibility_blockers"] = sorted(set(blockers))
    row["schema"] = SUBJECT_SCHEMA
    return row


def audit(raw_root: Path, *, fit8_selection: Path, expected_corpus_root_sha256: str = v1.EXPECTED_CORPUS_ROOT) -> dict:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    development_shas, development_aliases = v1._development_aliases(fit8_selection)
    scan_root = raw_root / "raw" / "quaternius"
    if not scan_root.is_dir():
        scan_root = raw_root
    paths = sorted({*scan_root.rglob("*.gltf"), *scan_root.rglob("*.glb")})
    file_rows, failures = [], []
    for index, path in enumerate(paths, 1):
        try:
            row = _audit_file_v2(path, raw_root, development_shas, development_aliases)
            file_rows.append(row)
            print(
                f"[TRUTH139_RAW_AUDIT_V2] {index}/{len(paths)} {row['source']['relative_path']} "
                f"rigged_animated={int(row['historical_rigged_animated_candidate'])} "
                f"full={int(row['capabilities']['FULL_TRUTH139_RAW_ELIGIBLE'])}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"relative_path": path.relative_to(raw_root).as_posix(), "error": f"{type(exc).__name__}:{exc}"})
            print(f"[TRUTH139_RAW_AUDIT_V2_FAIL] {index}/{len(paths)} {path.relative_to(raw_root)} {type(exc).__name__}:{exc}", flush=True)

    candidates = [row for row in file_rows if row["historical_rigged_animated_candidate"]]
    grouped: dict[str, list[dict]] = {}
    for row in candidates:
        grouped.setdefault(str(row["subject_identity_signature"]), []).append(row)

    subjects, duplicate_groups = [], []
    for identity, group in sorted(grouped.items()):
        ordered = sorted(group, key=_capability_rank)
        representative = dict(ordered[0])
        taint = _group_development_taint(group)
        lineage = _union_lineage(group)
        subject_id = "truth139_" + identity[:20]
        full = bool(representative["capabilities"]["FULL_TRUTH139_RAW_ELIGIBLE"])
        blockers = [b for b in list(representative.get("raw_eligibility_blockers") or ()) if b != "DEVELOPMENT_OR_REGRESSION_EXCLUSION"]
        if taint:
            blockers.append("DUPLICATE_FAMILY_DEVELOPMENT_OR_REGRESSION_EXCLUSION")
        row = {
            **representative,
            "schema": SUBJECT_SCHEMA,
            "subject_id": subject_id,
            "subject_identity_signature": identity,
            "lineage": lineage,
            "duplicate_source_files": [item["source"] for item in sorted(group, key=lambda item: item["source"]["relative_path"])],
            "duplicate_source_file_count": len(group),
            "duplicate_family_development_taint": taint,
            "development_exclusion": ("DUPLICATE_FAMILY_DEVELOPMENT_TAINT:" + ",".join(taint)) if taint else None,
            "raw_eligibility_blockers": sorted(set(blockers)),
            "representative_selection_policy": "MAX_TRUTH_CAPABILITY_THEN_TEXTURED_NONFLAT_THEN_PATH",
            "raw_eligible_for_truth139": bool(full and not taint),
            "admitted_for_truth139": False,
            "training_authorized": False,
        }
        subjects.append(row)
        if len(group) > 1:
            duplicate_groups.append({
                "subject_id": subject_id,
                "identity": identity,
                "representative": representative["source"]["relative_path"],
                "source_files": [item["source"]["relative_path"] for item in sorted(group, key=lambda item: item["source"]["relative_path"])],
                "development_taint": taint,
                "lineage_union": lineage,
            })

    counts = {
        "gltf_glb_files_scanned": len(paths),
        "files_audited": len(file_rows),
        "file_audit_failures": len(failures),
        "rigged_animated_files": len(candidates),
        "deduplicated_rigged_animated_subjects": len(subjects),
        "duplicate_files_collapsed": len(candidates) - len(subjects),
        "development_or_regression_excluded_subjects": sum(bool(row["development_exclusion"]) for row in subjects),
        "raw_full_family_eligible_subjects_before_development_exclusion": sum(bool(row["capabilities"]["FULL_TRUTH139_RAW_ELIGIBLE"]) for row in subjects),
        "raw_full_family_eligible_subjects_after_development_exclusion": sum(bool(row["raw_eligible_for_truth139"]) for row in subjects),
    }
    historical_match = (
        counts["gltf_glb_files_scanned"] == v1.EXPECTED_GLTF_FILES
        and counts["rigged_animated_files"] == v1.EXPECTED_RIGGED_ANIMATED_FILES
        and counts["deduplicated_rigged_animated_subjects"] == v1.EXPECTED_DEDUP_SUBJECTS
        and not failures
    )
    payload = {
        "schema": AUDIT_SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING" if historical_match else "FAIL_OR_INCOMPLETE_RAW_CAPABILITY_AUDIT_V2__DO_NOT_SPLIT_OR_TRAIN",
        "corpus_root_path": str(raw_root),
        "expected_corpus_root_sha256": expected_corpus_root_sha256,
        "corpus_root_hash_recomputed_by_this_tool": False,
        "fit8_selection_path": str(fit8_selection.resolve()),
        "historical_expectations": {
            "gltf_glb_files": v1.EXPECTED_GLTF_FILES,
            "rigged_animated_files": v1.EXPECTED_RIGGED_ANIMATED_FILES,
            "deduplicated_subjects": v1.EXPECTED_DEDUP_SUBJECTS,
        },
        "v2_repairs": {
            "tessa_scale_unit": "TOTAL_ASSET_RAW_PRECHECK",
            "duplicate_family_lineage": "UNION_ALL_VARIANTS",
            "duplicate_family_development_taint": "UNION_ALL_VARIANTS",
            "representative_selection": "CAPABILITY_FIRST",
        },
        "historical_cardinality_reproduced": historical_match,
        "counts": counts,
        "duplicate_groups": duplicate_groups,
        "subjects": subjects,
        "file_failures": failures,
        "split_authorized": False,
        "training_authorized": False,
        "next_gate": "PLATFORM_IMPORT_RENDER_STAGE47_STAGE48_APPARATUS",
    }
    payload["audit_identity_sha256"] = v1._digest_json(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--fit8-selection", default="canonical/FIT8_DIVERSITY_SELECTION_V1.json")
    parser.add_argument("--expected-corpus-root-sha256", default=v1.EXPECTED_CORPUS_ROOT)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit(
        Path(args.raw_root),
        fit8_selection=Path(args.fit8_selection).resolve(),
        expected_corpus_root_sha256=args.expected_corpus_root_sha256,
    )
    output = Path(args.output).resolve()
    v1._atomic_json(output, result)
    print(json.dumps({"output": str(output), "status": result["status"], "counts": result["counts"], "audit_identity_sha256": result["audit_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
