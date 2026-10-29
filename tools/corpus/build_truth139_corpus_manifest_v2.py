#!/usr/bin/env python3
"""Build the capability-aware Truth139 corpus manifest after exact sealed V2 audit.

This is the corpus identity layer, not an experiment projection.  All 139 historically
reproduced rigged+animated subject identities remain present, including development-tainted
subjects and subjects that lack one or more optional truth families.  The manifest records
what may be materialized for each subject; later experiment projections decide which rows
are legal for TRAIN or sealed UNSEEN.

Required parents:
* RealSaS.Truth139SealedRawAudit.v2 PASS
* RealSaS.Truth139HistoricalReproductionCourt.v1 PASS

This tool performs no render, no split, no training, and mints no qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

AUDIT_SCHEMA = "RealSaS.Truth139SealedRawAudit.v2"
AUDIT_STATUS = "PASS_SEALED_RAW_CAPABILITY_AUDIT_V2__PLATFORM_ADMISSION_PENDING"
COURT_SCHEMA = "RealSaS.Truth139HistoricalReproductionCourt.v1"
COURT_STATUS = "PASS_HISTORICAL_139_REPRODUCTION"
OUTPUT_SCHEMA = "RealSaS.Truth139CorpusManifest.v2"
EXPECTED_SUBJECT_COUNT = 139

CAPABILITY_NAMES = (
    "OBSERVATION_FSTAR",
    "SURFACE_NORMAL_TRUTH",
    "STAGE47_MECHANICS",
    "STAGE48_APPEARANCE_PRESENTATION",
    "TESSA_TOPOLOGY_TEACHER",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_binding(row: dict) -> dict:
    source = dict(row.get("source") or {})
    required = ("relative_path", "sha256", "size_bytes", "format", "creator", "pack_id")
    missing = [name for name in required if source.get(name) in (None, "")]
    if missing:
        raise ValueError("TRUTH139_CORPUS_SOURCE_BINDING_MISSING:" + ",".join(missing))
    deps = list(source.get("dependencies") or ())
    if int(source.get("dependency_count", len(deps))) != len(deps):
        raise ValueError("TRUTH139_CORPUS_DEPENDENCY_COUNT_DRIFT:" + str(row.get("subject_id")))
    snapshot = dict(source.get("source_page_snapshot") or {})
    if not str(snapshot.get("path") or "") or len(str(snapshot.get("sha256") or "")) != 64:
        raise ValueError("TRUTH139_CORPUS_PROVENANCE_SNAPSHOT_BINDING_MISSING:" + str(row.get("subject_id")))
    return {
        "corpus_relative_path": str(source["relative_path"]),
        "sha256": str(source["sha256"]),
        "byte_size": int(source["size_bytes"]),
        "format": str(source["format"]),
        "creator": str(source["creator"]),
        "pack_id": str(source["pack_id"]),
        "dependencies": deps,
        "dependency_count": len(deps),
        "source_page_snapshot": snapshot,
        "official_url": str(source.get("official_url") or ""),
        "drive_folder_id": str(source.get("drive_folder_id") or ""),
        "source_family_signature": list(source.get("source_family_signature") or ()),
    }


def _materialization_plan(capabilities: dict) -> dict:
    observation = bool(capabilities.get("OBSERVATION_FSTAR"))
    normals = bool(capabilities.get("SURFACE_NORMAL_TRUTH"))
    stage47 = bool(capabilities.get("STAGE47_MECHANICS"))
    stage48 = bool(capabilities.get("STAGE48_APPEARANCE_PRESENTATION"))
    tessa = bool(capabilities.get("TESSA_TOPOLOGY_TEACHER"))
    return {
        "CONTROLLED_8VIEW_OBSERVATION": observation,
        "DINO_AND_IMAGE_DERIVED_FSTAR_SHARDS": observation,
        "MULTIVIEW_NORMAL_DEPTH_TEACHER_TRUTH": observation and normals,
        "STAGE47_SOURCE_RESPONSE_TRUTH": stage47,
        "STAGE48_APPEARANCE_PRESENTATION_TRUTH": stage48,
        "TESSA_TOPOLOGY_TEACHER_CACHE": tessa,
    }


def build(audit: dict, court: dict) -> dict:
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("status") != AUDIT_STATUS:
        raise ValueError("TRUTH139_CORPUS_V2_AUDIT_NOT_PASS")
    if court.get("schema") != COURT_SCHEMA or court.get("status") != COURT_STATUS:
        raise ValueError("TRUTH139_CORPUS_V2_HISTORICAL_REPRODUCTION_NOT_PASS")
    if str(court.get("source_audit_identity_sha256") or "") != str(audit.get("audit_identity_sha256") or ""):
        raise ValueError("TRUTH139_CORPUS_V2_PARENT_IDENTITY_MISMATCH")
    corpus = dict(audit.get("corpus") or {})
    if corpus.get("seal_verified") is not True:
        raise ValueError("TRUTH139_CORPUS_V2_SEAL_NOT_VERIFIED")

    subjects = list(audit.get("subjects") or ())
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise ValueError(f"TRUTH139_CORPUS_V2_SUBJECT_COUNT:{len(subjects)}")

    rows = []
    capability_counts = Counter()
    development_count = 0
    full_family_clean_count = 0
    pack_counts = Counter()
    for ordinal, row in enumerate(sorted(subjects, key=lambda item: str(item.get("subject_id") or "")), 1):
        subject_id = str(row.get("subject_id") or "")
        if not subject_id:
            raise ValueError("TRUTH139_CORPUS_V2_SUBJECT_ID_MISSING")
        capabilities = {name: bool((row.get("capabilities") or {}).get(name)) for name in CAPABILITY_NAMES}
        for name, enabled in capabilities.items():
            capability_counts[name] += int(enabled)
        development_exclusion = row.get("development_exclusion")
        development_taint = list(row.get("duplicate_family_development_taint") or ())
        is_development = bool(development_exclusion or development_taint)
        development_count += int(is_development)
        full_family = all(capabilities.values())
        full_family_clean = full_family and not is_development
        full_family_clean_count += int(full_family_clean)
        source = _source_binding(row)
        pack_counts[source["pack_id"]] += 1
        rows.append(
            {
                "ordinal": ordinal,
                "subject_id": subject_id,
                "subject_identity_signature": str(row.get("subject_identity_signature") or ""),
                "source": source,
                "lineage": dict(row.get("lineage") or {}),
                "raw_structure": dict(row.get("raw_structure") or {}),
                "duplicate_source_file_count": int(row.get("duplicate_source_file_count", 1)),
                "duplicate_source_files": list(row.get("duplicate_source_files") or ()),
                "raw_capabilities": capabilities,
                "raw_eligibility_blockers": list(row.get("raw_eligibility_blockers") or ()),
                "development_exclusion": development_exclusion,
                "duplicate_family_development_taint": development_taint,
                "development_or_regression_tainted": is_development,
                "materialization_plan": _materialization_plan(capabilities),
                "full_family_raw_capable": full_family,
                "full_family_clean_experiment_candidate": full_family_clean,
                "platform_truth_admission": "PENDING",
                "train_membership": "UNASSIGNED",
                "unseen_membership": "UNASSIGNED",
                "training_authorized": False,
            }
        )

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": "SEALED_RAW_139_REPRODUCED__CAPABILITY_AWARE_MATERIALIZATION_PENDING",
        "parents": {
            "sealed_raw_audit_identity_sha256": str(audit["audit_identity_sha256"]),
            "historical_reproduction_court_identity_sha256": str(court["court_identity_sha256"]),
        },
        "corpus": {
            "root_sha256": str(corpus.get("root_sha256") or ""),
            "manifest_sha256": str(corpus.get("manifest_sha256") or ""),
            "ledger_sha256": str(corpus.get("ledger_sha256") or ""),
            "seal_json_sha256": str(corpus.get("seal_json_sha256") or ""),
            "historical_subject_count": EXPECTED_SUBJECT_COUNT,
            "subject_count": len(rows),
            "development_or_regression_tainted_subject_count": development_count,
            "clean_full_family_raw_candidate_count": full_family_clean_count,
            "capability_counts": dict(sorted(capability_counts.items())),
            "pack_counts": dict(sorted(pack_counts.items())),
        },
        "policy": {
            "corpus_identity_is_not_experiment_projection": True,
            "all_historical_139_subject_identities_retained": True,
            "development_taint_does_not_delete_corpus_rows": True,
            "missing_optional_truth_family_does_not_delete_corpus_rows": True,
            "heavy_truth_materialization_must_follow_per_subject_capability_plan": True,
            "split_before_platform_truth_admission_forbidden": True,
            "training_before_split_and_unseen_seal_forbidden": True,
            "subject_specific_code_forbidden": True,
        },
        "external_worker": {
            "contract": "canonical/TRUTH139_EXTERNAL_WORKER_NOTEBOOK_CONTRACT_V1.json",
            "intended_notebook_repo_path": "notebooks/Truth139_Corpus_Preparation_T4_CPU_v1.ipynb",
            "authority": "NON_AUTHORITATIVE_COMPUTE_WORKER",
            "platform_remains_lineage_authority": True,
        },
        "subjects": rows,
        "split_created": False,
        "unseen_sealed": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["corpus_manifest_identity_sha256"] = _digest(payload)
    return payload


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-raw-audit-v2", required=True)
    parser.add_argument("--historical-reproduction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = json.loads(Path(args.sealed_raw_audit_v2).resolve().read_text())
    court = json.loads(Path(args.historical_reproduction).resolve().read_text())
    result = build(audit, court)
    output = Path(args.output).resolve()
    _write(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": result["status"],
                "subject_count": result["corpus"]["subject_count"],
                "development_or_regression_tainted_subject_count": result["corpus"]["development_or_regression_tainted_subject_count"],
                "clean_full_family_raw_candidate_count": result["corpus"]["clean_full_family_raw_candidate_count"],
                "corpus_manifest_identity_sha256": result["corpus_manifest_identity_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
