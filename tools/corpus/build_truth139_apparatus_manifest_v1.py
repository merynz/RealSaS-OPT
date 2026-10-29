#!/usr/bin/env python3
"""Build the pre-Platform Truth139 source/import/render apparatus roster.

Consumes only the exact-seal PASS audit.  It does not split subjects and does not authorize
training.  The output deliberately matches the historical FIT8 ``selected`` row shape so the same
common observation producers can be reused without subject-specific branches, while preserving
external glTF dependency and source-provenance bindings discovered by the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


AUDIT_SCHEMA = "RealSaS.Truth139SealedRawAudit.v1"
OUTPUT_SCHEMA = "RealSaS.Truth139ApparatusManifest.v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def build(audit: dict) -> dict:
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("TRUTH139_APPARATUS_AUDIT_SCHEMA")
    if audit.get("status") != "PASS_SEALED_RAW_CAPABILITY_AUDIT__PLATFORM_ADMISSION_PENDING":
        raise ValueError("TRUTH139_APPARATUS_RAW_AUDIT_NOT_PASS")
    corpus=dict(audit.get("corpus") or {})
    if corpus.get("seal_verified") is not True:
        raise ValueError("TRUTH139_APPARATUS_CORPUS_SEAL_NOT_VERIFIED")
    subjects = [dict(row) for row in audit.get("subjects") or () if row.get("raw_eligible_for_truth139") is True]
    if not subjects:
        raise ValueError("TRUTH139_APPARATUS_NO_RAW_ELIGIBLE_SUBJECTS")

    selected = []
    for ordinal, row in enumerate(sorted(subjects, key=lambda item: str(item["subject_id"])), 1):
        source = dict(row["source"])
        capabilities = dict(row.get("capabilities") or {})
        required = (
            "OBSERVATION_FSTAR",
            "SURFACE_NORMAL_TRUTH",
            "STAGE47_MECHANICS",
            "STAGE48_APPEARANCE_PRESENTATION",
            "TESSA_TOPOLOGY_TEACHER",
            "FULL_TRUTH139_RAW_ELIGIBLE",
        )
        if any(capabilities.get(name) is not True for name in required):
            raise ValueError("TRUTH139_APPARATUS_CAPABILITY_DRIFT:" + str(row["subject_id"]))
        if row.get("development_exclusion") is not None:
            raise ValueError("TRUTH139_APPARATUS_DEVELOPMENT_SUBJECT_PRESENT:" + str(row["subject_id"]))
        snapshot=dict(source.get("source_page_snapshot") or {})
        if len(str(snapshot.get("sha256") or ""))!=64 or not str(snapshot.get("path") or ""):
            raise ValueError("TRUTH139_APPARATUS_SOURCE_PAGE_BINDING_MISSING:"+str(row["subject_id"]))
        dependencies=list(source.get("dependencies") or ())
        if int(source.get("dependency_count",-1))!=len(dependencies):
            raise ValueError("TRUTH139_APPARATUS_DEPENDENCY_COUNT_DRIFT:"+str(row["subject_id"]))
        selected.append(
            {
                "ordinal": ordinal,
                "subject_key": str(row["subject_id"]),
                "display_name": Path(str(source["relative_path"])).stem,
                "creator": str(source["creator"]),
                "pack_id": str(source["pack_id"]),
                "source": {
                    "corpus_relative_path": str(source["relative_path"]),
                    "sha256": str(source["sha256"]),
                    "byte_size": int(source["size_bytes"]),
                    "format": str(source["format"]),
                    "license": "CC0-1.0" if str(source["creator"]).lower() == "quaternius" else "AUDIT_REQUIRED",
                    "dependencies": dependencies,
                    "source_page_snapshot": snapshot,
                    "official_url": str(source.get("official_url") or ""),
                    "drive_folder_id": str(source.get("drive_folder_id") or ""),
                },
                "lineage": dict(row["lineage"]),
                "subject_identity_signature": str(row["subject_identity_signature"]),
                "raw_capabilities": capabilities,
                "raw_structure": dict(row.get("raw_structure") or {}),
                "raw_audit_source_count": int(row.get("duplicate_source_file_count", 1)),
                "import_render_admission": "PENDING",
                "stage47_truth": "PENDING",
                "stage48_truth": "PENDING",
                "training_authorized": False,
            }
        )

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": "READY_FOR_COMMON_SOURCE_IMPORT_RENDER_APPARATUS__NOT_PLATFORM_TRUTH_ADMITTED",
        "raw_audit_identity_sha256": str(audit["audit_identity_sha256"]),
        "raw_corpus_root_path": str(corpus.get("root_path") or ""),
        "expected_corpus_root_sha256": str(corpus.get("root_sha256") or ""),
        "corpus_manifest_sha256":str(corpus.get("manifest_sha256") or ""),
        "corpus_ledger_sha256":str(corpus.get("ledger_sha256") or ""),
        "subject_count": len(selected),
        "selected": selected,
        "observation_apparatus": {
            "producer": "tools/corpus/prepare_iris_observations_v1.py",
            "renderer": "tools/observation/blender_controlled_8view_v2.py",
            "independent_import_probe": "tools/observation/blender_source_parity_export_v1.py",
            "admission": "tools/corpus/iris_observation_admission_v1.py",
            "policy": "canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json",
            "view_count": 8,
            "source_dependency_closure_required":True,
            "subject_specific_code_forbidden": True
        },
        "post_observation_sequence": [
            "BUILD_GENERIC_IMAGE_DERIVED_FSTAR_AND_DINO_SHARDS",
            "BOOTSTRAP_PLATFORM_SOURCE_TO_STAGE48_CAMPAIGN",
            "COLLECT_STAGE47_STAGE48_TRUTH",
            "BUILD_SURFACE_NORMAL_TRUTH",
            "FINAL_TRUTH_ADMISSION_AND_LINEAGE_LEDGER",
            "ONLY_THEN_BUILD_TRAIN_UNSEEN_SPLIT"
        ],
        "split_created": False,
        "unseen_sealed": False,
        "training_authorized": False,
    }
    payload["apparatus_identity_sha256"] = _digest(payload)
    return payload


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".partial")
    staged.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit_path = Path(args.raw_audit).resolve()
    audit = json.loads(audit_path.read_text())
    result = build(audit)
    output = Path(args.output).resolve()
    _write(output, result)
    print(json.dumps({"output": str(output), "subject_count": result["subject_count"], "apparatus_identity_sha256": result["apparatus_identity_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
