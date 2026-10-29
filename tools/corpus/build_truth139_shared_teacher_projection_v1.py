#!/usr/bin/env python3
"""Project the sealed Truth139 corpus identity into model-specific teacher capabilities.

This is a planning/admission manifest only.  It never turns raw capability into derived truth,
never fabricates MIRA carrier weights, never creates a train/unseen split and never authorizes
training.  The historical raw-audit vocabulary remains unchanged so the October-4 reproduction
court can finish independently; this layer gives the same 139 identities explicit IRIS/TESSA/
AXIS/MIRA materialization obligations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

INPUT_SCHEMA = "RealSaS.Truth139CorpusManifest.v2"
OUTPUT_SCHEMA = "RealSaS.Truth139SharedTeacherProjectionManifest.v1"
EXPECTED_SUBJECT_COUNT = 139

CONTRACTS = {
    "shared": "canonical/TRUTH139_SHARED_TEACHER_SUBSTRATE_V1.json",
    "graphics": "canonical/TRUTH139_DENSE_GRAPHICS_TEACHER_V1.json",
    "response": "canonical/TRUTH139_IRIS_RESPONSE_EVIDENCE_TEACHER_V1.json",
    "axis": "canonical/TRUTH139_AXIS_TEACHER_V1.json",
    "mira": "canonical/TRUTH139_MIRA_TEACHER_V1.json",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _projection(raw: dict) -> dict[str, bool]:
    observation = bool(raw.get("OBSERVATION_FSTAR"))
    normals = bool(raw.get("SURFACE_NORMAL_TRUTH"))
    stage47 = bool(raw.get("STAGE47_MECHANICS"))
    stage48 = bool(raw.get("STAGE48_APPEARANCE_PRESENTATION"))
    tessa = bool(raw.get("TESSA_TOPOLOGY_TEACHER"))
    return {
        "IRIS_GEOMETRY_TEACHER_CANDIDATE": observation and normals,
        "IRIS_DENSE_GRAPHICS_MATERIALIZATION_CANDIDATE": observation and stage47,
        "IRIS_DENSE_GRAPHICS_FULL_COURT_CANDIDATE": observation and stage47 and stage48,
        "IRIS_RELATIONS_TEACHER_CANDIDATE": observation and stage47,
        "IRIS_RESPONSE_EVIDENCE_TEACHER_CANDIDATE": stage47,
        "TESSA_TOPOLOGY_TEACHER_CANDIDATE": tessa,
        "AXIS_RIG_TEACHER_CANDIDATE": stage47,
        "MIRA_SOURCE_SKIN_TEACHER_CANDIDATE": stage47,
        "MIRA_EXACT_STAGE19_W_M_TEACHER_READY": False,
        "POSE_DIVERSITY_TEACHER_CANDIDATE": stage47,
    }


def build(manifest: dict) -> dict:
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("TRUTH139_SHARED_TEACHER_PARENT_SCHEMA")
    subjects = list(manifest.get("subjects") or ())
    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        raise ValueError(f"TRUTH139_SHARED_TEACHER_SUBJECT_COUNT:{len(subjects)}")
    parent_identity = str(manifest.get("corpus_manifest_identity_sha256") or "")
    if len(parent_identity) != 64:
        raise ValueError("TRUTH139_SHARED_TEACHER_PARENT_IDENTITY")

    counts = Counter()
    rows = []
    for row in sorted(subjects, key=lambda value: str(value.get("subject_id") or "")):
        subject_id = str(row.get("subject_id") or "")
        if not subject_id:
            raise ValueError("TRUTH139_SHARED_TEACHER_SUBJECT_ID")
        raw = dict(row.get("raw_capabilities") or {})
        projection = _projection(raw)
        for name, enabled in projection.items():
            counts[name] += int(enabled)
        development = bool(row.get("development_or_regression_tainted"))
        statuses = {
            name: (
                "REQUIRES_EXACT_STAGE19_M_AND_AXIS_G_AFTER_MODEL_INDEPENDENT_CORPUS_MATERIALIZATION"
                if name == "MIRA_EXACT_STAGE19_W_M_TEACHER_READY"
                else ("PENDING_MATERIALIZATION_AND_INDEPENDENT_COURT" if enabled else "NOT_RAW_CAPABLE")
            )
            for name, enabled in projection.items()
        }
        rows.append(
            {
                "subject_id": subject_id,
                "source_sha256": str((row.get("source") or {}).get("sha256") or ""),
                "subject_identity_signature": str(row.get("subject_identity_signature") or ""),
                "development_or_regression_tainted": development,
                "eligible_for_future_train_or_unseen_projection_after_truth_admission": not development,
                "raw_capabilities": raw,
                "teacher_projection_capabilities": projection,
                "teacher_projection_status": statuses,
                "axis_teacher_authority": "PENDING" if projection["AXIS_RIG_TEACHER_CANDIDATE"] else "NOT_RAW_CAPABLE",
                "mira_source_skin_authority": "PENDING" if projection["MIRA_SOURCE_SKIN_TEACHER_CANDIDATE"] else "NOT_RAW_CAPABLE",
                "mira_exact_w_m_authority": "NOT_MINTED__REQUIRES_EXACT_STAGE19_CARRIER_AND_MAPPING_COURT",
                "training_authorized": False,
            }
        )

    payload = {
        "schema": OUTPUT_SCHEMA,
        "date": "2026-10-29",
        "status": "SHARED_TEACHER_PROJECTION_PLANNED__DERIVED_TRUTH_NOT_YET_ADMITTED",
        "parent_corpus_manifest_identity_sha256": parent_identity,
        "contracts": CONTRACTS,
        "policy": {
            "raw_audit_vocabulary_is_preserved": True,
            "stage47_historical_capability_name_is_not_axis_or_mira_model_authority": True,
            "stage47_is_used_here_only_as_a_strong_raw_precondition_for_rig_skin_response_and_pose_teacher_materialization": True,
            "mira_exact_w_m_may_not_be_minted_from_raw_source_weights": True,
            "axis_continuous_rig_truth_is_preserved_before_any_model_specific_xyz_quantization": True,
            "corpus_identity_is_not_experiment_projection": True,
            "no_split_or_training_authority": True,
        },
        "counts": dict(sorted(counts.items())),
        "subject_count": len(rows),
        "subjects": rows,
        "split_created": False,
        "training_authorized": False,
        "qualification_minted": False,
    }
    payload["shared_teacher_projection_identity_sha256"] = _digest(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest-v2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = json.loads(args.corpus_manifest_v2.resolve().read_text())
    result = build(value)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({
        "status": result["status"],
        "subject_count": result["subject_count"],
        "counts": result["counts"],
        "shared_teacher_projection_identity_sha256": result["shared_teacher_projection_identity_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
