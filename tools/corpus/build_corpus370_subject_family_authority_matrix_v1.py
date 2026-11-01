#!/usr/bin/env python3
"""Build fail-closed Corpus370 subject × truth-family authority matrix.

This tool aggregates already-courted evidence. It never infers scientific truth
from file presence. Missing family evidence becomes UNKNOWN, never PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

SCHEMA = "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1"
FAMILIES = (
    "SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF",
    "GLOBAL_SOLID_SDF", "SURFACE_NORMALS", "APPEARANCE_CAA",
    "VISIBILITY_OCCLUSION", "RAY_COMPOSITING",
    "MECHANICAL_SURFACE_TOPOLOGY", "CAUSAL_RESPONSE", "RIG_HIERARCHY",
    "SKIN_WEIGHTS", "ANIMATION_KINEMATICS", "TESSA_TEACHER",
)
ALLOWED = frozenset({"PASS", "MASKED", "FAIL", "QUARANTINED", "UNKNOWN"})
MODEL_POLICY: Mapping[str, Mapping[str, tuple[str, ...]]] = {
    "IRIS": {
        "required_pass": ("SOURCE_IMPORT", "CAMERA_OBSERVATION", "SURFACE_GEOMETRY_UDF"),
        "optional_maskable": ("GLOBAL_SOLID_SDF",),
    },
    "TESSA": {
        "required_pass": ("SOURCE_IMPORT", "MECHANICAL_SURFACE_TOPOLOGY", "TESSA_TEACHER"),
        "optional_maskable": (),
    },
    "MIRA": {
        "required_pass": ("SOURCE_IMPORT", "RIG_HIERARCHY", "SKIN_WEIGHTS"),
        "optional_maskable": (),
    },
    "AXIS": {
        "required_pass": ("SOURCE_IMPORT", "RIG_HIERARCHY"),
        "optional_maskable": ("ANIMATION_KINEMATICS",),
    },
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalize_cell(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {"status": "UNKNOWN", "reason_codes": ["NO_FAMILY_EVIDENCE"], "evidence": []}
    if not isinstance(raw, dict):
        raise ValueError("family cell must be an object")
    status = str(raw.get("status", "UNKNOWN")).upper()
    if status not in ALLOWED:
        raise ValueError(f"invalid family status {status}")
    reasons = raw.get("reason_codes", [])
    evidence = raw.get("evidence", [])
    if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
        raise ValueError("reason_codes must be list[str]")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be list")
    cell = {
        "status": status,
        "reason_codes": sorted(set(reasons)),
        "evidence": evidence,
    }
    for key in ("producer", "court", "receipt_sha256", "notes"):
        if key in raw:
            cell[key] = raw[key]
    return cell


def derive_model_disposition(families: Mapping[str, Mapping[str, Any]], model: str) -> Dict[str, Any]:
    policy = MODEL_POLICY[model]
    unresolved = []
    blockers = []
    for family in policy["required_pass"]:
        status = families[family]["status"]
        if status == "UNKNOWN":
            unresolved.append(family)
        elif status != "PASS":
            blockers.append({"family": family, "status": status})
    for family in policy["optional_maskable"]:
        status = families[family]["status"]
        if status == "UNKNOWN":
            unresolved.append(family)
        elif status not in {"PASS", "MASKED"}:
            blockers.append({"family": family, "status": status})
    if blockers:
        disposition = "INELIGIBLE"
    elif unresolved:
        disposition = "PENDING"
    else:
        disposition = "ELIGIBLE"
    return {"disposition": disposition, "unresolved": unresolved, "blockers": blockers}


def build(payload: Mapping[str, Any]) -> Dict[str, Any]:
    subjects = payload.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("input must contain subjects[]")
    ids = [str(row.get("subject_id", "")) for row in subjects]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("subject_id values must be non-empty and unique")

    out_rows = []
    family_counts = {family: {status: 0 for status in sorted(ALLOWED)} for family in FAMILIES}
    model_counts = {model: {"ELIGIBLE": 0, "PENDING": 0, "INELIGIBLE": 0} for model in MODEL_POLICY}

    for row in sorted(subjects, key=lambda x: str(x["subject_id"])):
        raw_families = row.get("families", {})
        if not isinstance(raw_families, dict):
            raise ValueError(f"{row['subject_id']}: families must be object")
        families = {family: normalize_cell(raw_families.get(family)) for family in FAMILIES}
        for family, cell in families.items():
            family_counts[family][cell["status"]] += 1
        models = {model: derive_model_disposition(families, model) for model in MODEL_POLICY}
        for model, decision in models.items():
            model_counts[model][decision["disposition"]] += 1
        out_rows.append({
            "subject_id": str(row["subject_id"]),
            "source_sha256": row.get("source_sha256"),
            "families": families,
            "models": models,
            "lineage": row.get("lineage", {}),
        })

    matrix = {
        "schema": SCHEMA,
        "status": "WORKING_MATRIX__NOT_TRAINING_AUTHORITY",
        "subject_count": len(out_rows),
        "families": list(FAMILIES),
        "rows": out_rows,
        "summary": {"family_counts": family_counts, "model_counts": model_counts},
        "training_authorized": False,
    }
    matrix["matrix_sha256"] = digest(matrix)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expect-subjects", type=int, default=370)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    matrix = build(payload)
    if matrix["subject_count"] != args.expect_subjects:
        raise SystemExit(f"subject count mismatch: {matrix['subject_count']} != {args.expect_subjects}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps(matrix["summary"], sort_keys=True))
    print(f"CORPUS370_AUTHORITY_MATRIX_WRITTEN sha256={matrix['matrix_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
