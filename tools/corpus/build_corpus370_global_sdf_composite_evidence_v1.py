#!/usr/bin/env python3
from __future__ import annotations

"""Bind explicit-solid numeric authority to current source/camera authority.

A passing SourceUnionSolidCandidateCourt is deliberately insufficient on its own:
its domain is source-world and declares camera/admission as downstream. This adapter
emits GLOBAL_SOLID_SDF=PASS only when the exact subject/source also has current
SOURCE_IMPORT and CAMERA_OBSERVATION PASS claims. It never invents MASKED outcomes;
an exhaustive solid-disposition receipt must do that separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

INDEX_SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
SOLID_BATCH_SCHEMA = "RealSaS.SourceUnionSolidCandidateCourtBatch.v1"
SOLID_COURT_SCHEMA = "RealSaS.SourceUnionSolidCandidateCourt.v1"
SOLID_COURT_STATUS = "PASS_SOURCE_UNION_BOUNDARY_AND_GLOBAL_SDF_NUMERICS__CAMERA_AND_ADMISSION_SEPARATE"
OUTPUT_SCHEMA = INDEX_SCHEMA


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "CORPUS370_GLOBAL_SDF_JSON_OBJECT")
    return value


def bound(ref: dict[str, Any], code: str) -> tuple[Path, dict[str, Any]]:
    ref = dict(ref or {})
    path = Path(str(ref.get("path") or "")).resolve()
    expected = str(ref.get("sha256") or "")
    require(path.is_file() and len(expected) == 64 and sha256(path) == expected, code)
    return path, load(path)


def _pass_claim(index: dict[str, Any], subject_id: str, family: str) -> dict[str, Any]:
    rows = {str(row.get("subject_id")): row for row in index.get("subjects") or ()}
    require(subject_id in rows, "CORPUS370_GLOBAL_SDF_SOURCE_AUTHORITY_SUBJECT")
    claim = dict((rows[subject_id].get("families") or {}).get(family) or {})
    require(claim.get("status") == "PASS", "CORPUS370_GLOBAL_SDF_SOURCE_AUTHORITY_" + family)
    court = dict(claim.get("court") or {})
    require(court.get("status") == "PASS", "CORPUS370_GLOBAL_SDF_SOURCE_COURT_" + family)
    require(len(str(court.get("sha256") or "")) == 64, "CORPUS370_GLOBAL_SDF_SOURCE_COURT_SHA_" + family)
    require(bool(court.get("path") or court.get("uri")), "CORPUS370_GLOBAL_SDF_SOURCE_COURT_REF_" + family)
    return claim


def build(roster_path: Path, source_index_path: Path, solid_batch_path: Path) -> dict[str, Any]:
    roster = load(roster_path)
    source_index = load(source_index_path)
    solid_batch = load(solid_batch_path)
    require(roster.get("schema") == ROSTER_SCHEMA, "CORPUS370_GLOBAL_SDF_ROSTER_SCHEMA")
    require(source_index.get("schema") == INDEX_SCHEMA, "CORPUS370_GLOBAL_SDF_SOURCE_INDEX_SCHEMA")
    require(solid_batch.get("schema") == SOLID_BATCH_SCHEMA, "CORPUS370_GLOBAL_SDF_BATCH_SCHEMA")
    require(roster.get("training_authorized") is False, "CORPUS370_GLOBAL_SDF_ROSTER_PREAUTHORIZED")
    require(source_index.get("training_authorized") is False, "CORPUS370_GLOBAL_SDF_INDEX_PREAUTHORIZED")
    require(solid_batch.get("training_authorized") is False, "CORPUS370_GLOBAL_SDF_BATCH_PREAUTHORIZED")

    roster_by_id = {str(row["subject_id"]): row for row in roster.get("subjects") or ()}
    require(len(roster_by_id) == int(roster.get("subject_count", -1)) == 370, "CORPUS370_GLOBAL_SDF_ROSTER_COUNT")
    output_rows = []
    ignored = []
    for row in solid_batch.get("subjects") or ():
        subject_id = str(row.get("subject_id") or "")
        require(subject_id in roster_by_id, "CORPUS370_GLOBAL_SDF_UNKNOWN_SUBJECT")
        if row.get("status") != SOLID_COURT_STATUS:
            ignored.append({"subject_id": subject_id, "status": row.get("status"), "reason": row.get("reason")})
            continue
        court_path, court = bound(dict(row.get("court") or {}), "CORPUS370_GLOBAL_SDF_SOLID_COURT_DRIFT")
        require(court.get("schema") == SOLID_COURT_SCHEMA and court.get("status") == SOLID_COURT_STATUS, "CORPUS370_GLOBAL_SDF_SOLID_COURT_CONTRACT")
        require(court.get("subject_id") == subject_id, "CORPUS370_GLOBAL_SDF_SOLID_COURT_SUBJECT")
        require(court.get("training_authorized") is False and court.get("camera_normalization_bound") is False, "CORPUS370_GLOBAL_SDF_SOLID_COURT_SCOPE")
        candidate_path, candidate = bound(dict(court.get("candidate") or {}), "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_DRIFT")
        require(candidate.get("schema") == "RealSaS.SourceUnionSolidCandidate.v1", "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_SCHEMA")
        require(candidate.get("subject_id") == subject_id, "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_SUBJECT")
        source_ref = dict((candidate.get("parents") or {}).get("source") or {})
        source_path = Path(str(source_ref.get("path") or "")).resolve()
        source_sha = str(source_ref.get("sha256") or "")
        require(source_path.is_file() and sha256(source_path) == source_sha, "CORPUS370_GLOBAL_SDF_RAW_SOURCE_DRIFT")
        require(source_sha == str(roster_by_id[subject_id]["source_sha256"]), "CORPUS370_GLOBAL_SDF_RAW_SOURCE_IDENTITY")

        source_claim = _pass_claim(source_index, subject_id, "SOURCE_IMPORT")
        camera_claim = _pass_claim(source_index, subject_id, "CAMERA_OBSERVATION")
        index_rows = {str(x.get("subject_id")): x for x in source_index.get("subjects") or ()}
        require(str(index_rows[subject_id].get("source_sha256") or "") == source_sha, "CORPUS370_GLOBAL_SDF_SOURCE_INDEX_IDENTITY")

        evidence_refs = [
            {"schema": SOLID_COURT_SCHEMA, "path": str(court_path), "sha256": sha256(court_path)},
            {"schema": "RealSaS.SourceUnionSolidCandidate.v1", "path": str(candidate_path), "sha256": sha256(candidate_path)},
        ]
        for parent_claim in (source_claim, camera_claim):
            for ref in parent_claim.get("evidence_refs") or ():
                evidence_refs.append(dict(ref))
        output_rows.append(
            {
                "subject_id": subject_id,
                "source_sha256": source_sha,
                "families": {
                    "GLOBAL_SOLID_SDF": {
                        "status": "PASS",
                        "reason_codes": [],
                        "evidence_refs": evidence_refs,
                        "court": {
                            "schema": "RealSaS.Corpus370ExplicitSolidCameraCompositeCourt.v1",
                            "path": str(court_path),
                            "sha256": sha256(court_path),
                            "status": "PASS",
                        },
                        "notes": "Explicit Boolean-union solid numerics PASS and exact current SOURCE_IMPORT/CAMERA_OBSERVATION PASS are jointly bound. Source-world SDF must be transformed by the sealed common camera normalization during materialization; no local component sign is consumed.",
                    }
                },
            }
        )

    return {
        "schema": OUTPUT_SCHEMA,
        "status": "GLOBAL_SOLID_SDF_COMPOSITE_PASS_ONLY__MASKED_DISPOSITION_SEPARATE",
        "subject_count": len(output_rows),
        "candidate_subject_count": 370,
        "subjects": sorted(output_rows, key=lambda row: row["subject_id"]),
        "solid_court_nonpass_rows": ignored,
        "parents": {
            "exact_source_roster": {"path": str(roster_path.resolve()), "sha256": sha256(roster_path)},
            "source_observation_evidence_index": {"path": str(source_index_path.resolve()), "sha256": sha256(source_index_path)},
            "solid_court_batch": {"path": str(solid_batch_path.resolve()), "sha256": sha256(solid_batch_path)},
        },
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--source-observation-evidence-index", type=Path, required=True)
    parser.add_argument("--solid-court-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.exact_source_roster, args.source_observation_evidence_index, args.solid_court_batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "status": result["status"], "subject_count": result["subject_count"], "solid_court_nonpass_count": len(result["solid_court_nonpass_rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
