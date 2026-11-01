#!/usr/bin/env python3
from __future__ import annotations

"""Bind explicit-solid numeric authority to current source/camera authority.

A passing SourceUnionSolidCandidateCourt is deliberately insufficient on its own:
its domain is source-world and declares camera/admission as downstream. This adapter
emits GLOBAL_SOLID_SDF=PASS only when the exact subject/source also has current
SOURCE_IMPORT and CAMERA_OBSERVATION PASS claims.

Every PASS is backed by a newly written, hash-bound composite court artifact. The
adapter never invents MASKED outcomes; an exhaustive solid-disposition receipt must
do that separately after all 370 subjects have been assessed under one fixed policy.
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
SOURCE_COURT_SCHEMA = "RealSaS.SourceObservationAuthorityCourt.v1"
SOURCE_COURT_STATUS = "PASS_SOURCE_OBSERVATION_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
COMPOSITE_SCHEMA = "RealSaS.Corpus370ExplicitSolidCameraCompositeCourt.v1"
COMPOSITE_STATUS = "PASS_EXPLICIT_SOLID_CAMERA_COMPOSITE__FINAL_ADMISSION_SEPARATE"
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


def ref(path: Path, schema: str) -> dict[str, Any]:
    path = path.resolve()
    return {"schema": schema, "path": str(path), "sha256": sha256(path)}


def bound(raw_ref: dict[str, Any], code: str) -> tuple[Path, dict[str, Any]]:
    raw_ref = dict(raw_ref or {})
    path = Path(str(raw_ref.get("path") or "")).resolve()
    expected = str(raw_ref.get("sha256") or "")
    require(path.is_file() and len(expected) == 64 and sha256(path) == expected, code)
    return path, load(path)


def _pass_claim(index: dict[str, Any], subject_id: str, family: str) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    rows = {str(row.get("subject_id")): row for row in index.get("subjects") or ()}
    require(subject_id in rows, "CORPUS370_GLOBAL_SDF_SOURCE_AUTHORITY_SUBJECT")
    claim = dict((rows[subject_id].get("families") or {}).get(family) or {})
    require(claim.get("status") == "PASS", "CORPUS370_GLOBAL_SDF_SOURCE_AUTHORITY_" + family)
    claim_court = dict(claim.get("court") or {})
    require(claim_court.get("status") == "PASS", "CORPUS370_GLOBAL_SDF_SOURCE_COURT_" + family)
    court_path, court = bound(claim_court, "CORPUS370_GLOBAL_SDF_SOURCE_COURT_DRIFT_" + family)
    require(court.get("schema") == SOURCE_COURT_SCHEMA and court.get("status") == SOURCE_COURT_STATUS,
            "CORPUS370_GLOBAL_SDF_SOURCE_COURT_CONTRACT_" + family)
    require(court.get("subject_id") == subject_id, "CORPUS370_GLOBAL_SDF_SOURCE_COURT_SUBJECT_" + family)
    require((court.get("authority") or {}).get("independent_court_passed") is True,
            "CORPUS370_GLOBAL_SDF_SOURCE_COURT_NOT_INDEPENDENT_" + family)
    require((court.get("authority") or {}).get("training_authorized") is False,
            "CORPUS370_GLOBAL_SDF_SOURCE_COURT_PREAUTHORIZED_" + family)
    return claim, court_path, court


def _write_composite(
    output_root: Path,
    *,
    subject_id: str,
    source_sha: str,
    solid_court_path: Path,
    solid_court: dict[str, Any],
    solid_candidate_path: Path,
    source_court_path: Path,
    source_court: dict[str, Any],
) -> Path:
    require(source_court.get("source_sha256") == source_sha, "CORPUS370_GLOBAL_SDF_CAMERA_SOURCE_SHA")
    source_outcomes = dict(source_court.get("family_candidate_outcomes") or {})
    require(source_outcomes == {
        "SOURCE_IMPORT": "QUALIFIED_CORE_CANDIDATE",
        "CAMERA_OBSERVATION": "QUALIFIED_CORE_CANDIDATE",
    }, "CORPUS370_GLOBAL_SDF_CAMERA_OUTCOMES")
    payload = {
        "schema": COMPOSITE_SCHEMA,
        "status": COMPOSITE_STATUS,
        "subject_id": subject_id,
        "source_sha256": source_sha,
        "parents": {
            "explicit_solid_numeric_court": ref(solid_court_path, SOLID_COURT_SCHEMA),
            "explicit_solid_candidate": ref(solid_candidate_path, "RealSaS.SourceUnionSolidCandidate.v1"),
            "source_observation_court": ref(source_court_path, SOURCE_COURT_SCHEMA),
        },
        "solid_domain": {
            "domain": "BOOLEAN_UNION_OF_CLOSED_EMBEDDED_POSITIVE_VOLUME_SOURCE_COMPONENT_SOLIDS",
            "coordinate_frame_input": "SOURCE_WORLD",
            "camera_normalization_binding": "EXACT_CURRENT_SOURCE_OBSERVATION_AUTHORITY",
            "sign_convention": "NEGATIVE_INSIDE__POSITIVE_OUTSIDE__ZERO_BOUNDARY",
            "distance_boundary": "ACTUAL_BOOLEAN_UNION_BOUNDARY",
            "component_local_sign_consumed": False,
            "raw_surface_replaced_for_other_families": False,
        },
        "validated": {
            "solid_numeric_court_passed": True,
            "source_import_court_passed": True,
            "camera_observation_court_passed": True,
            "same_exact_source_sha256": True,
            "source_world_to_canonical_binding_present": True,
        },
        "family_candidate_outcomes": {"GLOBAL_SOLID_SDF": "QUALIFIED_CORE_CANDIDATE"},
        "authority": {
            "independent_parent_courts_passed": True,
            "global_sdf_family_candidate": True,
            "final_family_authority_minted": False,
            "split_membership_minted": False,
            "training_authorized": False,
        },
    }
    out = output_root / "subjects" / subject_id / "CORPUS370_EXPLICIT_SOLID_CAMERA_COMPOSITE_COURT_V1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return out


def build(roster_path: Path, source_index_path: Path, solid_batch_path: Path, output_root: Path) -> dict[str, Any]:
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
    index_by_id = {str(row.get("subject_id")): row for row in source_index.get("subjects") or ()}
    require(len(roster_by_id) == int(roster.get("subject_count", -1)) == 370, "CORPUS370_GLOBAL_SDF_ROSTER_COUNT")
    output_rows = []
    nonpass = []
    for row in solid_batch.get("subjects") or ():
        subject_id = str(row.get("subject_id") or "")
        require(subject_id in roster_by_id, "CORPUS370_GLOBAL_SDF_UNKNOWN_SUBJECT")
        if row.get("status") != SOLID_COURT_STATUS:
            nonpass.append({"subject_id": subject_id, "status": row.get("status"), "reason": row.get("reason")})
            continue
        solid_court_path, solid_court = bound(dict(row.get("court") or {}), "CORPUS370_GLOBAL_SDF_SOLID_COURT_DRIFT")
        require(solid_court.get("schema") == SOLID_COURT_SCHEMA and solid_court.get("status") == SOLID_COURT_STATUS,
                "CORPUS370_GLOBAL_SDF_SOLID_COURT_CONTRACT")
        require(solid_court.get("subject_id") == subject_id, "CORPUS370_GLOBAL_SDF_SOLID_COURT_SUBJECT")
        require(solid_court.get("training_authorized") is False and solid_court.get("camera_normalization_bound") is False,
                "CORPUS370_GLOBAL_SDF_SOLID_COURT_SCOPE")
        solid_candidate_path, solid_candidate = bound(dict(solid_court.get("candidate") or {}), "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_DRIFT")
        require(solid_candidate.get("schema") == "RealSaS.SourceUnionSolidCandidate.v1",
                "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_SCHEMA")
        require(solid_candidate.get("subject_id") == subject_id, "CORPUS370_GLOBAL_SDF_SOLID_CANDIDATE_SUBJECT")
        source_ref = dict((solid_candidate.get("parents") or {}).get("source") or {})
        source_path = Path(str(source_ref.get("path") or "")).resolve()
        source_sha = str(source_ref.get("sha256") or "")
        require(source_path.is_file() and sha256(source_path) == source_sha, "CORPUS370_GLOBAL_SDF_RAW_SOURCE_DRIFT")
        require(source_sha == str(roster_by_id[subject_id]["source_sha256"]), "CORPUS370_GLOBAL_SDF_RAW_SOURCE_IDENTITY")
        require(subject_id in index_by_id and str(index_by_id[subject_id].get("source_sha256") or "") == source_sha,
                "CORPUS370_GLOBAL_SDF_SOURCE_INDEX_IDENTITY")

        source_claim, source_court_path, source_court = _pass_claim(source_index, subject_id, "SOURCE_IMPORT")
        camera_claim, camera_court_path, camera_court = _pass_claim(source_index, subject_id, "CAMERA_OBSERVATION")
        require(source_court_path == camera_court_path and source_court == camera_court,
                "CORPUS370_GLOBAL_SDF_SOURCE_CAMERA_COURT_NOT_SHARED")

        composite_path = _write_composite(
            output_root,
            subject_id=subject_id,
            source_sha=source_sha,
            solid_court_path=solid_court_path,
            solid_court=solid_court,
            solid_candidate_path=solid_candidate_path,
            source_court_path=source_court_path,
            source_court=source_court,
        )
        evidence_refs = [
            ref(composite_path, COMPOSITE_SCHEMA),
            ref(solid_court_path, SOLID_COURT_SCHEMA),
            ref(solid_candidate_path, "RealSaS.SourceUnionSolidCandidate.v1"),
            ref(source_court_path, SOURCE_COURT_SCHEMA),
        ]
        for parent_claim in (source_claim, camera_claim):
            for evidence_ref in parent_claim.get("evidence_refs") or ():
                if evidence_ref not in evidence_refs:
                    evidence_refs.append(dict(evidence_ref))
        output_rows.append({
            "subject_id": subject_id,
            "source_sha256": source_sha,
            "families": {
                "GLOBAL_SOLID_SDF": {
                    "status": "PASS",
                    "reason_codes": [],
                    "evidence_refs": evidence_refs,
                    "court": {
                        "schema": COMPOSITE_SCHEMA,
                        "path": str(composite_path.resolve()),
                        "sha256": sha256(composite_path),
                        "status": "PASS",
                    },
                    "notes": "Explicit-solid numerics and exact current SOURCE_IMPORT/CAMERA_OBSERVATION are jointly bound. No component-local sign is consumed.",
                }
            },
        })

    return {
        "schema": OUTPUT_SCHEMA,
        "status": "GLOBAL_SOLID_SDF_COMPOSITE_PASS_ONLY__MASKED_DISPOSITION_SEPARATE",
        "subject_count": len(output_rows),
        "candidate_subject_count": 370,
        "subjects": sorted(output_rows, key=lambda row: row["subject_id"]),
        "solid_court_nonpass_rows": nonpass,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.exact_source_roster, args.source_observation_evidence_index, args.solid_court_batch, args.output_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "status": result["status"], "subject_count": result["subject_count"], "solid_court_nonpass_count": len(result["solid_court_nonpass_rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
