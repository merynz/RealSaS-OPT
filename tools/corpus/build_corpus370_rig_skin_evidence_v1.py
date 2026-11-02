#!/usr/bin/env python3
from __future__ import annotations

"""Normalize current rig/skin courts into Corpus370 family-evidence claims.

Format-specific provenance remains visible in metadata, but model routing consumes
only RIG_HIERARCHY / SKIN_WEIGHTS authority. Partial row coverage is represented as
PASS plus an explicit row-validity mask contract; family MASKED means no usable
skin supervision exists for that subject.
"""

import argparse
import hashlib
import json
from pathlib import Path

OUTPUT_SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"
INDEX_SCHEMA = "RealSaS.Corpus370RigSkinCourtIndex.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
GLTF_SCHEMA = "RealSaS.GltfImportedRigSkinEquivalenceCourt.v1"
GLTF_STATUS = "PASS_GLTF_IMPORTED_RIG_SKIN_EQUIVALENCE__FINAL_ADMISSION_SEPARATE"
FBX_SCHEMA = "RealSaS.FbxImportedRigSkinAuthorityCourt.v1"
FBX_STATUS = "PASS_FBX_IMPORTED_RIG_SKIN_DECODE_EQUIVALENCE__FINAL_ADMISSION_SEPARATE"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "CORPUS370_RIG_SKIN_EVIDENCE_JSON_OBJECT")
    return value


def bound(ref: dict, code: str) -> tuple[Path, dict]:
    path = Path(str((ref or {}).get("path") or "")).resolve()
    expected = str((ref or {}).get("sha256") or "")
    require(path.is_file() and len(expected) == 64 and sha256(path) == expected, code)
    return path, load(path)


def court_ref(path: Path, schema: str) -> dict:
    return {"schema": schema, "path": str(path), "sha256": sha256(path), "status": "PASS"}


def evidence_ref(path: Path, schema: str) -> dict:
    return {"schema": schema, "path": str(path), "sha256": sha256(path)}


def source_from_court(doc: dict) -> dict:
    if doc.get("schema") == GLTF_SCHEMA:
        return dict(((doc.get("inputs") or {}).get("raw_source") or {}))
    return dict(doc.get("source") or {})


def normalize_court(subject_id: str, expected_source_sha: str, path: Path, doc: dict, lane: str) -> dict:
    schema = str(doc.get("schema") or "")
    status = str(doc.get("status") or "")
    require(doc.get("subject_id") == subject_id, "CORPUS370_RIG_SKIN_EVIDENCE_SUBJECT")
    source = source_from_court(doc)
    require(str(source.get("sha256") or "") == expected_source_sha, "CORPUS370_RIG_SKIN_EVIDENCE_SOURCE_SHA")
    source_path = Path(str(source.get("path") or "")).resolve()
    require(source_path.is_file() and sha256(source_path) == expected_source_sha, "CORPUS370_RIG_SKIN_EVIDENCE_SOURCE_BYTES")

    if schema == GLTF_SCHEMA:
        require(status == GLTF_STATUS, "CORPUS370_RIG_SKIN_EVIDENCE_GLTF_STATUS")
        outcomes = dict(doc.get("family_candidate_outcomes") or {})
        require(outcomes.get("RIG_HIERARCHY") == "QUALIFIED_CORE_CANDIDATE", "CORPUS370_RIG_SKIN_EVIDENCE_GLTF_RIG")
        require(outcomes.get("SKIN_WEIGHTS") == "QUALIFIED_CORE_CANDIDATE", "CORPUS370_RIG_SKIN_EVIDENCE_GLTF_SKIN")
        skin_metadata = {
            "authority_scope": "RAW_GLTF2_BYTES_PLUS_IMPORTED_EQUIVALENCE",
            "row_validity_mask_required": False,
            "valid_row_coverage_fraction": 1.0,
            "semantic_skin_transfer_used": False,
            "nearest_neighbor_skin_transfer_used": False,
        }
    elif schema == FBX_SCHEMA:
        require(status == FBX_STATUS, "CORPUS370_RIG_SKIN_EVIDENCE_FBX_STATUS")
        outcomes = dict(doc.get("family_candidate_outcomes") or {})
        require(outcomes.get("RIG_HIERARCHY") == "QUALIFIED_CORE_CANDIDATE", "CORPUS370_RIG_SKIN_EVIDENCE_FBX_RIG")
        skin_outcome = outcomes.get("SKIN_WEIGHTS")
        require(skin_outcome in {"QUALIFIED_CORE_CANDIDATE", "QUALIFIED_MASKED_CANDIDATE", "UNAVAILABLE_CANDIDATE"}, "CORPUS370_RIG_SKIN_EVIDENCE_FBX_SKIN")
        quality = dict(doc.get("skin_quality") or {})
        coverage = float(quality.get("simplex_valid_weight_coverage_fraction", 0.0))
        require(0.0 <= coverage <= 1.0, "CORPUS370_RIG_SKIN_EVIDENCE_FBX_COVERAGE")
        skin_metadata = {
            "authority_scope": str(doc.get("authority_scope") or ""),
            "row_validity_mask_required": skin_outcome == "QUALIFIED_MASKED_CANDIDATE",
            "valid_row_coverage_fraction": coverage,
            "authored_row_coverage_fraction": float(quality.get("authored_weight_coverage_fraction", 0.0)),
            "semantic_skin_transfer_used": False,
            "nearest_neighbor_skin_transfer_used": False,
            "raw_fbx_parser_authority_claimed": False,
        }
    else:
        raise ValueError("CORPUS370_RIG_SKIN_EVIDENCE_COURT_SCHEMA:" + schema)

    common_refs = [
        evidence_ref(path, schema),
        {"schema": "RealSaS.ImmutableRawSourceBytes.v1", "path": str(source_path), "sha256": expected_source_sha},
    ]
    court = court_ref(path, schema)
    families = {
        "RIG_HIERARCHY": {
            "status": "PASS",
            "reason_codes": [],
            "evidence_refs": common_refs,
            "court": court,
            "metadata": {
                "lane": lane,
                "authority_scope": "RAW_GLTF2_BYTES_PLUS_IMPORTED_EQUIVALENCE" if schema == GLTF_SCHEMA else str(doc.get("authority_scope") or ""),
            },
        }
    }
    if schema == FBX_SCHEMA and outcomes.get("SKIN_WEIGHTS") == "UNAVAILABLE_CANDIDATE":
        families["SKIN_WEIGHTS"] = {
            "status": "MASKED",
            "reason_codes": ["NO_SIMPLEX_VALID_AUTHORED_SKIN_ROWS"],
            "evidence_refs": common_refs,
            "court": court,
            "metadata": {"lane": lane, **skin_metadata},
        }
    else:
        families["SKIN_WEIGHTS"] = {
            "status": "PASS",
            "reason_codes": [],
            "evidence_refs": common_refs,
            "court": court,
            "metadata": {"lane": lane, **skin_metadata},
            "notes": "PASS may carry a hash-bound row-validity mask; invalid or absent rows are omitted from MIRA loss and are never encoded as zero targets.",
        }
    return {"subject_id": subject_id, "source_sha256": expected_source_sha, "families": families}


def build(roster_path: Path, court_index_path: Path) -> dict:
    roster = load(roster_path)
    index = load(court_index_path)
    require(roster.get("schema") == ROSTER_SCHEMA and int(roster.get("subject_count", -1)) == 370, "CORPUS370_RIG_SKIN_EVIDENCE_ROSTER")
    require(index.get("schema") == INDEX_SCHEMA, "CORPUS370_RIG_SKIN_EVIDENCE_INDEX")
    roster_rows = {str(row["subject_id"]): row for row in roster.get("subjects") or []}
    require(len(roster_rows) == 370, "CORPUS370_RIG_SKIN_EVIDENCE_ROSTER_SET")
    rows = []
    seen = set()
    for row in index.get("subjects") or []:
        subject_id = str(row.get("subject_id") or "")
        require(subject_id in roster_rows and subject_id not in seen, "CORPUS370_RIG_SKIN_EVIDENCE_INDEX_SUBJECT")
        seen.add(subject_id)
        expected_source_sha = str(roster_rows[subject_id]["source_sha256"])
        require(str(row.get("source_sha256") or "") == expected_source_sha, "CORPUS370_RIG_SKIN_EVIDENCE_INDEX_SOURCE")
        path, doc = bound(dict(row.get("court") or {}), "CORPUS370_RIG_SKIN_EVIDENCE_COURT_DRIFT")
        rows.append(normalize_court(subject_id, expected_source_sha, path, doc, str(row.get("lane") or "")))
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "CURRENT_RIG_SKIN_FAMILY_EVIDENCE__FINAL_MATRIX_AGGREGATION_REQUIRED",
        "subject_count": len(rows),
        "candidate_subject_count": 370,
        "subjects": sorted(rows, key=lambda row: row["subject_id"]),
        "parents": {
            "exact_source_roster": {"path": str(roster_path.resolve()), "sha256": sha256(roster_path)},
            "rig_skin_court_index": {"path": str(court_index_path.resolve()), "sha256": sha256(court_index_path)},
        },
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--court-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.exact_source_roster, args.court_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
