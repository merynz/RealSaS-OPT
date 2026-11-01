#!/usr/bin/env python3
from __future__ import annotations

"""Build the fail-closed Corpus370 subject × truth-family authority matrix.

The matrix is an aggregation court, not a truth producer. It consumes an exact
candidate universe plus one or more current family evidence indexes. Inventory,
producer output, historical training flags, and pack metadata never become PASS
implicitly. Missing claims remain UNKNOWN.

A PASS/MASKED claim must cite an independent current court with court_status=PASS.
A FAIL claim must cite a current court with court_status=FAIL. Conflicting claims
for one subject/family are quarantined instead of resolved by precedence.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.Corpus370SubjectFamilyAuthorityMatrix.v1"
EVIDENCE_SCHEMA = "RealSaS.Corpus370FamilyEvidenceIndex.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _candidate_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("subjects")
    require(isinstance(rows, list), "CORPUS370_CANDIDATE_SUBJECTS")
    expected = int(manifest.get("subject_count", len(rows)))
    require(len(rows) == expected, "CORPUS370_CANDIDATE_COUNT_DECLARATION")
    result = []
    seen = set()
    for raw in rows:
        subject_id = str(raw.get("subject_id") or "")
        require(subject_id and subject_id not in seen, "CORPUS370_CANDIDATE_SUBJECT_ID")
        seen.add(subject_id)
        source_sha = str(raw.get("source_sha256") or "")
        require(_SHA256_RE.fullmatch(source_sha) is not None, "CORPUS370_SOURCE_SHA256")
        result.append(
            {
                "subject_id": subject_id,
                "origin": raw.get("origin"),
                "source_id": raw.get("source_id"),
                "source_format": raw.get("source_format"),
                "source_relative_path": raw.get("source_relative_path"),
                "source_sha256": source_sha,
            }
        )
    return result


def _validate_ref(ref: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(ref, dict), "CORPUS370_EVIDENCE_REF_TYPE")
    digest = str(ref.get("sha256") or "")
    require(_SHA256_RE.fullmatch(digest) is not None, "CORPUS370_EVIDENCE_REF_SHA256")
    require(bool(ref.get("schema")), "CORPUS370_EVIDENCE_REF_SCHEMA")
    require(bool(ref.get("uri") or ref.get("path")), "CORPUS370_EVIDENCE_REF_LOCATION")
    return dict(ref)


def _normalize_claim(claim: dict[str, Any], statuses: set[str]) -> dict[str, Any]:
    status = str(claim.get("status") or "")
    require(status in statuses - {"UNKNOWN"}, "CORPUS370_EVIDENCE_DECISION_STATUS")
    reason_codes = sorted({str(x) for x in claim.get("reason_codes") or () if str(x)})
    refs = [_validate_ref(dict(x)) for x in claim.get("evidence_refs") or ()]
    court = dict(claim.get("court") or {})
    court_status = str(court.get("status") or "")
    require(court_status in {"PASS", "FAIL"}, "CORPUS370_EVIDENCE_COURT_STATUS")
    require(bool(court.get("schema")), "CORPUS370_EVIDENCE_COURT_SCHEMA")
    require(_SHA256_RE.fullmatch(str(court.get("sha256") or "")) is not None, "CORPUS370_EVIDENCE_COURT_SHA256")
    require(bool(court.get("uri") or court.get("path")), "CORPUS370_EVIDENCE_COURT_LOCATION")
    if status in {"PASS", "MASKED"}:
        require(court_status == "PASS", "CORPUS370_PASS_REQUIRES_PASS_COURT")
    if status == "FAIL":
        require(court_status == "FAIL", "CORPUS370_FAIL_REQUIRES_FAIL_COURT")
    require(refs, "CORPUS370_EVIDENCE_REFS_REQUIRED")
    if status != "PASS":
        require(reason_codes, "CORPUS370_NONPASS_REASON_REQUIRED")
    return {
        "status": status,
        "reason_codes": reason_codes,
        "evidence_refs": refs,
        "court": court,
        "producer_refs": [dict(x) for x in claim.get("producer_refs") or ()],
        "notes": claim.get("notes"),
    }


def _collect_claims(
    paths: list[Path], candidate_ids: set[str], families: set[str], statuses: set[str]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    claims: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in paths:
        doc = load_json(path)
        require(doc.get("schema") == EVIDENCE_SCHEMA, "CORPUS370_EVIDENCE_INDEX_SCHEMA")
        for subject in doc.get("subjects") or ():
            subject_id = str(subject.get("subject_id") or "")
            require(subject_id in candidate_ids, "CORPUS370_EVIDENCE_UNKNOWN_SUBJECT")
            source_sha = str(subject.get("source_sha256") or "")
            require(_SHA256_RE.fullmatch(source_sha) is not None, "CORPUS370_EVIDENCE_SOURCE_SHA")
            for family, raw_claim in (subject.get("families") or {}).items():
                require(family in families, "CORPUS370_EVIDENCE_UNKNOWN_FAMILY")
                claim = _normalize_claim(dict(raw_claim), statuses)
                claim["source_sha256"] = source_sha
                claim["index_path"] = str(path)
                claim["index_sha256"] = sha256(path)
                claims.setdefault((subject_id, family), []).append(claim)
    return claims


def _resolve_cell(
    subject: dict[str, Any], family: str, claims: list[dict[str, Any]] | None
) -> dict[str, Any]:
    if not claims:
        return {
            "status": "UNKNOWN",
            "reason_codes": ["NO_CURRENT_COURT_EVIDENCE"],
            "evidence_refs": [],
            "court_refs": [],
            "producer_refs": [],
        }
    source_sha = subject["source_sha256"]
    mismatched = [x for x in claims if x["source_sha256"] != source_sha]
    if mismatched:
        return {
            "status": "QUARANTINED",
            "reason_codes": ["SOURCE_SHA_MISMATCH"],
            "evidence_refs": [],
            "court_refs": [],
            "producer_refs": [],
        }
    distinct = {x["status"] for x in claims}
    if len(distinct) != 1:
        return {
            "status": "QUARANTINED",
            "reason_codes": ["CONFLICTING_CURRENT_AUTHORITY_CLAIMS"],
            "evidence_refs": [r for x in claims for r in x["evidence_refs"]],
            "court_refs": [x["court"] for x in claims],
            "producer_refs": [r for x in claims for r in x["producer_refs"]],
        }
    status = next(iter(distinct))
    return {
        "status": status,
        "reason_codes": sorted({r for x in claims for r in x["reason_codes"]}),
        "evidence_refs": [r for x in claims for r in x["evidence_refs"]],
        "court_refs": [x["court"] for x in claims],
        "producer_refs": [r for x in claims for r in x["producer_refs"]],
    }


def _model_eligibility(family_cells: dict[str, dict[str, Any]], route: dict[str, Any]) -> dict[str, Any]:
    required = list(route.get("required_pass") or ())
    blockers = [family for family in required if family_cells[family]["status"] != "PASS"]
    return {
        "corpus_eligible": not blockers,
        "required_pass": required,
        "blocking_families": blockers,
        "post_corpus_dependencies": list(route.get("post_corpus_dependencies") or ()),
    }


def build(policy_path: Path, candidate_path: Path, evidence_paths: list[Path]) -> dict[str, Any]:
    policy = load_json(policy_path)
    require(policy.get("schema") == "RealSaS.Corpus370SubjectFamilyAuthorityPolicy.v1", "CORPUS370_POLICY_SCHEMA")
    families = list(policy["families"])
    family_set = set(families)
    statuses = set(policy["cell_statuses"])
    candidates_doc = load_json(candidate_path)
    subjects = _candidate_rows(candidates_doc)
    expected = int(policy["candidate_universe"]["expected_subject_count"])
    require(len(subjects) == expected, "CORPUS370_EXPECTED_CANDIDATE_COUNT")
    candidate_ids = {x["subject_id"] for x in subjects}
    claims = _collect_claims(evidence_paths, candidate_ids, family_set, statuses)

    rows = []
    family_rollup = {family: Counter() for family in families}
    model_rollup = {name: Counter() for name in policy["model_routes"]}
    for subject in sorted(subjects, key=lambda x: x["subject_id"]):
        cells = {}
        for family in families:
            cell = _resolve_cell(subject, family, claims.get((subject["subject_id"], family)))
            cells[family] = cell
            family_rollup[family][cell["status"]] += 1
        models = {}
        for model, route in policy["model_routes"].items():
            models[model] = _model_eligibility(cells, route)
            model_rollup[model]["eligible" if models[model]["corpus_eligible"] else "blocked"] += 1
        rows.append({**subject, "families": cells, "model_eligibility": models})

    unresolved = sum(1 for row in rows for cell in row["families"].values() if cell["status"] == "UNKNOWN")
    quarantined = sum(1 for row in rows for cell in row["families"].values() if cell["status"] == "QUARANTINED")
    return {
        "schema": SCHEMA,
        "status": "OPEN" if unresolved or quarantined else "FAMILY_AUTHORITY_RESOLVED__SPLIT_AND_READINESS_STILL_REQUIRED",
        "training_authorized": False,
        "policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
        "candidate_manifest": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "evidence_indexes": [{"path": str(p), "sha256": sha256(p)} for p in evidence_paths],
        "subject_count": len(rows),
        "subjects": rows,
        "rollup": {
            "families": {family: dict(sorted(counter.items())) for family, counter in family_rollup.items()},
            "models": {model: dict(sorted(counter.items())) for model, counter in model_rollup.items()},
            "unknown_cell_count": unresolved,
            "quarantined_cell_count": quarantined,
        },
        "next_required_authority": [
            "resolve_all_required_family_UNKNOWN_or_QUARANTINED_cells",
            "seal_global_lineage_and_dedup",
            "isolate_global_end_to_end_unseen_pool",
            "derive_model_specific_train_val_rosters",
            "seal_materialization_manifest",
            "mint_training_readiness_receipt"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--evidence-index", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.policy, args.candidate_manifest, args.evidence_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": output["schema"], "status": output["status"], "subject_count": output["subject_count"], "rollup": output["rollup"]}, sort_keys=True))


if __name__ == "__main__":
    main()
