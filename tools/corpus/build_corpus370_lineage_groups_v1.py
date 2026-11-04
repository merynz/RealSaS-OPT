#!/usr/bin/env python3
from __future__ import annotations

"""Build leakage-safe Corpus370 lineage groups without fuzzy auto-merging.

Only two signals may merge subjects automatically:
1. identical exact source bytes, or
2. an independently courted high-precision structural identity fingerprint.

Path/authored-name equivalence is diagnostic only. Such collisions become explicit
REVIEW_REQUIRED candidates and block a sealed global split until resolved by a
separate review-decision artifact. This prevents near-named variants from being
silently collapsed while also preventing cross-format exports of one authored actor
from leaking across TRAIN and product-unseen.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

SCHEMA = "RealSaS.Corpus370LineageGroups.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
FINGERPRINT_SCHEMA = "RealSaS.Corpus370StructuralFingerprintIndex.v1"
REVIEW_SCHEMA = "RealSaS.Corpus370LineageReviewDecisions.v1"

EXPORT_LANE_TOKENS = {
    "fbx", "gltf", "glb", "obj", "blend", "blender", "models-fbx", "models-gltf",
    "models-glb", "fbx-format", "gltf-format", "glb-format", "obj-format", "fbx-unity",
    "gltf-godot-unreal", "gltf-godot-ue", "godot-ue", "unity", "raw", "sources",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "CORPUS370_LINEAGE_JSON_OBJECT")
    return value


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def authored_hint(row: dict) -> str:
    """Exact normalized path identity hint; never automatic lineage authority."""
    path = PurePosixPath(str(row.get("source_relative_path") or ""))
    parent_tokens = [slug(part) for part in path.parts[:-1]]
    parent_tokens = [token for token in parent_tokens if token and token not in EXPORT_LANE_TOKENS]
    # archive transport prefixes are not authored identity
    parent_tokens = [token for token in parent_tokens if token not in {"assets", "base", "publishers"}]
    stem = slug(path.stem)
    return "/".join([str(row.get("source_id") or ""), *parent_tokens, stem])


class UnionFind:
    def __init__(self, members: list[str]):
        self.parent = {member: member for member in members}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        lo, hi = sorted((a, b))
        self.parent[hi] = lo


def _fingerprints(path: Path | None, roster: dict[str, dict]) -> tuple[dict[str, str], dict[str, dict]]:
    if path is None:
        return {}, {}
    doc = load(path)
    require(doc.get("schema") == FINGERPRINT_SCHEMA, "CORPUS370_LINEAGE_FINGERPRINT_SCHEMA")
    rows = {}
    refs = {}
    for row in doc.get("subjects") or ():
        sid = str(row.get("subject_id") or "")
        require(sid in roster and sid not in rows, "CORPUS370_LINEAGE_FINGERPRINT_SUBJECT")
        require(row.get("source_sha256") == roster[sid]["source_sha256"], "CORPUS370_LINEAGE_FINGERPRINT_SOURCE")
        court = dict(row.get("court") or {})
        require(court.get("status") in {"PASS", "FAIL"}, "CORPUS370_LINEAGE_FINGERPRINT_COURT_STATUS")
        if court.get("status") != "PASS":
            continue
        require(row.get("identity_strength") == "EXACT_BIND_STRUCTURE_AND_RIG_SKIN", "CORPUS370_LINEAGE_FINGERPRINT_STRENGTH")
        fingerprint = str(row.get("fingerprint_sha256") or "")
        require(re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None, "CORPUS370_LINEAGE_FINGERPRINT_SHA")
        rows[sid] = fingerprint
        refs[sid] = {
            "fingerprint_sha256": fingerprint,
            "court": court,
            "evidence_refs": list(row.get("evidence_refs") or ()),
        }
    return rows, refs


def _review_decisions(path: Path | None, candidate_ids: set[str]) -> dict[str, dict]:
    if path is None:
        return {}
    doc = load(path)
    require(doc.get("schema") == REVIEW_SCHEMA, "CORPUS370_LINEAGE_REVIEW_SCHEMA")
    decisions = {}
    for row in doc.get("decisions") or ():
        candidate_id = str(row.get("candidate_id") or "")
        require(candidate_id in candidate_ids and candidate_id not in decisions, "CORPUS370_LINEAGE_REVIEW_CANDIDATE")
        decision = str(row.get("decision") or "")
        require(decision in {"SAME_LINEAGE", "DISTINCT_LINEAGES"}, "CORPUS370_LINEAGE_REVIEW_DECISION")
        reviewer = str(row.get("reviewer") or "")
        rationale = str(row.get("rationale") or "")
        require(reviewer and rationale, "CORPUS370_LINEAGE_REVIEW_PROVENANCE")
        members = sorted({str(value) for value in row.get("subject_ids") or ()})
        require(len(members) >= 2, "CORPUS370_LINEAGE_REVIEW_MEMBERS")
        decisions[candidate_id] = {"decision": decision, "subject_ids": members, "reviewer": reviewer, "rationale": rationale}
    return decisions


def build(
    roster_path: Path,
    structural_fingerprint_path: Path | None = None,
    review_decisions_path: Path | None = None,
    require_complete_structural_fingerprints: bool = False,
) -> dict:
    roster_doc = load(roster_path)
    require(roster_doc.get("schema") == ROSTER_SCHEMA, "CORPUS370_LINEAGE_ROSTER_SCHEMA")
    require(int(roster_doc.get("subject_count", -1)) == 370, "CORPUS370_LINEAGE_ROSTER_COUNT")
    require(roster_doc.get("training_authorized") is False, "CORPUS370_LINEAGE_ROSTER_PREAUTHORIZED")
    rows = list(roster_doc.get("subjects") or ())
    require(len(rows) == 370, "CORPUS370_LINEAGE_ROSTER_ROWS")
    roster = {str(row["subject_id"]): dict(row) for row in rows}
    require(len(roster) == 370, "CORPUS370_LINEAGE_SUBJECT_IDS")

    uf = UnionFind(sorted(roster))
    merge_reasons: dict[tuple[str, str], list[dict]] = defaultdict(list)

    # Exact source-byte equality is definitive identity at this corpus layer.
    by_source = defaultdict(list)
    for sid, row in roster.items():
        by_source[str(row["source_sha256"])].append(sid)
    for source_sha, members in by_source.items():
        if len(members) < 2:
            continue
        anchor = sorted(members)[0]
        for member in sorted(members)[1:]:
            uf.union(anchor, member)
            merge_reasons[(min(anchor, member), max(anchor, member))].append({
                "kind": "EXACT_SOURCE_BYTES",
                "source_sha256": source_sha,
            })

    fingerprints, fingerprint_refs = _fingerprints(structural_fingerprint_path, roster)
    structural_coverage_missing = sorted(set(roster) - set(fingerprints))
    by_fingerprint = defaultdict(list)
    for sid, fp in fingerprints.items():
        by_fingerprint[fp].append(sid)
    for fingerprint, members in by_fingerprint.items():
        if len(members) < 2:
            continue
        anchor = sorted(members)[0]
        for member in sorted(members)[1:]:
            uf.union(anchor, member)
            merge_reasons[(min(anchor, member), max(anchor, member))].append({
                "kind": "COURTED_EXACT_BIND_STRUCTURE_AND_RIG_SKIN",
                "fingerprint_sha256": fingerprint,
                "left_evidence": fingerprint_refs[anchor],
                "right_evidence": fingerprint_refs[member],
            })

    # Name/path equivalence is only a review candidate, never merge authority.
    by_hint = defaultdict(list)
    for sid, row in roster.items():
        by_hint[authored_hint(row)].append(sid)
    raw_candidates = []
    for hint, members in sorted(by_hint.items()):
        roots = {uf.find(sid) for sid in members}
        if len(members) < 2 or len(roots) <= 1:
            continue
        candidate_id = "lineage_review_" + digest({"hint": hint, "members": sorted(members)})[:20]
        raw_candidates.append({
            "candidate_id": candidate_id,
            "subject_ids": sorted(members),
            "hint": hint,
            "reason": "EXACT_NORMALIZED_AUTHORED_PATH_HINT_COLLISION__NOT_AUTOMATIC_AUTHORITY",
        })

    candidate_ids = {row["candidate_id"] for row in raw_candidates}
    decisions = _review_decisions(review_decisions_path, candidate_ids)
    unresolved = []
    resolved_reviews = []
    for candidate in raw_candidates:
        decision = decisions.get(candidate["candidate_id"])
        if decision is None:
            unresolved.append(candidate)
            continue
        require(sorted(decision["subject_ids"]) == candidate["subject_ids"], "CORPUS370_LINEAGE_REVIEW_MEMBER_DRIFT")
        if decision["decision"] == "SAME_LINEAGE":
            anchor = candidate["subject_ids"][0]
            for member in candidate["subject_ids"][1:]:
                uf.union(anchor, member)
                merge_reasons[(min(anchor, member), max(anchor, member))].append({
                    "kind": "EXPLICIT_LINEAGE_REVIEW_SAME_LINEAGE",
                    "candidate_id": candidate["candidate_id"],
                    "reviewer": decision["reviewer"],
                    "rationale": decision["rationale"],
                })
        resolved_reviews.append({**candidate, **decision})

    grouped = defaultdict(list)
    for sid in sorted(roster):
        grouped[uf.find(sid)].append(sid)

    groups = []
    subject_to_lineage = {}
    for members in sorted(grouped.values(), key=lambda values: values[0]):
        members = sorted(members)
        evidence = []
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                evidence.extend(merge_reasons.get((left, right), []))
        lineage_id = "lineage_" + digest({
            "subjects": members,
            "source_sha256": sorted(roster[sid]["source_sha256"] for sid in members),
        })[:24]
        for sid in members:
            subject_to_lineage[sid] = lineage_id
        groups.append({
            "lineage_id": lineage_id,
            "subject_ids": members,
            "subject_count": len(members),
            "source_sha256": sorted(roster[sid]["source_sha256"] for sid in members),
            "source_ids": sorted({str(roster[sid].get("source_id") or "") for sid in members}),
            "merge_evidence": evidence,
        })

    # A review decision can merge roots; ensure no unresolved candidate became moot.
    final_unresolved = []
    for candidate in unresolved:
        final_roots = {subject_to_lineage[sid] for sid in candidate["subject_ids"]}
        if len(final_roots) > 1:
            final_unresolved.append(candidate)

    return {
        "schema": SCHEMA,
        "status": (
            "BLOCKED_INDEPENDENT_STRUCTURAL_FINGERPRINT_COVERAGE_REQUIRED"
            if require_complete_structural_fingerprints and structural_coverage_missing
            else "BLOCKED_LINEAGE_REVIEW_REQUIRED"
            if final_unresolved
            else "PASS_LINEAGE_GROUPS_SEALED"
        ),
        "subject_count": 370,
        "lineage_count": len(groups),
        "multi_subject_lineage_count": sum(int(group["subject_count"] > 1) for group in groups),
        "exact_source_duplicate_group_count": sum(int(len(values) > 1) for values in by_source.values()),
        "courted_structural_duplicate_group_count": sum(int(len(values) > 1) for values in by_fingerprint.values()),
        "review_required_count": len(final_unresolved),
        "complete_independent_structural_fingerprint_required": bool(require_complete_structural_fingerprints),
        "missing_independently_courted_structural_subject_count": (
            len(structural_coverage_missing) if require_complete_structural_fingerprints else None
        ),
        "missing_independently_courted_structural_subject_ids": (
            structural_coverage_missing if require_complete_structural_fingerprints else None
        ),
        "groups": groups,
        "subject_to_lineage": subject_to_lineage,
        "resolved_reviews": resolved_reviews,
        "review_required": final_unresolved,
        "parents": {
            "exact_source_roster": {"path": str(roster_path.resolve()), "sha256": sha256(roster_path)},
            "structural_fingerprints": None if structural_fingerprint_path is None else {"path": str(structural_fingerprint_path.resolve()), "sha256": sha256(structural_fingerprint_path)},
            "review_decisions": None if review_decisions_path is None else {"path": str(review_decisions_path.resolve()), "sha256": sha256(review_decisions_path)},
        },
        "split_authorized": not final_unresolved and (
            not require_complete_structural_fingerprints or not structural_coverage_missing
        ),
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--structural-fingerprint-index", type=Path)
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--require-complete-structural-fingerprints", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.exact_source_roster, args.structural_fingerprint_index, args.review_decisions,
        require_complete_structural_fingerprints=args.require_complete_structural_fingerprints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("schema", "status", "subject_count", "lineage_count", "multi_subject_lineage_count", "review_required_count", "split_authorized", "training_authorized")}, sort_keys=True))


if __name__ == "__main__":
    main()
