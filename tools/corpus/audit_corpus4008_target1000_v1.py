#!/usr/bin/env python3
"""Fail-closed coverage audit for *distinct* independent per-head training subjects.

This is an AUDITOR, not an admission producer. It never authorizes training.
A candidate row, clip, pose, view, style, license hint, or rig count is NOT a
subject with qualified supervision. External admission receipts require source,
rights, split and target courts; their actual payload/authority must be
verified independently of this counting script before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.Corpus4008Target1000QuotaAudit.v1"
RECEIPT_SCHEMA = "RealSaS.IndependentHeadAdmissionEvidence.v1"
HEAD_TARGETS = {
    "IRIS_GEOMETRY": (
        "SOURCE_IMPORT", "CAMERA_OBSERVATION",
        "SURFACE_GEOMETRY_UDF", "SURFACE_NORMALS",
    ),
    "IRIS_SEMANTIC": (
        "SEMANTIC_CONCEPT", "COMPONENT_INSTANCE_SUPPORT",
        "FUNCTIONAL_AFFORDANCE_PRIOR", "STRUCTURAL_COMPONENT_RELATIONS",
        "SEMANTIC_VIEW_CONSISTENCY",
    ),
    "IRIS_GRAPHICS": (
        "APPEARANCE_CAA", "VISIBILITY_OCCLUSION",
        "RAY_COMPOSITING", "MOTION_CONDITIONED_RGB",
        "GRAPHIC_EDIT_REVEAL_CONSEQUENCE",
    ),
    "TESSA": (
        "MECHANICAL_SURFACE_TOPOLOGY",
        "ANIMATION_READY_DEFORMATION_CONSEQUENCE",
    ),
    "AXIS": (
        "RIG_HIERARCHY", "ANIMATION_KINEMATICS",
        "TYPED_CONTROLLABLE_RIG_PROGRAM_TEACHER",
        "CONTROL_RESPONSE_CONSEQUENCE",
    ),
    "MIRA": (
        "RIG_HIERARCHY", "SKIN_WEIGHTS",
        "SKIN_DEFORMATION_CONSEQUENCE",
    ),
}
SHA = re.compile(r"^[0-9a-f]{64}$")
# No training split leakage: only FIT subjects can count towards the 1000
# training-minimum; TUNE/DEV/CAL/EXTERNAL_HOLDOUT retain separate evidence.
SPLITS = frozenset(("FIT", "TUNE", "DEV", "CAL", "EXTERNAL_HOLDOUT"))
EVIDENCE_KINDS = ("source", "rights", "split", "character", "visual", "head")
REQUIRED_COURT_KEYS = ("schema", "asset_id", "source_sha256",
                       "court_kind", "status", "independent_review")


def require(value, reason):
    if not value:
        raise ValueError("TARGET1000_" + reason)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inside(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and relative != "", "EVIDENCE_PATH")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "EVIDENCE_PATH_ESCAPE")
    p = (root / rel).resolve()
    require(p.is_relative_to(root.resolve()), "EVIDENCE_PATH_SYMLINK_ESCAPE")
    return p


def evidence_courts(rec, root):
    """Check exact SHA-pinned distinct court artifacts; never invent a court."""
    evidence = rec.get("independent_evidence_files")
    require(isinstance(evidence, dict)
            and set(evidence) == set(EVIDENCE_KINDS), "COURT_FILE_SET")
    digests = set()
    for kind in EVIDENCE_KINDS:
        item = evidence[kind]
        require(isinstance(item, dict), "COURT_FILE_RECORD")
        expected = item.get("sha256")
        require(isinstance(expected, str) and SHA.fullmatch(expected),
                "COURT_SHA")
        path = inside(root, item.get("path"))
        require(path.is_file() and sha(path) == expected, "COURT_CONTENT_SHA")
        require(expected not in digests, "REUSED_COURT_CONTENT")
        digests.add(expected)
        doc = json.loads(path.read_text())
        require(isinstance(doc, dict) and
                all(k in doc for k in REQUIRED_COURT_KEYS), "COURT_PAYLOAD")
        require(doc["schema"] == "RealSaS.IndependentSourceCourt.v1",
                "COURT_SCHEMA")
        require(doc["court_kind"] == kind, "COURT_KIND")
        require(doc["asset_id"] == rec["canonical_asset_id"] and
                doc["source_sha256"] == rec["source_sha256"],
                "COURT_IDENTITY")
        require(doc["status"] == "PASS", "COURT_NOT_PASSED")
        require(doc["independent_review"] is True, "COURT_REVIEW")
        require(doc.get("training_authorized") is not True,
                "COURT_PREMATURE_ADMISSION")
    return True


def read_manifest(path):
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "MASTER_TYPE")
    for aid, row in value.items():
        require(isinstance(row, dict) and
                row.get("canonical_asset_id") == aid,
                "MASTER_ASSET_ID")
        require(row.get("split") in SPLITS, "MASTER_SPLIT")
    return value


def audit(selection: Path, receipts: Path, evidence_root: Path,
          output: Path, quota: int = 1000, expected_master_count: int = 3993):
    require(quota >= 1, "QUOTA")
    master = read_manifest(selection)
    require(len(master) == expected_master_count, "MASTER_SIZE")
    # Core requirements: stable canonical identity + immutable registered source.
    counts = defaultdict(set)
    per_head_strata = defaultdict(lambda: defaultdict(set))
    receipts_seen = set()
    rejected = Counter()
    warnings = []
    evaluated = 0
    files = sorted(receipts.rglob("*.json")) if receipts.is_dir() else []
    for f in files:
        evaluated += 1
        try:
            rec = json.loads(f.read_text())
            require(rec.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA")
            aid = rec.get("canonical_asset_id")
            require(aid in master, "ASSET_NOT_REGISTERED")
            row = master[aid]
            require(rec.get("candidate_id") == row.get("candidate_id"),
                    "CANDIDATE_ID")
            require(rec.get("split") == row.get("split"), "SPLIT_MISMATCH")
            require(rec.get("training_authorized") is False,
                    "MUST_NOT_AUTHORIZE_TRAINING")
            head = rec.get("head")
            require(head in HEAD_TARGETS, "UNKNOWN_HEAD")
            require((aid, head) not in receipts_seen, "DUPLICATE_RECEIPT")
            receipts_seen.add((aid, head))
            source_sha = rec.get("source_sha256")
            require(isinstance(source_sha, str) and SHA.fullmatch(source_sha),
                    "SOURCE_SHA")
            original_token = row["candidate_id"].split(":", 1)[-1]
            # Full SHA or 24-char source token only; unknown lineage NOT accepted.
            require(len(original_token) in (24, 64) and
                    source_sha.startswith(original_token), "SOURCE_TOKEN_DRIFT")
            require(rec.get("character_class") == "CHARACTER",
                    "NOT_CONFIRMED_CHARACTER")
            require(rec.get("commercial_ml_rights") == "INDEPENDENTLY_VERIFIED",
                    "RIGHTS_NOT_VERIFIED")
            require(rec.get("source_scene_court") == "PASS"
                    and rec.get("visual_court") == "PASS"
                    and rec.get("head_independent_court") == "PASS",
                    "UNQUALIFIED_SOURCE_OR_HEAD")
            targets = rec.get("target_truth")
            require(isinstance(targets, dict) and
                    all(targets.get(t) == "PASS" for t in HEAD_TARGETS[head]),
                    "INCOMPLETE_HEAD_TARGETS")
            require(evidence_courts(rec, evidence_root), "NO_EVIDENCE")
            lineage = rec.get("source_identity_cluster")
            require(isinstance(lineage, str) and len(lineage) >= 12,
                    "NO_DEDUP_CLUSTER")
            require(rec.get("cross_split_lineage_court") == "PASS",
                    "CROSS_SPLIT_NOT_COURTED")
            if row["split"] == "FIT":
                counts[head].add(lineage)
                strata = rec.get("morphology_stratum", "UNKNOWN")
                per_head_strata[head][strata].add(lineage)
            # TUNE / DEV / CAL / HOLDOUT are not FIT subject quota.
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
            reason = str(exc)
            rejected[reason] += 1
            if len(warnings) < 40:
                warnings.append({"filename": f.name, "reason": reason})
    per_head = {}
    for head in HEAD_TARGETS:
        n = len(counts[head])
        per_head[head] = {
            "target_distinct_fit_subjects": quota,
            "evidence_receipt_supported_distinct_fit_clusters": n,
            "missing_to_quota": max(0, quota - n),
            "quota_met_by_receipts": n >= quota,
            "required_target_families": list(HEAD_TARGETS[head]),
            "morphology_strata_receipt_supported": {
                k: len(v) for k, v in sorted(per_head_strata[head].items())
            },
            "training_authorized": False,
        }
    report = {
        "schema": SCHEMA,
        "master_selection_sha256": sha(selection),
        "master_count": len(master),
        "independent_receipt_files_inspected": evaluated,
        "rejected_receipt_reason_counts": dict(sorted(rejected.items())),
        "rejected_receipt_examples": warnings,
        "per_head": per_head,
        "cross_head_joint_cohort_proven": False,
        "rights_checked_from_source": False,
        "authored_control_ik_courts_completed": False,
        "all_legacy_admit_decisions_are_non_authoritative": True,
        "model_training_authorized": False,
        "warning": (
            "QUOTA MET is only a SHA-bound receipt coverage claim, not an "
            "independent re-audit of biological correctness, legal rights, "
            "semantic meaning or trainer/loss readiness."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--admission-receipts", type=Path, required=True)
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--quota", type=int, default=1000)
    p.add_argument("--expected-master-count", type=int, default=3993)
    a = p.parse_args()
    report = audit(a.selection, a.admission_receipts,
                   a.evidence_root, a.output, a.quota,
                   a.expected_master_count)
    print(json.dumps({
        "master_count": report["master_count"],
        "independent_receipt_files_inspected":
        report["independent_receipt_files_inspected"],
        "per_head": {k: v["evidence_receipt_supported_distinct_fit_clusters"]
                     for k, v in report["per_head"].items()},
        "training_authorized": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
