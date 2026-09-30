#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


FOCUS_PATTERNS = (
    r"^compiler/realsas_compiler_core/substrate/",
    r"^compiler/realsas_compiler_core/geometry_",
    r"^compiler/realsas_compiler_core/canonical_mesh",
    r"^compiler/realsas_compiler_core/mechanical_",
    r"^compiler/realsas_compiler_core/mesh/",
    r"^compiler/realsas_compiler_core/product_mesh_skin",
    r"^compiler/realsas_compiler_core/skin\.py$",
    r"^compiler/realsas_compiler_core/rig\.py$",
    r"^compiler/realsas_compiler_services/orchestrator/adapters/iris_geometry_v2\.py$",
    r"^compiler/realsas_compiler_services/orchestrator/adapters/learned_mechanics_v2\.py$",
    r"^compiler/realsas_compiler_services/orchestrator/adapters/mesh_v2\.py$",
    r"^compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture\.py$",
    r"^compiler/realsas_compiler_services/orchestrator/mainline\.py$",
    r"^canonical/MAINLINE_EXECUTION_PLAN_V2\.json$",
    r"^tests/compiler/.*(mesh|skin|rig|substrate|geometry|topology|g3|g5|provenance|mechanical).*\.py$",
    r"^tests/repository/test_mainline.*\.py$",
    r"^\.github/workflows/.*(mesh|skin|topology|g3|geometry|dense|repartition|mainline).*\.yml$",
)

KEYWORDS = (
    "falsif", "negative", "revert", "replace", "supersed", "deprecat",
    "wrong", "bug", "fix", "repair", "correct", "drift", "authority",
    "lineage", "fail closed", "fail-closed", "proof", "adversar",
    "oracle", "closure", "promote", "qualif", "holeless", "topology",
    "weight", "skin", "repartition", "dense face", "normal", "g3", "g3b",
)

SUBJECT_TOKENS = ("knight", "mage", "fit1", "demo")


def sh(*args: str, check: bool = True) -> str:
    p = subprocess.run(
        list(args), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if check and p.returncode:
        raise RuntimeError(
            "COMMAND_FAIL\n" + " ".join(args) + "\nSTDOUT:\n" + p.stdout + "\nSTDERR:\n" + p.stderr
        )
    return p.stdout


def git(*args: str, check: bool = True) -> str:
    return sh("git", *args, check=check)


def matches_focus(path: str) -> bool:
    return any(re.search(pat, path) for pat in FOCUS_PATTERNS)


def parse_commit_block(raw: str) -> list[dict]:
    commits = []
    current = None
    for line in raw.splitlines():
        if line.startswith("@@@"):
            if current:
                commits.append(current)
            _, sha, parents, date, subject = line.split("\t", 4)
            current = {
                "sha": sha,
                "parents": parents.split() if parents else [],
                "date": date,
                "subject": subject,
                "files": [],
            }
        elif current is not None and line.strip():
            parts = line.split("\t")
            status = parts[0]
            paths = parts[1:]
            current["files"].append({"status": status, "paths": paths})
    if current:
        commits.append(current)
    return commits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ref", required=True)
    ap.add_argument("--research-ref", required=True)
    ap.add_argument("--report", default="lineage_report.json")
    args = ap.parse_args()

    merge_base = git("merge-base", args.baseline_ref, args.research_ref).strip()
    ahead_count = int(git("rev-list", "--count", f"{merge_base}..{args.research_ref}").strip())

    raw = git(
        "log",
        "--reverse",
        "--date=iso-strict",
        "--format=@@@%x09%H%x09%P%x09%cI%x09%s",
        "--name-status",
        "--find-renames",
        f"{merge_base}..{args.research_ref}",
    )
    all_commits = parse_commit_block(raw)

    focused = []
    path_history = defaultdict(list)
    keyword_counts = defaultdict(int)
    subject_specific_commits = []
    merge_commits = []

    for idx, c in enumerate(all_commits, start=1):
        touched = []
        for row in c["files"]:
            for path in row["paths"]:
                if matches_focus(path):
                    touched.append(path)
        touched = sorted(set(touched))
        if len(c["parents"]) > 1:
            merge_commits.append(c["sha"])
        if not touched:
            continue

        subject_l = c["subject"].lower()
        flags = sorted(k for k in KEYWORDS if k in subject_l)
        for k in flags:
            keyword_counts[k] += 1
        subject_specific = any(t in subject_l for t in SUBJECT_TOKENS)
        if subject_specific:
            subject_specific_commits.append(c["sha"])

        row = {
            "ordinal_in_research_lineage": idx,
            "sha": c["sha"],
            "parents": c["parents"],
            "date": c["date"],
            "subject": c["subject"],
            "focus_paths": touched,
            "message_flags": flags,
            "subject_specific_message": subject_specific,
        }
        focused.append(row)
        for path in touched:
            path_history[path].append({
                "ordinal_in_research_lineage": idx,
                "sha": c["sha"],
                "date": c["date"],
                "subject": c["subject"],
                "message_flags": flags,
                "subject_specific_message": subject_specific,
            })

    # Detect explicit first-parent reversals of focused files by comparing changed
    # path sets and commit subjects. This is only a candidate list; humans must
    # inspect actual patches before promotion.
    reversal_candidates = []
    for c in focused:
        s = c["subject"].lower()
        if any(token in s for token in ("revert", "replace", "supersed", "rollback", "restore", "wrong", "falsif")):
            reversal_candidates.append(c)

    current_tips = []
    for path, hist in sorted(path_history.items()):
        last = hist[-1]
        current_tips.append({
            "path": path,
            "last_touch": last,
            "history_length": len(hist),
        })

    # Mechanical/geometric proof staircases: preserve every focused commit whose
    # subject indicates proof/repair/authority evolution, in chronological order.
    proof_staircase = [
        c for c in focused
        if set(c["message_flags"]) & {
            "falsif", "negative", "revert", "replace", "supersed", "fix",
            "repair", "correct", "drift", "authority", "lineage", "proof",
            "adversar", "oracle", "closure", "promote", "qualif", "holeless",
            "topology", "weight", "skin", "repartition", "dense face", "normal",
            "g3", "g3b",
        }
    ]

    report = {
        "schema": "RealSaS.GenericMechanicalGeometryLineageAudit.v1",
        "baseline_ref": args.baseline_ref,
        "research_ref": args.research_ref,
        "merge_base": merge_base,
        "research_commit_count_from_merge_base": ahead_count,
        "parsed_commit_count": len(all_commits),
        "focused_commit_count": len(focused),
        "focused_path_count": len(path_history),
        "merge_commit_count": len(merge_commits),
        "keyword_counts": dict(sorted(keyword_counts.items())),
        "subject_specific_focused_commit_count": len(subject_specific_commits),
        "reversal_candidate_count": len(reversal_candidates),
        "reversal_candidates": reversal_candidates,
        "proof_staircase": proof_staircase,
        "current_path_tips": current_tips,
        "path_history": dict(sorted(path_history.items())),
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps({
        "merge_base": merge_base,
        "research_commit_count_from_merge_base": ahead_count,
        "parsed_commit_count": len(all_commits),
        "focused_commit_count": len(focused),
        "focused_path_count": len(path_history),
        "reversal_candidate_count": len(reversal_candidates),
        "subject_specific_focused_commit_count": len(subject_specific_commits),
        "report": args.report,
    }, indent=2, sort_keys=True))

    print("\n=== LAST 80 PROOF/REPAIR/AUTHORITY COMMITS ===")
    for row in proof_staircase[-80:]:
        print(
            f"{row['ordinal_in_research_lineage']:04d} "
            f"{row['sha'][:12]} {row['date']} {row['subject']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
