#!/usr/bin/env python3
"""Render exhaustive commit-level lineage from the RealSaS FIT1 gate anchor.

This is a discovery/provenance ledger, not independent scientific authority.  It
ensures every commit descended from the FIT1 gate anchor on any live branch is
visible to future chats/agents.  Semantic interpretation lives in
canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md and the experiment/architecture ledgers.
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIT1_GATE_ANCHOR = "f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee"
FIRST_EXECUTABLE_FIT_BASE = "de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3"
OUT_JSON = ROOT / "canonical" / "FIT1_COMMIT_LINEAGE_V1.json"
OUT_MD = ROOT / "canonical" / "FIT1_COMMIT_LINEAGE_V1.md"


def run(*args: str, check: bool = True) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def is_ancestor(base: str, head: str) -> bool:
    p = subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT)
    return p.returncode == 0


def refs() -> dict[str, str]:
    # Local checkout is expected to have full history.  Include local + remote
    # branch refs but de-duplicate origin/foo versus foo when they point together.
    raw = run("git", "for-each-ref", "--format=%(refname)%09%(objectname)", "refs/heads", "refs/remotes/origin")
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        ref, sha = line.split("\t", 1)
        if ref == "refs/remotes/origin/HEAD":
            continue
        if ref.startswith("refs/remotes/origin/"):
            name = ref[len("refs/remotes/origin/"):]
        elif ref.startswith("refs/heads/"):
            name = ref[len("refs/heads/"):]
        else:
            name = ref
        out[name] = sha
    return dict(sorted(out.items()))


def changed_paths(sha: str) -> list[str]:
    if sha == FIT1_GATE_ANCHOR:
        parent = run("git", "rev-parse", f"{sha}^", check=False)
        if parent:
            raw = run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", parent, sha)
        else:
            raw = run("git", "show", "--pretty=", "--name-only", sha)
    else:
        raw = run("git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
    return sorted({x for x in raw.splitlines() if x.strip()})


def subsystem_tags(paths: list[str], subject: str) -> list[str]:
    tags: set[str] = set()
    s = subject.lower()
    joined = "\n".join(paths).lower()
    blob = s + "\n" + joined
    rules = [
        ("IRIS_GSA", ("iris", "riggingsurface", "substrate", "scene_first_signed")),
        ("GEPPETTO", ("geppetto", "skeleton", "riganything", "ar01")),
        ("ARACHNE_SKIN", ("arachne", "skinfield", "skin_field", "skintokens", "weight")),
        ("COMPILER_PROOF", ("compiler", "proof", "canonicalpuppet", "solver", "qualification")),
        ("RUNTIME_PRODUCT", ("runtime", "export", "product/living_compile", "living compile")),
        ("FIT_DATA", ("fit", "mage", "family", "observation_authority")),
        ("GOVERNANCE_CONTINUITY", ("canonical/", "current_state", "agents.md", "authority", "continuity", "rehydration")),
        ("CI_TEST", (".github/workflows", "tests/", "test_", "ci(")),
    ]
    for tag, needles in rules:
        if any(n in blob for n in needles):
            tags.add(tag)
    if not tags:
        tags.add("OTHER")
    return sorted(tags)


def commit_meta(sha: str) -> dict:
    fmt = "%H%x00%P%x00%aI%x00%cI%x00%s"
    raw = run("git", "show", "-s", f"--format={fmt}", sha)
    parts = raw.split("\x00")
    if len(parts) != 5:
        raise RuntimeError(f"unexpected commit metadata for {sha}")
    return {
        "sha": parts[0],
        "parents": parts[1].split() if parts[1] else [],
        "author_time": parts[2],
        "committer_time": parts[3],
        "subject": parts[4],
    }


def main() -> int:
    run("git", "cat-file", "-e", f"{FIT1_GATE_ANCHOR}^{{commit}}")
    run("git", "cat-file", "-e", f"{FIRST_EXECUTABLE_FIT_BASE}^{{commit}}")

    branch_heads = refs()
    eligible = {name: sha for name, sha in branch_heads.items() if is_ancestor(FIT1_GATE_ANCHOR, sha)}

    commit_to_branches: dict[str, set[str]] = defaultdict(set)
    commits: set[str] = {FIT1_GATE_ANCHOR}
    for name, head in eligible.items():
        raw = run("git", "rev-list", "--reverse", "--topo-order", f"{FIT1_GATE_ANCHOR}..{head}")
        for sha in raw.splitlines():
            if not sha.strip():
                continue
            commits.add(sha)
            commit_to_branches[sha].add(name)
        commit_to_branches[FIT1_GATE_ANCHOR].add(name)

    # Sort globally by committer timestamp, then SHA. Topology is retained via parents.
    rows = []
    for sha in commits:
        meta = commit_meta(sha)
        paths = changed_paths(sha)
        meta["branches_containing"] = sorted(commit_to_branches.get(sha, set()))
        meta["main_reachable"] = "main" in meta["branches_containing"]
        meta["changed_paths"] = paths
        meta["subsystem_tags"] = subsystem_tags(paths, meta["subject"])
        meta["is_fit1_gate_anchor"] = sha == FIT1_GATE_ANCHOR
        meta["is_first_executable_fit_base"] = sha == FIRST_EXECUTABLE_FIT_BASE
        rows.append(meta)
    rows.sort(key=lambda x: (x["committer_time"], x["sha"]))

    payload = {
        "schema": "RealSaS.FIT1CommitLineage.v1",
        "authority": "DISCOVERY_PROVENANCE_NOT_INDEPENDENT_SCIENTIFIC_AUTHORITY",
        "fit1_gate_anchor": FIT1_GATE_ANCHOR,
        "first_executable_fit_base": FIRST_EXECUTABLE_FIT_BASE,
        "semantic_lineage": "canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md",
        "ownership_envelopes": "canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md",
        "eligible_live_branches": eligible,
        "commit_count": len(rows),
        "commits": rows,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RealSaS — FIT1 Commit Lineage V1",
        "",
        "> **GENERATED EXHAUSTIVE DISCOVERY/PROVENANCE VIEW — NOT INDEPENDENT SCIENTIFIC AUTHORITY.**",
        f"> FIT1 gate anchor: `{FIT1_GATE_ANCHOR}`",
        f"> First executable FIT base: `{FIRST_EXECUTABLE_FIT_BASE}`",
        "> Semantic interpretation: `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`",
        "> Ownership interpretation: `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`",
        "",
        f"Commits descended from the FIT1 gate anchor across eligible live refs: **{len(rows)}**.",
        "",
        "Every row is discoverable context. A row does **not** imply that its subject-line claim is scientifically promoted; inspect the linked prereg/result/authority ledger before making a scientific claim.",
        "",
        "| Committer time | SHA | Main? | Subsystem tags | Subject | Changed paths | Branch refs |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        path_text = "<br>".join(f"`{p}`" for p in row["changed_paths"][:12])
        if len(row["changed_paths"]) > 12:
            path_text += f"<br>… +{len(row['changed_paths']) - 12}"
        branch_text = "<br>".join(f"`{b}`" for b in row["branches_containing"])
        tags = ", ".join(row["subsystem_tags"])
        subject = row["subject"].replace("|", "\\|")
        lines.append(
            f"| `{row['committer_time']}` | `{row['sha'][:12]}` | {'yes' if row['main_reachable'] else 'no'} | {tags} | {subject} | {path_text or '_none_'} | {branch_text or '_none_'} |"
        )
    lines += [
        "",
        "## Use rule",
        "",
        "- Start with `FIT1_SCIENTIFIC_LINEAGE_V1.md` for the scientific flow.",
        "- Use this ledger to locate the exact commit when a detail, repair, regression, branch-only intervention, or chronology question matters.",
        "- Never infer `implemented`, `executed`, `promoted`, or `generalized` from a commit subject alone.",
        "- After AOA closure, a FIT1-descendant commit missing from this ledger is a continuity regression.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"fit1_commit_count": len(rows), "eligible_live_branches": len(eligible)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
