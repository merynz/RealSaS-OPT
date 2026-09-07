#!/usr/bin/env python3
"""Audit whether high-signal repository knowledge is indexed by the continuity spine.

This is deliberately conservative. During bootstrap it reports unexplained artifacts
without pretending they are absent. After BOOTSTRAP_AUDIT_CLOSED, newly unindexed
high-signal artifacts become a fail-closed continuity error.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "canonical" / "BOOTSTRAP_COVERAGE_STATE_V1.json"
OUT_JSON = ROOT / "canonical" / "CONTEXT_COVERAGE_AUDIT.json"
OUT_MD = ROOT / "canonical" / "CONTEXT_COVERAGE_AUDIT.md"

INDEX_FILES = [
    "CURRENT_STATE.md",
    "README.md",
    "REPOSITORY_MAP.md",
    "SYSTEM_INDEX.md",
    "canonical/README.md",
    "canonical/CONTEXT_STATE_V1.json",
    "canonical/AUTHORITY_MAP_V1.json",
    "canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md",
    "canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md",
    "canonical/EXPERIMENT_REGISTRY_V1.json",
    "canonical/SCIENTIFIC_JOURNAL_V1.jsonl",
    "canonical/BRANCH_AUTHORITY_V1.md",
]

HIGH_SIGNAL_TOKENS = re.compile(
    r"(PREREG|REPORT|RESULT|AUDIT|DECISION|VERDICT|CLOSURE|MANIFEST|LEDGER|SEAL|"
    r"RETRACTION|PROMOTION|AUTHORITY|STATE|PLAN|MATRIX|DISPOSITION|GATE|FREEZE|"
    r"EXPERIMENT|FIT|FAMILY|CAUSAL|CHALLENGER)",
    re.IGNORECASE,
)

WORKFLOW_SIGNAL = re.compile(
    r"(geppetto|iris|arachne|fit|family|audit|closure|experiment|promotion|freeze|restoration)",
    re.IGNORECASE,
)


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def tracked_files() -> list[str]:
    return [line for line in run("git", "ls-files").splitlines() if line.strip()]


def live_branches() -> list[dict]:
    raw = run("git", "ls-remote", "--heads", "origin")
    out = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split("\t", 1)
        prefix = "refs/heads/"
        if ref.startswith(prefix):
            out.append({"name": ref[len(prefix):], "sha": sha})
    return sorted(out, key=lambda x: x["name"])


def is_high_signal(path: str) -> bool:
    p = Path(path)
    name = p.name

    if path in {"CURRENT_STATE.md", "RESTORATION_STATE.md", "REPOSITORY_MAP.md", "SYSTEM_INDEX.md"}:
        return True

    if path.startswith("prereg/"):
        return True

    if path.startswith("canonical/"):
        if path in {
            "canonical/LIVE_AUTHORITY_MAP.md",
            "canonical/REHYDRATION_PACKET.md",
            "canonical/CONTEXT_COVERAGE_AUDIT.md",
            "canonical/CONTEXT_COVERAGE_AUDIT.json",
        }:
            return False
        return bool(HIGH_SIGNAL_TOKENS.search(name))

    if path.startswith("experiments/"):
        return p.suffix.lower() == ".ipynb" or bool(HIGH_SIGNAL_TOKENS.search(name))

    if path.startswith("models/") and "/challengers/" in path and p.suffix.lower() == ".py":
        return True

    if path.startswith(".github/workflows/") and p.suffix.lower() in {".yml", ".yaml"}:
        return bool(WORKFLOW_SIGNAL.search(name))

    return False


def continuity_corpus() -> str:
    chunks: list[str] = []
    for rel in INDEX_FILES:
        path = ROOT / rel
        if path.exists():
            try:
                chunks.append(path.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                pass
    return "\n".join(chunks)


def referenced(path: str, corpus: str) -> tuple[bool, str]:
    if path in INDEX_FILES:
        return True, "continuity spine file"
    if path in corpus:
        return True, "exact path referenced by continuity spine"
    basename = Path(path).name
    if basename and basename in corpus:
        return True, "basename referenced by continuity spine"
    return False, "no path/basename reference found in continuity spine"


def branch_coverage(branches: list[dict]) -> list[dict]:
    auth_path = ROOT / "canonical" / "AUTHORITY_MAP_V1.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    active = {x["branch"]: x["id"] for x in auth.get("active_experiments", [])}
    overrides = {x["branch"]: x.get("class", "OVERRIDE") for x in auth.get("branch_overrides", [])}
    canonical = auth["branch_policy"]["canonical_branch"]
    rows = []
    for item in branches:
        name = item["name"]
        if name == canonical:
            cls, reason = "CANONICAL", "canonical branch"
        elif name in active:
            cls, reason = "ACTIVE_EXPERIMENT", active[name]
        elif name in overrides:
            cls, reason = overrides[name], "explicit authority-map override"
        else:
            cls, reason = "EVIDENCE_ONLY_UNREGISTERED", "safe default; needs disposition only if scientifically material"
        rows.append({**item, "class": cls, "reason": reason})
    return rows


def main() -> int:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    files = tracked_files()
    corpus = continuity_corpus()

    candidates = []
    for path in files:
        if not is_high_signal(path):
            continue
        ok, why = referenced(path, corpus)
        candidates.append({"path": path, "indexed": ok, "reason": why})

    indexed = [x for x in candidates if x["indexed"]]
    unindexed = [x for x in candidates if not x["indexed"]]
    branches = branch_coverage(live_branches())
    unregistered_branches = [x for x in branches if x["class"] == "EVIDENCE_ONLY_UNREGISTERED"]

    payload = {
        "schema_version": 1,
        "bootstrap_status": bootstrap["status"],
        "candidate_rule": "high-signal canonical/prereg/experiment/challenger/workflow artifacts",
        "tracked_file_count": len(files),
        "high_signal_artifact_count": len(candidates),
        "indexed_high_signal_count": len(indexed),
        "unindexed_high_signal_count": len(unindexed),
        "live_branch_count": len(branches),
        "unregistered_branch_count": len(unregistered_branches),
        "coverage_fraction": (len(indexed) / len(candidates)) if candidates else 1.0,
        "unindexed_high_signal_artifacts": unindexed,
        "branches": branches,
        "closure_rule": "During bootstrap, unindexed means UNKNOWN/NEEDS_AUDIT. After BOOTSTRAP_AUDIT_CLOSED, any new unindexed high-signal artifact is a continuity failure."
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    lines = [
        "# RealSaS — Context Coverage Audit",
        "",
        "> **GENERATED AUDIT-OF-AUDITS VIEW.** Do not hand-edit.",
        f"> Bootstrap status: `{bootstrap['status']}`",
        "",
        "## Coverage",
        "",
        f"- Git-tracked files: **{len(files)}**",
        f"- High-signal knowledge artifacts: **{len(candidates)}**",
        f"- Indexed/referenced by continuity spine: **{len(indexed)}**",
        f"- Unindexed high-signal artifacts: **{len(unindexed)}**",
        f"- Coverage: **{payload['coverage_fraction']:.1%}**",
        f"- Live branches: **{len(branches)}**",
        f"- Safe-default unregistered branches: **{len(unregistered_branches)}**",
        "",
        "During bootstrap, **UNINDEXED does not mean irrelevant or false**. It means the continuity system has not yet audited/disposed that artifact.",
        "",
        "## Unindexed high-signal artifacts",
        "",
    ]
    if unindexed:
        for item in unindexed:
            lines.append(f"- `{item['path']}` — {item['reason']}")
    else:
        lines.append("_None._")

    lines += ["", "## Live branch disposition", "", "| Branch | Head | Class | Reason |", "|---|---|---|---|"]
    for item in branches:
        lines.append(f"| `{item['name']}` | `{item['sha'][:12]}` | `{item['class']}` | {item['reason']} |")

    lines += [
        "",
        "## Closure semantics",
        "",
        "- Bootstrap closes only after every high-signal residual is either indexed, explicitly historical/superseded/duplicate, or accepted as a documented residual.",
        "- After closure, this scanner becomes a regression guard: newly added high-signal artifacts must enter the continuity spine or CI fails.",
        "- Branch existence is not authority. Unregistered branches remain fail-safe evidence-only until explicitly promoted/activated.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "bootstrap_status": bootstrap["status"],
        "high_signal": len(candidates),
        "indexed": len(indexed),
        "unindexed": len(unindexed),
        "branches": len(branches),
        "unregistered_branches": len(unregistered_branches),
    }, indent=2))

    if bootstrap["status"] == "BOOTSTRAP_AUDIT_CLOSED" and unindexed:
        print("CONTEXT COVERAGE FAILURE: new unindexed high-signal artifacts after bootstrap closure", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
