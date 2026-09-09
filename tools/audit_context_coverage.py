#!/usr/bin/env python3
"""Audit high-signal repository knowledge against the continuity spine.

AOA closure uses two honest coverage classes in addition to explicit semantic
indexing:

1. high-signal artifacts already present at the FIT1 gate anchor are accepted as
   historical provenance residuals unless separately indexed;
2. any high-signal artifact introduced or changed after the FIT1 gate anchor must
   be discoverable through the exhaustive FIT1 commit ledger.

After BOOTSTRAP_AUDIT_CLOSED, an artifact satisfying neither rule is an unexplained
continuity regression and this script fails closed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "canonical" / "BOOTSTRAP_COVERAGE_STATE_V1.json"
DISPOSITION = ROOT / "canonical" / "AOA_ARTIFACT_DISPOSITION_V1.json"
FIT1_LEDGER = ROOT / "canonical" / "FIT1_COMMIT_LINEAGE_V1.json"
OUT_JSON = ROOT / "canonical" / "CONTEXT_COVERAGE_AUDIT.json"
OUT_MD = ROOT / "canonical" / "CONTEXT_COVERAGE_AUDIT.md"
FIT1_GATE_ANCHOR = "f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee"

# Current continuity sources come first. V1 registry/journal remain in the corpus as
# immutable historical provenance, but no longer define current navigation.
INDEX_FILES = [
    "AGENTS.md",
    "CURRENT_STATE.md",
    "README.md",
    "REPOSITORY_MAP.md",
    "SYSTEM_INDEX.md",
    "canonical/README.md",
    "canonical/CONTEXT_STATE_V1.json",
    "canonical/AUTHORITY_MAP_V1.json",
    "canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md",
    "canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md",
    "canonical/EXPERIMENT_REGISTRY_V2.json",
    "canonical/SCIENTIFIC_JOURNAL_V2_20260909.jsonl",
    "canonical/EXPERIMENT_REGISTRY_V1.json",
    "canonical/SCIENTIFIC_JOURNAL_V1.jsonl",
    "canonical/BRANCH_AUTHORITY_V1.md",
    "canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md",
    "canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md",
    "canonical/FIT1_EVIDENCE_INDEX_20260909.md",
    "canonical/GEPPETTO_REFERENCE_STRENGTH_FIT1_CLOSURE_20260908.md",
    "canonical/GEPPETTO_REFERENCE_STRENGTH_MAINLINE_PROMOTION_20260909.md",
    "canonical/FIT1_COMMIT_LINEAGE_V1.md",
    "canonical/AOA_ARTIFACT_DISPOSITION_V1.json",
    "canonical/AUDIT_OF_AUDITS_CLOSURE_20260907.md",
]

HIGH_SIGNAL_TOKENS = re.compile(
    r"(PREREG|REPORT|RESULT|AUDIT|DECISION|VERDICT|CLOSURE|MANIFEST|LEDGER|SEAL|"
    r"RETRACTION|PROMOTION|AUTHORITY|STATE|PLAN|MATRIX|DISPOSITION|GATE|FREEZE|"
    r"EXPERIMENT|FIT|FAMILY|CAUSAL|CHALLENGER|LINEAGE|OWNERSHIP)",
    re.IGNORECASE,
)

WORKFLOW_SIGNAL = re.compile(
    r"(geppetto|iris|arachne|fit|family|audit|closure|experiment|promotion|freeze|restoration|authority|continuity)",
    re.IGNORECASE,
)

GENERATED_SELF = {
    "canonical/LIVE_AUTHORITY_MAP.md",
    "canonical/REHYDRATION_PACKET.md",
    "canonical/CONTEXT_COVERAGE_AUDIT.md",
    "canonical/CONTEXT_COVERAGE_AUDIT.json",
    "canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json",
    "canonical/BOOTSTRAP_AUDIT_QUEUE.md",
    "canonical/FIT1_COMMIT_LINEAGE_V1.md",
    "canonical/FIT1_COMMIT_LINEAGE_V1.json",
}


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_files() -> list[str]:
    return [line for line in run("git", "ls-files").splitlines() if line.strip()]


def files_at_anchor() -> set[str]:
    raw = run("git", "ls-tree", "-r", "--name-only", FIT1_GATE_ANCHOR)
    return {line for line in raw.splitlines() if line.strip()}


def postfit_changed_paths() -> set[str]:
    if not FIT1_LEDGER.exists():
        return set()
    payload = load(FIT1_LEDGER)
    if payload.get("fit1_gate_anchor") != FIT1_GATE_ANCHOR:
        raise RuntimeError("FIT1 commit ledger anchor mismatch")
    out: set[str] = set()
    for row in payload.get("commits", []):
        if row.get("is_fit1_gate_anchor"):
            continue
        out.update(row.get("changed_paths", []))
    return out


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

    if path in {"AGENTS.md", "CURRENT_STATE.md", "RESTORATION_STATE.md", "REPOSITORY_MAP.md", "SYSTEM_INDEX.md"}:
        return True
    if path.startswith("prereg/"):
        return True
    if path.startswith("canonical/"):
        if path in GENERATED_SELF:
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


def explicit_reference(path: str, corpus: str) -> tuple[bool, str]:
    if path in INDEX_FILES:
        return True, "continuity spine file"
    if path in corpus:
        return True, "exact path referenced by continuity spine"
    basename = Path(path).name
    if basename and basename in corpus:
        return True, "basename referenced by continuity spine"
    return False, "not explicitly referenced"


def classify(path: str, corpus: str, anchor_files: set[str], changed: set[str]) -> dict:
    explicit, why = explicit_reference(path, corpus)
    if explicit:
        return {"path": path, "class": "INDEXED_EXPLICIT", "reason": why, "explained": True}
    if path in changed:
        return {
            "path": path,
            "class": "FIT1_COMMIT_LEDGER_COVERED",
            "reason": "introduced or changed in FIT1-descendant history; exact commit discoverable in FIT1 ledger",
            "explained": True,
        }
    if path in anchor_files:
        return {
            "path": path,
            "class": "HISTORICAL_PROVENANCE_ACCEPTED_RESIDUAL",
            "reason": "already present at FIT1 gate anchor; preserved as discoverable historical provenance, not current authority by default",
            "explained": True,
        }
    return {
        "path": path,
        "class": "UNEXPLAINED",
        "reason": "neither explicitly indexed, present at FIT1 anchor, nor covered by FIT1 descendant commit ledger",
        "explained": False,
    }


def branch_coverage(branches: list[dict]) -> list[dict]:
    auth = load(ROOT / "canonical" / "AUTHORITY_MAP_V1.json")
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
            cls, reason = "EVIDENCE_ONLY_UNREGISTERED", "safe default under branch authority policy"
        rows.append({**item, "class": cls, "reason": reason})
    return rows


def main() -> int:
    bootstrap = load(BOOTSTRAP)
    if bootstrap["status"] == "BOOTSTRAP_AUDIT_CLOSED" and not DISPOSITION.exists():
        raise RuntimeError("AOA closed but disposition manifest is missing")
    if bootstrap["status"] == "BOOTSTRAP_AUDIT_CLOSED" and not FIT1_LEDGER.exists():
        raise RuntimeError("AOA closed but FIT1 commit ledger is missing")

    files = tracked_files()
    corpus = continuity_corpus()
    anchor_files = files_at_anchor()
    changed = postfit_changed_paths()

    candidates = [classify(path, corpus, anchor_files, changed) for path in files if is_high_signal(path)]
    unexplained = [x for x in candidates if not x["explained"]]
    classes: dict[str, int] = {}
    for row in candidates:
        classes[row["class"]] = classes.get(row["class"], 0) + 1

    branches = branch_coverage(live_branches())
    unregistered_branches = [x for x in branches if x["class"] == "EVIDENCE_ONLY_UNREGISTERED"]

    payload = {
        "schema_version": 2,
        "bootstrap_status": bootstrap["status"],
        "fit1_gate_anchor": FIT1_GATE_ANCHOR,
        "candidate_rule": "high-signal canonical/prereg/experiment/challenger/workflow artifacts",
        "tracked_file_count": len(files),
        "high_signal_artifact_count": len(candidates),
        "classification_counts": classes,
        "explained_high_signal_count": len(candidates) - len(unexplained),
        "unexplained_high_signal_count": len(unexplained),
        "coverage_fraction": ((len(candidates) - len(unexplained)) / len(candidates)) if candidates else 1.0,
        "live_branch_count": len(branches),
        "safe_default_evidence_only_branch_count": len(unregistered_branches),
        "unexplained_high_signal_artifacts": unexplained,
        "branches": branches,
        "semantics": {
            "INDEXED_EXPLICIT": "continuity spine contains an explicit path/basename or the artifact is itself a spine file",
            "FIT1_COMMIT_LEDGER_COVERED": "post-FIT1 change is exhaustively discoverable by commit; exact scientific claims still require reading source/prereg/result",
            "HISTORICAL_PROVENANCE_ACCEPTED_RESIDUAL": "pre-existing at FIT1 gate; retained for provenance but not current authority by default",
            "UNEXPLAINED": "continuity regression after closure",
        },
        "closure_rule": "After BOOTSTRAP_AUDIT_CLOSED, UNEXPLAINED must remain zero. Ledger coverage is discoverability, not scientific promotion.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RealSaS — Context Coverage Audit",
        "",
        "> **GENERATED AUDIT-OF-AUDITS VIEW. DO NOT HAND-EDIT.**",
        f"> Bootstrap status: `{bootstrap['status']}`",
        f"> FIT1 gate anchor: `{FIT1_GATE_ANCHOR}`",
        "",
        "## Coverage",
        "",
        f"- Git-tracked files: **{len(files)}**",
        f"- High-signal knowledge artifacts: **{len(candidates)}**",
        f"- Explained by continuity policy: **{len(candidates) - len(unexplained)}**",
        f"- Unexplained high-signal artifacts: **{len(unexplained)}**",
        f"- Coverage: **{payload['coverage_fraction']:.1%}**",
        f"- Live branches: **{len(branches)}**",
        f"- Safe-default evidence-only branches: **{len(unregistered_branches)}**",
        "",
        "### Classification counts",
        "",
    ]
    for key in sorted(classes):
        lines.append(f"- `{key}`: **{classes[key]}**")
    lines += [
        "",
        "`FIT1_COMMIT_LEDGER_COVERED` means exact provenance is recoverable; it does **not** mean the artifact's scientific claim is promoted. `HISTORICAL_PROVENANCE_ACCEPTED_RESIDUAL` means the artifact predates the FIT1 gate and remains evidence/provenance unless another authority explicitly promotes it.",
        "",
        "## Unexplained high-signal artifacts",
        "",
    ]
    if unexplained:
        for item in unexplained:
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
        "- AOA closure is a **context/provenance closure**, not retroactive scientific validation of every historical report.",
        "- FIT1-to-now changes are recoverable through `FIT1_COMMIT_LINEAGE_V1`; semantic epochs are summarized in `FIT1_SCIENTIFIC_LINEAGE_V1.md`.",
        "- Pre-FIT residuals remain discoverable historical evidence and cannot outrank current authority without an explicit promotion/reconciliation.",
        "- Branch existence is not authority. Unregistered branches remain fail-safe evidence-only.",
        "- After closure, any newly unexplained high-signal artifact fails CI.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "bootstrap_status": bootstrap["status"],
        "high_signal": len(candidates),
        "classes": classes,
        "unexplained": len(unexplained),
        "branches": len(branches),
    }, indent=2))

    if bootstrap["status"] == "BOOTSTRAP_AUDIT_CLOSED" and unexplained:
        print("CONTEXT COVERAGE FAILURE: unexplained high-signal artifacts after AOA closure", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
