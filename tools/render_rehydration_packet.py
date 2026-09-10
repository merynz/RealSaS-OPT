#!/usr/bin/env python3
"""Render the compact RealSaS cross-chat/cross-agent rehydration packet.

The packet is navigation/cache, not independent scientific authority. It is valid
with either one explicitly active experiment or no active experiment at all.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = ROOT / "canonical" / "CONTEXT_STATE_V1.json"
AUTH_PATH = ROOT / "canonical" / "AUTHORITY_MAP_V1.json"
BOOTSTRAP_PATH = ROOT / "canonical" / "BOOTSTRAP_COVERAGE_STATE_V1.json"
CATALOG_PATH = ROOT / "canonical" / "KNOWLEDGE_ARTIFACT_CATALOG_V1.json"
COVERAGE_PATH = ROOT / "canonical" / "CONTEXT_COVERAGE_AUDIT.json"
OWNERSHIP_PATH = ROOT / "canonical" / "SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md"
FIT1_SCIENCE_PATH = ROOT / "canonical" / "FIT1_SCIENTIFIC_LINEAGE_V1.md"
FIT1_COMMIT_PATH = ROOT / "canonical" / "FIT1_COMMIT_LINEAGE_V1.md"
FIT1_EVIDENCE_PATH = ROOT / "canonical" / "FIT1_EVIDENCE_INDEX_20260909.md"
GEPPETTO_EVIDENCE_PATH = ROOT / "canonical" / "GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json"
AOA_CLOSURE_PATH = ROOT / "canonical" / "AUDIT_OF_AUDITS_CLOSURE_20260907.md"
AOA_DISPOSITION_PATH = ROOT / "canonical" / "AOA_ARTIFACT_DISPOSITION_V1.json"
CURRENT_PATH = ROOT / "CURRENT_STATE.md"
OUTPUT_PATH = ROOT / "canonical" / "REHYDRATION_PACKET.md"


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_optional(path: Path) -> dict | None:
    return load(path) if path.exists() else None


def live_heads() -> dict[str, str]:
    raw = run("git", "ls-remote", "--heads", "origin")
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, ref = line.split("\t", 1)
        prefix = "refs/heads/"
        if ref.startswith(prefix):
            out[ref[len(prefix):]] = sha
    return out


def validate(context: dict, authority: dict, current_text: str, heads: dict[str, str], bootstrap: dict) -> list[str]:
    errors: list[str] = []
    focus = context["current_focus"]
    active = authority.get("active_experiments", [])
    active_id = focus.get("active_experiment")

    if active_id is None:
        if active:
            errors.append("context says no active experiment but authority manifest lists active experiment(s)")
        if "Active experiment gate:** `NONE`" not in current_text:
            errors.append("CURRENT_STATE.md does not explicitly record active experiment gate NONE")
    else:
        matches = [e for e in active if e.get("id") == active_id]
        if len(matches) != 1:
            errors.append(f"context active experiment must match exactly one authority experiment: {active_id}")
        else:
            branch = matches[0].get("branch")
            if branch and branch not in heads:
                errors.append(f"active experiment branch missing on origin: {branch}")
        if active_id not in current_text:
            errors.append("CURRENT_STATE.md does not name context active experiment")

    state = focus.get("state")
    if state and state not in current_text:
        errors.append("CURRENT_STATE.md does not name context current state")
    closed_gate = focus.get("most_recent_closed_gate")
    if closed_gate and closed_gate not in current_text:
        errors.append("CURRENT_STATE.md does not name most recent closed gate")

    canonical = authority["branch_policy"]["canonical_branch"]
    if canonical not in heads:
        errors.append(f"canonical branch missing from origin: {canonical}")

    required = [OWNERSHIP_PATH, FIT1_SCIENCE_PATH, FIT1_EVIDENCE_PATH, GEPPETTO_EVIDENCE_PATH]
    for key in ("experiment_registry", "scientific_journal"):
        rel = context.get("authority", {}).get(key)
        if rel and not (ROOT / rel).exists():
            errors.append(f"current authority target missing: {rel}")
    if bootstrap.get("status") == "BOOTSTRAP_AUDIT_CLOSED":
        required += [FIT1_COMMIT_PATH, AOA_CLOSURE_PATH, AOA_DISPOSITION_PATH]
    for path in required:
        if not path.exists():
            errors.append(f"required continuity artifact missing: {path.relative_to(ROOT)}")
    return errors


def bullets(items: list[str]) -> list[str]:
    return [f"- {x}" for x in items]


def render(context: dict, authority: dict, heads: dict[str, str], errors: list[str], bootstrap: dict, catalog: dict | None, coverage: dict | None) -> str:
    product = context["product"]
    focus = context["current_focus"]
    fit = context["fit_science"]
    env = context["execution_environment"]
    canonical_branch = authority["branch_policy"]["canonical_branch"]
    main_head = heads.get(canonical_branch, "MISSING")
    active_id = focus.get("active_experiment")
    auth = context.get("authority", {})
    geppetto = context.get("geppetto_fit1_closed_result", {})
    arachne = context.get("arachne_current_state", {})

    lines: list[str] = [
        "# RealSaS — Rehydration Packet",
        "",
        "> **GENERATED NAVIGATION/CACHE — NOT INDEPENDENT SCIENTIFIC AUTHORITY.**  ",
        "> Continuation authority remains `CURRENT_STATE.md`; scientific claims require the referenced source/prereg/result authority.",
        "",
        "## 60-second state",
        "",
        f"- **Product:** {product['target']}",
        f"- **Current witness:** `{product['demo_witness']}`",
        f"- **Current module:** `{focus['module']}`",
        f"- **Current state:** `{focus['state']}`",
        f"- **Active experiment:** `{active_id if active_id is not None else 'NONE'}`",
        f"- **Most recent closed gate:** `{focus.get('most_recent_closed_gate', 'NONE')}`",
        f"- **Canonical main:** `{main_head[:12] if main_head != 'MISSING' else main_head}`",
        f"- **Promotion block:** {focus['promotion_block']}",
        f"- **Scope warning:** {focus['scope_warning']}",
        f"- **Next visible product milestone:** {product['next_visible_product_milestone']}",
        "",
        "## Current machine authority",
        "",
        f"- Experiment registry: `{auth.get('experiment_registry', 'UNSPECIFIED')}`",
        f"- Scientific journal: `{auth.get('scientific_journal', 'UNSPECIFIED')}`",
        f"- Historical registry: `{auth.get('historical_experiment_registry', 'UNSPECIFIED')}`",
        f"- Historical journal: `{auth.get('historical_scientific_journal', 'UNSPECIFIED')}`",
        "- Current pointers above outrank version guesses from filenames.",
        "",
        "## Mandatory ownership memory",
        "",
        "Read `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` before moving responsibilities between learned and deterministic layers.",
        "",
        "- **IRIS shorthand:** observations/cameras -> learned evidence -> deterministic GSA/RiggingSurfaceIR assembly/provenance. Current Mage FIT1 signed witness uses promoted scene-first V3; V2 remains its promoted observation/foundation support layer where referenced by lineage.",
        "- **Geppetto shorthand:** lossless RiggingSurfaceIR -> learned SkeletonProposalIR/evidence -> Compiler exact graph qualification/canonical IDs.",
        "- **Arachne shorthand:** qualified surface+skeleton -> learned skin/deformation proposal -> Compiler skin/mesh qualification.",
        "- **Memory guard:** **Geppetto is proposal, not canonical rig authority.** Compiler may constrain legality; it may not secretly repair missing Geppetto semantics.",
        "",
        "## FIT1 scientific memory",
        "",
        "- Investor/auditor evidence index: `canonical/FIT1_EVIDENCE_INDEX_20260909.md`.",
        "- Machine Geppetto evidence manifest: `canonical/GEPPETTO_FIT1_EVIDENCE_MANIFEST_V1.json`.",
        "- Semantic spine: `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md`.",
        "- Exhaustive commit/provenance ledger: `canonical/FIT1_COMMIT_LINEAGE_V1.md`.",
        "- FIT1 gate anchor: `f6ce5dbc8719d6b6c592a4e060d8f1b38056b8ee`.",
        "- First executable FIT base: `de1a44cae1195dd9cbad3b23ef75d58ae80aa9b3`.",
        "- FIT1 is one-witness architecture/mechanism qualification; it is not generalization proof.",
    ]

    if geppetto:
        lines += [
            "",
            "### Most recent promoted closure — Geppetto",
            "",
            f"- Status: `{geppetto.get('status', 'UNKNOWN')}`",
            f"- Frozen source commit: `{geppetto.get('frozen_source_commit', 'UNKNOWN')}`",
            f"- Seal commit: `{geppetto.get('seal_commit', 'UNKNOWN')}`",
            f"- Mainline home: `{geppetto.get('mainline_home', 'UNKNOWN')}`",
            f"- Closure step / terminal streak: `{geppetto.get('closure_step', '?')}` / `{geppetto.get('terminal_streak_checks', '?')}/48`",
            f"- Qualified controls / deform roots: `{geppetto.get('qualified_control_count', '?')}` / `{geppetto.get('qualified_deform_root_count', '?')}`",
            f"- Checkpoint SHA-256: `{geppetto.get('checkpoint_sha256', 'UNKNOWN')}`",
            f"- QualifiedSkeletonIR SHA-256: `{geppetto.get('qualified_skeleton_ir_sha256', 'UNKNOWN')}`",
            f"- Result SHA-256: `{geppetto.get('result_json_sha256', 'UNKNOWN')}`",
            f"- Generalization claimed: `{str(geppetto.get('generalization_claim', False)).lower()}`",
            f"- Product PASS claimed: `{str(geppetto.get('product_pass_claim', False)).lower()}`",
        ]

    if arachne:
        lines += [
            "",
            "### Current open learned-skinning gate",
            "",
            f"- Scope: `{arachne.get('scope', 'UNKNOWN')}`",
            f"- Research branch: `{arachne.get('branch', 'UNKNOWN')}`",
            f"- Architecture: `{arachne.get('architecture', 'UNKNOWN')}`",
            f"- Parameters: `{arachne.get('parameters', 'UNKNOWN')}`",
            f"- A1 authorized: `{str(arachne.get('a1_authorized', False)).lower()}`",
            f"- A1 rule: {arachne.get('a1_rule', 'UNKNOWN')}",
        ]

    lines += [
        "",
        "## Historical-memory health",
        "",
        f"- **Bootstrap/AOA:** `{bootstrap['status']}`",
    ]

    if coverage:
        lines += [
            f"- **High-signal artifacts:** {coverage.get('high_signal_artifact_count', '?')}",
            f"- **Explained by continuity policy:** {coverage.get('explained_high_signal_count', coverage.get('indexed_high_signal_count', '?'))}",
            f"- **Unexplained:** {coverage.get('unexplained_high_signal_count', coverage.get('unindexed_high_signal_count', '?'))}",
            f"- **Coverage:** {coverage.get('coverage_fraction', 0.0):.1%}",
        ]
    elif catalog:
        lines.append(f"- **Artifact census:** {catalog.get('artifact_count', '?')} discovered; regenerate coverage before claiming closure health.")
    else:
        lines.append("- **Coverage views missing:** run local continuity refresh.")

    if bootstrap["status"] == "BOOTSTRAP_AUDIT_CLOSED":
        lines += [
            "- Closure authority: `canonical/AUDIT_OF_AUDITS_CLOSURE_20260907.md`.",
            "- Residual/disposition policy: `canonical/AOA_ARTIFACT_DISPOSITION_V1.json`.",
            "- Closure means context/provenance coverage, **not** retroactive validation of every historical artifact.",
        ]
    else:
        lines += [
            "- Missing historical detail remains `UNKNOWN / NEEDS AUDIT`; never infer absence.",
            "- Use `canonical/BOOTSTRAP_AUDIT_QUEUE.md` + artifact catalog for unresolved evidence.",
        ]

    lines += [
        "",
        "## Current scientific question",
        "",
        focus["scientific_question"],
        "",
        "Do **not** widen the result beyond exact gate semantics in the experiment ledger/result authority.",
        "",
        "## Pipeline ownership",
        "",
        "| Module | Role | Binding/current rule |",
        "|---|---|---|",
    ]
    for item in context["pipeline"]:
        lines.append(f"| **{item['name']}** | {item['role']} | {item['canonical_rule']} |")

    lines += ["", "## Critical RigAnything memory", ""]
    ra = context["riganything_state"]
    lines += [
        f"- Fuller challenger already exists: **{str(ra['fuller_research_challenger_already_exists']).upper()}**",
        f"- Creation implementation: `{ra['implementation_at_creation_commit']}`",
        f"- Creation commit: `{ra['creation_commit']}`",
        "- Contains:",
    ]
    lines += [f"  - {x}" for x in ra["contains"]]
    lines += [
        f"- **Memory guard:** {ra['critical_memory_rule']}",
        f"- Canonical status: {ra['canonical_status']}",
        "",
        "## FIT science guardrails",
        "",
        f"- {fit['principle']}",
        f"- FIT1 **is:** {fit['fit1_is']}",
        f"- FIT1 **is not:** {fit['fit1_is_not']}",
        f"- FIT-specific optimizer interventions: {fit['fit_specific_optimizer_interventions']}",
        f"- Ladder: {' → '.join(fit['next_ladder'])}",
        "",
        "## Settled invariants",
        "",
    ]
    lines += bullets(context["settled_invariants"])

    lines += ["", "## Context traps", ""]
    for trap in context["known_context_traps"]:
        lines.append(f"- **{trap['trap']}** → {trap['defense']}")

    lines += [
        "",
        "## Execution environment",
        "",
        f"- Actions: **{env['actions']}**",
        f"- Required labels: `{', '.join(env['labels'])}`",
        f"- Known runner: `{env['runner']}`",
        f"- Operator path: `{env['operator_path']}`",
        f"- Why: {env['reason']}",
        "",
        "## Rehydration drill-down",
        "",
        "1. `canonical/FIT1_EVIDENCE_INDEX_20260909.md` — technical/investor proof path and current claim boundary.",
        "2. current scientific journal from `canonical/CONTEXT_STATE_V1.json` — recent causal decision sequence.",
        "3. `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md` — learned/deterministic responsibility envelope.",
        "4. `canonical/FIT1_SCIENTIFIC_LINEAGE_V1.md` — FIT1-to-now scientific flow.",
        "5. `CURRENT_STATE.md` — current stop/go authority.",
        "6. `canonical/FIT1_COMMIT_LINEAGE_V1.md` — exact FIT1-descendant commit discovery.",
        "7. `canonical/CONTEXT_COVERAGE_AUDIT.md` + `canonical/LIVE_AUTHORITY_MAP.md` — coverage and live branch authority.",
        "8. architecture + experiment ledgers/current registry — implementation/test/promotion distinctions.",
        "9. exact prereg/result/source artifacts only as needed.",
        "",
        "## Completion transaction",
        "",
        context["continuation_protocol"]["experiment_completion_definition"],
        "",
        "## Packet validity",
        "",
    ]
    if errors:
        lines.append("**INVALID — context/authority/coverage drift detected.**")
        lines += [f"- {e}" for e in errors]
    else:
        lines.append("**VALID — current state, authority manifest, ownership/FIT1 continuity guards, and live refs agree.**")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    context = load(CONTEXT_PATH)
    authority = load(AUTH_PATH)
    bootstrap = load(BOOTSTRAP_PATH)
    catalog = load_optional(CATALOG_PATH)
    coverage = load_optional(COVERAGE_PATH)
    current_text = CURRENT_PATH.read_text(encoding="utf-8")
    heads = live_heads()
    errors = validate(context, authority, current_text, heads, bootstrap)
    text = render(context, authority, heads, errors, bootstrap, catalog, coverage)

    if args.write:
        OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)

    if errors:
        print("REHYDRATION PACKET INVALID", file=sys.stderr)
        for e in errors:
            print(f"- {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
