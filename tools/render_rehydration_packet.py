#!/usr/bin/env python3
"""Render the compact RealSaS cross-chat/cross-agent rehydration packet.

Source state is canonical/CONTEXT_STATE_V1.json plus the machine authority manifest,
bootstrap/census state, and live remote branch heads. The generated packet is
navigation/cache, not independent scientific authority.
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
CURRENT_PATH = ROOT / "CURRENT_STATE.md"
OUTPUT_PATH = ROOT / "canonical" / "REHYDRATION_PACKET.md"


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return load(path)


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


def validate(context: dict, authority: dict, current_text: str, heads: dict[str, str]) -> list[str]:
    errors: list[str] = []
    focus = context["current_focus"]
    active = authority.get("active_experiments", [])
    matches = [e for e in active if e.get("id") == focus.get("gate")]
    if len(matches) != 1:
        errors.append(f"context current gate must match exactly one active authority experiment: {focus.get('gate')}")
    else:
        exp = matches[0]
        if exp.get("branch") != focus.get("branch"):
            errors.append("context current branch disagrees with authority manifest")
    if focus.get("gate") not in current_text:
        errors.append("CURRENT_STATE.md does not name context current gate")
    if focus.get("branch") not in current_text:
        errors.append("CURRENT_STATE.md does not name context current branch")
    if focus.get("branch") not in heads:
        errors.append("context current branch does not exist on origin")
    canonical = authority["branch_policy"]["canonical_branch"]
    if canonical not in heads:
        errors.append(f"canonical branch missing on origin: {canonical}")
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
    active_head = heads.get(focus["branch"], "MISSING")

    lines: list[str] = [
        "# RealSaS — Rehydration Packet",
        "",
        "> **GENERATED NAVIGATION/CACHE — DO NOT TREAT AS INDEPENDENT SCIENTIFIC AUTHORITY.**  ",
        "> Source state: `canonical/CONTEXT_STATE_V1.json` + `canonical/AUTHORITY_MAP_V1.json` + bootstrap/census state + live remote refs.  ",
        "> If this packet conflicts with `CURRENT_STATE.md`, `CURRENT_STATE.md` wins.",
        "",
        "## 60-second state",
        "",
        f"- **Product:** {product['target']}",
        f"- **Current witness:** `{product['demo_witness']}`",
        f"- **Current module:** `{focus['module']}`",
        f"- **Active gate:** `{focus['gate']}`",
        f"- **Active branch:** `{focus['branch']}` @ `{active_head[:12] if active_head != 'MISSING' else active_head}`",
        f"- **Canonical main:** `{main_head[:12] if main_head != 'MISSING' else main_head}`",
        f"- **Promotion block:** {focus['promotion_block']}",
        f"- **Scope warning:** {focus['scope_warning']}",
        f"- **Next visible product milestone:** {product['next_visible_product_milestone']}",
        "",
        "## Historical-memory health",
        "",
        f"- **Bootstrap:** `{bootstrap['status']}`",
    ]
    if catalog:
        lines += [
            f"- **Artifact census:** {catalog.get('artifact_count', '?')} high-signal artifacts discovered; census coverage **100% by construction**.",
            f"- **Semantically reconciled:** {catalog.get('semantic_indexed_count', '?')}.",
            f"- **Catalogued but unreviewed:** {catalog.get('semantic_unreviewed_count', '?')}.",
            f"- **Semantic coverage:** {catalog.get('semantic_coverage_fraction', 0.0):.1%}.",
        ]
    elif coverage:
        lines += [
            f"- **High-signal artifacts:** {coverage.get('high_signal_artifact_count', '?')}.",
            f"- **Semantically indexed:** {coverage.get('indexed_high_signal_count', '?')}.",
            f"- **Unreviewed:** {coverage.get('unindexed_high_signal_count', '?')}.",
        ]
    else:
        lines.append("- **Coverage views missing:** run the local continuity refresh before making historical completeness claims.")
    if bootstrap["status"] != "BOOTSTRAP_AUDIT_CLOSED":
        lines += [
            "- **Honesty rule:** missing historical details are `UNKNOWN / NEEDS AUDIT`, never inferred absent from the registry.",
            "- Use `canonical/BOOTSTRAP_AUDIT_QUEUE.md` + `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` to locate unreviewed evidence.",
        ]

    lines += [
        "",
        "## What we are testing right now",
        "",
        focus["scientific_question"],
        "",
        "Do **not** widen the result beyond the exact gate semantics in `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md`.",
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
        f"- Clean-room studied: **{str(ra['paper_mechanisms_cleanroom_studied']).upper()}**",
        f"- Fuller challenger already exists: **{str(ra['fuller_research_challenger_already_exists']).upper()}**",
        f"- Implementation: `{ra['implementation']}`",
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

    lines += ["", "## Context traps — check these before claiming something is missing", ""]
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
        "Read only as needed, in this order:",
        "",
        "1. `CURRENT_STATE.md` — stop/go and continuation authority.",
        "2. `canonical/CONTEXT_COVERAGE_AUDIT.md` + `canonical/BOOTSTRAP_AUDIT_QUEUE.md` — know what memory remains unresolved.",
        "3. `canonical/LIVE_AUTHORITY_MAP.md` or run `python3 tools/render_authority_map.py --write` — live branch/active-experiment navigation.",
        "4. `canonical/ARCHITECTURE_AUTHORITY_LEDGER_V1.md` — mechanism implementation vs test vs canonical status.",
        "5. `canonical/EXPERIMENT_AUTHORITY_LEDGER_V1.md` — exact gate meaning and explicit non-claims.",
        "6. `canonical/EXPERIMENT_REGISTRY_V1.json` — experiment structure and scientific-flow dependencies.",
        "7. `canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json` — discover historical evidence not yet semantically reconciled.",
        "8. Only then descend into referenced reports, notebooks, source commits and historical branches.",
        "",
        "## Completion transaction",
        "",
        context["continuation_protocol"]["experiment_completion_definition"],
        "",
        "## Packet validity",
        "",
    ]
    if errors:
        lines.append("**INVALID — context/authority drift detected.**")
        lines += [f"- {e}" for e in errors]
    else:
        lines.append("**VALID — compact context, active experiment manifest, `CURRENT_STATE.md`, and live branch refs agree.**")
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
    errors = validate(context, authority, current_text, heads)
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
