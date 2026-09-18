#!/usr/bin/env python3
"""Render the current RealSaS cross-chat/cross-agent rehydration packet.

Generated navigation only. Current continuation authority is CURRENT_STATE.md,
AUTHORITY_MAP_V1.json, ACTIVE_RUN_V1.json and the exact pipeline plan.
Historical FIT/restoration artifacts remain evidence, never current pointers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compiler.realsas_compiler_services.orchestrator.mainline import validate_ledger

ROOT = Path(__file__).resolve().parents[1]
CONTEXT_PATH = ROOT / "canonical" / "CONTEXT_STATE_V2.json"
AUTH_PATH = ROOT / "canonical" / "AUTHORITY_MAP_V1.json"
PLAN_PATH = ROOT / "canonical" / "MAINLINE_EXECUTION_PLAN_V1.json"
RUN_PATH = ROOT / "canonical" / "ACTIVE_RUN_V1.json"
CURRENT_PATH = ROOT / "CURRENT_STATE.md"
OUTPUT_PATH = ROOT / "canonical" / "REHYDRATION_PACKET.md"


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def live_heads() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in run("git", "ls-remote", "--heads", "origin").splitlines():
        if not line.strip():
            continue
        sha, ref = line.split("\t", 1)
        prefix = "refs/heads/"
        if ref.startswith(prefix):
            out[ref[len(prefix):]] = sha
    return out


def validate(context: dict, authority: dict, plan: dict, ledger: dict, current_text: str, heads: dict[str, str]) -> list[str]:
    errors: list[str] = []
    try:
        validate_ledger(plan, ledger)
    except Exception as exc:
        errors.append(f"active run / plan validation failed: {type(exc).__name__}: {exc}")

    canonical = authority["branch_policy"]["canonical_branch"]
    if canonical != "main" or canonical not in heads:
        errors.append(f"canonical main missing/drifted: {canonical}")

    focus = context.get("current_focus", {})
    active_id = focus.get("active_experiment")
    active = [x for x in authority.get("active_experiments", []) if x.get("id") == active_id]
    if len(active) != 1:
        errors.append(f"context active experiment does not bind exactly one authority record: {active_id}")
    elif active[0].get("branch") != canonical:
        errors.append("active experiment must execute on canonical main")

    for token in (
        str(active_id or ""),
        str(focus.get("state") or ""),
        "QualifiedMeshIR",
        "QualifiedPresentationGraphIR",
    ):
        if token and token not in current_text:
            errors.append(f"CURRENT_STATE.md missing current token: {token}")

    for rel in authority.get("required_files", []):
        if not (ROOT / rel).exists():
            errors.append(f"required authority file missing: {rel}")

    if authority.get("context_state") != "canonical/CONTEXT_STATE_V2.json":
        errors.append("authority map context pointer is not CONTEXT_STATE_V2")
    if authority.get("experiment_registry") != "canonical/EXPERIMENT_REGISTRY_V3.json":
        errors.append("authority map current registry pointer is not V3")
    return errors


def render(context: dict, authority: dict, plan: dict, ledger: dict, heads: dict[str, str], errors: list[str]) -> str:
    product = context["product"]
    focus = context["current_focus"]
    env = context["execution_environment"]
    auth = context["authority"]
    canonical = authority["branch_policy"]["canonical_branch"]
    main_head = heads.get(canonical, "MISSING")

    lines = [
        "# RealSaS — Rehydration Packet",
        "",
        "> GENERATED NAVIGATION/CACHE — NOT INDEPENDENT SCIENTIFIC AUTHORITY.",
        "> Current continuation is defined by the active-run ledger, exact plan hash, authority map and CURRENT_STATE.md.",
        "",
        "## 60-second state",
        "",
        f"- Product: {product['target']}",
        f"- Current witness: {product['demo_witness']}",
        f"- Current module: {focus['module']}",
        f"- Current state: {focus['state']}",
        f"- Active experiment: {focus['active_experiment']}",
        f"- Most recent closed gate: {focus['most_recent_closed_gate']}",
        f"- Canonical main: {main_head[:12] if main_head != 'MISSING' else main_head}",
        f"- Active run: {ledger['run_id']} — {ledger['completed_count']}/{ledger['total_count']}; next {ledger.get('next_stage') or 'NONE'}",
        f"- Plan SHA-256: {ledger['pipeline_plan_sha256']}",
        f"- Promotion block: {focus['promotion_block']}",
        f"- Scope warning: {focus['scope_warning']}",
        "",
        "## Current machine authority",
        "",
        "- Context: canonical/CONTEXT_STATE_V2.json",
        f"- Experiment registry: {auth['experiment_registry']}",
        f"- Scientific journal: {auth['scientific_journal']}",
        f"- Active run: {auth['active_run_ledger']}",
        f"- Pipeline plan: {authority['pipeline_plan']}",
        "- Product geometry/presentation contract: canonical/QUALIFIED_MESH_PRESENTATION_AUTHORITY_V1_20260918.md",
        f"- Historical detail registry: {auth['historical_experiment_registry']}",
        f"- Historical scientific journal: {auth['historical_scientific_journal']}",
        "",
        "## Frozen product authority",
        "",
        "- RiggingSurfaceIR S is immutable admitted mechanical/evidence substrate.",
        "- MechanicalPartitionIR declares structural membership plus SEPARATE / PRESERVE_CONTINUITY / UNKNOWN boundaries without mutating S.",
        "- QualifiedMeshIR M is view-independent and is the single product geometry authority after independent G1–G5 PASS.",
        "- DeformationCapabilityEnvelopeIR binds mesh conditioning, consequential-UNKNOWN and motion-admissibility policy.",
        "- QualifiedPresentationGraphIR owns slots, attachments, carrier classes, composition evidence and 8-view overlays.",
        "- Runtime/export consumes these exact authorities and may not create a second topology or presentation truth.",
        "",
        "## Pipeline",
        "",
        "| Layer | Role | Current rule |",
        "|---|---|---|",
    ]
    for item in context["pipeline"]:
        lines.append(f"| **{item['name']}** | {item['role']} | {item['canonical_rule']} |")

    lines += [
        "",
        "## Current scientific question",
        "",
        focus["scientific_question"],
        "",
        "## Settled invariants",
        "",
    ]
    lines += [f"- {x}" for x in context["settled_invariants"]]
    lines += [
        "",
        "## Execution environment",
        "",
        f"- Actions: {env['actions']}",
        f"- Runner labels: {', '.join(env['labels'])}",
        f"- Known runner: {env['runner']}",
        f"- Authority root: {env['operator_path']}",
        "",
        "## Resume",
        "",
        "Read in this order:",
    ]
    for i, rel in enumerate(authority.get("rehydration_order", []), 1):
        lines.append(f"{i}. {rel}")
    lines += [
        "",
        f"Resume rule: {context['continuation_protocol']['experiment_completion_definition']}",
        "",
        "## Packet validity",
        "",
    ]
    if errors:
        lines.append("INVALID — current authority drift detected.")
        lines += [f"- {x}" for x in errors]
    else:
        lines.append("VALID — current V2/V3 context, active run, plan hash, product authority and live main agree.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    context = load(CONTEXT_PATH)
    authority = load(AUTH_PATH)
    plan = load(PLAN_PATH)
    ledger = load(RUN_PATH)
    current_text = CURRENT_PATH.read_text(encoding="utf-8")
    heads = live_heads()
    errors = validate(context, authority, plan, ledger, current_text, heads)
    text = render(context, authority, plan, ledger, heads, errors)

    if args.write:
        OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)

    if errors:
        print("REHYDRATION PACKET INVALID", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
