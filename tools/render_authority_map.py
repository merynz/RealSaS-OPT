#!/usr/bin/env python3
"""Render the RealSaS live authority map from repository truth.

The machine-readable policy lives in canonical/AUTHORITY_MAP_V1.json.
Repository branch existence/head/date are observed live from git refs.
The generated canonical/LIVE_AUTHORITY_MAP.md is a view, never the source of truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "canonical" / "AUTHORITY_MAP_V1.json"


@dataclass(frozen=True)
class Branch:
    name: str
    sha: str
    committed: str


def run(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def load_manifest() -> dict:
    with MANIFEST.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_refs() -> None:
    # Fail closed when origin cannot be refreshed: a "live" map must not silently use stale refs.
    run(
        "git",
        "fetch",
        "--prune",
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
    )


def branches() -> List[Branch]:
    raw = run(
        "git",
        "for-each-ref",
        "--format=%(refname:short)|%(objectname)|%(committerdate:iso8601-strict)",
        "refs/remotes/origin",
    )
    out: List[Branch] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        ref, sha, committed = line.split("|", 2)
        if ref == "origin/HEAD":
            continue
        name = ref.removeprefix("origin/")
        out.append(Branch(name=name, sha=sha, committed=committed))
    return sorted(out, key=lambda b: b.name)


def classify(branch: Branch, manifest: dict) -> Tuple[str, str]:
    policy = manifest["branch_policy"]
    if branch.name == policy["canonical_branch"]:
        return "CANONICAL", "canonical continuation branch"

    for exp in manifest.get("active_experiments", []):
        if branch.name == exp["branch"]:
            return "ACTIVE_EXPERIMENT", exp["id"]

    for item in manifest.get("branch_overrides", []):
        if branch.name == item["branch"]:
            return item["class"], item.get("reason", "explicit manifest override")

    for pattern in policy.get("delete_candidate_regex", []):
        if re.fullmatch(pattern, branch.name):
            return "DELETE_CANDIDATE", f"matches /{pattern}/"

    return policy["default_non_main_class"], "observed live; not explicitly registered active"


def validate(manifest: dict, observed: List[Branch]) -> List[str]:
    errors: List[str] = []
    names = {b.name for b in observed}

    for rel in manifest.get("required_files", []):
        if not (ROOT / rel).exists():
            errors.append(f"required file missing: {rel}")

    canonical = manifest["branch_policy"]["canonical_branch"]
    if canonical not in names:
        errors.append(f"canonical branch missing from origin: {canonical}")

    current_state_path = ROOT / manifest["continuation_authority"]
    current_state = (
        current_state_path.read_text(encoding="utf-8") if current_state_path.exists() else ""
    )

    seen_active: set[str] = set()
    for exp in manifest.get("active_experiments", []):
        exp_id = exp["id"]
        branch = exp["branch"]
        if exp_id in seen_active:
            errors.append(f"duplicate active experiment id: {exp_id}")
        seen_active.add(exp_id)
        if branch not in names:
            errors.append(f"active experiment branch missing: {exp_id} -> {branch}")
        if exp_id not in current_state:
            errors.append(f"CURRENT_STATE.md does not name active gate id: {exp_id}")
        if branch not in current_state:
            errors.append(f"CURRENT_STATE.md does not name active experiment branch: {branch}")

    override_names: set[str] = set()
    for item in manifest.get("branch_overrides", []):
        name = item["branch"]
        if name in override_names:
            errors.append(f"duplicate branch override: {name}")
        override_names.add(name)

    return errors


def esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def fingerprint(manifest: dict, observed: Iterable[Branch]) -> str:
    payload = {
        "manifest": manifest,
        "branches": [
            {"name": b.name, "sha": b.sha, "committed": b.committed} for b in observed
        ],
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def render(manifest: dict, observed: List[Branch], errors: List[str]) -> str:
    fp = fingerprint(manifest, observed)
    rows: Dict[str, List[Tuple[Branch, str]]] = {}
    for b in observed:
        cls, reason = classify(b, manifest)
        rows.setdefault(cls, []).append((b, reason))

    order = [
        "CANONICAL",
        "ACTIVE_EXPERIMENT",
        "EVIDENCE_ONLY",
        "EVIDENCE_ONLY_UNREGISTERED",
        "DELETE_CANDIDATE",
    ]

    lines: List[str] = []
    lines += [
        "# RealSaS — Live Authority Map",
        "",
        "> **GENERATED FILE — DO NOT HAND EDIT.**  ",
        "> Policy/source of truth: `canonical/AUTHORITY_MAP_V1.json`. Repository facts are read live from `origin/*` refs.  ",
        f"> State fingerprint: `{fp}`",
        "",
        f"**Continuation authority:** `{manifest['continuation_authority']}` on `{manifest['branch_policy']['canonical_branch']}`.  ",
        "A recent branch, green Action, notebook, report, or source file is **not** continuation authority unless the manifest + `CURRENT_STATE.md` explicitly say so.",
        "",
        "## Rehydration order",
        "",
    ]
    for idx, path in enumerate(manifest.get("rehydration_order", []), start=1):
        lines.append(f"{idx}. `{path}`")

    lines += ["", "## Live experiment register", ""]
    active = manifest.get("active_experiments", [])
    if not active:
        lines.append("_No active experiments registered._")
    else:
        lines += [
            "| Gate | Status | Branch | Live head | Question | Does not prove |",
            "|---|---|---|---|---|---|",
        ]
        by_name = {b.name: b for b in observed}
        for exp in active:
            b = by_name.get(exp["branch"])
            live_head = f"`{b.sha[:12]}`" if b else "**MISSING**"
            not_prove = "; ".join(exp.get("does_not_prove", []))
            lines.append(
                "| `{}` | `{}` | `{}` | {} | {} | {} |".format(
                    esc(exp["id"]),
                    esc(exp["status"]),
                    esc(exp["branch"]),
                    live_head,
                    esc(exp["question"]),
                    esc(not_prove),
                )
            )

    lines += ["", "## Branch inventory — observed live", ""]
    counts = {cls: len(rows.get(cls, [])) for cls in order}
    lines.append(
        " / ".join(f"**{cls}: {counts[cls]}**" for cls in order if counts[cls])
        or "No branches observed."
    )
    lines.append("")

    for cls in order:
        items = rows.get(cls, [])
        if not items:
            continue
        lines += [f"### {cls}", "", "| Branch | Head | Last commit | Classification reason |", "|---|---|---|---|"]
        for b, reason in items:
            lines.append(
                f"| `{esc(b.name)}` | `{b.sha[:12]}` | `{esc(b.committed)}` | {esc(reason)} |"
            )
        lines.append("")

    lines += ["## Drift / validity", ""]
    if errors:
        lines.append("**INVALID — live authority drift detected.**")
        lines.append("")
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("**VALID — manifest, active branches, required authority files, and `CURRENT_STATE.md` references agree.**")

    lines += ["", "## Binding anti-conflation rules", ""]
    for invariant in manifest.get("invariants", []):
        lines.append(f"- {invariant}")

    lines += [
        "",
        "## Update semantics",
        "",
        "- Branch heads/dates/classification are regenerated from live git refs; do not manually copy branch SHAs into authority prose.",
        "- Creating a new non-main branch is safe by default: it appears as `EVIDENCE_ONLY_UNREGISTERED` until explicitly registered.",
        "- Starting an experiment requires adding it to `active_experiments` **and** naming its gate + branch in `CURRENT_STATE.md`; otherwise validation fails.",
        "- Closing/promoting an experiment requires one atomic reconciliation of the machine manifest, human ledgers, result/provenance, source/tests, and `CURRENT_STATE.md`.",
        "- `DELETE_CANDIDATE` is advisory only. No branch is deleted automatically.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write generated live map")
    parser.add_argument("--no-fetch", action="store_true", help="use current remote refs")
    args = parser.parse_args()

    manifest = load_manifest()
    if not args.no_fetch:
        fetch_refs()
    observed = branches()
    errors = validate(manifest, observed)
    output = render(manifest, observed, errors)
    target = ROOT / manifest["generated_live_map"]

    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output + "\n", encoding="utf-8", newline="\n")
    else:
        print(output)

    if errors:
        print("\nAUTHORITY MAP INVALID:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
