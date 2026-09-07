#!/usr/bin/env python3
"""Build a discovery census of high-signal RealSaS knowledge artifacts.

Census is not semantic authority. It guarantees discoverability inside the explicitly
scanned scope while preserving a separate SEMANTICALLY_INDEXED vs
CATALOGUED_UNREVIEWED distinction.

Scan scope:
- canonical `main` checkout;
- every ACTIVE_EXPERIMENT branch registered in AUTHORITY_MAP_V1.json;
- every explicit EVIDENCE_ONLY branch override in AUTHORITY_MAP_V1.json.

Branch scans are shallow and isolated under refs/context-census/*; they never change
working branch or experiment refs. Branch artifacts are emitted only when their blob
is absent from or differs from the same path on main.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "canonical" / "KNOWLEDGE_ARTIFACT_CATALOG_V1.json"
OUT_MD = ROOT / "canonical" / "BOOTSTRAP_AUDIT_QUEUE.md"
BOOTSTRAP = ROOT / "canonical" / "BOOTSTRAP_COVERAGE_STATE_V1.json"
AUTHORITY = ROOT / "canonical" / "AUTHORITY_MAP_V1.json"

SEMANTIC_SPINE = [
    "CURRENT_STATE.md",
    "README.md",
    "REPOSITORY_MAP.md",
    "SYSTEM_INDEX.md",
    "AGENTS.md",
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
    r"EXPERIMENT|FIT|FAMILY|CAUSAL|CHALLENGER)", re.IGNORECASE
)
WORKFLOW_SIGNAL = re.compile(
    r"(geppetto|iris|arachne|fit|family|audit|closure|experiment|promotion|freeze|restoration)",
    re.IGNORECASE,
)
DATE8 = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
DATE_DASH = re.compile(r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)")

SUBSYSTEM_PATTERNS = [
    ("GEPPETTO", re.compile(r"geppetto|riganything", re.I)),
    ("IRIS", re.compile(r"iris|dino|n1d|depth|reprojection|m4r?|corr|quotient", re.I)),
    ("ARACHNE_SKIN", re.compile(r"arachne|skin.?field|skintokens|weight", re.I)),
    ("COMPILER", re.compile(r"compiler|solver|mwb|repair|proof|substrate|mesh|appearance|motion", re.I)),
    ("RUNTIME_EXPORT", re.compile(r"runtime|export|consumer|deploy|bake", re.I)),
    ("FIT_PRODUCT", re.compile(r"fit|family|mage|product|living_compile", re.I)),
    ("REPOSITORY_GOVERNANCE", re.compile(r"repository|authority|restoration|architecture|completion", re.I)),
]

ROLE_PATTERNS = [
    ("PREREG", re.compile(r"PREREG", re.I)),
    ("RESULT", re.compile(r"RESULT", re.I)),
    ("CLOSURE", re.compile(r"CLOSURE|VERDICT", re.I)),
    ("AUDIT", re.compile(r"AUDIT|REPORT", re.I)),
    ("PROMOTION_RETRACTION", re.compile(r"PROMOTION|RETRACTION", re.I)),
    ("AUTHORITY_DECISION", re.compile(r"AUTHORITY|DECISION|DISPOSITION|FREEZE|SEAL", re.I)),
    ("PLAN_MATRIX", re.compile(r"PLAN|MATRIX|MANIFEST|LEDGER|STATE", re.I)),
]


def run(*args: str) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def tree(ref: str) -> dict[str, str]:
    raw = run("git", "ls-tree", "-r", "--full-tree", ref)
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        _mode, kind, sha = meta.split(" ", 2)
        if kind == "blob":
            out[path] = sha
    return out


def registered_branch_scope(authority: dict) -> list[str]:
    names: set[str] = set()
    for exp in authority.get("active_experiments", []):
        branch = exp.get("branch")
        if branch:
            names.add(branch)
    for item in authority.get("branch_overrides", []):
        if item.get("class") == "EVIDENCE_ONLY" and item.get("branch"):
            names.add(item["branch"])
    names.discard(authority["branch_policy"]["canonical_branch"])
    return sorted(names)


def fetch_branch_for_census(branch: str) -> str:
    digest = hashlib.sha256(branch.encode("utf-8")).hexdigest()[:16]
    local_ref = f"refs/context-census/{digest}"
    run(
        "git", "fetch", "--no-tags", "--depth=1", "origin",
        f"refs/heads/{branch}:{local_ref}",
    )
    return local_ref


def is_high_signal(path: str) -> bool:
    p = Path(path)
    name = p.name
    if path in {"CURRENT_STATE.md", "RESTORATION_STATE.md", "REPOSITORY_MAP.md", "SYSTEM_INDEX.md", "AGENTS.md"}:
        return True
    if path.startswith("prereg/"):
        return True
    if path.startswith("canonical/"):
        if path in {
            "canonical/LIVE_AUTHORITY_MAP.md",
            "canonical/REHYDRATION_PACKET.md",
            "canonical/CONTEXT_COVERAGE_AUDIT.md",
            "canonical/CONTEXT_COVERAGE_AUDIT.json",
            "canonical/KNOWLEDGE_ARTIFACT_CATALOG_V1.json",
            "canonical/BOOTSTRAP_AUDIT_QUEUE.md",
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


def semantic_corpus() -> str:
    chunks = []
    for rel in SEMANTIC_SPINE:
        p = ROOT / rel
        if p.exists():
            try:
                chunks.append(p.read_text(encoding="utf-8"))
            except UnicodeDecodeError:
                pass
    return "\n".join(chunks)


def semantic_status(path: str, source_ref: str, blob_sha: str, corpus: str) -> tuple[str, str]:
    if source_ref == "main" and path in SEMANTIC_SPINE:
        return "SEMANTICALLY_INDEXED", "continuity spine file"

    path_named = path in corpus or (Path(path).name and Path(path).name in corpus)
    if source_ref == "main" and path_named:
        return "SEMANTICALLY_INDEXED", "main artifact referenced by semantic spine"

    # Branch evidence is not considered semantically reconciled merely because the same
    # path name appears in prose. Bind it by branch identity or exact blob SHA.
    if source_ref != "main" and path_named and (source_ref in corpus or blob_sha in corpus):
        return "SEMANTICALLY_INDEXED", "branch artifact explicitly provenance-bound by semantic spine"

    return "CATALOGUED_UNREVIEWED", "present in census; semantic role/provenance not yet reconciled"


def date_from_path(path: str) -> tuple[str | None, str]:
    m = DATE8.search(path)
    if m:
        s = m.group(1)
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}", "filename"
    m = DATE_DASH.search(path)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "filename"
    return None, "unknown"


def role(path: str) -> str:
    name = Path(path).name
    if path.startswith(".github/workflows/"):
        return "WORKFLOW"
    if "/challengers/" in path:
        return "CHALLENGER_SOURCE"
    if Path(path).suffix.lower() == ".ipynb":
        return "NOTEBOOK"
    for label, rx in ROLE_PATTERNS:
        if rx.search(name):
            return label
    return "HIGH_SIGNAL_OTHER"


def subsystems(path: str) -> list[str]:
    tags = [label for label, rx in SUBSYSTEM_PATTERNS if rx.search(path)]
    return tags or ["CROSS_CUTTING_OTHER"]


def stable_id(source_ref: str, path: str, blob_sha: str) -> str:
    raw = f"{source_ref}\0{path}\0{blob_sha}"
    return "KA-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12].upper()


def priority(item: dict) -> tuple:
    p = item["path"].lower()
    if "ar01" in p or "riganything" in p or "geppetto" in p:
        band = 0
    elif item["date"] and item["date"] >= "2026-09-04":
        band = 1
    elif item["date"] and item["date"] >= "2026-09-01":
        band = 2
    else:
        band = 3
    return (band, item["date"] or "9999-99-99", item["source_ref"], item["path"])


def make_item(path: str, blob_sha: str, source_ref: str, source_head: str, corpus: str) -> dict:
    status, why = semantic_status(path, source_ref, blob_sha, corpus)
    d, dsrc = date_from_path(path)
    return {
        "artifact_id": stable_id(source_ref, path, blob_sha),
        "source_ref": source_ref,
        "source_head": source_head,
        "blob_sha": blob_sha,
        "path": path,
        "role": role(path),
        "subsystems": subsystems(path),
        "date": d,
        "date_source": dsrc,
        "semantic_status": status,
        "semantic_reason": why,
    }


def main() -> int:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    corpus = semantic_corpus()

    canonical_branch = authority["branch_policy"]["canonical_branch"]
    main_head = run("git", "rev-parse", "HEAD")
    main_tree = tree("HEAD")

    artifacts: list[dict] = []
    for path, blob in main_tree.items():
        if is_high_signal(path):
            artifacts.append(make_item(path, blob, canonical_branch, main_head, corpus))

    scanned_branches: list[dict] = []
    for branch in registered_branch_scope(authority):
        local_ref = fetch_branch_for_census(branch)
        head = run("git", "rev-parse", local_ref)
        branch_tree = tree(local_ref)
        differing = 0
        high_signal_differing = 0
        for path, blob in branch_tree.items():
            if main_tree.get(path) == blob:
                continue
            differing += 1
            if not is_high_signal(path):
                continue
            high_signal_differing += 1
            artifacts.append(make_item(path, blob, branch, head, corpus))
        scanned_branches.append({
            "branch": branch,
            "head": head,
            "differing_blob_paths_vs_main": differing,
            "high_signal_differing_artifacts": high_signal_differing,
        })

    artifacts.sort(key=priority)
    counts = Counter(x["semantic_status"] for x in artifacts)
    roles = Counter(x["role"] for x in artifacts)
    subsystem_counts = Counter(tag for x in artifacts for tag in x["subsystems"])
    payload = {
        "schema_version": 2,
        "bootstrap_status": bootstrap["status"],
        "authority": "DISCOVERY_CENSUS_ONLY__NOT_SCIENTIFIC_OR_CONTINUATION_AUTHORITY",
        "scope_note": "100% census coverage refers only to main plus explicitly registered ACTIVE_EXPERIMENT/EVIDENCE_ONLY branches. Bootstrap remains incomplete until historical branch disposition/audit closes.",
        "canonical_branch": canonical_branch,
        "canonical_head": main_head,
        "registered_branch_scans": scanned_branches,
        "artifact_count": len(artifacts),
        "census_coverage_fraction_within_declared_scope": 1.0,
        "semantic_indexed_count": counts.get("SEMANTICALLY_INDEXED", 0),
        "semantic_unreviewed_count": counts.get("CATALOGUED_UNREVIEWED", 0),
        "semantic_coverage_fraction": (counts.get("SEMANTICALLY_INDEXED", 0) / len(artifacts)) if artifacts else 1.0,
        "role_counts": dict(sorted(roles.items())),
        "subsystem_counts": dict(sorted(subsystem_counts.items())),
        "artifacts": artifacts,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for item in artifacts:
        if item["semantic_status"] == "SEMANTICALLY_INDEXED":
            continue
        date = item["date"] or "UNKNOWN_DATE"
        primary = item["subsystems"][0]
        by_cluster[f"{date} :: {primary}"].append(item)

    lines = [
        "# RealSaS — Bootstrap Audit Queue",
        "",
        "> **GENERATED DISCOVERY VIEW — NOT SCIENTIFIC AUTHORITY.**",
        f"> Bootstrap: `{bootstrap['status']}`",
        "",
        f"- Declared census scope: `{canonical_branch}` + {len(scanned_branches)} registered active/evidence branch(es)",
        f"- Census artifacts in declared scope: **{len(artifacts)} / {len(artifacts)} discovered (100%)**",
        f"- Semantically reconciled: **{counts.get('SEMANTICALLY_INDEXED', 0)}**",
        f"- Catalogued but unreviewed: **{counts.get('CATALOGUED_UNREVIEWED', 0)}**",
        f"- Semantic coverage: **{payload['semantic_coverage_fraction']:.1%}**",
        "",
        "**Important:** 100% is discovery coverage only inside the declared scan scope. It is not a claim that all historical branches have been audited or that the repository's scientific history is semantically complete.",
        "",
        "## Registered branch scan scope",
        "",
    ]
    if not scanned_branches:
        lines.append("_No registered non-main branches scanned._")
    else:
        lines += ["| Branch | Head | Different blobs vs main | High-signal differing artifacts |", "|---|---|---:|---:|"]
        for b in scanned_branches:
            lines.append(
                f"| `{b['branch']}` | `{b['head'][:12]}` | {b['differing_blob_paths_vs_main']} | {b['high_signal_differing_artifacts']} |"
            )

    lines += [
        "",
        "The queue is a discovery aid. A path being listed does not establish what it proves, whether it is current, or whether it was ever executed.",
        "",
        "## Backfill clusters",
        "",
    ]
    for key in sorted(by_cluster):
        items = by_cluster[key]
        lines.append(f"### {key} ({len(items)})")
        lines.append("")
        for item in items:
            src = "main" if item["source_ref"] == canonical_branch else item["source_ref"]
            lines.append(
                f"- `{item['artifact_id']}` `{item['role']}` — `{src}` :: `{item['path']}` @ blob `{item['blob_sha'][:12]}`"
            )
        lines.append("")

    lines += [
        "## Completion rule",
        "",
        "A cluster closes only after artifacts are dispositioned from source evidence into the experiment/architecture authority system: purpose, arms/mechanisms, result, falsification boundary, current effect, provenance, and supersession where applicable.",
        "",
        "Bootstrap cannot close until unregistered historical branches have also received explicit provenance disposition; branch-aware registered scanning prevents known evidence branches from becoming invisible in the meantime.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "scope_branches": 1 + len(scanned_branches),
        "census": len(artifacts),
        "semantic_indexed": counts.get("SEMANTICALLY_INDEXED", 0),
        "semantic_unreviewed": counts.get("CATALOGUED_UNREVIEWED", 0),
        "semantic_coverage": payload["semantic_coverage_fraction"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
