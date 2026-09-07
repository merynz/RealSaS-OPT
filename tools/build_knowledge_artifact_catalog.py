#!/usr/bin/env python3
"""Build a complete census of high-signal RealSaS knowledge artifacts.

Census is not semantic authority. It guarantees discoverability of repository knowledge
while preserving a separate SEMANTICALLY_INDEXED vs CATALOGUED_UNREVIEWED distinction.
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

SEMANTIC_SPINE = [
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


def tracked_files() -> list[str]:
    return [x for x in run("git", "ls-files").splitlines() if x.strip()]


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


def semantic_status(path: str, corpus: str) -> tuple[str, str]:
    if path in SEMANTIC_SPINE:
        return "SEMANTICALLY_INDEXED", "continuity spine file"
    if path in corpus:
        return "SEMANTICALLY_INDEXED", "exact path referenced by semantic spine"
    base = Path(path).name
    if base and base in corpus:
        return "SEMANTICALLY_INDEXED", "basename referenced by semantic spine"
    return "CATALOGUED_UNREVIEWED", "present in census; semantic role not yet reconciled"


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


def stable_id(path: str) -> str:
    return "KA-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:12].upper()


def priority(item: dict) -> tuple:
    # Discovery priority only; never scientific importance authority.
    p = item["path"].lower()
    if "ar01" in p or "riganything" in p or "geppetto" in p:
        band = 0
    elif item["date"] and item["date"] >= "2026-09-04":
        band = 1
    elif item["date"] and item["date"] >= "2026-09-01":
        band = 2
    else:
        band = 3
    return (band, item["date"] or "9999-99-99", item["path"])


def main() -> int:
    bootstrap = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    corpus = semantic_corpus()
    artifacts = []
    for path in tracked_files():
        if not is_high_signal(path):
            continue
        status, why = semantic_status(path, corpus)
        d, dsrc = date_from_path(path)
        artifacts.append({
            "artifact_id": stable_id(path),
            "path": path,
            "role": role(path),
            "subsystems": subsystems(path),
            "date": d,
            "date_source": dsrc,
            "semantic_status": status,
            "semantic_reason": why,
        })
    artifacts.sort(key=priority)

    counts = Counter(x["semantic_status"] for x in artifacts)
    roles = Counter(x["role"] for x in artifacts)
    subsystem_counts = Counter(tag for x in artifacts for tag in x["subsystems"])
    payload = {
        "schema_version": 1,
        "bootstrap_status": bootstrap["status"],
        "authority": "DISCOVERY_CENSUS_ONLY__NOT_SCIENTIFIC_OR_CONTINUATION_AUTHORITY",
        "artifact_count": len(artifacts),
        "census_coverage_fraction": 1.0,
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
        f"- Census artifacts: **{len(artifacts)} / {len(artifacts)} discovered (100%)**",
        f"- Semantically reconciled: **{counts.get('SEMANTICALLY_INDEXED', 0)}**",
        f"- Catalogued but unreviewed: **{counts.get('CATALOGUED_UNREVIEWED', 0)}**",
        f"- Semantic coverage: **{payload['semantic_coverage_fraction']:.1%}**",
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
            lines.append(f"- `{item['artifact_id']}` `{item['role']}` — `{item['path']}`")
        lines.append("")

    lines += [
        "## Completion rule",
        "",
        "A cluster closes only after artifacts are dispositioned from source evidence into the experiment/architecture authority system: purpose, arms/mechanisms, result, falsification boundary, current effect, provenance, and supersession where applicable.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "census": len(artifacts),
        "semantic_indexed": counts.get("SEMANTICALLY_INDEXED", 0),
        "semantic_unreviewed": counts.get("CATALOGUED_UNREVIEWED", 0),
        "semantic_coverage": payload["semantic_coverage_fraction"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
