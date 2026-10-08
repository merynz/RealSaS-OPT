#!/usr/bin/env python3
"""Catalog dated narrative records without changing sealed bytes or paths."""

from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ENTRY_FILES = (
    "README.md", "CURRENT_STATE.md", "AGENTS.md", "SYSTEM_INDEX.md",
    "canonical/README.md",
)


def render() -> str:
    entries = "\n".join((ROOT / p).read_text() for p in ENTRY_FILES)
    grouped = defaultdict(list)
    for path in sorted((ROOT / "canonical").glob("*.md")):
        match = re.search(r"20\d{6}", path.name)
        if match and path.name not in entries:
            grouped[match.group()[:6]].append(path)
    count = sum(map(len, grouped.values()))
    lines = [
        "# Historical Record Index", "",
        "Current work starts at [CURRENT_STATE.md](../CURRENT_STATE.md) and",
        "[Execution lanes](../docs/platform/EXECUTION_LANES.md). This index is",
        "navigation only; it does not grant execution or scientific authority.", "",
        "## Repository snapshots", "",
        "- [Restoration state, 2026-09-04](repository/RESTORATION_STATE_20260904.md)",
        "- [Pre-platform repository map, 2026-10-01](repository/REPOSITORY_MAP_20261001_PRE_PLATFORM.md)", "",
        "## Dated narrative collection", "",
        f"{count} dated canonical narratives are outside the current entry documents.",
        "Their original paths and bytes are retained because workflows, hashes and",
        "scientific lineage indexes refer to them. Some retain narrow contractual",
        "scope: consult current authority before interpreting an older result or",
        "stop/go statement. Date and preservation alone do not establish authority.", "",
        "Rebuild this catalog with `python tools/build_documentation_index.py`.", "",
    ]
    for month, paths in sorted(grouped.items(), reverse=True):
        lines.extend([f"### {month[:4]}-{month[4:]}", ""])
        for path in paths:
            lines.append(f"- [{path.stem}](../canonical/{path.name})")
        lines.append("")
    lines.extend([
        "## Research and unpromoted material", "",
        "- [Experiments](../experiments/README.md): bounded apparatus and research records.",
        "- `prospective/`: unpromoted hypotheses and plans.",
        "- `restoration/`: restoration-era provenance and source audits.", "",
        "The external v0.5 archive remains byte authority for its own historical",
        "source. Only explicitly promoted, regression-gated mechanisms belong in",
        "current executable homes.", "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    destination = ROOT / "historical" / "README.md"
    destination.write_text(render())
    print(destination.relative_to(ROOT))
