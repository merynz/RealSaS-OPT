"""Reproducible inventory of Git-tracked source, docs and workflow LOC.

Physical LOC counts every line, including comments and blanks. Nonblank LOC
excludes only empty/whitespace lines. Neither is a language-aware SLOC metric.
Generated files, binaries, archives and Jupyter JSON are not code LOC.
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi", ".go", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".rs", ".cs",
    ".sh", ".bash", ".zsh", ".sql", ".html", ".css", ".scss", ".java",
    ".kt", ".swift", ".rb", ".php", ".lua", ".glsl", ".wgsl",
})
MARKDOWN_SUFFIXES = frozenset({".md", ".mdx", ".markdown"})
WORKFLOW_SUFFIXES = frozenset({".yml", ".yaml"})


def _lines(data: bytes) -> tuple[int, int]:
    if b"\x00" in data:
        raise ValueError("binary input")
    decoded = data.decode("utf-8")
    rows = decoded.splitlines()
    return len(rows), sum(bool(row.strip()) for row in rows)


def inventory(root: Path) -> dict:
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], check=True,
        capture_output=True,
    ).stdout.split(b"\x00")
    result = {
        "schema": "RealSaS.RepositoryCensus.v1",
        "line_metric": "UTF8_PHYSICAL_LINES_IN_GIT_TRACKED_FILES",
        "tracked_files": 0,
        "tracked_bytes": 0,
        "source_files": 0,
        "source_physical_loc": 0,
        "source_nonblank_loc": 0,
        "markdown_files": 0,
        "markdown_physical_lines": 0,
        "workflow_files": 0,
        "workflow_physical_lines": 0,
        "source_by_root": {},
        "source_by_extension": {},
        "markdown_by_root": {},
        "unreadable_tracked_paths": [],
    }
    by_root = defaultdict(lambda: {"files": 0, "physical_loc": 0, "nonblank_loc": 0})
    by_ext = defaultdict(lambda: {"files": 0, "physical_loc": 0, "nonblank_loc": 0})
    md_roots = defaultdict(int)
    for encoded in tracked:
        if not encoded:
            continue
        rel = encoded.decode("utf-8", errors="surrogateescape")
        path = root / rel
        if not path.is_file() or path.is_symlink():
            result["unreadable_tracked_paths"].append(rel)
            continue
        payload = path.read_bytes()
        result["tracked_files"] += 1
        result["tracked_bytes"] += len(payload)
        suffix = path.suffix.lower()
        source = suffix in SOURCE_SUFFIXES
        markdown = suffix in MARKDOWN_SUFFIXES
        workflow = (rel.startswith(".github/workflows/")
                    and suffix in WORKFLOW_SUFFIXES)
        if not (source or markdown or workflow):
            continue
        try:
            physical, nonblank = _lines(payload)
        except (UnicodeDecodeError, ValueError):
            result["unreadable_tracked_paths"].append(rel)
            continue
        root_name = rel.split("/", 1)[0]
        if source:
            result["source_files"] += 1
            result["source_physical_loc"] += physical
            result["source_nonblank_loc"] += nonblank
            for group in (by_root[root_name], by_ext[suffix]):
                group["files"] += 1
                group["physical_loc"] += physical
                group["nonblank_loc"] += nonblank
        if markdown:
            result["markdown_files"] += 1
            result["markdown_physical_lines"] += physical
            md_roots[root_name] += 1
        if workflow:
            result["workflow_files"] += 1
            result["workflow_physical_lines"] += physical
    result["source_by_root"] = dict(sorted(by_root.items()))
    result["source_by_extension"] = dict(sorted(by_ext.items()))
    result["markdown_by_root"] = dict(sorted(md_roots.items()))
    return result


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    result = inventory(root)
    print("RSS_VNEXT_REPOSITORY_CENSUS=" + json.dumps(
        result, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ))


if __name__ == "__main__":
    main()
