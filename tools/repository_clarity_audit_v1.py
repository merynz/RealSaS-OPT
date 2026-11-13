"""Read-only, source-derived RSS.vNext repository and CI classification.

No file is marked safe to move/delete by this tool: sealed artifact paths,
dynamic imports, workflow dispatch callers and external links need separate
consumer audits. The report describes repository shape, not product PASS.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT_DISPOSITIONS = {
    ".github": "CI_AND_REPOSITORY_ENFORCEMENT",
    "assets": "SHARED_BINARY_ASSETS",
    "canonical": "MIXED_HISTORIC_EVIDENCE_AND_PATH_BOUND_MACHINE_POLICIES",
    "compiler": "PRODUCT_COMPILER_AND_COMPATIBILITY",
    "docs": "HUMAN_DOCS_AND_HISTORICAL_REFERENCE",
    "experiments": "RESEARCH_APPARATUS",
    "historical": "HISTORICAL_PROVENANCE",
    "models": "PRODUCT_EXPERTS_AND_LEGACY_COMPATIBILITY",
    "notebooks": "RESEARCH_COMPUTE_NOTEBOOKS",
    "platform": "ACTIVE_PLATFORM_AND_COMPATIBILITY",
    "prereg": "SCIENTIFIC_PREREGISTRATION",
    "product": "ACTIVE_BUSTED_PRODUCT_UI_AND_GATEWAY",
    "prospective": "UNPROMOTED_RESEARCH_PROPOSALS",
    "requirements": "DEPENDENCY_LOCKS_AND_REQUIREMENTS",
    "research": "SCIENTIFIC_REFERENCE",
    "restoration": "RESTORATION_PROVENANCE",
    "runtime": "PRODUCT_RUNTIME_AND_COMPATIBILITY",
    "tests": "EXECUTABLE_REGRESSIONS_ALL_ERAS",
    "tools": "OPERATORS_ALL_ERAS",
}
LEGACY_MODEL_PREFIXES = ("models.arachne", "models.geppetto", "models.skin_field_codec")
PRODUCT_READER_ROOTS = (
    "models/rss_vnext/",
    "models/iris/vnext/",
    "models/tessa/",
    "models/axis/",
    "models/mira/",
    "compiler/realsas_compiler_services/platform_worker/",
)
WORKFLOW_EVENTS = frozenset({
    "push", "pull_request", "pull_request_target", "schedule",
    "workflow_dispatch", "workflow_call", "repository_dispatch", "release",
    "issues", "issue_comment", "workflow_run", "create", "delete",
    "merge_group", "check_suite", "check_run", "status",
})
TOP_LEVEL_ON = re.compile(r"^(?:on|['\"]on['\"]):(?:\s*(.*))?$")
EVENT_KEY = re.compile(r"^\s+([a-z_]+)\s*:")


def tracked_paths(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
        check=True,
    ).stdout
    return sorted(
        raw.decode("utf-8", "surrogateescape") for raw in out.split(b"\0") if raw
    )


def workflow_events(source: str) -> list[str]:
    """Conservative static event inventory; unknown forms remain unclassified."""
    rows = source.splitlines()
    events: set[str] = set()
    for i, raw in enumerate(rows):
        m = TOP_LEVEL_ON.fullmatch(raw.strip()) if raw and not raw[0].isspace() else None
        if m is None:
            continue
        inline = (m.group(1) or "").strip()
        if inline:
            events.update(event for event in WORKFLOW_EVENTS
                          if re.search(r"(?<![\w-])" + event + r"(?![\w-])", inline))
        for row in rows[i + 1:]:
            if row.strip() and not row[0].isspace() and not row.lstrip().startswith("#"):
                break
            match = EVENT_KEY.match(row)
            if match and match.group(1) in WORKFLOW_EVENTS:
                events.add(match.group(1))
        break
    return sorted(events)


def static_model_imports(source: str) -> set[str]:
    """Direct AST imports only; not transitive or dynamic-import assurance."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set()
    discovered = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            discovered.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            discovered.add(node.module)
    return {mod for mod in discovered if any(
        mod == p or mod.startswith(p + ".") for p in LEGACY_MODEL_PREFIXES
    )}


def census(root: Path) -> dict:
    files = tracked_paths(root)
    file_set = set(files)
    specs_path = root / "platform/internal/capability/contracts/product_capabilities_v1.json"
    ci_path = root / "canonical/CI_EXECUTION_AUTHORITY_V1.json"
    catalog = json.loads(specs_path.read_text(encoding="utf-8"))
    policy = json.loads(ci_path.read_text(encoding="utf-8"))
    budgets = policy["pr_policy"]
    fast = set(budgets["fast_on_synchronize"])
    full = set(budgets["full_on_open_reopen_ready_or_explicit_dispatch"])
    named = fast | full
    root_counts = Counter()
    legacy_imports: list[dict] = []
    workflow_class = Counter()
    events_count = Counter()
    workflows_unparsed: list[str] = []
    workflows_automatic_outside_named: list[str] = []
    workflows_dispatch_outside_named: list[str] = []
    workflow_files = []
    for rel in files:
        root_counts[rel.split("/", 1)[0]] += 1
        path = root / rel
        if not path.is_file():
            continue
        if rel.startswith(PRODUCT_READER_ROOTS) and rel.endswith(".py"):
            imports = static_model_imports(path.read_text(encoding="utf-8", errors="replace"))
            if imports:
                legacy_imports.append({"consumer": rel, "imports": sorted(imports)})
        if rel.startswith(".github/workflows/") and rel.endswith((".yml", ".yaml")):
            workflow_files.append(rel)
            name = path.name
            group = "RSS_VNEXT_FAST" if name in fast else (
                "RSS_VNEXT_FULL" if name in full else "OTHER_UNREVIEWED"
            )
            workflow_class[group] += 1
            extracted = workflow_events(path.read_text(encoding="utf-8", errors="replace"))
            if not extracted:
                workflows_unparsed.append(rel)
            events_count.update(extracted)
            if group == "OTHER_UNREVIEWED":
                if any(x in extracted for x in ("push", "pull_request", "pull_request_target",
                                                  "schedule", "workflow_run", "merge_group")):
                    workflows_automatic_outside_named.append(rel)
                if "workflow_dispatch" in extracted:
                    workflows_dispatch_outside_named.append(rel)
    roles = {
        role: {
            "entry_module": data["entry_module"],
            "dependencies": data.get("policy", {}).get("dependencies", []),
            "model_requirements": data.get("metadata", {}).get("required_models", []),
            "product_pass_authority": data.get("policy", {}).get("product_pass_authority", False),
        }
        for role, data in sorted(catalog["capabilities"].items())
    }
    missing_named_workflows = sorted(
        name for name in named if ".github/workflows/" + name not in file_set
    )
    actual_directories = {
        rel.split("/", 1)[0] for rel in files if "/" in rel
    }
    return {
        "schema": "RealSaS.RepositoryClarityAudit.v1",
        "scope": "TRACKED_CHECKOUT_READ_ONLY",
        "tracked_files": len(files),
        "root_file_counts": dict(sorted(root_counts.items())),
        "directory_dispositions": {
            name: ROOT_DISPOSITIONS.get(name, "UNCLASSIFIED_NEEDS_OWNER")
            for name in sorted(actual_directories)
        },
        "unclassified_directories": sorted(actual_directories - ROOT_DISPOSITIONS.keys()),
        "active_catalog_roles": roles,
        "active_product_role_count": len(roles),
        "workflows": {
            "yaml_count": len(workflow_files),
            "named_fast_count": len(fast),
            "named_full_count": len(full),
            "actual_by_class": dict(sorted(workflow_class.items())),
            "missing_named_workflows": missing_named_workflows,
            "event_occurrences": dict(sorted(events_count.items())),
            "unparsed_events": workflows_unparsed,
            "other_automatic": workflows_automatic_outside_named,
            "other_dispatch": workflows_dispatch_outside_named,
        },
        "legacy_model_static_imports_from_product_readers": legacy_imports,
        "warnings": [
            "Historical-named model packages with present imports cannot be relocated blindly.",
            "Canonical includes active machine contracts and sealed historical evidence; move neither without consumers.",
            "Workflows outside the 26 named RSS.vNext courts are not automatically unused.",
            "Static AST and YAML event scans cannot prove dynamic-loader or external-client safety.",
            "Current catalog coverage and successful CI do not prove ProductRevision qualification.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print complete machine-readable audit")
    args = parser.parse_args()
    data = census(REPO)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print("Tracked files:", data["tracked_files"])
        print("Unclassified top-level directories:", data["unclassified_directories"])
        print("Active product roles:", ", ".join(data["active_catalog_roles"]))
        print("Workflows:", data["workflows"]["yaml_count"], data["workflows"]["actual_by_class"])
        print("Unreviewed auto-trigger workflows:", len(data["workflows"]["other_automatic"]))
        print("Unreviewed dispatch-capable workflows:", len(data["workflows"]["other_dispatch"]))
        print("Legacy direct model imports:", len(data["legacy_model_static_imports_from_product_readers"]))
        print("Run with --json for paths and exact classification.")


if __name__ == "__main__":
    main()
