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
    "models/rss_vnext/experts/iris/vnext/",
    "models/rss_vnext/experts/tessa/",
    "models/rss_vnext/experts/axis/",
    "models/rss_vnext/experts/mira/",
    "compiler/realsas_compiler_services/platform_worker/",
    "compiler/rss_vnext/workers/",
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



def static_return_statuses(source: str, function_name: str) -> list[str]:
    """Find literal return statuses in an executable function, not in comments."""
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        return []
    statuses = set()
    for node in ast.walk(matches[0]):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if (
                isinstance(key, ast.Constant) and key.value == "status"
                and isinstance(value, ast.Constant) and isinstance(value.value, str)
            ):
                statuses.add(value.value)
    return sorted(statuses)


def product_readiness(root: Path, capabilities: dict) -> dict:
    """Read-only closure inventory from the live product code, never ProductPASS.

    Static inspection can identify missing product wiring. It cannot establish
    source truth, native equivalence, a usable checkpoint or a ProductRevision.
    Even after the last static blocker disappears, independent executable
    qualification is still mandatory.
    """
    def source(path: str) -> str:
        return (root / path).read_text(encoding="utf-8")

    proof = source(
        "compiler/rss_vnext/workers/product_downstream_executor_v1.py"
    )
    server = source("product/busted_site/server.py")
    parent = source("product/living_compile/server.py")
    frontend = "\n".join(source(name) for name in (
        "product/busted_site/static/js/awaken.js",
        "product/busted_site/static/js/studio_persistence.js",
        "product/busted_site/static/js/studio_requalification_ui.js",
    ))
    input_contract = source(
        "compiler/rss_vnext/core/rss_vnext_source_observation_v1.py"
    )
    runtime = source(
        "compiler/realsas_compiler_core/controllable_puppet_runtime_v1.py"
    )
    proof_statuses = static_return_statuses(
        proof, "execute_proof_capability_v1"
    )
    proof_missing = sorted(set(re.findall(
        r'"(SOURCE_APPEARANCE_AND_POSE_VISIBILITY|'
        r'NATIVE_GRAPHICS_RENDERER_PARITY|'
        r'CONTACT_AND_ATTACHMENT_BEHAVIOR|'
        r'EDIT_CONSEQUENCE_AND_RUNTIME_RSS_REOPEN|'
        r'SOURCE_AND_MODEL_RELEASE_ADMISSION)"', proof
    )))
    developer_only = sorted(
        key for key, item in capabilities.items()
        if item.get("policy", {}).get("allowed_modes") == ["DEVELOPER"]
    )
    frontend_routes = set(re.findall(
        r"/api/(?:compile|studio)/[a-z-]+", frontend
    ))
    # A mentioned route is NOT a working handler: the inherited Living Compile
    # server explicitly raises on several compile endpoints. An overriding
    # BUSTED handler may implement one, but a rejected parent route is absent.
    server_literals = {
        node.value for node in ast.walk(ast.parse(server))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    parent_ast = ast.parse(parent)
    parent_literals = {
        node.value for node in ast.walk(parent_ast)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    parent_rejected = {
        node.value
        for conditional in ast.walk(parent_ast)
        if isinstance(conditional, ast.If) and any(
            isinstance(child, ast.Raise)
            for statement in conditional.body for child in ast.walk(statement)
        )
        for node in ast.walk(conditional.test)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("/api/")
    }
    handled_routes = server_literals | (parent_literals - parent_rejected)
    missing_routes = sorted(frontend_routes - handled_routes)
    checks = []

    def block(code: str, owner: str, reason: str, details: dict) -> None:
        checks.append({
            "id": code, "status": "OPEN", "owner": owner,
            "reason": reason, "source_evidence": details,
        })

    if developer_only:
        block(
            "PRODUCT_CAPABILITIES_DEVELOPER_ONLY", "platform",
            "A developer capability DAG is not an admitted product workflow.",
            {"roles": developer_only},
        )
    if "FAIL" in proof_statuses and "PASS" not in proof_statuses:
        block(
            "PROOF_HAS_NO_PASS_EXECUTION_PATH", "compiler",
            "The active Proof executor always returns a failed qualification.",
            {"literal_return_statuses": proof_statuses},
        )
    if proof_missing:
        block(
            "PROOF_INDEPENDENT_WITNESSES_MISSING", "compiler/runtime",
            "The active Proof executor names independently required witnesses.",
            {"requirements": proof_missing},
        )
    if '"public_compile_gateway": False' in server or '"compile_available": False' in server:
        block(
            "PUBLIC_COMPILE_GATEWAY_DISABLED", "product",
            "The public API reports compile unavailable even with valid input.",
            {"api": "/api/project"},
        )
    if missing_routes:
        block(
            "PRODUCT_HTTP_CONSUMERS_UNBOUND", "product/platform",
            "The BUSTED frontend calls endpoints not handled by its HTTP server.",
            {"unhandled_frontend_routes": missing_routes},
        )
    if (
        "camera_features" in input_contract
        and "(2, 13)" in input_contract
        and "camera_features" not in source("product/busted_site/static/js/awaken.js")
    ):
        block(
            "TWO_VIEW_CAMERA_POLICY_UNBOUND", "product/models",
            "Front/back files do not supply the required canonical 13D cameras; "
            "a versioned assumption/estimation policy must be explicit.",
            {"observations": 2, "camera_features_per_view": 13},
        )
    if "PUPPET_RUNTIME_EIGHT_CAMERAS_REQUIRED" in runtime:
        block(
            "TWO_VIEW_TO_RUNTIME_VIEW_BRIDGE_UNVERIFIED", "runtime",
            "The current puppet consumer requires eight output cameras; "
            "two observed source views alone do not prove the required view set.",
            {"source_view_count": 2, "runtime_requires_eight_views": True},
        )
    return {
        "schema": "RealSaS.ProductReadinessSourceAudit.v1",
        "scope": "SOURCE_STATIC_ONLY_NOT_PRODUCT_QUALIFICATION",
        "readiness_claim": "BLOCKED" if checks else "UNVERIFIED_REQUIRES_EXECUTABLE_PROOF",
        "product_capability_count": len(capabilities),
        "proof_literal_return_statuses": proof_statuses,
        "checks": checks,
        "open_count": len(checks),
        "training_and_unseen": "EXTERNAL_NOT_EVALUATED",
        "guarantees": {
            "no_product_revision_minted": True,
            "no_source_or_model_authority_inferred": True,
            "no_native_parity_claimed": True,
            "static_absence_of_blockers_is_not_deploy_pass": True,
        },
    }

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
        "product_readiness": product_readiness(root, catalog["capabilities"]),
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
        print("Product deploy blockers (source audit):", data["product_readiness"]["open_count"])
        print("Workflows:", data["workflows"]["yaml_count"], data["workflows"]["actual_by_class"])
        print("Unreviewed auto-trigger workflows:", len(data["workflows"]["other_automatic"]))
        print("Unreviewed dispatch-capable workflows:", len(data["workflows"]["other_dispatch"]))
        print("Legacy direct model imports:", len(data["legacy_model_static_imports_from_product_readers"]))
        print("Run with --json for paths and exact classification.")


if __name__ == "__main__":
    main()
