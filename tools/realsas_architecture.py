#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT / "platform"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_services.orchestrator import mainline  # noqa: E402


LIVE_CODE_PREFIXES = ("compiler/", "models/", "runtime/", "platform/")
LIVE_CODE_SUFFIXES = (".py", ".go", ".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh")
DYNAMIC_IMPORT_MARKERS = (
    "importlib.",
    "__import__(",
    "spec_from_file_location",
    "exec_module(",
)


def _run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_files() -> list[str]:
    out = _run(["git", "ls-files", "-z"]).stdout
    return sorted(path for path in out.split("\0") if path)


def architecture_registry() -> dict[str, Any]:
    result = _run(["go", "run", "./cmd/realsas-architecture"], cwd=PLATFORM)
    return json.loads(result.stdout)


def module_path(module_name: str) -> str | None:
    rel = module_name.replace(".", "/")
    for candidate in (ROOT / f"{rel}.py", ROOT / rel / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(ROOT).as_posix()
    return None


def owner_roots(registry: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for domain in registry.get("domains", []):
        for module in domain.get("modules", []):
            module_id = str(module["id"])
            for root in module.get("code_roots", []):
                rows.append((str(root).rstrip("/"), module_id))
    return sorted(rows, key=lambda row: (-len(row[0]), row[0], row[1]))


def declared_owners(path: str, roots: list[tuple[str, str]]) -> list[str]:
    owners: list[str] = []
    for root, module_id in roots:
        if path == root or path.startswith(root + "/"):
            owners.append(module_id)
    return sorted(set(owners))


def stage_owner_map(registry: dict[str, Any]) -> dict[str, str]:
    return {str(row["stage_id"]): str(row["owner_module_id"]) for row in registry.get("stages", [])}


def descendants(plan: dict[str, Any], root_stage: str) -> list[str]:
    children: dict[str, set[str]] = defaultdict(set)
    order: dict[str, int] = {}
    for row in plan["stages"]:
        stage_id = str(row["id"])
        order[stage_id] = int(row["ordinal"])
        for dep in row.get("depends_on", []):
            children[str(dep)].add(stage_id)
    if root_stage not in order:
        raise KeyError(root_stage)
    seen = {root_stage}
    pending = [root_stage]
    while pending:
        current = pending.pop()
        for child in children[current]:
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return sorted(seen, key=lambda stage_id: (order[stage_id], stage_id))


def support_references(stage_id: str, adapter: str) -> list[str]:
    tokens = {stage_id, adapter}
    module_name, _, symbol = adapter.partition(":")
    if symbol:
        tokens.add(symbol)
    found: set[str] = set()
    search_roots = [root for root in ("tests", ".github/workflows", "tools", "docs") if (ROOT / root).exists()]
    for token in sorted(tokens):
        if not token:
            continue
        cmd = ["git", "grep", "-l", "-F", token, "--", *search_roots]
        proc = _run(cmd, check=False)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.strip())
        found.update(path for path in proc.stdout.splitlines() if path)
    return sorted(found)


def lightweight_implementation_closure(plan: dict[str, Any]) -> dict[str, Any]:
    """Recompute the compiler implementation closure without importing adapters.

    This intentionally mirrors mainline._adapter_impl_hash semantics from source
    hashes and AST import closure only, so architecture discovery never requires
    numpy/torch/blender or other runtime dependencies.
    """
    plan_hash = mainline.validate_plan(plan)
    adapter_rows: list[dict[str, Any]] = []
    imported_modules: set[str] = set()

    for stage in sorted(plan["stages"], key=lambda row: int(row["ordinal"])):
        adapter = str(stage["adapter"])
        module_name, separator, function_name = adapter.partition(":")
        if not separator or not module_name or not function_name:
            raise RuntimeError(f"MAINLINE_V2_ADAPTER_ID_INVALID:{adapter}")
        module_file = mainline._local_module_path(module_name)
        if module_file is None:
            raise RuntimeError(f"MAINLINE_V2_ADAPTER_MODULE_MISSING:{adapter}")

        tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
        callable_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if function_name not in callable_names:
            raise RuntimeError(f"MAINLINE_V2_ADAPTER_CALLABLE_MISSING_SOURCE:{adapter}")

        local_closure = mainline._local_import_closure(module_name)
        if not local_closure:
            raise RuntimeError(f"MAINLINE_V2_IMPLEMENTATION_CLOSURE_EMPTY:{adapter}")
        imported_modules.update(name for name, _digest in local_closure)

        implementation_hash = mainline.content_sha256(
            {
                "schema": "RealSaS.AdapterImplementationClosure.v2",
                "adapter": adapter,
                "local_python_import_closure": [
                    {"module": name, "sha256": digest}
                    for name, digest in local_closure
                ],
            }
        )
        adapter_rows.append(
            {
                "stage_id": str(stage["id"]),
                "adapter": adapter,
                "implementation_hash": implementation_hash,
                "local_python_import_closure": [
                    {"module": name, "sha256": digest}
                    for name, digest in local_closure
                ],
            }
        )

    forbidden = sorted(
        module
        for module in imported_modules
        if module in mainline.CURRENT_V2_FORBIDDEN_IMPORT_MODULES
    )
    if forbidden:
        raise RuntimeError(
            "V2_CURRENT_CLOSURE_IMPORTS_DONOR_ERA_MODULE:" + ",".join(forbidden)
        )

    closure_files = (
        tuple(mainline.IMPLEMENTATION_CLOSURE_STATIC_PATHS)
        + mainline._implementation_closure_dynamic_files()
        + mainline._implementation_closure_test_files()
    )
    if len(closure_files) != len(set(closure_files)):
        raise RuntimeError("V2_IMPLEMENTATION_CLOSURE_DUPLICATE_FILE")
    file_rows: list[dict[str, str]] = []
    for rel in closure_files:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"V2_IMPLEMENTATION_CLOSURE_FILE_MISSING:{rel}")
        file_rows.append({"path": rel, "sha256": mainline.sha256_file(path)})

    return {
        "schema": "RealSaS.DynamicImplementationClosure.v1",
        "pipeline_plan_sha256": plan_hash,
        "adapter_implementation_closures": adapter_rows,
        "imported_module_count": len(imported_modules),
        "forbidden_import_modules": sorted(mainline.CURRENT_V2_FORBIDDEN_IMPORT_MODULES),
        "dynamic_governance_files": list(mainline._implementation_closure_dynamic_files()),
        "critical_files": file_rows,
    }


def build_discovery() -> dict[str, Any]:
    registry = architecture_registry()
    plan = mainline.load_json(mainline.PLAN_PATH)
    mainline.validate_plan(plan)
    closure = lightweight_implementation_closure(plan)
    tracked = git_files()
    tracked_set = set(tracked)
    roots = owner_roots(registry)
    stage_owners = stage_owner_map(registry)

    closure_by_stage: dict[str, dict[str, Any]] = {}
    file_consumers: dict[str, set[str]] = defaultdict(set)
    missing_module_paths: list[dict[str, str]] = []

    for row in closure["adapter_implementation_closures"]:
        stage_id = str(row["stage_id"])
        adapter = str(row["adapter"])
        modules = [str(item["module"]) for item in row.get("local_python_import_closure", [])]
        adapter_module = adapter.split(":", 1)[0]
        if adapter_module not in modules:
            modules.insert(0, adapter_module)
        files: list[str] = []
        for module_name in modules:
            path = module_path(module_name)
            if path is None:
                missing_module_paths.append({"stage_id": stage_id, "module": module_name})
                continue
            files.append(path)
            file_consumers[path].add(stage_id)
        closure_by_stage[stage_id] = {
            "stage_id": stage_id,
            "adapter": adapter,
            "implementation_hash": str(row["implementation_hash"]),
            "dependency_files": sorted(set(files)),
        }

    dependency_rows: list[dict[str, Any]] = []
    ownership_debt: list[dict[str, Any]] = []
    shared_dependencies: list[dict[str, Any]] = []
    for path in sorted(file_consumers):
        stages = sorted(file_consumers[path])
        consumer_modules = sorted({stage_owners[stage] for stage in stages if stage in stage_owners})
        declared = declared_owners(path, roots)
        item = {
            "path": path,
            "consumer_stages": stages,
            "consumer_modules": consumer_modules,
            "declared_owner_modules": declared,
            "status": "DECLARED" if declared else "DYNAMIC_DEPENDENCY_UNDECLARED_OWNER",
        }
        dependency_rows.append(item)
        if not declared:
            ownership_debt.append(item)
        if len(consumer_modules) > 1:
            shared_dependencies.append(item)

    dynamic_import_sites: list[dict[str, Any]] = []
    for path in sorted(file_consumers):
        full = ROOT / path
        try:
            text = full.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        markers = sorted(marker for marker in DYNAMIC_IMPORT_MARKERS if marker in text)
        if markers:
            dynamic_import_sites.append({
                "path": path,
                "markers": markers,
                "consumer_stages": sorted(file_consumers[path]),
            })

    dependency_paths = set(file_consumers)
    declared_live_paths = {
        path
        for path in tracked
        if declared_owners(path, roots)
    }
    outside_current_closure = [
        path
        for path in tracked
        if path.startswith(LIVE_CODE_PREFIXES)
        and path.endswith(LIVE_CODE_SUFFIXES)
        and path not in dependency_paths
        and path not in declared_live_paths
    ]

    platform_unclaimed = [
        path
        for path in tracked
        if path.startswith("platform/")
        and path.endswith(".go")
        and not declared_owners(path, roots)
    ]

    critical_files = [str(row["path"]) for row in closure.get("critical_files", [])]
    missing_critical_files = [path for path in critical_files if path not in tracked_set]

    return {
        "schema": "RealSaS.DynamicArchitectureDiscovery.v1",
        "authority_note": "Computed from the current checkout on every invocation; this output is discovery evidence, not a cached authority.",
        "registry_contract_version": registry.get("contract_version"),
        "pipeline_plan_sha256": closure["pipeline_plan_sha256"],
        "tracked_file_count": len(tracked),
        "stage_count": len(plan["stages"]),
        "stage_dependency_file_count": len(dependency_rows),
        "ownership_debt_count": len(ownership_debt),
        "shared_dependency_file_count": len(shared_dependencies),
        "dynamic_import_site_count": len(dynamic_import_sites),
        "outside_current_stage_closure_count": len(outside_current_closure),
        "platform_unclaimed_go_file_count": len(platform_unclaimed),
        "missing_critical_file_count": len(missing_critical_files),
        "missing_module_path_count": len(missing_module_paths),
        "stage_closures": closure_by_stage,
        "dependency_files": dependency_rows,
        "ownership_debt": ownership_debt,
        "shared_dependencies": shared_dependencies,
        "dynamic_import_sites": dynamic_import_sites,
        "outside_current_stage_closure": outside_current_closure,
        "platform_unclaimed_go_files": platform_unclaimed,
        "critical_files": critical_files,
        "missing_critical_files": missing_critical_files,
        "missing_module_paths": missing_module_paths,
        "registry": registry,
    }


def stage_query(discovery: dict[str, Any], stage_id: str) -> dict[str, Any]:
    registry = discovery["registry"]
    stage = next((row for row in registry["stages"] if row["stage_id"] == stage_id), None)
    if stage is None:
        raise KeyError(f"unknown stage {stage_id}")
    module = next(
        module
        for domain in registry["domains"]
        for module in domain["modules"]
        if module["id"] == stage["owner_module_id"]
    )
    closure = discovery["stage_closures"][stage_id]
    dep_paths = set(closure["dependency_files"])
    dynamic_rows = [row for row in discovery["dependency_files"] if row["path"] in dep_paths]
    plan = mainline.load_json(mainline.PLAN_PATH)
    return {
        "schema": "RealSaS.ArchitectureStageQuery.v1",
        "stage": stage,
        "owner_module": module,
        "implementation_hash": closure["implementation_hash"],
        "dynamic_dependency_files": dynamic_rows,
        "upstream_stage_ids": list(stage.get("depends_on", [])),
        "downstream_stage_ids": descendants(plan, stage_id),
        "support_references": support_references(stage_id, str(stage["adapter"])),
    }


def module_query(discovery: dict[str, Any], module_id: str) -> dict[str, Any]:
    registry = discovery["registry"]
    module = next(
        (module for domain in registry["domains"] for module in domain["modules"] if module["id"] == module_id),
        None,
    )
    if module is None:
        raise KeyError(f"unknown module {module_id}")
    stages = [row for row in registry["stages"] if row["owner_module_id"] == module_id]
    stage_ids = {row["stage_id"] for row in stages}
    deps = [
        row
        for row in discovery["dependency_files"]
        if stage_ids.intersection(row["consumer_stages"])
    ]
    return {
        "schema": "RealSaS.ArchitectureModuleQuery.v1",
        "module": module,
        "owned_stages": stages,
        "dynamic_dependency_files": deps,
    }


def search_query(discovery: dict[str, Any], term: str) -> dict[str, Any]:
    needle = term.lower()
    hits: list[dict[str, str]] = []
    registry = discovery["registry"]
    for domain in registry["domains"]:
        if needle in str(domain["id"]).lower() or needle in str(domain.get("purpose", "")).lower():
            hits.append({"kind": "domain", "id": str(domain["id"])})
        for module in domain["modules"]:
            hay = " ".join([str(module["id"]), str(module.get("purpose", "")), *map(str, module.get("code_roots", []))]).lower()
            if needle in hay:
                hits.append({"kind": "module", "id": str(module["id"])})
    for stage in registry["stages"]:
        hay = " ".join([str(stage["stage_id"]), str(stage.get("group", "")), str(stage.get("adapter", ""))]).lower()
        if needle in hay:
            hits.append({"kind": "stage", "id": str(stage["stage_id"])})
    for row in discovery["dependency_files"]:
        if needle in str(row["path"]).lower():
            hits.append({"kind": "file", "id": str(row["path"])})
    unique = {(row["kind"], row["id"]): row for row in hits}
    return {
        "schema": "RealSaS.ArchitectureSearch.v1",
        "term": term,
        "hits": [unique[key] for key in sorted(unique)],
    }


def audit_summary(discovery: dict[str, Any]) -> dict[str, Any]:
    return {
        key: discovery[key]
        for key in (
            "schema",
            "authority_note",
            "pipeline_plan_sha256",
            "tracked_file_count",
            "stage_count",
            "stage_dependency_file_count",
            "ownership_debt_count",
            "shared_dependency_file_count",
            "dynamic_import_site_count",
            "outside_current_stage_closure_count",
            "platform_unclaimed_go_file_count",
            "missing_critical_file_count",
            "missing_module_path_count",
            "ownership_debt",
            "shared_dependencies",
            "dynamic_import_sites",
            "outside_current_stage_closure",
            "platform_unclaimed_go_files",
            "missing_critical_files",
            "missing_module_paths",
        )
    }


def write_output(value: dict[str, Any], path: str | None) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic RealSaS architecture discovery/query tool.")
    parser.add_argument("--json-out", help="optional output path")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("audit")
    stage = sub.add_parser("stage")
    stage.add_argument("stage_id")
    module = sub.add_parser("module")
    module.add_argument("module_id")
    search = sub.add_parser("search")
    search.add_argument("term")

    args = parser.parse_args()
    discovery = build_discovery()
    if args.command == "audit":
        value = audit_summary(discovery)
    elif args.command == "stage":
        value = stage_query(discovery, args.stage_id)
    elif args.command == "module":
        value = module_query(discovery, args.module_id)
    else:
        value = search_query(discovery, args.term)

    write_output(value, args.json_out)

    # Hard failures are only conditions that mean the dynamic discovery itself
    # lost part of the current executable closure. Ownership debt is surfaced
    # explicitly but is not hidden or guessed.
    if discovery["missing_module_path_count"] or discovery["missing_critical_file_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
