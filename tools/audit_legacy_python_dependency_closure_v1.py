#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from collections import deque
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Iterable

SCHEMA = "RealSaS.LegacyPythonDependencyClosure.v1"

REUSABLE_ROOTS = (
    "models/iris/vnext/structure_v4.py",
    "models/iris/vnext/structure_v3.py",
    "models/iris/vnext/presentation_evidence_v1.py",
    "models/tessa/v3/model_v3.py",
    "models/tessa/v2/model_v2.py",
    "models/tessa/v2/generation_v2.py",
    "models/tessa/v2/windowed_incremental_generation_v2.py",
    "compiler/realsas_compiler_core/graphics_runtime_v1.py",
    "compiler/realsas_compiler_core/runtime_graphics_binary_v1.py",
    "tools/verify_native_runtime_source_seal_v2.py",
)

EXPERIMENT_ROOTS = (
    "tools/iris_vnext/train_fit8_tessa_v3_structural_evidence_ab_v1.py",
    "tools/iris_vnext/evaluate_structural_evidence_ab_directional_screen_v1.py",
)


def _run(repo: Path, *args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _tree_paths(repo: Path, ref: str) -> list[str]:
    return [
        line.strip()
        for line in _run(repo, "ls-tree", "-r", "--name-only", ref).splitlines()
        if line.strip().endswith(".py")
    ]


def _module_for_path(path: str) -> str:
    p = PurePosixPath(path)
    if p.name == "__init__.py":
        return ".".join(p.parent.parts)
    return ".".join(p.with_suffix("").parts)


def _path_index(paths: Iterable[str]) -> tuple[dict[str, str], dict[str, str]]:
    module_to_path: dict[str, str] = {}
    path_to_module: dict[str, str] = {}
    for path in paths:
        module = _module_for_path(path)
        module_to_path[module] = path
        path_to_module[path] = module
    return module_to_path, path_to_module


def _show(repo: Path, ref: str, path: str) -> str:
    return _run(repo, "show", f"{ref}:{path}")


def _relative_target(current_module: str, current_path: str, level: int, module: str | None) -> str:
    package = current_module if current_path.endswith("/__init__.py") else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    if level <= 0:
        base = []
    else:
        up = level - 1
        if up > len(parts):
            return ""
        base = parts[: len(parts) - up]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _resolve_imports(
    *,
    tree: ast.AST,
    current_module: str,
    current_path: str,
    module_to_path: dict[str, str],
) -> tuple[set[str], set[str]]:
    internal: set[str] = set()
    unresolved: set[str] = set()

    def add_module(name: str) -> bool:
        if not name:
            return False
        candidate = name
        while candidate:
            if candidate in module_to_path:
                internal.add(module_to_path[candidate])
                return True
            candidate = candidate.rpartition(".")[0]
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not add_module(alias.name):
                    unresolved.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _relative_target(
                    current_module,
                    current_path,
                    node.level,
                    node.module,
                )
            else:
                base = node.module or ""
            base_is_internal = add_module(base)
            matched_alias = False
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = f"{base}.{alias.name}" if base else alias.name
                if add_module(target):
                    matched_alias = True
            if not base_is_internal and not matched_alias and base:
                unresolved.add(base)
    internal.discard(current_path)
    return internal, unresolved


def _graph(repo: Path, ref: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    paths = _tree_paths(repo, ref)
    module_to_path, path_to_module = _path_index(paths)
    graph: dict[str, set[str]] = {}
    unresolved: dict[str, set[str]] = {}
    for path in paths:
        try:
            tree = ast.parse(_show(repo, ref, path), filename=path)
        except SyntaxError as exc:
            raise RuntimeError(f"syntax error while auditing {path}: {exc}") from exc
        deps, missing = _resolve_imports(
            tree=tree,
            current_module=path_to_module[path],
            current_path=path,
            module_to_path=module_to_path,
        )
        graph[path] = deps
        if missing:
            unresolved[path] = missing
    return graph, unresolved


def _closure(graph: dict[str, set[str]], roots: Iterable[str]) -> set[str]:
    missing_roots = [root for root in roots if root not in graph]
    if missing_roots:
        raise RuntimeError("DEPENDENCY_AUDIT_ROOT_MISSING:" + ",".join(missing_roots))
    seen: set[str] = set()
    queue = deque(roots)
    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)
        for dep in sorted(graph.get(path, ())):
            if dep not in seen:
                queue.append(dep)
    return seen


def _load_inventory(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "RealSaS.LegacyResearchBranchDrainPathInventory.v1":
        raise RuntimeError("DEPENDENCY_AUDIT_INVENTORY_SCHEMA_INVALID")
    if not value.get("complete_path_classification"):
        raise RuntimeError("DEPENDENCY_AUDIT_REQUIRES_COMPLETE_PATH_INVENTORY")
    return value


def build(repo: Path, ref: str, inventory: dict) -> dict[str, object]:
    graph, unresolved = _graph(repo, ref)
    reusable = _closure(graph, REUSABLE_ROOTS)
    experiment = _closure(graph, EXPERIMENT_ROOTS)
    experiment_only = experiment - reusable

    changed = {str(row["path"]): row for row in inventory["entries"]}
    h_paths = {
        path
        for path, row in changed.items()
        if row.get("disposition") == "H_UNRESOLVED_DEPENDENCY"
    }

    h_reusable = sorted(h_paths & reusable)
    h_experiment_only = sorted(h_paths & experiment_only)
    h_unreached = sorted(h_paths - reusable - experiment_only)

    changed_reusable = sorted(set(changed) & reusable)
    changed_experiment_only = sorted(set(changed) & experiment_only)

    internal_unresolved_imports = {
        path: sorted(names)
        for path, names in unresolved.items()
        if path in reusable or path in experiment
    }

    return {
        "schema": SCHEMA,
        "legacy_ref": ref,
        "legacy_commit": _run(repo, "rev-parse", ref).strip(),
        "reusable_roots": list(REUSABLE_ROOTS),
        "experiment_roots": list(EXPERIMENT_ROOTS),
        "reusable_closure_path_count": len(reusable),
        "experiment_closure_path_count": len(experiment),
        "experiment_only_path_count": len(experiment_only),
        "changed_paths_in_reusable_closure": changed_reusable,
        "changed_paths_in_experiment_only_closure": changed_experiment_only,
        "h_reachable_from_reusable_roots": h_reusable,
        "h_reachable_only_from_experiment_roots": h_experiment_only,
        "h_not_reached_by_active_python_roots": h_unreached,
        "h_counts": {
            "reusable": len(h_reusable),
            "experiment_only": len(h_experiment_only),
            "unreached": len(h_unreached),
            "total": len(h_paths),
        },
        "unresolved_import_names_in_active_closures": internal_unresolved_imports,
        "note": "Unresolved import names include external/stdlib packages and are telemetry; classification decisions apply only to repository paths resolved by the graph.",
    }


def self_test() -> None:
    module_to_path = {
        "pkg": "pkg/__init__.py",
        "pkg.a": "pkg/a.py",
        "pkg.b": "pkg/b.py",
        "other.c": "other/c.py",
    }
    tree = ast.parse("from . import b\nfrom other.c import value\nimport json\n")
    deps, _ = _resolve_imports(
        tree=tree,
        current_module="pkg.a",
        current_path="pkg/a.py",
        module_to_path=module_to_path,
    )
    if deps != {"pkg/__init__.py", "pkg/b.py", "other/c.py"}:
        raise AssertionError(f"dependency resolver drift: {sorted(deps)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ref")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("legacy python dependency closure self-test: PASS")
        if not args.ref and not args.inventory:
            return 0

    if not args.ref or not args.inventory:
        parser.error("--ref and --inventory are required unless only --self-test is used")

    result = build(args.repo_root.resolve(), args.ref, _load_inventory(args.inventory))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "legacy_commit": result["legacy_commit"],
        "reusable_closure_path_count": result["reusable_closure_path_count"],
        "experiment_closure_path_count": result["experiment_closure_path_count"],
        "h_counts": result["h_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
