#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
from collections import deque
from pathlib import Path


LOCAL_PREFIXES = ("compiler.", "models.", "runtime.", "tools.", "tests.")


def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode:
        raise RuntimeError(
            "git " + " ".join(args) + "\nSTDOUT:\n" + p.stdout + "\nSTDERR:\n" + p.stderr
        )
    return p.stdout


def ref_file_exists(ref: str, path: str) -> bool:
    p = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return p.returncode == 0


def show_text(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def blob_sha(ref: str, path: str) -> str | None:
    if not ref_file_exists(ref, path):
        return None
    return git("rev-parse", f"{ref}:{path}").strip()


def list_paths(ref: str) -> list[str]:
    return [
        row
        for row in git("ls-tree", "-r", "--name-only", ref).splitlines()
        if row
    ]


def module_candidates(module: str) -> tuple[str, ...]:
    rel = module.replace(".", "/")
    return (f"{rel}.py", f"{rel}/__init__.py")


def module_path(ref: str, module: str) -> str | None:
    for path in module_candidates(module):
        if ref_file_exists(ref, path):
            return path
    return None


def path_module(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")].replace("/", ".")
    return path[:-3].replace("/", ".")


def resolve_from_module(current_module: str, level: int, target: str | None) -> str:
    if level <= 0:
        return target or ""
    package = current_module.split(".")[:-1]
    up = max(level - 1, 0)
    if up:
        package = package[:-up] if up <= len(package) else []
    if target:
        package.extend(target.split("."))
    return ".".join(package)


def imported_local_modules(ref: str, path: str, source: str) -> set[str]:
    current_module = path_module(path) or ""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise RuntimeError(f"AST_PARSE_FAIL:{path}:{exc}") from exc
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(LOCAL_PREFIXES) and module_path(ref, name):
                    out.add(name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_from_module(current_module, int(node.level or 0), node.module)
            if base.startswith(LOCAL_PREFIXES) and module_path(ref, base):
                out.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                if child.startswith(LOCAL_PREFIXES) and module_path(ref, child):
                    out.add(child)
    return out


def extract_literal_constant(source: str, name: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return ast.literal_eval(node.value)
    raise KeyError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-ref", required=True)
    ap.add_argument("--baseline-ref", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--report", default="normalization_report.json")
    args = ap.parse_args()

    research = args.research_ref
    baseline = args.baseline_ref
    research_paths = set(list_paths(research))

    plan_path = "canonical/MAINLINE_EXECUTION_PLAN_V2.json"
    mainline_path = "compiler/realsas_compiler_services/orchestrator/mainline.py"
    plan = json.loads(show_text(research, plan_path))
    mainline_src = show_text(research, mainline_path)

    if plan.get("schema") != "RealSaS.MainlineExecutionPlan.v2":
        raise RuntimeError("CURRENT_PLAN_SCHEMA_DRIFT")
    if int(plan.get("stage_count", -1)) != 46 or len(plan.get("stages") or ()) != 46:
        raise RuntimeError("CURRENT_PLAN_NOT_EXACT_46_STAGE")
    if plan.get("canonical_branch") != "main":
        raise RuntimeError("CURRENT_PLAN_CANONICAL_BRANCH_DRIFT")
    if plan.get("subject_specific_code_forbidden") is not True:
        raise RuntimeError("CURRENT_PLAN_GENERICITY_FLAG_MISSING")

    static_paths = tuple(extract_literal_constant(mainline_src, "IMPLEMENTATION_CLOSURE_STATIC_PATHS"))
    test_roots = tuple(extract_literal_constant(mainline_src, "IMPLEMENTATION_CLOSURE_TEST_ROOTS"))
    dynamic_globs = tuple(extract_literal_constant(mainline_src, "IMPLEMENTATION_CLOSURE_DYNAMIC_GLOBS"))
    forbidden_modules = set(extract_literal_constant(mainline_src, "CURRENT_V2_FORBIDDEN_IMPORT_MODULES"))

    selected: set[str] = {plan_path, mainline_path}
    selected.update(p for p in static_paths if p in research_paths)
    for pattern in dynamic_globs:
        selected.update(p for p in research_paths if fnmatch.fnmatch(p, pattern))
    for root in test_roots:
        prefix = root.rstrip("/") + "/"
        selected.update(
            p for p in research_paths
            if p.startswith(prefix) and p.endswith(".py")
        )

    adapter_modules = []
    adapter_seed_paths: set[str] = set()
    for stage in sorted(plan["stages"], key=lambda row: int(row["ordinal"])):
        adapter = str(stage["adapter"])
        module, sep, func = adapter.partition(":")
        if not sep or not module or not func:
            raise RuntimeError(f"BAD_ADAPTER_ID:{adapter}")
        adapter_modules.append(module)
        path = module_path(research, module)
        if not path:
            raise RuntimeError(f"ADAPTER_MODULE_MISSING:{module}")
        selected.add(path)
        adapter_seed_paths.add(path)

    # Match canonical mainline.py semantics exactly: forbidden donor imports are
    # checked only across the local import closure of the 46 stage adapters.
    # Test files are closure-hashed governance inputs but are not runtime import
    # roots and must not pull historical donor modules into the product closure.
    queue = deque(sorted(adapter_seed_paths))
    adapter_parsed: set[str] = set()
    adapter_imported_modules: set[str] = set()
    while queue:
        path = queue.popleft()
        if path in adapter_parsed:
            continue
        adapter_parsed.add(path)
        source = show_text(research, path)
        for module in imported_local_modules(research, path, source):
            adapter_imported_modules.add(module)
            child = module_path(research, module)
            if child and child not in selected:
                selected.add(child)
            if child and child not in adapter_parsed:
                queue.append(child)

    forbidden_used = sorted(adapter_imported_modules & forbidden_modules)
    if forbidden_used:
        raise RuntimeError("CURRENT_V2_IMPORTS_FORBIDDEN_DONOR_MODULE:" + ",".join(forbidden_used))

    # The final Stage18 contract must consume observation authority because the
    # source-owned visual mesh is built from qualified source foreground masks.
    stage18 = next(row for row in plan["stages"] if int(row["ordinal"]) == 18)
    stage18_deps = set(map(str, stage18.get("depends_on") or ()))
    if "07_OBSERVATION_CONTRACT_QUALIFIED" not in stage18_deps:
        raise RuntimeError("STAGE18_SOURCE_VISUAL_AUTHORITY_DEPENDENCY_MISSING")

    code_paths = sorted(
        p for p in selected
        if p.startswith(("compiler/", "models/", "runtime/"))
    )
    subject_named_paths = [
        p for p in code_paths
        if re.search(r"(knight|mage|fit1|subject[_-]?2)", p, re.I)
    ]
    if subject_named_paths:
        raise RuntimeError("SUBJECT_NAMED_PRODUCT_CODE:" + ",".join(subject_named_paths))

    changed = []
    new = []
    same = []
    for path in sorted(selected):
        rs = blob_sha(research, path)
        bs = blob_sha(baseline, path)
        row = {"path": path, "baseline_sha": bs, "research_sha": rs}
        if rs == bs:
            same.append(row)
        elif bs is None:
            new.append(row)
            changed.append(row)
        else:
            changed.append(row)

    if args.apply:
        for row in changed:
            path = row["path"]
            if ref_file_exists(research, path):
                git("checkout", research, "--", path)

    # Report literal subject references for human review without treating policy
    # sentinel strings (e.g. KNIGHT_RESULT forbidden-input declarations) as a
    # product-code failure.
    subject_literal_hits = []
    for path in code_paths:
        text = show_text(research, path)
        hits = sorted(set(re.findall(r"(?i)\b(knight|mage|fit1|subject[_-]?2)\b", text)))
        if hits:
            subject_literal_hits.append({"path": path, "tokens": hits})

    visual_mesh_paths = sorted(
        p for p in selected
        if "visual_mesh" in p.lower() or "arap" in p.lower()
    )
    appearance_paths = sorted(
        p for p in selected
        if "appearance" in p.lower() or "visibility" in p.lower()
    )
    runtime_paths = sorted(
        p for p in selected
        if "runtime" in p.lower()
    )

    report = {
        "schema": "RealSaS.GenericMainNormalizationAudit.v1",
        "research_ref": research,
        "baseline_ref": baseline,
        "plan": {
            "stage_count": plan["stage_count"],
            "canonical_branch": plan["canonical_branch"],
            "subject_specific_code_forbidden": plan["subject_specific_code_forbidden"],
            "stage18_depends_on": sorted(stage18_deps),
            "stage42_adapter": next(
                row["adapter"] for row in plan["stages"] if int(row["ordinal"]) == 42
            ),
        },
        "closure": {
            "selected_file_count": len(selected),
            "changed_file_count": len(changed),
            "new_file_count": len(new),
            "same_file_count": len(same),
            "adapter_import_closure_python_file_count": len(adapter_parsed),
            "adapter_imported_local_module_count": len(adapter_imported_modules),
            "forbidden_donor_modules_used": forbidden_used,
            "adapter_modules": adapter_modules,
        },
        "changed_files": changed,
        "new_files": new,
        "subject_named_product_code_paths": subject_named_paths,
        "subject_literal_hits_for_review": subject_literal_hits,
        "visual_mesh_or_arap_closure_paths": visual_mesh_paths,
        "appearance_or_visibility_closure_paths": appearance_paths,
        "runtime_closure_paths": runtime_paths,
        "readiness": {
            "baseline_blob_sha": blob_sha(baseline, "canonical/V2_IMPLEMENTATION_READINESS.json"),
            "research_blob_sha": blob_sha(research, "canonical/V2_IMPLEMENTATION_READINESS.json"),
            "note": "Readiness is audited separately and must be rebound only after the candidate implementation closure is green.",
        },
    }
    Path(args.report).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "selected_file_count": len(selected),
        "changed_file_count": len(changed),
        "new_file_count": len(new),
        "forbidden_donor_modules_used": forbidden_used,
        "subject_named_product_code_paths": subject_named_paths,
        "visual_mesh_or_arap_closure_paths": visual_mesh_paths,
        "report": args.report,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
