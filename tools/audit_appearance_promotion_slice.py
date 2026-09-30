#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import deque
from pathlib import Path

LOCAL_PREFIXES = ("compiler.", "models.", "runtime.", "tools.")
APPEARANCE_STAGE_ORDINALS = tuple(range(20, 26))
TEST_PATTERNS = (
    r"test_.*caa",
    r"test_.*appearance",
    r"test_.*visibility",
    r"test_.*visual",
    r"test_.*source_fidelity",
    r"test_.*rest_render",
)

MECHANICS_FROZEN_PREFIXES = (
    "compiler/realsas_compiler_core/canonical_mesh_candidate_v1.py",
    "compiler/realsas_compiler_core/canonical_mesh_quality_repair_v1.py",
    "compiler/realsas_compiler_core/mechanical_",
    "compiler/realsas_compiler_core/mesh/",
    "compiler/realsas_compiler_core/skin.py",
    "compiler/realsas_compiler_core/rig.py",
    "compiler/realsas_compiler_core/substrate/",
    "compiler/realsas_compiler_services/orchestrator/adapters/mesh_v2.py",
    "compiler/realsas_compiler_services/orchestrator/adapters/v2_architecture.py",
)

def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(["git", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and p.returncode:
        raise RuntimeError(
            "git " + " ".join(args) + "\nSTDOUT:\n" + p.stdout + "\nSTDERR:\n" + p.stderr
        )
    return p.stdout

def exists(ref: str, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0

def text_at(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")

def sha(ref: str, path: str) -> str | None:
    return git("rev-parse", f"{ref}:{path}").strip() if exists(ref, path) else None

def all_paths(ref: str) -> tuple[str, ...]:
    return tuple(x for x in git("ls-tree", "-r", "--name-only", ref).splitlines() if x)

def module_path(ref: str, module: str) -> str | None:
    rel = module.replace(".", "/")
    for path in (rel + ".py", rel + "/__init__.py"):
        if exists(ref, path):
            return path
    return None

def path_module(path: str) -> str:
    if path.endswith("/__init__.py"):
        return path[:-12].replace("/", ".")
    return path[:-3].replace("/", ".")

def resolve_from(current_module: str, level: int, target: str | None) -> str:
    if level <= 0:
        return target or ""
    parts = current_module.split(".")[:-1]
    up = max(level - 1, 0)
    if up:
        parts = parts[:-up] if up <= len(parts) else []
    if target:
        parts.extend(target.split("."))
    return ".".join(parts)

def local_imports(ref: str, path: str) -> set[str]:
    source = text_at(ref, path)
    tree = ast.parse(source, filename=path)
    current = path_module(path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(LOCAL_PREFIXES) and module_path(ref, name):
                    out.add(name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_from(current, int(node.level or 0), node.module)
            if base.startswith(LOCAL_PREFIXES) and module_path(ref, base):
                out.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                if child.startswith(LOCAL_PREFIXES) and module_path(ref, child):
                    out.add(child)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ref", required=True)
    ap.add_argument("--research-ref", required=True)
    ap.add_argument("--report", default="appearance_promotion_slice.json")
    args = ap.parse_args()

    plan = json.loads(text_at(args.research_ref, "canonical/MAINLINE_EXECUTION_PLAN_V2.json"))
    stages = [
        s for s in sorted(plan["stages"], key=lambda r: int(r["ordinal"]))
        if int(s["ordinal"]) in APPEARANCE_STAGE_ORDINALS
    ]
    if [int(s["ordinal"]) for s in stages] != list(APPEARANCE_STAGE_ORDINALS):
        raise RuntimeError("APPEARANCE_STAGE_SLICE_INCOMPLETE")

    adapter_paths: set[str] = set()
    stage_rows = []
    for stage in stages:
        module = str(stage["adapter"]).split(":", 1)[0]
        path = module_path(args.research_ref, module)
        if not path:
            raise RuntimeError(f"MISSING_STAGE_MODULE:{stage['id']}:{module}")
        adapter_paths.add(path)
        stage_rows.append({
            "ordinal": int(stage["ordinal"]),
            "id": stage["id"],
            "adapter": stage["adapter"],
            "depends_on": list(stage.get("depends_on") or ()),
        })

    closure = set(adapter_paths)
    queue = deque(sorted(adapter_paths))
    parsed = set()
    imported_modules = set()
    while queue:
        path = queue.popleft()
        if path in parsed:
            continue
        parsed.add(path)
        for module in local_imports(args.research_ref, path):
            imported_modules.add(module)
            child = module_path(args.research_ref, module)
            if child and child not in closure:
                closure.add(child)
                queue.append(child)

    repo_paths = all_paths(args.research_ref)
    tests = sorted(
        p for p in repo_paths
        if p.startswith("tests/compiler/")
        and p.endswith(".py")
        and any(re.search(pattern, Path(p).name, re.I) for pattern in TEST_PATTERNS)
    )
    # Current end-to-end Stage20-25 policy test is a named node inside this
    # mixed historical/current file; keep the file as evidence but gate by node id.
    mixed_test = "tests/compiler/test_product_mesh_stage_adapters_v1.py"
    if exists(args.research_ref, mixed_test):
        tests.append(mixed_test)
    tests = sorted(set(tests))

    workflow_names = (
        "vf23_production_policy_e2e_bank.yml",
        "current_mainline_self_hosted_ci.yml",
    )
    workflows = [
        f".github/workflows/{name}"
        for name in workflow_names
        if exists(args.research_ref, f".github/workflows/{name}")
    ]

    selected = sorted(closure | set(tests) | set(workflows))
    changed, same, new = [], [], []
    for path in selected:
        rs, bs = sha(args.research_ref, path), sha(args.baseline_ref, path)
        row = {"path": path, "baseline_sha": bs, "research_sha": rs}
        if rs == bs:
            same.append(row)
        else:
            changed.append(row)
            if bs is None:
                new.append(row)

    frozen_touches = sorted(
        p for p in closure
        if any(p == prefix or p.startswith(prefix) for prefix in MECHANICS_FROZEN_PREFIXES)
    )
    subject_named_product_paths = sorted(
        p for p in closure
        if re.search(r"(knight|mage|fit1|subject[_-]?2)", p, re.I)
    )
    if subject_named_product_paths:
        raise RuntimeError("SUBJECT_NAMED_APPEARANCE_PRODUCT_PATH:" + ",".join(subject_named_product_paths))

    report = {
        "schema": "RealSaS.AppearancePromotionSlice.v1",
        "baseline_ref": args.baseline_ref,
        "research_ref": args.research_ref,
        "stage_ordinals": list(APPEARANCE_STAGE_ORDINALS),
        "stages": stage_rows,
        "adapter_path_count": len(adapter_paths),
        "implementation_closure_path_count": len(closure),
        "imported_local_module_count": len(imported_modules),
        "test_path_count": len(tests),
        "workflow_path_count": len(workflows),
        "selected_path_count": len(selected),
        "changed_path_count": len(changed),
        "new_path_count": len(new),
        "frozen_mechanics_closure_touches": frozen_touches,
        "subject_named_product_paths": subject_named_product_paths,
        "implementation_closure_paths": sorted(closure),
        "test_paths": tests,
        "workflow_paths": workflows,
        "changed_files": changed,
        "new_files": new,
        "same_files": same,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage_count": len(stages),
        "implementation_closure_path_count": len(closure),
        "changed_path_count": len(changed),
        "new_path_count": len(new),
        "frozen_mechanics_closure_touches": frozen_touches,
        "subject_named_product_paths": subject_named_product_paths,
        "report": args.report,
    }, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
