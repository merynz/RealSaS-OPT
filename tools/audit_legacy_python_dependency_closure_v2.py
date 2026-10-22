#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SCHEMA = "RealSaS.LegacyPythonDependencyClosure.v2"

DOCUMENTED_REUSABLE_ROOTS = (
    # Post-2026-10-19 topology-factored IRIS evidence model. TESSA remains external.
    "models/iris/vnext/model_v3.py",
    "models/iris/vnext/training_v3.py",
    # Current Structural-Evidence architecture: V3 evidence + V4 presentation-conditioned TESSA V3 bridge.
    "models/iris/vnext/structure_v4.py",
    "models/iris/vnext/presentation_evidence_v1.py",
    # Current role-free relation/visibility truth semantics used by the post-V8 program.
    "models/iris/vnext/graphics_truth_v3.py",
    "models/iris/vnext/relation_truth_v3_adapter_v1.py",
    # Current post-M1 training/qualification policy sources recorded by the architecture index.
    "models/iris/vnext/quality_policy_v1.py",
    "models/iris/vnext/optimization_v2.py",
    "models/iris/vnext/relation_support_v2.py",
    # Current TESSA V3 proposal decoder plus fail-closed exact generation path.
    "models/tessa/v3/model_v3.py",
    "models/tessa/v2/generation_v2.py",
    "models/tessa/v2/windowed_incremental_generation_v2.py",
    # Current behaviorally-consumed Graphics V1 runtime path.
    "compiler/realsas_compiler_core/graphics_runtime_v1.py",
    "compiler/realsas_compiler_core/runtime_graphics_binary_v1.py",
    "tools/verify_native_runtime_source_seal_v2.py",
)

EXPLICITLY_NOT_ROOTS = {
    "models/iris/vnext/stack_v3.py": "optional TESSA-V2 global-history composition predates current V4/TESSA-V3 structural-evidence bridge; inspect as supersession candidate instead of assuming latest/current",
    "models/iris/vnext/structure_global_v3.py": "belongs to the older TESSA-V2 global-history composition/preflight path",
}


def _load_v1():
    here = Path(__file__).resolve().parent
    path = here / "audit_legacy_python_dependency_closure_v1.py"
    spec = importlib.util.spec_from_file_location("legacy_dep_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("DEPENDENCY_AUDIT_V1_IMPORT_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ref")
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    v1 = _load_v1()
    if args.self_test:
        v1.self_test()
        if len(set(DOCUMENTED_REUSABLE_ROOTS)) != len(DOCUMENTED_REUSABLE_ROOTS):
            raise AssertionError("duplicate documented reusable root")
        print("legacy python dependency closure v2 self-test: PASS")
        if not args.ref and not args.inventory:
            return 0

    if not args.ref or not args.inventory:
        parser.error("--ref and --inventory are required unless only --self-test is used")

    v1.REUSABLE_ROOTS = DOCUMENTED_REUSABLE_ROOTS
    result = v1.build(args.repo_root.resolve(), args.ref, v1._load_inventory(args.inventory))
    result["schema"] = SCHEMA
    result["root_policy"] = {
        "basis": [
            "2026-10-19 topology-factored IRIS V3 evidence-model commits",
            "2026-10-20 shared Structural-Evidence plan with TESSA V3 and role-free presentation memory",
            "legacy SYSTEM_INDEX documented post-M1 policy and Graphics V1 behavioral runtime path"
        ],
        "documented_reusable_roots": list(DOCUMENTED_REUSABLE_ROOTS),
        "explicitly_not_roots": EXPLICITLY_NOT_ROOTS,
        "rule": "A path is not reusable merely because its version number is high or because it is reachable from a package __init__. Current architecture/behavioral evidence must justify root status."
    }

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
