#!/usr/bin/env python3
"""Audit whether current IRIS/TESSA V3 objectives are wired into a real trainer.

This is a repository-source fact, not a criticism: V3 may intentionally still be
architecture/objective source awaiting a preregistered training program.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3TrainingRouteProbe.v1"
LOSS_NAMES = {
    "mechanical_demand_loss_v3",
    "appearance_loss_v3",
    "graphics_relation_loss_v3",
    "ray_visibility_loss_v3",
}


def _call_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def run(repo_root: Path) -> dict:
    training_source = repo_root / "models/iris/vnext/training_v3.py"
    if not training_source.is_file():
        raise RuntimeError("V3_TRAINING_ROUTE_OBJECTIVE_SOURCE_MISSING")

    call_sites: dict[str, list[str]] = {name: [] for name in sorted(LOSS_NAMES)}
    optimizer_steps: list[str] = []
    v3_constructor_sites: list[str] = []

    roots = [repo_root / x for x in ("models", "compiler", "tools", "experiments")]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(repo_root).as_posix()
            if rel.startswith("tests/") or "iris_tessa_v3_apparatus_audit_20261025" in rel:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _call_name(node)
                    if name in LOSS_NAMES and rel != "models/iris/vnext/training_v3.py":
                        call_sites[name].append(f"{rel}:{getattr(node, 'lineno', 0)}")
                    if name == "IRISVNextEvidenceModelV3":
                        v3_constructor_sites.append(f"{rel}:{getattr(node, 'lineno', 0)}")
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "step":
                        owner = ast.unparse(node.func.value) if hasattr(ast, "unparse") else "unknown"
                        if "optimizer" in owner.lower() or "optim" in owner.lower():
                            optimizer_steps.append(f"{rel}:{getattr(node, 'lineno', 0)}:{owner}")

    nonempty_loss_consumers = {k: v for k, v in call_sites.items() if v}
    if nonempty_loss_consumers:
        status = "V3_OBJECTIVES_HAVE_NONTEST_CONSUMERS__REVIEW_REQUIRED"
    else:
        status = "NO_CURRENT_NONTEST_V3_OBJECTIVE_CONSUMER_FOUND"

    return {
        "schema": SCHEMA,
        "status": status,
        "objective_call_sites_outside_definition_and_tests": call_sites,
        "iris_v3_constructor_sites": sorted(v3_constructor_sites),
        "optimizer_step_sites_repository_wide_non_test": sorted(optimizer_steps),
        "interpretation": [
            "the four V3 objective functions are source-level objective contracts but currently have no non-test call site in the audited roots",
            "the repository may contain optimizers for other historical/current model families; their existence does not constitute a V3 training program",
            "therefore no claim should be made that the current V3 hypothesis has already been trained or falsified under these objectives",
        ],
        "required_before_first_v3_training": [
            "version a single explicit V3 training orchestration that binds exact truth schema, sampling, objective-family weights, optimizer semantics, checkpoint selection and evaluation courts",
            "seal gradient/update semantics before opening FIT8 or causal-truth model outcomes",
            "keep training runner distinct from product inference and ModelRelease promotion",
        ],
        "training_executed_by_this_probe": False,
        "model_inference_executed": False,
        "gpu_required": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.repo_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"], "status": report["status"], "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
