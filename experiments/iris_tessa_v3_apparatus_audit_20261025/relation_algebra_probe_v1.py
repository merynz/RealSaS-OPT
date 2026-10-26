#!/usr/bin/env python3
"""Static audit of algebraic constraints in the current V3 pair parameterization.

The probe distinguishes what the architecture *can* represent from what it
*guarantees* by construction. No model is loaded and no scientific quality claim
is made.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

SCHEMA = "RealSaS.IRISTESSAV3RelationAlgebraProbe.v1"
REL_PATH = "models/iris/vnext/heads_v3.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise RuntimeError("RELATION_ALGEBRA_CLASS_MISSING:" + name)


def _function_source(text: str, tree: ast.Module, class_name: str, function_name: str) -> str:
    cls = _class(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = text.splitlines()
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise RuntimeError("RELATION_ALGEBRA_FUNCTION_MISSING:" + function_name)


def _class_tuple(tree: ast.Module, class_name: str, field: str) -> tuple[str, ...]:
    cls = _class(tree, class_name)
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == field for t in node.targets):
            value = ast.literal_eval(node.value)
            return tuple(str(x) for x in value)
    raise RuntimeError("RELATION_ALGEBRA_FIELD_MISSING:" + field)


def run(repo_root: Path) -> dict:
    path = repo_root / REL_PATH
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    pair = _function_source(text, tree, "GraphicsRelationHeadV3", "_pair")
    static = _class_tuple(tree, "GraphicsRelationHeadV3", "STATIC_OUTPUTS")
    view = _class_tuple(tree, "GraphicsRelationHeadV3", "VIEW_OUTPUTS")

    required = ("a, b, a - b, a * b",)
    if not any(fragment in pair for fragment in required):
        raise RuntimeError("RELATION_ALGEBRA_PAIR_PARAMETERIZATION_DRIFT")

    return {
        "schema": SCHEMA,
        "status": "ORDERED_PAIR_PARAMETERIZATION_CONFIRMED__RELATION_ALGEBRA_NOT_ENFORCED",
        "source": REL_PATH,
        "facts": {
            "pair_features": ["a", "b", "a-b", "a*b"],
            "swap_changes_representation_in_general": True,
            "symmetric_output_constraint_by_construction": False,
            "front_complement_constraint_by_construction": False,
            "directed_relation_representable": True,
            "static_outputs": list(static),
            "view_outputs": list(view),
        },
        "interpretation": [
            "ordered pair features are appropriate for future directed dependency evidence",
            "the same parameterization does not guarantee symmetry for quantities that are mathematically symmetric",
            "front(a,b) and front(b,a) are not constrained by construction to be complementary",
            "this is an apparatus/design fact rather than evidence that a trained model necessarily violates those relations",
        ],
        "required_before_causal_truth_training": [
            "assign an explicit algebra class to every truth family: symmetric, directed, complementary/tie-safe, or ray-global",
            "either enforce the algebra in parameterization or preregister algebra-consistency losses/courts",
            "do not retain same_drawable/cut/attachment as one undifferentiated static-output family merely for checkpoint convenience if causal truth identifies different algebras",
        ],
        "training_executed": False,
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
