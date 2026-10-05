from __future__ import annotations

"""MIRA v6 teacher23 preflight using exact decoder AST closure extraction.

The sealed historical v6 runner imports a superseded v5 training helper at module
import time. That helper supplied the historical 28-joint *court guard*, but the
sealed ArachneV6RawReadout itself has no dependency on that training module and
no fixed joint vocabulary. We therefore extract only the exact decoder closure
from the SHA-pinned runner and keep the historical guard as independently
verified provenance rather than executable inference authority.

All teacher23 qualification, checkpoint strict-load, and J=23 smoke semantics
remain owned by the v1 preflight implementation.
"""

import ast
import builtins
import math
from dataclasses import dataclass
from pathlib import Path
import types
import typing

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tools.research import materialize_teacher23_mira_query_preflight_v1 as base


ROOT_SYMBOLS = (
    "ArachneV6RawReadout",
    "RELATION_DIM",
)
HISTORICAL_EXPECTED_JOINTS = 28


def _target_names(node: ast.AST) -> tuple[str, ...]:
    out: list[str] = []

    def visit_target(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            out.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                visit_target(elt)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.append(node.name)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            visit_target(target)
    elif isinstance(node, ast.AnnAssign):
        visit_target(node.target)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            out.append(alias.asname or alias.name.split(".", 1)[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            out.append(alias.asname or alias.name)
    return tuple(out)


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _base_attributes(node: ast.AST) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "base"
    }


def _verify_historical_guard_assignment(definitions: dict[str, ast.AST]) -> None:
    node = definitions.get("EXPECTED_JOINTS")
    if node is None:
        raise RuntimeError("MIRA_V6_HISTORICAL_EXPECTED_JOINTS_ASSIGNMENT_MISSING")
    attrs = _base_attributes(node)
    if attrs != {"EXPECTED_JOINTS"}:
        raise RuntimeError(f"MIRA_V6_HISTORICAL_GUARD_ASSIGNMENT_DRIFT::{sorted(attrs)}")


def extract_exact_v6_symbol_closure(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        for name in _target_names(node):
            definitions.setdefault(name, node)

    missing_roots = [name for name in ROOT_SYMBOLS if name not in definitions]
    if missing_roots:
        raise RuntimeError(f"MIRA_V6_AST_ROOT_SYMBOL_MISSING::{missing_roots}")
    _verify_historical_guard_assignment(definitions)

    selected_nodes: set[int] = set()
    selected_symbols: set[str] = set()
    pending = list(ROOT_SYMBOLS)
    blocked_imports: set[str] = set()
    selected_base_attributes: set[str] = set()

    while pending:
        symbol = pending.pop()
        if symbol in selected_symbols:
            continue
        selected_symbols.add(symbol)
        node = definitions.get(symbol)
        if node is None:
            continue
        selected_nodes.add(id(node))
        selected_base_attributes.update(_base_attributes(node))

        for dependency in sorted(_loaded_names(node)):
            dep_node = definitions.get(dependency)
            if dep_node is None or dependency in selected_symbols:
                continue
            if isinstance(dep_node, ast.ImportFrom):
                module = str(dep_node.module or "")
                if module.startswith("tools.training"):
                    blocked_imports.add(dependency)
                    continue
            pending.append(dependency)

    if selected_base_attributes:
        raise RuntimeError(
            "MIRA_V6_DECODER_ITSELF_REFERENCES_HISTORICAL_BASE::"
            + ",".join(sorted(selected_base_attributes))
        )
    if blocked_imports:
        raise RuntimeError(
            "MIRA_V6_DECODER_SYMBOL_CLOSURE_REQUIRES_HISTORICAL_TRAINING_IMPORT::"
            + ",".join(sorted(blocked_imports))
        )

    body: list[ast.stmt] = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ]
    for node in tree.body:
        if id(node) in selected_nodes:
            body.append(node)

    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "__builtins__": builtins.__dict__,
        "np": np,
        "numpy": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "dataclass": dataclass,
        "Path": Path,
        "typing": typing,
    }
    exec(compile(module, str(path) + "::<decoder-closure>", "exec"), namespace)

    missing_after_exec = [name for name in ROOT_SYMBOLS if name not in namespace]
    if missing_after_exec:
        raise RuntimeError(f"MIRA_V6_AST_ROOT_EXEC_MISSING::{missing_after_exec}")

    return types.SimpleNamespace(
        ArachneV6RawReadout=namespace["ArachneV6RawReadout"],
        RELATION_DIM=namespace["RELATION_DIM"],
        # Historical closure expected 28 derived target joints. This value is a
        # report/provenance guard only and is deliberately not part of decoder
        # execution. The J=23 smoke below is what tests runtime cardinality.
        EXPECTED_JOINTS=HISTORICAL_EXPECTED_JOINTS,
        _ast_selected_symbols=tuple(sorted(selected_symbols)),
        _ast_selected_node_count=int(len(selected_nodes)),
        _historical_training_module_imported=False,
        _decoder_historical_base_attributes=tuple(),
        _historical_guard_source="EXPECTED_JOINTS = base.EXPECTED_JOINTS",
    )


def main(argv=None) -> int:
    # The v1 court remains the single owner of scientific assertions. Only its
    # historical whole-module import is replaced with exact inference closure
    # extraction from the already SHA-pinned runner bytes.
    base.import_exact_runner = extract_exact_v6_symbol_closure
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
