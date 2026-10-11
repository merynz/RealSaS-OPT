"""Bounded source observations of typed stage reads, without importing adapters.

Observations name requested payloads and source locations, not runtime execution.
Conditional and unresolved reads remain visible; no producer payload is invented.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ADAPTERS = "compiler.realsas_compiler_services.orchestrator.adapters."
READERS = {"stage_output_payload", "_stage_output_payload", "stage_payload"}


class ContractScanner:
    def __init__(self, root: Path):
        self.root = root
        self.modules = {}

    def module(self, name):
        if name not in self.modules:
            rel = name.replace(".", "/")
            path = next(
                (
                    p
                    for p in [
                        self.root / (rel + ".py"),
                        self.root / rel / "__init__.py",
                    ]
                    if p.is_file()
                ),
                None,
            )
            self.modules[name] = (
                (path, ast.parse(path.read_text())) if path else (None, None)
            )
        return self.modules[name]

    def resolve(self, module, symbol, seen=None):
        seen = set() if seen is None else seen
        if (module, symbol) in seen:
            return None
        seen.add((module, symbol))
        path, tree = self.module(module)
        if tree is None:
            return None
        # Python's last binding wins, including an explicit override after '*'.
        for node in reversed(tree.body):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol
            ):
                return module, node
            if isinstance(node, ast.ImportFrom):
                target = (
                    importlib.util.resolve_name(
                        "." * node.level + (node.module or ""),
                        module.rpartition(".")[0],
                    )
                    if node.level
                    else node.module
                )
                if not target:
                    continue
                for alias in node.names:
                    if alias.name == "*" or (alias.asname or alias.name) == symbol:
                        found = self.resolve(
                            target,
                            symbol if alias.name == "*" else alias.name,
                            seen.copy(),
                        )
                        if found:
                            return found
        return None

    @staticmethod
    def value(node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return tuple(ContractScanner.value(n, env) for n in node.elts)
        if isinstance(node, ast.Dict):
            return {
                ContractScanner.value(k, env): ContractScanner.value(v, env)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        if isinstance(node, ast.Subscript):
            obj = ContractScanner.value(node.value, env)
            key = ContractScanner.value(node.slice, env)
            if isinstance(obj, dict):
                return obj.get(key)
        return None

    def inspect(self, adapter, stage_ids):
        module, symbol = adapter.split(":", 1)
        reads, writes, unresolved = [], [], []
        visited = set()

        def walk(mod, sym, arguments=None):
            found = self.resolve(mod, sym)
            if found is None:
                return
            mod, fn = found
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return
            env = {}
            defaults = dict(
                zip(
                    (
                        [a.arg for a in fn.args.args[-len(fn.args.defaults) :]]
                        if fn.args.defaults
                        else []
                    ),
                    fn.args.defaults,
                )
            )
            defaults.update(
                {
                    a.arg: d
                    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)
                    if d is not None
                }
            )
            env.update({k: self.value(v, env) for k, v in defaults.items()})
            env.update(arguments or {})
            key = (mod, sym, repr(sorted(env.items())))
            if key in visited:
                return
            visited.add(key)
            path = self.module(mod)[0].relative_to(self.root).as_posix()

            def lexical(node):
                # Nested definitions are not calls. Follow named helper calls
                # separately, and retain both branches as source observations.
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                ):
                    return
                yield node
                for child in ast.iter_child_nodes(node):
                    yield from lexical(child)

            for node in (n for statement in fn.body for n in lexical(statement)):
                if isinstance(node, ast.Assign):
                    value = self.value(node.value, env)
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            env[target.id] = value
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Name
                ):
                    continue
                name = node.func.id
                location = {"path": path, "line": node.lineno, "function": fn.name}
                if name in READERS:
                    args = [self.value(n, env) for n in node.args]
                    kwargs = {
                        k.arg: self.value(k.value, env) for k in node.keywords if k.arg
                    }
                    producer = args[1] if len(args) > 1 else kwargs.get("stage_id")
                    schema = args[2] if len(args) > 2 else kwargs.get("schema")
                    if isinstance(producer, str) and isinstance(schema, str):
                        reads.append(
                            {
                                "producer": producer,
                                "schema": schema,
                                "location": location,
                                "evidence": "SOURCE_SCHEMA_READ",
                                "in_current_plan": producer in stage_ids,
                            }
                        )
                    else:
                        unresolved.append(
                            {
                                **location,
                                "expression": ast.unparse(node),
                                "reason": "DYNAMIC_STAGE_OR_SCHEMA",
                            }
                        )
                    continue
                if name in {"write_json", "_write_json"}:
                    schema_node = next(
                        (k.value for k in node.keywords if k.arg == "schema"), None
                    )
                    schema = self.value(schema_node, env)
                    if isinstance(schema, str):
                        writes.append(
                            {
                                "schema": schema,
                                "location": location,
                                "evidence": "SOURCE_JSON_WRITE",
                            }
                        )
                found_call = self.resolve(mod, name)
                if found_call and found_call[0].startswith(ADAPTERS):
                    call_mod, call_fn = found_call
                    if isinstance(call_fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = {
                            a.arg: self.value(v, env)
                            for a, v in zip(call_fn.args.args, node.args)
                        }
                        args.update(
                            {
                                k.arg: self.value(k.value, env)
                                for k in node.keywords
                                if k.arg
                            }
                        )
                        # Don't descend into generic writers/readers or runtime IO.
                        if name not in {
                            "write_ir",
                            "_write_ir",
                            "write_json",
                            "_write_json",
                            "read_json",
                            "load_file_ref",
                            "_load_file_ref",
                        }:
                            walk(call_mod, call_fn.name, args)

        walk(module, symbol)
        unique = lambda rows: list({repr(row): row for row in rows}.values())
        return {
            "reads": unique(reads),
            "observed_json_writes": unique(writes),
            "unresolved_reads": unique(unresolved),
            "analysis_scope": "Bounded adapter/helper source scan; conditional reads are included. Dynamic calls, IR writes and field use can remain unresolved. This is not runtime consumption proof.",
        }


def payload_contracts(plan, root):
    scanner = ContractScanner(root)
    ids = {s["id"] for s in plan["stages"]}
    result = {s["id"]: scanner.inspect(s["adapter"], ids) for s in plan["stages"]}
    for stage in plan["stages"]:
        allowed = set(stage["depends_on"])
        ancestors = set()
        by_id = {s["id"]: s for s in plan["stages"]}

        def visit(sid):
            if sid in ancestors or sid not in by_id:
                return
            ancestors.add(sid)
            for parent in by_id[sid]["depends_on"]:
                visit(parent)

        for parent in allowed:
            visit(parent)
        for read in result[stage["id"]]["reads"]:
            read["declared_direct_dependency"] = read["producer"] in allowed
            read["binding"] = (
                "DIRECT"
                if read["producer"] in allowed
                else "ANCESTOR" if read["producer"] in ancestors else "UNBOUND"
            )
    return result


def compare_maps(before, after):
    old = {n["id"]: n for n in before["nodes"]}
    new = {n["id"]: n for n in after["nodes"]}
    edges = lambda m: {(e["producer"], e["consumer"]) for e in m["edges"]}
    files = lambda m: {
        r["path"]: r["sha256"]
        for r in m.get("source", {}).get("fingerprinted_files", [])
    }
    old_files, new_files = files(before), files(after)
    changed_files = sorted(
        p
        for p in old_files.keys() | new_files.keys()
        if old_files.get(p) != new_files.get(p)
    )
    impact = []
    for path in changed_files:
        direct = [
            n["id"] for n in after["nodes"] if path in n.get("dependency_files", [])
        ]
        descendants = sorted(
            {
                sid
                for n in after["nodes"]
                if n["id"] in direct
                for sid in n.get("downstream_stage_ids", [])
            }
        )
        impact.append(
            {
                "path": path,
                "direct_consumers": direct,
                "descendants": descendants,
                "classification": (
                    "SOURCE_CLOSURE" if direct else "UNCLASSIFIED_OR_GLOBAL__INSPECT"
                ),
            }
        )
    changes = []
    for sid in sorted(old.keys() & new.keys()):
        fields = [
            k
            for k in [
                "adapter",
                "depends_on",
                "manifest_keys",
                "implementation_hash",
                "payload_contracts",
            ]
            if old[sid].get(k) != new[sid].get(k)
        ]
        if fields:
            changes.append({"stage": sid, "changed_fields": fields})
    return {
        "added_stages": sorted(new.keys() - old.keys()),
        "removed_stages": sorted(old.keys() - new.keys()),
        "changed_files": impact,
        "changed_stages": changes,
        "added_edges": sorted(edges(after) - edges(before)),
        "removed_edges": sorted(edges(before) - edges(after)),
        "before_source": {
            k: before.get("source", {}).get(k)
            for k in ("head_sha", "source_fingerprint_sha256")
        },
        "after_source": {
            k: after.get("source", {}).get(k)
            for k in ("head_sha", "source_fingerprint_sha256")
        },
        "scope": "Source/declared-contract difference; reuse qualification is decided by the released graph resolver.",
    }
