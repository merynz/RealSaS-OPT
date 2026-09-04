from __future__ import annotations

"""Exact global optimization for the compiler-owned canonical mechanical graph.

This module is not a second graph pipeline.  It is the mathematical resolver
used by ``mechanical_graph_hypotheses`` after proposal validation, duplicate
fusion and bounded node completion have produced a sparse candidate graph.

The normal route is a deterministic Chu-Liu/Edmonds maximum-spanning
arborescence.  A bounded MILP solve may run as a shadow/escalation proof using
the same candidate graph and constraints.  Until model calibration is locked,
the MILP result is diagnostic and cannot silently replace the arborescence.
"""

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Any, Iterable

from realsas_contracts.execution import canonical_sha256
from realsas_contracts.technical_part_graph import (
    CANONICAL_GRAPH_CANDIDATE_SCHEMA_VERSION,
    CanonicalGraphConstraint,
    CanonicalGraphEdgeCandidate,
    CanonicalGraphOptimizationRequest,
    CanonicalGraphOptimizationResult,
)


@dataclass(frozen=True)
class _Edge:
    edge_key: str
    parent: str
    child: str
    weight: float
    payload: CanonicalGraphEdgeCandidate
    # During contraction, ``original`` points to the edge in the caller's graph.
    original: "_Edge | None" = None
    enter_target: str | None = None
    leave_source: str | None = None


@dataclass(frozen=True)
class _Solve:
    root: str | None
    selected_nodes: tuple[str, ...]
    edges: tuple[_Edge, ...]
    objective: float
    feasible: bool
    status: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    optimality_proven: bool = False
    optimality_gap: float | None = None
    elapsed_seconds: float = 0.0


def _edge_order(edge: _Edge) -> tuple[float, str, str, str]:
    # Higher weight wins.  Exact ties use stable semantic IDs rather than input
    # order, preserving proposal-permutation invariance.
    return (-float(edge.weight), edge.parent, edge.child, edge.edge_key)


def _best(edges: Iterable[_Edge]) -> _Edge | None:
    rows = tuple(edges)
    if not rows:
        return None
    return sorted(rows, key=_edge_order)[0]


def _deduplicate_parallel(edges: Iterable[_Edge]) -> tuple[_Edge, ...]:
    by_pair: dict[tuple[str, str], list[_Edge]] = {}
    for edge in edges:
        if edge.parent == edge.child:
            continue
        by_pair.setdefault((edge.parent, edge.child), []).append(edge)
    return tuple(
        sorted(
            (_best(rows) for rows in by_pair.values()),
            key=lambda row: (row.parent, row.child, row.edge_key) if row is not None else ("", "", ""),
        )
    )  # type: ignore[return-value]


def _find_cycle(root: str, incoming: dict[str, _Edge]) -> tuple[str, ...]:
    for start in sorted(incoming):
        if start == root:
            continue
        cursor = start
        path: list[str] = []
        seen_at: dict[str, int] = {}
        while cursor != root and cursor in incoming:
            if cursor in seen_at:
                return tuple(path[seen_at[cursor] :])
            seen_at[cursor] = len(path)
            path.append(cursor)
            cursor = incoming[cursor].parent
    return ()


def _chu_liu_edmonds(
    *,
    nodes: tuple[str, ...],
    edges: tuple[_Edge, ...],
    root: str,
    depth: int = 0,
) -> tuple[_Edge, ...] | None:
    if root not in nodes:
        return None
    if len(nodes) <= 1:
        return ()

    incoming: dict[str, _Edge] = {}
    for node in sorted(nodes):
        if node == root:
            continue
        selected = _best(edge for edge in edges if edge.child == node and edge.parent != node)
        if selected is None:
            return None
        incoming[node] = selected

    cycle = _find_cycle(root, incoming)
    if not cycle:
        return tuple(sorted(incoming.values(), key=lambda edge: (edge.child, edge.parent, edge.edge_key)))

    cycle_set = set(cycle)
    supernode = f"__CYCLE__:{depth}:{canonical_sha256({'cycle': sorted(cycle)})[:16]}"
    contracted_nodes = tuple(sorted((set(nodes) - cycle_set) | {supernode}))
    contracted_edges: list[_Edge] = []

    for edge in edges:
        parent_in = edge.parent in cycle_set
        child_in = edge.child in cycle_set
        if parent_in and child_in:
            continue
        if not parent_in and child_in:
            # Entering a contracted cycle replaces the chosen incoming edge of
            # the target node, hence the reduced-cost adjustment.
            adjusted = float(edge.weight) - float(incoming[edge.child].weight)
            contracted_edges.append(
                _Edge(
                    edge_key=f"CONTRACT:IN:{edge.edge_key}:{supernode}",
                    parent=edge.parent,
                    child=supernode,
                    weight=adjusted,
                    payload=edge.payload,
                    original=edge,
                    enter_target=edge.child,
                )
            )
        elif parent_in and not child_in:
            contracted_edges.append(
                _Edge(
                    edge_key=f"CONTRACT:OUT:{edge.edge_key}:{supernode}",
                    parent=supernode,
                    child=edge.child,
                    weight=float(edge.weight),
                    payload=edge.payload,
                    original=edge,
                    leave_source=edge.parent,
                )
            )
        else:
            contracted_edges.append(
                _Edge(
                    edge_key=f"CONTRACT:KEEP:{edge.edge_key}:{supernode}",
                    parent=edge.parent,
                    child=edge.child,
                    weight=float(edge.weight),
                    payload=edge.payload,
                    original=edge,
                )
            )

    selected_contracted = _chu_liu_edmonds(
        nodes=contracted_nodes,
        edges=_deduplicate_parallel(contracted_edges),
        root=root,
        depth=depth + 1,
    )
    if selected_contracted is None:
        return None

    expanded: list[_Edge] = []
    entering_target: str | None = None
    for edge in selected_contracted:
        original = edge.original
        if original is None:
            return None
        if edge.child == supernode:
            entering_target = edge.enter_target
            expanded.append(original)
        elif edge.parent == supernode:
            expanded.append(original)
        else:
            expanded.append(original)

    if entering_target is None:
        # The contracted cycle cannot be the root (the root has no incoming
        # edge), so a valid arborescence must enter it exactly once.
        return None
    for node in cycle:
        if node != entering_target:
            expanded.append(incoming[node])

    return tuple(sorted(expanded, key=lambda edge: (edge.child, edge.parent, edge.edge_key)))


def _constraint_index(
    constraints: Iterable[CanonicalGraphConstraint],
) -> tuple[
    set[tuple[str, str]],
    set[tuple[str, str]],
    set[str],
    set[str],
    set[str],
    set[str],
    list[dict[str, Any]],
    list[str],
]:
    required_edges: set[tuple[str, str]] = set()
    forbidden_edges: set[tuple[str, str]] = set()
    required_roots: set[str] = set()
    forbidden_roots: set[str] = set()
    required_nodes: set[str] = set()
    forbidden_nodes: set[str] = set()
    records: list[dict[str, Any]] = []
    blockers: list[str] = []
    for constraint in sorted(constraints, key=lambda row: row.constraint_id):
        kind = str(constraint.constraint_kind).strip().lower()
        parent = None if constraint.parent_control_id is None else str(constraint.parent_control_id)
        child = None if constraint.child_control_id is None else str(constraint.child_control_id)
        records.append(
            {
                "constraint_id": constraint.constraint_id,
                "kind": kind,
                "hard": bool(constraint.hard),
                "parent_control_id": parent,
                "child_control_id": child,
                "owner": constraint.owner,
                "reason": constraint.reason,
                "penalty": float(constraint.penalty),
                "source_evidence_refs": tuple(constraint.source_evidence_refs),
            }
        )
        if not constraint.hard:
            # Soft requests are typed and preserved for calibrated objective or
            # advisory owners. Unknown soft kinds are not allowed to become
            # decorative fields that appear consumed without route influence.
            if kind == "request_completion" and child:
                continue
            blockers.append(
                f"unsupported_or_incomplete_soft_constraint:{constraint.constraint_id}:{kind}"
            )
            continue
        if kind == "require_edge" and parent and child:
            required_edges.add((parent, child))
        elif kind == "forbid_edge" and parent and child:
            forbidden_edges.add((parent, child))
        elif kind == "require_root" and child:
            required_roots.add(child)
            required_nodes.add(child)
        elif kind == "forbid_root" and child:
            forbidden_roots.add(child)
        elif kind == "require_node" and child:
            required_nodes.add(child)
        elif kind == "forbid_node" and child:
            forbidden_nodes.add(child)
        else:
            blockers.append(f"unsupported_or_incomplete_hard_constraint:{constraint.constraint_id}:{kind}")
    if required_edges & forbidden_edges:
        blockers.append("same_edge_required_and_forbidden")
    if len(required_roots) > 1:
        blockers.append("multiple_required_roots")
    if required_roots & forbidden_roots:
        blockers.append("same_root_required_and_forbidden")
    if required_nodes & forbidden_nodes:
        blockers.append("same_node_required_and_forbidden")
    for parent, child in required_edges:
        required_nodes.update((parent, child))
    return (
        required_edges,
        forbidden_edges,
        required_roots,
        forbidden_roots,
        required_nodes,
        forbidden_nodes,
        records,
        blockers,
    )


def _request_contract_validation(
    request: CanonicalGraphOptimizationRequest,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if str(request.schema_version) != CANONICAL_GRAPH_CANDIDATE_SCHEMA_VERSION:
        blockers.append(f"unsupported_candidate_graph_schema:{request.schema_version}")
    if not str(request.request_id).strip():
        blockers.append("candidate_graph_request_id_missing")
    if not str(request.attempt_id).strip():
        blockers.append("candidate_graph_attempt_id_missing")
    request_numeric = {
        "ilp_time_limit_seconds": float(request.ilp_time_limit_seconds),
        "ilp_relative_gap": float(request.ilp_relative_gap),
        "qualified_ilp_max_gap": float(request.qualified_ilp_max_gap),
    }
    for name, value in request_numeric.items():
        if not isfinite(value):
            blockers.append(f"request_nonfinite:{name}")
    if isfinite(request_numeric["ilp_time_limit_seconds"]) and request_numeric["ilp_time_limit_seconds"] <= 0.0:
        blockers.append("request_nonpositive_ilp_time_limit")
    for name in ("ilp_relative_gap", "qualified_ilp_max_gap"):
        value = request_numeric[name]
        if isfinite(value) and value < 0.0:
            blockers.append(f"request_negative:{name}")
    if int(request.minimum_selected_nodes) < 1:
        blockers.append("request_minimum_selected_nodes_below_one")
    if int(request.max_synthesized_nodes) < 0:
        blockers.append("request_negative_max_synthesized_nodes")

    node_ids = tuple(str(row.canonical_control_id) for row in request.nodes)
    node_set = set(node_ids)
    if any(not node_id.strip() for node_id in node_ids):
        blockers.append("canonical_node_candidate_id_missing")
    if len(node_set) != len(node_ids):
        blockers.append("duplicate_canonical_node_candidate_id")
    root_candidate_ids = tuple(str(node) for node in request.root_candidate_ids)
    if len(set(root_candidate_ids)) != len(root_candidate_ids):
        blockers.append("duplicate_root_candidate_id")
    edge_keys = tuple(str(row.edge_key) for row in request.edges)
    if any(not edge_key.strip() for edge_key in edge_keys):
        blockers.append("canonical_edge_candidate_key_missing")
    if len(set(edge_keys)) != len(edge_keys):
        blockers.append("duplicate_canonical_edge_candidate_key")
    constraint_ids = tuple(str(row.constraint_id) for row in request.constraints)
    if any(not constraint_id.strip() for constraint_id in constraint_ids):
        blockers.append("canonical_graph_constraint_id_missing")
    if len(set(constraint_ids)) != len(constraint_ids):
        blockers.append("duplicate_canonical_graph_constraint_id")

    unknown_roots = sorted(set(root_candidate_ids) - node_set)
    for node in unknown_roots:
        blockers.append(f"root_candidate_endpoint_missing:{node}")

    for row in request.nodes:
        node_id = str(row.canonical_control_id)
        numeric = {
            "root_score01": float(row.root_score01),
            "confidence01": float(row.confidence01),
            "uncertainty01": float(row.uncertainty01),
            "acceptance_score": float(row.acceptance_score),
            "complexity_penalty": float(row.complexity_penalty),
        }
        for name, value in numeric.items():
            if not isfinite(value):
                blockers.append(f"node_nonfinite:{node_id}:{name}")
        for name in ("root_score01", "confidence01", "uncertainty01"):
            value = numeric[name]
            if isfinite(value) and not 0.0 <= value <= 1.0:
                blockers.append(f"node_probability_out_of_range:{node_id}:{name}")
        if isfinite(numeric["complexity_penalty"]) and numeric["complexity_penalty"] < 0.0:
            blockers.append(f"node_negative_complexity_penalty:{node_id}")

    for row in request.edges:
        parent = str(row.parent_control_id)
        child = str(row.child_control_id)
        if parent not in node_set or child not in node_set:
            blockers.append(f"candidate_edge_endpoint_missing:{row.edge_key}:{parent}->{child}")
        if parent == child:
            blockers.append(f"candidate_self_edge:{row.edge_key}:{parent}")
        numeric = {
            "evidence_score01": float(row.evidence_score01),
            "objective_score": float(row.objective_score),
            "uncertainty01": float(row.uncertainty01),
            "prior_contribution01": float(row.prior_contribution01),
            "fallback_contribution01": float(row.fallback_contribution01),
        }
        for name, value in numeric.items():
            if not isfinite(value):
                blockers.append(f"edge_nonfinite:{row.edge_key}:{name}")
        for name in (
            "evidence_score01",
            "uncertainty01",
            "prior_contribution01",
            "fallback_contribution01",
        ):
            value = numeric[name]
            if isfinite(value) and not 0.0 <= value <= 1.0:
                blockers.append(f"edge_probability_out_of_range:{row.edge_key}:{name}")

    for row in request.constraints:
        penalty = float(row.penalty)
        if not isfinite(penalty):
            blockers.append(f"constraint_nonfinite_penalty:{row.constraint_id}")
        elif penalty < 0.0:
            blockers.append(f"constraint_negative_penalty:{row.constraint_id}")
        parent = None if row.parent_control_id is None else str(row.parent_control_id)
        child = None if row.child_control_id is None else str(row.child_control_id)
        missing = tuple(
            endpoint for endpoint in (parent, child)
            if endpoint is not None and endpoint not in node_set
        )
        if not missing:
            continue
        label = f"constraint_endpoint_missing:{row.constraint_id}:{','.join(sorted(set(missing)))}"
        if row.hard:
            blockers.append(label)
        else:
            warnings.append(label)
    return tuple(sorted(set(blockers))), tuple(sorted(set(warnings)))


def _prepare_edges(
    request: CanonicalGraphOptimizationRequest,
    *,
    required_edges: set[tuple[str, str]],
    forbidden_edges: set[tuple[str, str]],
) -> tuple[_Edge, ...]:
    node_ids = {row.canonical_control_id for row in request.nodes}
    rows: list[_Edge] = []
    for candidate in request.edges:
        pair = (candidate.parent_control_id, candidate.child_control_id)
        if candidate.parent_control_id not in node_ids or candidate.child_control_id not in node_ids:
            continue
        if pair in forbidden_edges or candidate.hard_forbidden:
            continue
        rows.append(
            _Edge(
                edge_key=candidate.edge_key,
                parent=candidate.parent_control_id,
                child=candidate.child_control_id,
                weight=float(candidate.objective_score),
                payload=replace(
                    candidate,
                    hard_required=bool(candidate.hard_required or pair in required_edges),
                ),
            )
        )
    return _deduplicate_parallel(rows)


def _required_edge_errors(
    *,
    nodes: set[str],
    required_edges: set[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    required_by_child: dict[str, list[str]] = {}
    for parent, child in sorted(required_edges):
        if parent not in nodes or child not in nodes:
            errors.append(f"required_edge_endpoint_missing:{parent}->{child}")
        if parent == child:
            errors.append(f"required_self_edge:{parent}")
        required_by_child.setdefault(child, []).append(parent)
    for child, parents in required_by_child.items():
        if len(parents) > 1:
            errors.append(f"multiple_required_parents:{child}")
    return errors


def _solve_arborescence(
    request: CanonicalGraphOptimizationRequest,
    *,
    edges: tuple[_Edge, ...],
    required_edges: set[tuple[str, str]],
    required_roots: set[str],
    forbidden_roots: set[str],
    required_nodes: set[str],
    forbidden_nodes: set[str],
) -> _Solve:
    """Solve root and parent tree jointly with one exact Edmonds call.

    A synthetic super-root receives one heavily penalized edge to each allowed
    root candidate.  The penalty is larger than the complete objective span of
    all real edges, so an ordinary single-root arborescence always beats any
    branching that uses two super-root edges.  This preserves the exact global
    objective while avoiding O(root_candidate_count) separate Edmonds solves.
    """

    started = perf_counter()
    node_rows = {row.canonical_control_id: row for row in request.nodes}
    all_nodes = set(node_rows)
    nodes = tuple(sorted(all_nodes - forbidden_nodes))
    node_set = set(nodes)
    blockers = _required_edge_errors(nodes=node_set, required_edges=required_edges)
    for required_node in sorted(required_nodes - node_set):
        blockers.append(f"required_node_missing_or_forbidden:{required_node}")
    minimum_selected_nodes = max(1, int(request.minimum_selected_nodes))
    if len(nodes) < minimum_selected_nodes:
        blockers.append(
            f"fixed_node_set_below_minimum:{len(nodes)}<{minimum_selected_nodes}"
        )
    synthesized_count = sum(bool(node_rows[node].synthesized) for node in nodes)
    if synthesized_count > max(0, int(request.max_synthesized_nodes)):
        blockers.append(
            "fixed_node_set_exceeds_synthesized_budget:"
            f"{synthesized_count}>{max(0, int(request.max_synthesized_nodes))}"
        )
    exclusive_groups: dict[str, list[str]] = {}
    for node in nodes:
        group = node_rows[node].exclusive_group_id
        if group:
            exclusive_groups.setdefault(str(group), []).append(node)
    for group, members in sorted(exclusive_groups.items()):
        if len(members) > 1:
            blockers.append(
                f"fixed_node_set_exclusive_group_conflict:{group}:{','.join(sorted(members))}"
            )
    if len(required_roots) > 1:
        blockers.append("multiple_required_roots_for_single_root_graph")
    if blockers:
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "invalid_constraints",
            tuple(blockers),
            elapsed_seconds=perf_counter() - started,
        )

    root_scores = {row.canonical_control_id: float(row.root_score01) for row in request.nodes}
    roots = set(request.root_candidate_ids or nodes) & node_set
    if required_roots:
        roots &= required_roots
    roots -= forbidden_roots
    required_by_child = {child: parent for parent, child in required_edges}
    roots -= set(required_by_child)
    if not roots:
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "no_root_candidate",
            ("no_root_candidate",),
            elapsed_seconds=perf_counter() - started,
        )

    usable_edges: list[_Edge] = []
    for edge in edges:
        if edge.parent not in node_set or edge.child not in node_set:
            continue
        required_parent = required_by_child.get(edge.child)
        if required_parent is not None and edge.parent != required_parent:
            continue
        usable_edges.append(edge)

    for node in nodes:
        if node in roots:
            continue
        if not any(edge.child == node for edge in usable_edges):
            return _Solve(
                None,
                (),
                (),
                float("-inf"),
                False,
                "arborescence_infeasible",
                (f"node_has_no_incoming_candidate:{node}",),
                elapsed_seconds=perf_counter() - started,
            )

    super_root = "__REALSAS_CANONICAL_SUPER_ROOT__"
    while super_root in node_set:
        super_root += "_"
    # Any additional super-root edge must lose more than the entire possible
    # real-edge/root-score objective span.  Therefore, when a connected
    # single-root tree exists, exactly one super-root edge is globally optimal.
    objective_span = (
        sum(abs(float(edge.weight)) for edge in usable_edges)
        + sum(abs(float(root_scores.get(root, 0.0))) for root in roots)
        + 1.0
    )
    root_penalty = 2.0 * objective_span + 1.0
    super_edges: list[_Edge] = []
    for root in sorted(roots):
        edge_key = f"SUPER_ROOT>{root}"
        payload = CanonicalGraphEdgeCandidate(
            edge_key=edge_key,
            parent_control_id=super_root,
            child_control_id=root,
            evidence_score01=float(root_scores.get(root, 0.0)),
            authority_tier="compiler_global_root_selection",
            objective_score=float(root_scores.get(root, 0.0)) - root_penalty,
            uncertainty01=0.0,
            reason="joint_single_root_selection",
            metadata={"synthetic_optimizer_edge": True},
        )
        super_edges.append(
            _Edge(
                edge_key=edge_key,
                parent=super_root,
                child=root,
                weight=float(root_scores.get(root, 0.0)) - root_penalty,
                payload=payload,
            )
        )

    selected = _chu_liu_edmonds(
        nodes=tuple(sorted((*nodes, super_root))),
        edges=_deduplicate_parallel((*usable_edges, *super_edges)),
        root=super_root,
    )
    if selected is None:
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "arborescence_infeasible",
            ("arborescence_infeasible",),
            elapsed_seconds=perf_counter() - started,
        )
    selected_super_edges = tuple(edge for edge in selected if edge.parent == super_root)
    if len(selected_super_edges) != 1:
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "single_root_connectivity_infeasible",
            (f"selected_super_root_edge_count:{len(selected_super_edges)}",),
            elapsed_seconds=perf_counter() - started,
        )
    selected_root = selected_super_edges[0].child
    selected_real_edges = tuple(
        sorted(
            (edge for edge in selected if edge.parent != super_root),
            key=lambda edge: (edge.child, edge.parent, edge.edge_key),
        )
    )
    selected_pairs = {(edge.parent, edge.child) for edge in selected_real_edges}
    if not required_edges.issubset(selected_pairs):
        missing = tuple(sorted(required_edges - selected_pairs))
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "required_edge_missing",
            tuple(f"required_edge_missing:{parent}->{child}" for parent, child in missing),
            elapsed_seconds=perf_counter() - started,
        )
    if len(selected_real_edges) != max(0, len(nodes) - 1):
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "edge_cardinality_mismatch",
            ("edge_cardinality_mismatch",),
            elapsed_seconds=perf_counter() - started,
        )
    objective = (
        sum(float(edge.weight) for edge in selected_real_edges)
        + float(root_scores.get(selected_root, 0.0))
    )
    return _Solve(
        root=selected_root,
        selected_nodes=nodes,
        edges=selected_real_edges,
        objective=objective,
        feasible=True,
        status="optimal_arborescence_super_root",
        optimality_proven=True,
        optimality_gap=0.0,
        elapsed_seconds=perf_counter() - started,
    )


def _verify_tree(
    *,
    nodes: tuple[str, ...],
    root: str | None,
    edges: tuple[_Edge, ...],
    required_edges: set[tuple[str, str]],
    forbidden_edges: set[tuple[str, str]],
) -> tuple[bool, tuple[str, ...], dict[str, str | None]]:
    blockers: list[str] = []
    node_set = set(nodes)
    if root is None or root not in node_set:
        blockers.append("root_missing")
    parent_by_child: dict[str, str | None] = {root: None} if root is not None else {}
    for edge in edges:
        if edge.parent not in node_set or edge.child not in node_set:
            blockers.append(f"edge_endpoint_missing:{edge.edge_key}")
            continue
        if edge.child in parent_by_child:
            blockers.append(f"multiple_parents:{edge.child}")
        parent_by_child[edge.child] = edge.parent
        if (edge.parent, edge.child) in forbidden_edges:
            blockers.append(f"forbidden_edge_selected:{edge.parent}->{edge.child}")
    if len(edges) != max(0, len(nodes) - 1):
        blockers.append("edge_cardinality_mismatch")
    for node in nodes:
        if node not in parent_by_child:
            blockers.append(f"node_without_parent_or_root:{node}")
    selected_pairs = {(edge.parent, edge.child) for edge in edges}
    for pair in sorted(required_edges - selected_pairs):
        blockers.append(f"required_edge_missing:{pair[0]}->{pair[1]}")
    # Reachability and cycle proof.
    if root is not None:
        for node in nodes:
            cursor = node
            seen: set[str] = set()
            while cursor != root:
                if cursor in seen:
                    blockers.append(f"cycle_detected:{node}")
                    break
                seen.add(cursor)
                parent = parent_by_child.get(cursor)
                if parent is None:
                    blockers.append(f"not_connected_to_root:{node}")
                    break
                cursor = parent
    return not blockers, tuple(sorted(set(blockers))), parent_by_child


def _solve_milp(
    request: CanonicalGraphOptimizationRequest,
    *,
    edges: tuple[_Edge, ...],
    required_edges: set[tuple[str, str]],
    forbidden_edges: set[tuple[str, str]],
    required_roots: set[str],
    forbidden_roots: set[str],
    required_nodes: set[str],
    forbidden_nodes: set[str],
) -> _Solve:
    started = perf_counter()
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except Exception as exc:  # pragma: no cover - environment-dependent
        return _Solve(
            None,
            (),
            (),
            float("-inf"),
            False,
            "solver_unavailable",
            blockers=(f"scipy_milp_unavailable:{type(exc).__name__}",),
            elapsed_seconds=perf_counter() - started,
        )

    node_rows = {row.canonical_control_id: row for row in request.nodes}
    nodes = tuple(sorted(node_rows))
    node_set = set(nodes)
    allowed_root_candidates = set(request.root_candidate_ids or nodes) & node_set
    errors = _required_edge_errors(nodes=node_set, required_edges=required_edges)
    for required_node in sorted(required_nodes - node_set):
        errors.append(f"required_node_missing:{required_node}")
    for required_root in sorted(required_roots - allowed_root_candidates):
        errors.append(f"required_root_not_in_candidate_set:{required_root}")
    if not (allowed_root_candidates - forbidden_roots - forbidden_nodes):
        errors.append("no_allowed_root_candidate")
    if errors:
        return _Solve(None, (), (), float("-inf"), False, "invalid_constraints", tuple(errors), elapsed_seconds=perf_counter() - started)
    if not nodes:
        return _Solve(None, (), (), float("-inf"), False, "empty_graph", ("empty_graph",), elapsed_seconds=perf_counter() - started)

    normal_edges = tuple(sorted(edges, key=lambda edge: edge.edge_key))
    root_arcs = tuple((f"ROOT::{node}", node, float(node_rows[node].root_score01)) for node in nodes)
    arc_count = len(normal_edges) + len(root_arcs)
    x_offset = 0
    y_offset = arc_count
    flow_offset = y_offset + len(nodes)
    variable_count = flow_offset + arc_count

    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    upper[flow_offset:] = float(len(nodes))
    integrality = np.zeros(variable_count, dtype=int)
    integrality[x_offset:y_offset] = 1
    integrality[y_offset:flow_offset] = 1
    objective = np.zeros(variable_count, dtype=float)

    node_index = {node: index for index, node in enumerate(nodes)}
    edge_index_by_pair = {(edge.parent, edge.child): index for index, edge in enumerate(normal_edges)}

    for index, edge in enumerate(normal_edges):
        objective[x_offset + index] = -float(edge.weight)
        pair = (edge.parent, edge.child)
        if pair in forbidden_edges or edge.payload.hard_forbidden:
            upper[x_offset + index] = 0.0
        if pair in required_edges or edge.payload.hard_required:
            lower[x_offset + index] = 1.0
            upper[x_offset + index] = 1.0

    root_offset = len(normal_edges)
    for offset, (_, node, score) in enumerate(root_arcs):
        index = x_offset + root_offset + offset
        objective[index] = -float(score)
        if node not in allowed_root_candidates or node in forbidden_roots or node in forbidden_nodes:
            upper[index] = 0.0
        if node in required_roots:
            lower[index] = 1.0
            upper[index] = 1.0

    for node, row in node_rows.items():
        index = y_offset + node_index[node]
        net_acceptance = float(row.acceptance_score) - float(row.complexity_penalty)
        objective[index] = -net_acceptance
        if row.required or node in required_nodes:
            lower[index] = 1.0
            upper[index] = 1.0
        if node in forbidden_nodes:
            lower[index] = 0.0
            upper[index] = 0.0

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    lb: list[float] = []
    ub: list[float] = []

    def add_row(coefficients: dict[int, float], lower_bound: float, upper_bound: float) -> None:
        row_index = len(lb)
        for column, value in coefficients.items():
            row_indices.append(row_index)
            col_indices.append(column)
            data.append(float(value))
        lb.append(float(lower_bound))
        ub.append(float(upper_bound))

    # A selected node has exactly one incoming structural arc; an unselected
    # optional node has none.  One selected node receives the super-root arc.
    for node in nodes:
        coeffs: dict[int, float] = {y_offset + node_index[node]: -1.0}
        for index, edge in enumerate(normal_edges):
            if edge.child == node:
                coeffs[x_offset + index] = 1.0
        coeffs[x_offset + root_offset + node_index[node]] = 1.0
        add_row(coeffs, 0.0, 0.0)
    add_row({x_offset + root_offset + index: 1.0 for index in range(len(nodes))}, 1.0, 1.0)

    # An edge may be selected only when both endpoints survive node selection.
    for index, edge in enumerate(normal_edges):
        add_row(
            {x_offset + index: 1.0, y_offset + node_index[edge.parent]: -1.0},
            -np.inf,
            0.0,
        )
        add_row(
            {x_offset + index: 1.0, y_offset + node_index[edge.child]: -1.0},
            -np.inf,
            0.0,
        )
    for index, (_, node, _) in enumerate(root_arcs):
        add_row(
            {x_offset + root_offset + index: 1.0, y_offset + node_index[node]: -1.0},
            -np.inf,
            0.0,
        )

    # Rooted single-commodity flow proves connectivity and acyclicity for the
    # selected subset.  Each selected node consumes one unit of flow.
    for node in nodes:
        coeffs: dict[int, float] = {y_offset + node_index[node]: -1.0}
        for index, edge in enumerate(normal_edges):
            flow_index = flow_offset + index
            if edge.child == node:
                coeffs[flow_index] = coeffs.get(flow_index, 0.0) + 1.0
            if edge.parent == node:
                coeffs[flow_index] = coeffs.get(flow_index, 0.0) - 1.0
        root_flow_index = flow_offset + root_offset + node_index[node]
        coeffs[root_flow_index] = coeffs.get(root_flow_index, 0.0) + 1.0
        add_row(coeffs, 0.0, 0.0)
    super_coeffs = {
        flow_offset + root_offset + index: 1.0 for index in range(len(nodes))
    }
    for node in nodes:
        super_coeffs[y_offset + node_index[node]] = -1.0
    add_row(super_coeffs, 0.0, 0.0)

    capacity = float(len(nodes))
    for index in range(arc_count):
        add_row(
            {flow_offset + index: 1.0, x_offset + index: -capacity},
            -np.inf,
            0.0,
        )

    # Bounded optional-node and completion decisions are the NP-hard escalation
    # scope.  High-confidence observed nodes remain fixed by ``required``.
    add_row(
        {y_offset + index: 1.0 for index in range(len(nodes))},
        float(max(1, min(len(nodes), int(request.minimum_selected_nodes)))),
        np.inf,
    )
    synthesized_indices = [
        y_offset + node_index[node]
        for node, row in node_rows.items()
        if row.synthesized
    ]
    if synthesized_indices:
        add_row(
            {index: 1.0 for index in synthesized_indices},
            -np.inf,
            float(max(0, int(request.max_synthesized_nodes))),
        )
    exclusive_groups: dict[str, list[int]] = {}
    for node, row in node_rows.items():
        if row.exclusive_group_id:
            exclusive_groups.setdefault(str(row.exclusive_group_id), []).append(y_offset + node_index[node])
    for indices in exclusive_groups.values():
        if len(indices) > 1:
            add_row({index: 1.0 for index in indices}, -np.inf, 1.0)

    matrix = coo_matrix((data, (row_indices, col_indices)), shape=(len(lb), variable_count)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lb), np.asarray(ub))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={
            "time_limit": max(0.01, float(request.ilp_time_limit_seconds)),
            "mip_rel_gap": max(0.0, float(request.ilp_relative_gap)),
            "presolve": True,
            "disp": False,
        },
    )

    selected_edges: list[_Edge] = []
    selected_nodes: list[str] = []
    selected_root: str | None = None
    if getattr(result, "x", None) is not None:
        values = result.x
        for node in nodes:
            if float(values[y_offset + node_index[node]]) >= 0.5:
                selected_nodes.append(node)
        for index, edge in enumerate(normal_edges):
            if float(values[x_offset + index]) >= 0.5:
                selected_edges.append(edge)
        for offset, (_, node, _) in enumerate(root_arcs):
            if float(values[x_offset + root_offset + offset]) >= 0.5:
                selected_root = node
                break

    selected_node_tuple = tuple(sorted(selected_nodes))
    selected_node_set = set(selected_node_tuple)
    selected_required_edges = {
        pair for pair in required_edges if pair[0] in selected_node_set and pair[1] in selected_node_set
    }
    verified, verification_blockers, _ = _verify_tree(
        nodes=selected_node_tuple,
        root=selected_root,
        edges=tuple(selected_edges),
        required_edges=selected_required_edges,
        forbidden_edges=forbidden_edges,
    )
    for required_node in sorted(required_nodes - selected_node_set):
        verification_blockers = tuple(sorted(set(verification_blockers + (f"required_node_not_selected:{required_node}",))))
        verified = False

    status_code = int(getattr(result, "status", 4))
    gap_raw = getattr(result, "mip_gap", None)
    gap = None if gap_raw is None else float(gap_raw)
    if status_code == 0 and verified:
        status = "optimal"
        optimality = True
    elif verified:
        status = "feasible_with_gap"
        optimality = False
    elif status_code == 2:
        status = "infeasible"
        optimality = False
    elif status_code == 1:
        status = "timeout_no_certified_solution"
        optimality = False
    else:
        status = "solver_failure"
        optimality = False

    objective_value = float("-inf")
    if verified:
        objective_value = (
            sum(float(edge.weight) for edge in selected_edges)
            + float(node_rows[selected_root].root_score01 if selected_root else 0.0)
            + sum(
                float(node_rows[node].acceptance_score) - float(node_rows[node].complexity_penalty)
                for node in selected_node_tuple
            )
        )
    blockers = verification_blockers if not verified else ()
    if not verified and status == "infeasible":
        blockers = tuple(sorted(set(blockers + ("milp_infeasible",))))
    return _Solve(
        root=selected_root,
        selected_nodes=selected_node_tuple,
        edges=tuple(sorted(selected_edges, key=lambda edge: (edge.child, edge.parent, edge.edge_key))),
        objective=objective_value,
        feasible=verified,
        status=status,
        blockers=blockers,
        optimality_proven=optimality,
        optimality_gap=gap,
        elapsed_seconds=perf_counter() - started,
    )


def build_dynamic_graph_constraints_v18_98(
    *,
    mechanical_graph_ir: Any,
    signature_records: Iterable[dict[str, Any]],
) -> tuple[CanonicalGraphConstraint, ...]:
    """Translate qualified runtime findings into typed next-revision cuts.

    Runtime remains a validation/diagnosis owner.  It cannot mutate the graph
    directly; it can only emit constraints that the same canonical graph owner
    may consume on a subsequent revision.
    """

    edge_by_child = {
        str(getattr(edge, "child_control_id", "")): edge
        for edge in tuple(getattr(mechanical_graph_ir, "edges", ()) or ())
        if getattr(edge, "child_control_id", None)
    }
    constraints: list[CanonicalGraphConstraint] = []
    seen: set[str] = set()
    hard_forbid_causes = {
        "wrong_parent_hypothesis",
        "rig_parent_mismatch",
        "cross_component_attachment",
        "edge_exits_foreground",
    }
    completion_causes = {
        "missing_intermediate_control",
        "insufficient_support",
        "deformation_discontinuity",
    }
    for record in sorted(signature_records, key=lambda row: str(row.get("signature_id", ""))):
        signature_id = str(record.get("signature_id", ""))
        diagnosis = dict(record.get("diagnosis", {}) or {})
        attribution = dict(record.get("owner_attribution", {}) or {})
        cause = str(diagnosis.get("selected_cause_code") or record.get("failure_family") or "")
        owner = str(attribution.get("selected_owner_id") or diagnosis.get("selected_owner_id") or "")
        confidence = max(
            float(diagnosis.get("confidence01", 0.0) or 0.0),
            float(attribution.get("confidence01", 0.0) or 0.0),
        )
        canonical_ids = tuple(sorted(str(value) for value in tuple(record.get("canonical_control_ids", ()) or ()) if value))
        if confidence < 0.70 or owner not in {"realsas_rig", "realsas_synthesis", "rig", "compiler"}:
            continue
        if cause in hard_forbid_causes:
            for child_id in canonical_ids:
                edge = edge_by_child.get(child_id)
                parent_id = None if edge is None else getattr(edge, "parent_control_id", None)
                if not parent_id:
                    continue
                constraint_id = f"DYNAMIC_CUT:{canonical_sha256({'signature': signature_id, 'cause': cause, 'parent': parent_id, 'child': child_id})[:22]}"
                if constraint_id in seen:
                    continue
                seen.add(constraint_id)
                constraints.append(
                    CanonicalGraphConstraint(
                        constraint_id=constraint_id,
                        constraint_kind="forbid_edge",
                        parent_control_id=str(parent_id),
                        child_control_id=child_id,
                        hard=True,
                        owner="runtime_dynamic_witness",
                        reason=cause,
                        source_evidence_refs=(signature_id,),
                        metadata={
                            "confidence01": confidence,
                            "next_revision_only": True,
                            "runtime_cannot_mutate_graph_directly": True,
                        },
                    )
                )
        elif cause in completion_causes:
            for child_id in canonical_ids:
                constraint_id = f"DYNAMIC_HINT:{canonical_sha256({'signature': signature_id, 'cause': cause, 'child': child_id})[:22]}"
                if constraint_id in seen:
                    continue
                seen.add(constraint_id)
                constraints.append(
                    CanonicalGraphConstraint(
                        constraint_id=constraint_id,
                        constraint_kind="request_completion",
                        child_control_id=child_id,
                        hard=False,
                        penalty=confidence,
                        owner="runtime_dynamic_witness",
                        reason=cause,
                        source_evidence_refs=(signature_id,),
                        metadata={
                            "confidence01": confidence,
                            "next_revision_only": True,
                            "requires_bounded_completion_candidate": True,
                        },
                    )
                )
    return tuple(constraints)

def optimize_canonical_graph_v18_98(
    request: CanonicalGraphOptimizationRequest,
) -> CanonicalGraphOptimizationResult:
    """Resolve one sparse canonical candidate graph with exact global methods."""

    started = perf_counter()
    (
        required_edges,
        forbidden_edges,
        required_roots,
        forbidden_roots,
        required_nodes,
        forbidden_nodes,
        constraint_records,
        constraint_blockers,
    ) = _constraint_index(request.constraints)
    candidate_required_edges = {
        (row.parent_control_id, row.child_control_id)
        for row in request.edges
        if row.hard_required
    }
    candidate_forbidden_edges = {
        (row.parent_control_id, row.child_control_id)
        for row in request.edges
        if row.hard_forbidden
    }
    required_edges.update(candidate_required_edges)
    forbidden_edges.update(candidate_forbidden_edges)
    required_nodes.update(
        row.canonical_control_id for row in request.nodes if row.required
    )
    required_nodes.update(
        endpoint for pair in required_edges for endpoint in pair
    )
    contract_blockers, contract_warnings = _request_contract_validation(request)
    blockers = [*constraint_blockers, *contract_blockers]
    warnings: list[str] = list(contract_warnings)
    if required_edges & forbidden_edges:
        blockers.append("same_edge_required_and_forbidden")
    if required_nodes & forbidden_nodes:
        blockers.append("same_node_required_and_forbidden")
    if candidate_required_edges:
        constraint_records.append(
            {
                "decision": "candidate_hard_required_edges_consumed",
                "edge_pairs": tuple(sorted(candidate_required_edges)),
                "owner": "canonical_candidate_graph_contract",
            }
        )
    if candidate_forbidden_edges:
        constraint_records.append(
            {
                "decision": "candidate_hard_forbidden_edges_consumed",
                "edge_pairs": tuple(sorted(candidate_forbidden_edges)),
                "owner": "canonical_candidate_graph_contract",
            }
        )
    edges = _prepare_edges(
        request,
        required_edges=required_edges,
        forbidden_edges=forbidden_edges,
    )
    node_ids = tuple(sorted(row.canonical_control_id for row in request.nodes))
    if not node_ids:
        blockers.append("canonical_candidate_graph_empty")
    if not edges and len(node_ids) > 1:
        warnings.append("candidate_edge_pool_empty_normal_route_may_be_infeasible")

    if blockers:
        return CanonicalGraphOptimizationResult(
            result_id=f"GRAPH_OPT:{canonical_sha256({'request': request.request_id, 'blockers': sorted(blockers)})[:24]}",
            request_id=request.request_id,
            selected_root_control_id=None,
            selected_node_ids=(),
            selected_edge_keys=(),
            parent_by_child={},
            solver="none",
            status="invalid_request",
            objective_value=float("-inf"),
            optimality_proven=False,
            feasible=False,
            elapsed_seconds=perf_counter() - started,
            blockers=tuple(sorted(set(blockers))),
            decision_records=tuple(constraint_records),
            metadata={"single_canonical_graph_owner": True},
        )

    msa = _solve_arborescence(
        request,
        edges=edges,
        required_edges=required_edges,
        required_roots=required_roots,
        forbidden_roots=forbidden_roots,
        required_nodes=required_nodes,
        forbidden_nodes=forbidden_nodes,
    )
    msa_verified, msa_verification_blockers, parent_by_child = _verify_tree(
        nodes=msa.selected_nodes,
        root=msa.root,
        edges=msa.edges,
        required_edges=required_edges,
        forbidden_edges=forbidden_edges,
    )
    msa_verified = bool(msa.feasible and msa_verified)
    msa_failure_blockers = tuple(
        sorted(set((*msa.blockers, *msa_verification_blockers)))
    )

    ilp = _Solve(None, (), (), float("-inf"), False, "not_run")
    if request.run_ilp_shadow or (request.ilp_authority_enabled and request.calibration_locked):
        ilp = _solve_milp(
            request,
            edges=edges,
            required_edges=required_edges,
            forbidden_edges=forbidden_edges,
            required_roots=required_roots,
            forbidden_roots=forbidden_roots,
            required_nodes=required_nodes,
            forbidden_nodes=forbidden_nodes,
        )
        if not ilp.feasible:
            warnings.append(f"ilp_shadow_not_feasible:{ilp.status}")

    selected = msa
    selected_verified = msa_verified
    solver = "chu_liu_edmonds_maximum_spanning_arborescence"
    if request.ilp_authority_enabled:
        if not request.calibration_locked:
            warnings.append("ilp_authority_requested_without_locked_calibration_diagnostic_only")
        elif ilp.feasible and (
            ilp.optimality_proven
            or (
                float(request.qualified_ilp_max_gap) > 0.0
                and ilp.optimality_gap is not None
                and float(ilp.optimality_gap) <= float(request.qualified_ilp_max_gap)
            )
        ):
            if (
                set(ilp.selected_nodes) != set(msa.selected_nodes)
                and not request.ilp_node_selection_authority_enabled
            ):
                warnings.append("ilp_node_selection_result_diagnostic_only")
            else:
                selected = ilp
                solver = "scipy_highs_milp_escalation"
                if not msa_verified:
                    warnings.append("normal_arborescence_infeasible_resolved_by_qualified_ilp")
            selected_verified, selected_blockers, parent_by_child = _verify_tree(
                nodes=selected.selected_nodes,
                root=selected.root,
                edges=selected.edges,
                required_edges=required_edges,
                forbidden_edges=forbidden_edges,
            )
            selected_verified = bool(selected.feasible and selected_verified)
            if not selected_verified:
                blockers.extend(selected.blockers)
                blockers.extend(selected_blockers)
        elif request.calibration_locked:
            warnings.append("ilp_escalation_did_not_meet_qualified_optimality_gap_using_arborescence")

    if selected is msa and not selected_verified:
        blockers.extend(msa_failure_blockers)

    agreement = None
    if msa.feasible and ilp.feasible:
        agreement = (
            msa.root == ilp.root
            and set(msa.selected_nodes) == set(ilp.selected_nodes)
            and {edge.edge_key for edge in msa.edges} == {edge.edge_key for edge in ilp.edges}
        )
        if not agreement:
            warnings.append("arborescence_ilp_shadow_disagreement")

    selected_node_ids = tuple(sorted(selected.selected_nodes))
    selected_edge_keys = tuple(sorted(edge.edge_key for edge in selected.edges))
    result_identity = {
        "request_id": request.request_id,
        "root": selected.root,
        "selected_node_ids": selected_node_ids,
        "selected_edge_keys": selected_edge_keys,
        "solver": solver,
        "status": selected.status,
    }
    records = list(constraint_records)
    records.append(
        {
            "decision": "global_graph_optimization",
            "normal_solver": "chu_liu_edmonds",
            "normal_status": msa.status,
            "normal_objective": msa.objective,
            "ilp_shadow_status": ilp.status,
            "ilp_shadow_objective": None if not ilp.feasible else ilp.objective,
            "ilp_shadow_selected_node_ids": ilp.selected_nodes,
            "ilp_shadow_agrees": agreement,
            "ilp_authority_enabled": bool(request.ilp_authority_enabled),
            "calibration_locked": bool(request.calibration_locked),
            "selected_solver": solver,
            "candidate_node_count": len(node_ids),
            "candidate_edge_count": len(edges),
        }
    )
    return CanonicalGraphOptimizationResult(
        result_id=f"GRAPH_OPT:{canonical_sha256(result_identity)[:24]}",
        request_id=request.request_id,
        selected_root_control_id=selected.root,
        selected_node_ids=selected_node_ids,
        selected_edge_keys=selected_edge_keys,
        parent_by_child={key: value for key, value in sorted(parent_by_child.items())},
        solver=solver,
        status=selected.status,
        objective_value=float(selected.objective),
        optimality_proven=bool(selected.optimality_proven),
        feasible=bool(selected.feasible and not blockers),
        ilp_shadow_status=ilp.status,
        ilp_shadow_objective_value=None if not ilp.feasible else float(ilp.objective),
        ilp_shadow_selected_node_ids=tuple(sorted(ilp.selected_nodes)),
        ilp_shadow_selected_edge_keys=tuple(sorted(edge.edge_key for edge in ilp.edges)),
        ilp_shadow_agrees=agreement,
        optimality_gap=selected.optimality_gap,
        elapsed_seconds=perf_counter() - started,
        blockers=tuple(sorted(set(blockers))),
        warnings=tuple(sorted(set(warnings))),
        decision_records=tuple(records),
        metadata={
            "single_canonical_graph_owner": True,
            "normal_route_polynomial": True,
            "normal_route_exact_for_fixed_candidate_graph": True,
            "ilp_is_bounded_shadow_or_escalation": True,
            "ilp_product_authority_requires_locked_calibration": True,
            "ilp_node_selection_authority_enabled": bool(request.ilp_node_selection_authority_enabled),
            "optional_node_selection_is_np_hard_escalation_scope": True,
            "raw_proposal_order_consumed": False,
        },
    )
