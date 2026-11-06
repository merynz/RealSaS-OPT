"""Finite capability-relative representation decisions; never artist ownership.

This module has no source names, anatomy, render targets or per-subject rules.
Evidence, required capabilities and ranking policy are separate inputs.
"""
from __future__ import annotations

from itertools import product
from collections.abc import Mapping, Sequence
import math

STATES = frozenset(('SATISFIED', 'VIOLATED', 'UNKNOWN'))
AUTHORITY = {k: False for k in ('graphics_owner_authorized',
    'iris_training_authorized', 'tessa_training_authorized',
    'materialization_authorized', 'training_authorized')}


def partitions(n: int):
    """All restricted-growth labelings, deterministic and label invariant."""
    if not isinstance(n, int) or n < 1 or n > 9:
        raise ValueError('bounded nonempty address domain required')
    def extend(prefix):
        if len(prefix) == n:
            yield tuple(prefix)
        else:
            for label in range(max(prefix) + 2):
                yield from extend(prefix + [label])
    yield from extend([0])


def reachable_states(groups: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Independent Boolean controls; addresses in each group are tied."""
    g = tuple(groups)
    if not g or any(type(x) is not int or x < 0 for x in g):
        raise ValueError('invalid control group')
    labels = sorted(set(g))
    if len(labels) > 16:
        raise ValueError('finite enumeration budget exceeded')
    return tuple(sorted(tuple(s[labels.index(i)] for i in g)
        for s in product((0, 1), repeat=len(labels))))


def compile_edit(groups, request):
    """Compile without observing a target image. Return a conflict witness."""
    g, q = tuple(groups), tuple(request)
    reachable_states(g)  # validate
    if len(q) != len(g) or any(type(x) is not int or x not in (0, 1) for x in q):
        raise ValueError('request must be one bit per address')
    assignment = {}
    for i, (control, bit) in enumerate(zip(g, q)):
        if control in assignment and assignment[control][0] != bit:
            return {'status': 'VIOLATED', 'conflicting_addresses':
                [assignment[control][1], i], 'control': control}
        assignment[control] = (bit, i)
    return {'status': 'SATISFIED', 'controls':
        {str(k): v[0] for k, v in sorted(assignment.items())},
        'result_address_state': list(q)}


def feasible_total_orders(precedence: Sequence[Sequence[int]], n: int):
    """Enumerate topological orders; a cycle makes the flat class infeasible."""
    if type(n) is not int or not 1 <= n <= 9:
        raise ValueError('bounded layer domain required')
    edges = set()
    for edge in precedence:
        if len(edge) != 2:
            raise ValueError('precedence must be ordered pairs')
        a, b = edge
        if any(type(x) is not int or not 0 <= x < n for x in (a, b)) or a == b:
            raise ValueError('invalid precedence')
        edges.add((a, b))
    def extend(prefix, remaining):
        if not remaining:
            yield tuple(prefix)
        else:
            for node in sorted(remaining):
                if not any(b == node and a in remaining for a, b in edges):
                    yield from extend(prefix + [node], remaining - {node})
    return tuple(extend([], set(range(n))))


def compatible_pairs(max_errors, tolerance: float):
    """A graph, NOT transitive clustering. None denotes unobserved comparison."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('invalid tolerance')
    n = len(max_errors)
    if any(len(row) != n for row in max_errors):
        raise ValueError('pair table must be square')
    result = []
    for a in range(n):
        for b in range(a + 1, n):
            x, y = max_errors[a][b], max_errors[b][a]
            if x != y:
                raise ValueError('comparison table must be symmetric')
            if x is not None and (not math.isfinite(x) or x < 0):
                raise ValueError('invalid comparison error')
            result.append({'a': a, 'b': b, 'status': 'UNKNOWN' if x is None
                else ('COMPATIBLE' if x <= tolerance else 'DISTINGUISHED')})
    return result


def decide(candidates: Sequence[Mapping], required: Sequence[str],
           cost_axes: Sequence[str] = ()):
    """Three-valued finite decision with explicit optional Pareto policy.

    Each candidate supplies independently established per-capability verdicts.
    This function is a decision layer, not evidence authentication. The caller
    must bind its evidence/court identity (the physical runner does so).
    UNKNOWN never enters the qualified Pareto frontier. Equal costs retain ties.
    """
    if not required or len(set(required)) != len(required):
        raise ValueError('unique nonempty required capabilities needed')
    if len(set(cost_axes)) != len(cost_axes):
        raise ValueError('duplicate cost axis')
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate identity')
    rows = []
    for c in candidates:
        evidence = c.get('capabilities', {})
        if any(v not in STATES for v in evidence.values()):
            raise ValueError('invalid capability verdict')
        verdicts = {k: evidence.get(k, 'UNKNOWN') for k in required}
        state = ('VIOLATED' if 'VIOLATED' in verdicts.values() else
                 'UNKNOWN' if 'UNKNOWN' in verdicts.values() else 'SATISFIED')
        costs = dict(c.get('costs', {}))
        for a in cost_axes:
            if a not in costs or type(costs[a]) not in (int, float) or \
                    not math.isfinite(costs[a]) or costs[a] < 0:
                raise ValueError('policy requires explicit finite costs')
        rows.append({'id': c['id'], 'status': state,
                     'capabilities': verdicts, 'costs': costs})
    admitted = [r for r in rows if r['status'] == 'SATISFIED']
    frontier = []
    for r in admitted:
        dominated = bool(cost_axes) and any(
            all(other['costs'][a] <= r['costs'][a] for a in cost_axes) and
            any(other['costs'][a] < r['costs'][a] for a in cost_axes)
            for other in admitted)
        if not dominated:
            frontier.append(r['id'])
    return {'schema': 'RealSaS.DrawableCapabilityDecision.v2',
        'candidate_decisions': rows, 'satisfying_alternatives': [r['id'] for r in admitted],
        'unresolved_alternatives': [r['id'] for r in rows if r['status'] == 'UNKNOWN'],
        'policy_cost_axes': list(cost_axes), 'policy_pareto_frontier': frontier,
        'policy_is_artist_ownership': False, 'same_drawable_pairs': [],
        'different_drawable_pairs': [], **AUTHORITY}
