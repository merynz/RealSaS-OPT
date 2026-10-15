from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

from tools.corpus.fit8_static_draw_order_capacity_court_v1 import (
    COMPONENT_TOTAL_ORDER_FALSIFIED,
    COMPONENT_TOTAL_ORDER_NOT_FALSIFIED,
)

STATIC_RAY_SCHEMA = "RealSaS.FIT8StaticRayCensus.v3"
SCHEMA_VERSION = "RealSaS.FIT8StaticDrawOrderCapacityCourt.v2"


@dataclass(frozen=True)
class ViewDirectedCycleWitnessV2:
    view_index: int
    cycle: tuple[int, ...]

    def validate(self) -> None:
        if not 0 <= self.view_index < 8:
            raise ValueError("FIT8_DRAW_ORDER_V2_WITNESS_VIEW_INVALID")
        if len(self.cycle) < 3 or self.cycle[0] != self.cycle[-1]:
            raise ValueError("FIT8_DRAW_ORDER_V2_WITNESS_CYCLE_INVALID")
        if any(value < 0 for value in self.cycle):
            raise ValueError("FIT8_DRAW_ORDER_V2_WITNESS_COMPONENT_INVALID")


@dataclass(frozen=True)
class SubjectDrawOrderCapacityV2:
    subject_key: str
    resolution: int
    component_count: int
    mixed_pair_view_count: int
    mixed_pair_count_sum: int
    directed_cycle_view_count: int
    ordered_relation_count_sum: int
    maximum_distinct_component_stack: int
    cycle_witnesses: tuple[ViewDirectedCycleWitnessV2, ...]
    component_total_order_state: str

    def validate(self) -> None:
        if not self.subject_key or self.resolution < 32 or self.component_count < 1:
            raise ValueError("FIT8_DRAW_ORDER_V2_DOMAIN_INVALID")
        counts = (
            self.mixed_pair_view_count,
            self.mixed_pair_count_sum,
            self.directed_cycle_view_count,
            self.ordered_relation_count_sum,
        )
        if any(value < 0 for value in counts):
            raise ValueError("FIT8_DRAW_ORDER_V2_COUNT_INVALID")
        if self.mixed_pair_view_count > self.directed_cycle_view_count:
            raise ValueError("FIT8_DRAW_ORDER_V2_MIXED_VIEW_MUST_BE_CYCLIC")
        if self.maximum_distinct_component_stack < 1:
            raise ValueError("FIT8_DRAW_ORDER_V2_STACK_INVALID")
        if len(self.cycle_witnesses) != self.directed_cycle_view_count:
            raise ValueError("FIT8_DRAW_ORDER_V2_WITNESS_COUNT_MISMATCH")
        views = []
        for witness in self.cycle_witnesses:
            witness.validate()
            views.append(witness.view_index)
        if tuple(sorted(views)) != tuple(views) or len(set(views)) != len(views):
            raise ValueError("FIT8_DRAW_ORDER_V2_WITNESS_VIEW_SET_INVALID")
        expected = (
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if self.directed_cycle_view_count > 0
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        )
        if self.component_total_order_state != expected:
            raise ValueError("FIT8_DRAW_ORDER_V2_STATE_MISMATCH")


@dataclass(frozen=True)
class CohortDrawOrderCapacityV2:
    subjects: tuple[SubjectDrawOrderCapacityV2, ...]
    component_total_order_state: str
    replacement_mechanism_frozen: bool
    replacement_candidates: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.subjects:
            raise ValueError("FIT8_DRAW_ORDER_V2_COHORT_INVALID")
        for row in self.subjects:
            row.validate()
        expected = (
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if any(row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED for row in self.subjects)
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        )
        if self.component_total_order_state != expected:
            raise ValueError("FIT8_DRAW_ORDER_V2_COHORT_STATE_MISMATCH")
        if self.replacement_mechanism_frozen:
            raise ValueError("FIT8_DRAW_ORDER_V2_REPLACEMENT_MUST_REMAIN_OPEN")
        if not self.replacement_candidates:
            raise ValueError("FIT8_DRAW_ORDER_V2_REPLACEMENT_CANDIDATES_EMPTY")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _ordered_edges_and_validate_mixed(row: Mapping) -> set[tuple[int, int]]:
    ordered = row.get("ordered_pairs")
    if not isinstance(ordered, list) or int(row.get("ordered_pair_count", -1)) != len(ordered):
        raise ValueError("FIT8_DRAW_ORDER_V2_ORDERED_PAIR_EVIDENCE_INVALID")
    counts: dict[tuple[int, int], int] = {}
    for raw in ordered:
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("FIT8_DRAW_ORDER_V2_ORDERED_PAIR_ROW_INVALID")
        a, b, count = map(int, raw)
        if a < 0 or b < 0 or a == b or count <= 0 or (a, b) in counts:
            raise ValueError("FIT8_DRAW_ORDER_V2_ORDERED_PAIR_VALUE_INVALID")
        counts[(a, b)] = count

    expected_mixed = []
    unordered = sorted({tuple(sorted(edge)) for edge in counts})
    for lo, hi in unordered:
        forward = counts.get((lo, hi), 0)
        reverse = counts.get((hi, lo), 0)
        if forward and reverse:
            expected_mixed.append([lo, hi, forward, reverse])
    observed_mixed = row.get("mixed_pairs")
    if not isinstance(observed_mixed, list):
        raise ValueError("FIT8_DRAW_ORDER_V2_MIXED_PAIR_EVIDENCE_INVALID")
    normalized_mixed = sorted([list(map(int, raw)) for raw in observed_mixed])
    if normalized_mixed != expected_mixed or int(row.get("mixed_pair_count", -1)) != len(expected_mixed):
        raise ValueError("FIT8_DRAW_ORDER_V2_MIXED_PAIR_EVIDENCE_MISMATCH")
    return set(counts)


def _deterministic_cycle_witness(edges: set[tuple[int, int]]) -> tuple[int, ...] | None:
    nodes = sorted({value for edge in edges for value in edge})
    adjacency = {node: [] for node in nodes}
    for a, b in sorted(edges):
        adjacency[a].append(b)
    state = {node: 0 for node in nodes}
    stack: list[int] = []
    stack_index: dict[int, int] = {}

    def visit(node: int) -> tuple[int, ...] | None:
        state[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for neighbor in adjacency[node]:
            if state[neighbor] == 0:
                found = visit(neighbor)
                if found is not None:
                    return found
            elif state[neighbor] == 1:
                start = stack_index[neighbor]
                return tuple(stack[start:] + [neighbor])
        stack.pop()
        stack_index.pop(node, None)
        state[node] = 2
        return None

    for node in nodes:
        if state[node] == 0:
            found = visit(node)
            if found is not None:
                return found
    return None


def evaluate_subject_static_draw_order_capacity_v2(summary: Mapping) -> SubjectDrawOrderCapacityV2:
    if summary.get("schema") != STATIC_RAY_SCHEMA:
        raise ValueError("FIT8_DRAW_ORDER_V2_STATIC_RAY_SCHEMA_INVALID")
    subject_key = str(summary.get("subject_key") or "")
    policy = summary.get("policy") or {}
    resolution = int(policy.get("resolution", 0))
    component_count = int(summary.get("component_count", 0))
    static_views = summary.get("static_views") or []
    pair_evidence = summary.get("pair_evidence") or {}
    if len(static_views) != 8 or sorted(pair_evidence) != [str(i) for i in range(8)]:
        raise ValueError("FIT8_DRAW_ORDER_V2_VIEW_SET_INVALID")

    mixed_pair_view_count = 0
    mixed_pair_count_sum = 0
    ordered_relation_count_sum = 0
    witnesses: list[ViewDirectedCycleWitnessV2] = []
    for view_index in range(8):
        row = pair_evidence[str(view_index)]
        if not isinstance(row, Mapping):
            raise ValueError("FIT8_DRAW_ORDER_V2_PAIR_EVIDENCE_INVALID")
        edges = _ordered_edges_and_validate_mixed(row)
        ordered_relation_count_sum += len(edges)
        mixed_count = int(row["mixed_pair_count"])
        mixed_pair_count_sum += mixed_count
        mixed_pair_view_count += int(mixed_count > 0)
        cycle = _deterministic_cycle_witness(edges)
        if cycle is not None:
            witnesses.append(ViewDirectedCycleWitnessV2(view_index=view_index, cycle=cycle))

    max_stack = max(int(row.get("max_distinct_component_stack", 0)) for row in static_views)
    result = SubjectDrawOrderCapacityV2(
        subject_key=subject_key,
        resolution=resolution,
        component_count=component_count,
        mixed_pair_view_count=mixed_pair_view_count,
        mixed_pair_count_sum=mixed_pair_count_sum,
        directed_cycle_view_count=len(witnesses),
        ordered_relation_count_sum=ordered_relation_count_sum,
        maximum_distinct_component_stack=max_stack,
        cycle_witnesses=tuple(witnesses),
        component_total_order_state=(
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if witnesses
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        ),
    )
    result.validate()
    return result


def evaluate_cohort_static_draw_order_capacity_v2(
    summaries: Sequence[Mapping],
) -> CohortDrawOrderCapacityV2:
    subjects = tuple(evaluate_subject_static_draw_order_capacity_v2(row) for row in summaries)
    result = CohortDrawOrderCapacityV2(
        subjects=subjects,
        component_total_order_state=(
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if any(row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED for row in subjects)
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        ),
        replacement_mechanism_frozen=False,
        replacement_candidates=(
            "PER_PIXEL_OR_GEOMETRIC_DEPTH_RESOLUTION",
            "QUALIFIED_SUBCOMPONENT_FRAGMENT_ORDER",
            "VIEW_POSE_CONDITIONED_FRAGMENT_DAG",
        ),
    )
    result.validate()
    return result


__all__ = [
    "CohortDrawOrderCapacityV2",
    "SCHEMA_VERSION",
    "STATIC_RAY_SCHEMA",
    "SubjectDrawOrderCapacityV2",
    "ViewDirectedCycleWitnessV2",
    "evaluate_cohort_static_draw_order_capacity_v2",
    "evaluate_subject_static_draw_order_capacity_v2",
]
