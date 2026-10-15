from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

STATIC_RAY_SCHEMAS = {
    "RealSaS.FIT8StaticRayCensus.v1",
    "RealSaS.FIT8StaticRayCensus.v2",
}
SCHEMA_VERSION = "RealSaS.FIT8StaticDrawOrderCapacityCourt.v1"

COMPONENT_TOTAL_ORDER_FALSIFIED = "COMPONENT_TOTAL_ORDER_FALSIFIED"
COMPONENT_TOTAL_ORDER_NOT_FALSIFIED = "COMPONENT_TOTAL_ORDER_NOT_FALSIFIED"


@dataclass(frozen=True)
class SubjectDrawOrderCapacityV1:
    subject_key: str
    resolution: int
    component_count: int
    mixed_pair_view_count: int
    mixed_pair_count_sum: int
    maximum_distinct_component_stack: int
    component_total_order_state: str

    def validate(self) -> None:
        if not self.subject_key:
            raise ValueError("FIT8_DRAW_ORDER_SUBJECT_MISSING")
        if self.resolution < 32 or self.component_count < 1:
            raise ValueError("FIT8_DRAW_ORDER_DOMAIN_INVALID")
        if self.mixed_pair_view_count < 0 or self.mixed_pair_count_sum < 0:
            raise ValueError("FIT8_DRAW_ORDER_MIXED_COUNT_INVALID")
        if self.maximum_distinct_component_stack < 1:
            raise ValueError("FIT8_DRAW_ORDER_STACK_INVALID")
        expected = (
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if self.mixed_pair_view_count > 0
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        )
        if self.component_total_order_state != expected:
            raise ValueError("FIT8_DRAW_ORDER_STATE_MISMATCH")


@dataclass(frozen=True)
class CohortDrawOrderCapacityV1:
    subjects: tuple[SubjectDrawOrderCapacityV1, ...]
    component_total_order_state: str
    replacement_mechanism_frozen: bool
    replacement_candidates: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.subjects:
            raise ValueError("FIT8_DRAW_ORDER_COHORT_INVALID")
        for row in self.subjects:
            row.validate()
        expected = (
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if any(row.component_total_order_state == COMPONENT_TOTAL_ORDER_FALSIFIED for row in self.subjects)
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        )
        if self.component_total_order_state != expected:
            raise ValueError("FIT8_DRAW_ORDER_COHORT_STATE_MISMATCH")
        if self.replacement_mechanism_frozen:
            raise ValueError("FIT8_DRAW_ORDER_REPLACEMENT_MUST_REMAIN_OPEN")
        if not self.replacement_candidates:
            raise ValueError("FIT8_DRAW_ORDER_REPLACEMENT_CANDIDATES_EMPTY")

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


def evaluate_subject_static_draw_order_capacity_v1(summary: Mapping) -> SubjectDrawOrderCapacityV1:
    if summary.get("schema") not in STATIC_RAY_SCHEMAS:
        raise ValueError("FIT8_DRAW_ORDER_STATIC_RAY_SCHEMA_INVALID")
    subject_key = str(summary.get("subject_key") or "")
    policy = summary.get("policy") or {}
    resolution = int(policy.get("resolution", 0))
    component_count = int(summary.get("component_count", 0))
    static_views = summary.get("static_views") or []
    pair_evidence = summary.get("pair_evidence") or {}
    if len(static_views) != 8:
        raise ValueError("FIT8_DRAW_ORDER_VIEW_COUNT_INVALID")
    mixed_counts = []
    for view_index in range(8):
        row = pair_evidence.get(str(view_index))
        if not isinstance(row, Mapping):
            raise ValueError("FIT8_DRAW_ORDER_PAIR_EVIDENCE_MISSING")
        count = int(row.get("mixed_pair_count", -1))
        mixed = row.get("mixed_pairs")
        if count < 0 or not isinstance(mixed, list) or count != len(mixed):
            raise ValueError("FIT8_DRAW_ORDER_PAIR_EVIDENCE_INVALID")
        mixed_counts.append(count)
    max_stack = max(int(row.get("max_distinct_component_stack", 0)) for row in static_views)
    result = SubjectDrawOrderCapacityV1(
        subject_key=subject_key,
        resolution=resolution,
        component_count=component_count,
        mixed_pair_view_count=sum(value > 0 for value in mixed_counts),
        mixed_pair_count_sum=sum(mixed_counts),
        maximum_distinct_component_stack=max_stack,
        component_total_order_state=(
            COMPONENT_TOTAL_ORDER_FALSIFIED
            if any(value > 0 for value in mixed_counts)
            else COMPONENT_TOTAL_ORDER_NOT_FALSIFIED
        ),
    )
    result.validate()
    return result


def evaluate_cohort_static_draw_order_capacity_v1(
    summaries: Sequence[Mapping],
) -> CohortDrawOrderCapacityV1:
    subjects = tuple(evaluate_subject_static_draw_order_capacity_v1(row) for row in summaries)
    result = CohortDrawOrderCapacityV1(
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
    "COMPONENT_TOTAL_ORDER_FALSIFIED",
    "COMPONENT_TOTAL_ORDER_NOT_FALSIFIED",
    "CohortDrawOrderCapacityV1",
    "SCHEMA_VERSION",
    "STATIC_RAY_SCHEMAS",
    "SubjectDrawOrderCapacityV1",
    "evaluate_cohort_static_draw_order_capacity_v1",
    "evaluate_subject_static_draw_order_capacity_v1",
]
