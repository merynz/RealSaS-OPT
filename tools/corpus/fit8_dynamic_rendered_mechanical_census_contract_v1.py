from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence


SCHEMA_VERSION = "RealSaS.FIT8DynamicRenderedMechanicalCensus.v1"
AUTHORITY_STATES = {"DIAGNOSTIC", "QUALIFIED"}


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


@dataclass(frozen=True)
class DynamicPoseSampleV1:
    clip_name: str
    clip_index: int
    normalized_time: float
    authored_time: float
    quotient_worst_separation: float
    edge_stretch_p99: float
    face_condition_p99: float
    face_area_ratio_p01: float
    face_area_ratio_p99: float
    face_area_ratio_min: float
    face_area_ratio_max: float
    directed_cycle_view_count: int | None = None
    mixed_pair_view_count: int | None = None
    mixed_pair_count_sum: int | None = None
    ordered_relation_count_sum: int | None = None
    maximum_distinct_component_stack: int | None = None
    new_directed_edges_vs_static: int | None = None
    reversed_directed_edges_vs_static: int | None = None
    strict_pair_flips_vs_static: int | None = None

    def validate(self) -> None:
        if not self.clip_name or self.clip_index < 0:
            raise ValueError("FIT8_DYNAMIC_POSE_CLIP_INVALID")
        if not 0.0 <= float(self.normalized_time) <= 1.0 or not _finite(self.authored_time):
            raise ValueError("FIT8_DYNAMIC_POSE_TIME_INVALID")
        values = (
            self.quotient_worst_separation,
            self.edge_stretch_p99,
            self.face_condition_p99,
            self.face_area_ratio_p01,
            self.face_area_ratio_p99,
            self.face_area_ratio_min,
            self.face_area_ratio_max,
        )
        if not all(_finite(v) for v in values):
            raise ValueError("FIT8_DYNAMIC_POSE_MECHANICS_NONFINITE")
        if self.quotient_worst_separation < 0.0 or self.edge_stretch_p99 <= 0.0:
            raise ValueError("FIT8_DYNAMIC_POSE_MECHANICS_INVALID")
        if self.face_condition_p99 < 1.0:
            raise ValueError("FIT8_DYNAMIC_POSE_CONDITION_LT_ONE")
        if not (0.0 <= self.face_area_ratio_min <= self.face_area_ratio_p01 <= self.face_area_ratio_p99 <= self.face_area_ratio_max):
            raise ValueError("FIT8_DYNAMIC_POSE_AREA_ORDER_INVALID")
        presentation = (
            self.directed_cycle_view_count,
            self.mixed_pair_view_count,
            self.mixed_pair_count_sum,
            self.ordered_relation_count_sum,
            self.maximum_distinct_component_stack,
            self.new_directed_edges_vs_static,
            self.reversed_directed_edges_vs_static,
            self.strict_pair_flips_vs_static,
        )
        present = [value is not None for value in presentation]
        if any(present) and not all(present):
            raise ValueError("FIT8_DYNAMIC_POSE_PRESENTATION_PARTIAL")
        if all(present):
            if not 0 <= int(self.directed_cycle_view_count) <= 8:
                raise ValueError("FIT8_DYNAMIC_POSE_CYCLE_VIEW_INVALID")
            if not 0 <= int(self.mixed_pair_view_count) <= 8:
                raise ValueError("FIT8_DYNAMIC_POSE_MIXED_VIEW_INVALID")
            for value in presentation[2:]:
                if int(value) < 0:
                    raise ValueError("FIT8_DYNAMIC_POSE_PRESENTATION_NEGATIVE")
            if int(self.maximum_distinct_component_stack) < 1:
                raise ValueError("FIT8_DYNAMIC_POSE_STACK_INVALID")


@dataclass(frozen=True)
class DynamicSubjectCensusV1:
    subject_key: str
    source_sha256: str
    carrier_sha256: str
    animation_evaluator_sha256: str
    static_ray_producer_sha256: str | None
    sampling_policy_sha256: str
    mechanics_authority_state: str
    presentation_authority_state: str
    samples: tuple[DynamicPoseSampleV1, ...]
    all_native_clips_covered: bool
    quotient_motion_witness_required: bool = True
    raw_maxima_diagnostic_only: bool = True
    fit8_pass_claimed: bool = False
    product_pass_claimed: bool = False
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.subject_key:
            raise ValueError("FIT8_DYNAMIC_SUBJECT_SCHEMA_INVALID")
        for value in (self.source_sha256, self.carrier_sha256, self.animation_evaluator_sha256, self.sampling_policy_sha256):
            if len(value) != 64:
                raise ValueError("FIT8_DYNAMIC_SUBJECT_SHA_INVALID")
        if self.static_ray_producer_sha256 is not None and len(self.static_ray_producer_sha256) != 64:
            raise ValueError("FIT8_DYNAMIC_STATIC_RAY_SHA_INVALID")
        if self.mechanics_authority_state not in AUTHORITY_STATES:
            raise ValueError("FIT8_DYNAMIC_MECHANICS_AUTHORITY_INVALID")
        if self.presentation_authority_state not in AUTHORITY_STATES:
            raise ValueError("FIT8_DYNAMIC_PRESENTATION_AUTHORITY_INVALID")
        if not self.samples:
            raise ValueError("FIT8_DYNAMIC_SAMPLES_EMPTY")
        for sample in self.samples:
            sample.validate()
        if not self.quotient_motion_witness_required:
            raise ValueError("FIT8_DYNAMIC_QUOTIENT_WITNESS_MUST_BE_REQUIRED")
        if not self.raw_maxima_diagnostic_only:
            raise ValueError("FIT8_DYNAMIC_RAW_MAXIMA_MUST_REMAIN_DIAGNOSTIC")
        if self.fit8_pass_claimed or self.product_pass_claimed:
            raise ValueError("FIT8_DYNAMIC_UNSUPPORTED_PASS_CLAIM")
        presentation_present = [sample.directed_cycle_view_count is not None for sample in self.samples]
        if self.presentation_authority_state == "QUALIFIED" and not all(presentation_present):
            raise ValueError("FIT8_DYNAMIC_QUALIFIED_PRESENTATION_REQUIRES_ALL_SAMPLES")

    def canonical_payload(self) -> dict:
        self.validate()
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def validate_cohort_v1(rows: Sequence[DynamicSubjectCensusV1]) -> None:
    if len(rows) != 8 or len({row.subject_key for row in rows}) != 8:
        raise ValueError("FIT8_DYNAMIC_COHORT_MUST_HAVE_EIGHT_UNIQUE_SUBJECTS")
    for row in rows:
        row.validate()


__all__ = [
    "AUTHORITY_STATES",
    "DynamicPoseSampleV1",
    "DynamicSubjectCensusV1",
    "SCHEMA_VERSION",
    "validate_cohort_v1",
]
