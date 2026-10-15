from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Sequence

from tools.corpus.fit8_dynamic_rendered_mechanical_census_contract_v1 import (
    AUTHORITY_STATES,
    DynamicPoseSampleV1,
)
from tools.corpus.fit8_dynamic_fixed_frame_presentation_v1 import FRAME_SCOPE


SCHEMA_VERSION = "RealSaS.FIT8DynamicRenderedMechanicalCensus.v2"


@dataclass(frozen=True)
class DynamicSubjectCensusV2:
    subject_key: str
    source_sha256: str
    carrier_sha256: str
    animation_evaluator_sha256: str
    fixed_frame_presentation_producer_sha256: str
    sampling_policy_sha256: str
    dynamic_frame_sha256: str
    dynamic_frame_scope: str
    mechanics_authority_state: str
    presentation_authority_state: str
    samples: tuple[DynamicPoseSampleV1, ...]
    all_native_clips_covered: bool
    per_pose_recenter_forbidden: bool = True
    quotient_motion_witness_required: bool = True
    raw_maxima_diagnostic_only: bool = True
    fit8_pass_claimed: bool = False
    product_pass_claimed: bool = False
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.subject_key:
            raise ValueError("FIT8_DYNAMIC_V2_SUBJECT_SCHEMA_INVALID")
        for value in (
            self.source_sha256,
            self.carrier_sha256,
            self.animation_evaluator_sha256,
            self.fixed_frame_presentation_producer_sha256,
            self.sampling_policy_sha256,
            self.dynamic_frame_sha256,
        ):
            if len(value) != 64:
                raise ValueError("FIT8_DYNAMIC_V2_SHA_INVALID")
        if self.dynamic_frame_scope != FRAME_SCOPE:
            raise ValueError("FIT8_DYNAMIC_V2_FRAME_SCOPE_INVALID")
        if not self.per_pose_recenter_forbidden:
            raise ValueError("FIT8_DYNAMIC_V2_PER_POSE_RECENTER_MUST_BE_FORBIDDEN")
        if self.mechanics_authority_state not in AUTHORITY_STATES:
            raise ValueError("FIT8_DYNAMIC_V2_MECHANICS_AUTHORITY_INVALID")
        if self.presentation_authority_state not in AUTHORITY_STATES:
            raise ValueError("FIT8_DYNAMIC_V2_PRESENTATION_AUTHORITY_INVALID")
        if not self.samples:
            raise ValueError("FIT8_DYNAMIC_V2_SAMPLES_EMPTY")
        for sample in self.samples:
            sample.validate()
        if not self.quotient_motion_witness_required:
            raise ValueError("FIT8_DYNAMIC_V2_QUOTIENT_WITNESS_MUST_BE_REQUIRED")
        if not self.raw_maxima_diagnostic_only:
            raise ValueError("FIT8_DYNAMIC_V2_RAW_MAXIMA_MUST_REMAIN_DIAGNOSTIC")
        if self.fit8_pass_claimed or self.product_pass_claimed:
            raise ValueError("FIT8_DYNAMIC_V2_UNSUPPORTED_PASS_CLAIM")
        presentation_present = [sample.directed_cycle_view_count is not None for sample in self.samples]
        if self.presentation_authority_state == "QUALIFIED" and not all(presentation_present):
            raise ValueError("FIT8_DYNAMIC_V2_QUALIFIED_PRESENTATION_REQUIRES_ALL_SAMPLES")

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


def validate_cohort_v2(rows: Sequence[DynamicSubjectCensusV2]) -> None:
    if len(rows) != 8 or len({row.subject_key for row in rows}) != 8:
        raise ValueError("FIT8_DYNAMIC_V2_COHORT_MUST_HAVE_EIGHT_UNIQUE_SUBJECTS")
    for row in rows:
        row.validate()


__all__ = [
    "DynamicSubjectCensusV2",
    "SCHEMA_VERSION",
    "validate_cohort_v2",
]
