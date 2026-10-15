from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "RealSaS.Fit8SemanticPartitionCensus.v1"
AUTHORITY_SCOPE = "FIT8_ARCHITECTURE_ACCEPTANCE_ONLY"
FIT8_SUBJECTS = (
    "Snake",
    "Bat",
    "T-Rex",
    "Horse",
    "Spider",
    "GreenBlob",
    "Shark",
    "George",
)

QUALIFIED = "QUALIFIED"
ABSTAINED = "ABSTAINED"
NOT_EVALUATED = "NOT_EVALUATED"

CAPACITY_NOT_EVALUATED = "NOT_EVALUATED"
CAPACITY_FAIL_CLOSED = "FAIL_CLOSED_QUERY_CAPACITY_BELOW_OBSERVED_OWNER_COUNT"
CAPACITY_INCOMPLETE = "ABSTAINED_INCOMPLETE_FIT8_OWNER_COHORT"
CAPACITY_COVERED = "FIT8_OBSERVED_OWNER_CAPACITY_COVERED"


@dataclass(frozen=True)
class SemanticPartitionSubjectTruthV1:
    """Qualified teacher truth used by the semantic-partition census.

    Semantic owner, material region, authored material identity, geometric
    component and mechanical joint owner are intentionally represented as
    different axes. None may substitute for another when an authoritative axis
    is unavailable.
    """

    subject_key: str
    source_ref: str
    teacher_bundle_schema_version: str
    teacher_bundle_manifest_sha256: str
    owner_partition_manifest: Mapping[str, Any] | None
    surface_owner_region_id: Sequence[int] | None
    surface_material_region_id: Sequence[int] | None = None
    surface_material_region_valid: Sequence[bool] | None = None
    surface_material_id: Sequence[Any] | None = None
    surface_geometric_component_id: Sequence[int] | None = None
    surface_mechanical_owner_joint_id: Sequence[int] | None = None
    expected_surface_sample_count: int | None = None
    qualification_state: str = QUALIFIED


@dataclass(frozen=True)
class AxisCensusV1:
    status: str
    distinct_count: int | None
    assigned_sample_count: int | None
    total_sample_count: int | None
    reason: str | None = None


@dataclass(frozen=True)
class SubjectSemanticPartitionCensusV1:
    subject_key: str
    source_ref: str
    teacher_bundle_manifest_sha256: str
    owner: AxisCensusV1
    material_region: AxisCensusV1
    material_identity: AxisCensusV1
    geometric_component: AxisCensusV1
    mechanical_owner_joint: AxisCensusV1
    substitution_for_owner_forbidden: bool = True


@dataclass(frozen=True)
class CohortSemanticPartitionCensusV1:
    schema_version: str
    authority_scope: str
    expected_subjects: tuple[str, ...]
    subjects: tuple[SubjectSemanticPartitionCensusV1, ...]
    missing_subjects: tuple[str, ...]
    unexpected_subjects: tuple[str, ...]
    qualified_owner_subject_count: int
    complete_owner_cohort: bool
    observed_owner_count_max: int | None
    observed_owner_count_median: float | None
    minimum_query_capacity_for_observed_fit8: int | None
    supplied_query_capacity: int | None
    supplied_query_capacity_state: str
    global_query_cap_freeze_authority: bool = False
    unseen_generalization_authority: bool = False
    silent_region_merge_permitted: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


def _expected_count(record: SemanticPartitionSubjectTruthV1) -> int | None:
    if record.expected_surface_sample_count is not None:
        if record.expected_surface_sample_count < 0:
            raise ValueError("SEMANTIC_CENSUS_NEGATIVE_EXPECTED_SAMPLE_COUNT")
        return int(record.expected_surface_sample_count)
    for values in (
        record.surface_owner_region_id,
        record.surface_material_region_id,
        record.surface_material_id,
        record.surface_geometric_component_id,
        record.surface_mechanical_owner_joint_id,
    ):
        if values is not None:
            return len(values)
    return None


def _manifest_region_ids(manifest: Mapping[str, Any] | None) -> set[int] | None:
    if manifest is None:
        return None
    regions = manifest.get("regions")
    if not isinstance(regions, Mapping):
        return None
    result: set[int] = set()
    for key, payload in regions.items():
        candidates = [key]
        if isinstance(payload, Mapping):
            candidates.extend(
                payload.get(name)
                for name in ("owner_region_id", "region_id", "id")
                if name in payload
            )
        parsed: int | None = None
        for candidate in candidates:
            try:
                parsed = int(candidate)
                break
            except (TypeError, ValueError):
                continue
        if parsed is None or parsed < 0:
            return None
        result.add(parsed)
    return result


def _abstained(total: int | None, reason: str) -> AxisCensusV1:
    return AxisCensusV1(
        status=ABSTAINED,
        distinct_count=None,
        assigned_sample_count=None,
        total_sample_count=total,
        reason=reason,
    )


def _plain_axis(
    values: Sequence[Any] | None,
    *,
    total: int | None,
    missing_reason: str,
) -> AxisCensusV1:
    if values is None:
        return _abstained(total, missing_reason)
    if total is not None and len(values) != total:
        return _abstained(total, "SAMPLE_COUNT_MISMATCH")
    assigned = [value for value in values if value is not None and value != -1]
    return AxisCensusV1(
        status=QUALIFIED,
        distinct_count=len(set(assigned)),
        assigned_sample_count=len(assigned),
        total_sample_count=len(values) if total is None else total,
    )


def _owner_axis(
    record: SemanticPartitionSubjectTruthV1,
    *,
    total: int | None,
) -> AxisCensusV1:
    if record.qualification_state != QUALIFIED:
        return _abstained(total, "TEACHER_BUNDLE_NOT_QUALIFIED")
    if record.owner_partition_manifest is None:
        return _abstained(total, "QUALIFIED_OWNER_PARTITION_MANIFEST_MISSING")
    if record.surface_owner_region_id is None:
        return _abstained(total, "QUALIFIED_OWNER_REGION_CHANNEL_MISSING")
    if total is not None and len(record.surface_owner_region_id) != total:
        return _abstained(total, "OWNER_SAMPLE_COUNT_MISMATCH")

    manifest_ids = _manifest_region_ids(record.owner_partition_manifest)
    if manifest_ids is None:
        return _abstained(total, "OWNER_PARTITION_MANIFEST_INVALID")

    values = [int(value) for value in record.surface_owner_region_id]
    if any(value < 0 for value in values):
        return _abstained(total, "OWNER_REGION_UNASSIGNED_SAMPLE_PRESENT")
    observed_ids = set(values)
    if not observed_ids.issubset(manifest_ids):
        return _abstained(total, "OWNER_REGION_ID_NOT_IN_MANIFEST")
    if manifest_ids != observed_ids:
        return _abstained(total, "OWNER_MANIFEST_REGION_WITHOUT_SURFACE_SUPPORT")

    resolved_total = len(values) if total is None else total
    return AxisCensusV1(
        status=QUALIFIED,
        distinct_count=len(observed_ids),
        assigned_sample_count=len(values),
        total_sample_count=resolved_total,
    )


def _material_region_axis(
    record: SemanticPartitionSubjectTruthV1,
    *,
    total: int | None,
) -> AxisCensusV1:
    values = record.surface_material_region_id
    validity = record.surface_material_region_valid
    if values is None:
        return _abstained(total, "MATERIAL_REGION_CHANNEL_MISSING")
    if validity is None:
        return _abstained(total, "MATERIAL_REGION_VALIDITY_MISSING")
    if len(values) != len(validity):
        return _abstained(total, "MATERIAL_REGION_VALIDITY_SHAPE_MISMATCH")
    if total is not None and len(values) != total:
        return _abstained(total, "MATERIAL_REGION_SAMPLE_COUNT_MISMATCH")

    qualified = [
        int(value)
        for value, valid in zip(values, validity)
        if bool(valid) and int(value) >= 0
    ]
    resolved_total = len(values) if total is None else total
    return AxisCensusV1(
        status=QUALIFIED,
        distinct_count=len(set(qualified)),
        assigned_sample_count=len(qualified),
        total_sample_count=resolved_total,
    )


def census_subject_semantic_partition_v1(
    record: SemanticPartitionSubjectTruthV1,
) -> SubjectSemanticPartitionCensusV1:
    total = _expected_count(record)
    return SubjectSemanticPartitionCensusV1(
        subject_key=record.subject_key,
        source_ref=record.source_ref,
        teacher_bundle_manifest_sha256=record.teacher_bundle_manifest_sha256,
        owner=_owner_axis(record, total=total),
        material_region=_material_region_axis(record, total=total),
        material_identity=_plain_axis(
            record.surface_material_id,
            total=total,
            missing_reason="AUTHORED_MATERIAL_IDENTITY_CHANNEL_MISSING",
        ),
        geometric_component=_plain_axis(
            record.surface_geometric_component_id,
            total=total,
            missing_reason="GEOMETRIC_COMPONENT_CHANNEL_MISSING",
        ),
        mechanical_owner_joint=_plain_axis(
            record.surface_mechanical_owner_joint_id,
            total=total,
            missing_reason="MECHANICAL_OWNER_JOINT_CHANNEL_MISSING",
        ),
    )


def census_fit8_semantic_partitions_v1(
    records: Sequence[SemanticPartitionSubjectTruthV1],
    *,
    supplied_query_capacity: int | None = None,
    strict_subject_set: bool = True,
) -> CohortSemanticPartitionCensusV1:
    if supplied_query_capacity is not None and supplied_query_capacity <= 0:
        raise ValueError("SEMANTIC_CENSUS_QUERY_CAPACITY_MUST_BE_POSITIVE")

    by_subject: dict[str, SemanticPartitionSubjectTruthV1] = {}
    for record in records:
        if record.subject_key in by_subject:
            raise ValueError(f"SEMANTIC_CENSUS_DUPLICATE_SUBJECT:{record.subject_key}")
        by_subject[record.subject_key] = record

    expected = set(FIT8_SUBJECTS)
    seen = set(by_subject)
    missing = tuple(subject for subject in FIT8_SUBJECTS if subject not in seen)
    unexpected = tuple(sorted(seen - expected))
    if strict_subject_set and unexpected:
        raise ValueError(f"SEMANTIC_CENSUS_UNEXPECTED_SUBJECTS:{unexpected}")

    subject_results = tuple(
        census_subject_semantic_partition_v1(by_subject[subject])
        for subject in FIT8_SUBJECTS
        if subject in by_subject
    )
    owner_counts = [
        result.owner.distinct_count
        for result in subject_results
        if result.owner.status == QUALIFIED and result.owner.distinct_count is not None
    ]
    qualified_owner_subject_count = len(owner_counts)
    complete_owner_cohort = (
        not missing
        and not unexpected
        and qualified_owner_subject_count == len(FIT8_SUBJECTS)
    )
    observed_max = max(owner_counts) if owner_counts else None
    observed_median = float(median(owner_counts)) if owner_counts else None
    minimum_capacity = observed_max if complete_owner_cohort else None

    if supplied_query_capacity is None:
        capacity_state = CAPACITY_NOT_EVALUATED
    elif observed_max is not None and supplied_query_capacity < observed_max:
        capacity_state = CAPACITY_FAIL_CLOSED
    elif not complete_owner_cohort:
        capacity_state = CAPACITY_INCOMPLETE
    else:
        capacity_state = CAPACITY_COVERED

    notes = (
        "Semantic owner count is measured only from qualified owner-partition truth.",
        "Material region, material identity, geometric component and mechanical joint owner are separate evidence axes and never substitute for semantic owner.",
        "A complete Fit8 census establishes only the minimum capacity needed for the observed Fit8 teacher truth; it does not freeze a global query cap or establish unseen/generalization authority.",
    )

    return CohortSemanticPartitionCensusV1(
        schema_version=SCHEMA_VERSION,
        authority_scope=AUTHORITY_SCOPE,
        expected_subjects=FIT8_SUBJECTS,
        subjects=subject_results,
        missing_subjects=missing,
        unexpected_subjects=unexpected,
        qualified_owner_subject_count=qualified_owner_subject_count,
        complete_owner_cohort=complete_owner_cohort,
        observed_owner_count_max=observed_max,
        observed_owner_count_median=observed_median,
        minimum_query_capacity_for_observed_fit8=minimum_capacity,
        supplied_query_capacity=supplied_query_capacity,
        supplied_query_capacity_state=capacity_state,
        global_query_cap_freeze_authority=False,
        unseen_generalization_authority=False,
        silent_region_merge_permitted=False,
        notes=notes,
    )


__all__ = [
    "ABSTAINED",
    "AUTHORITY_SCOPE",
    "CAPACITY_COVERED",
    "CAPACITY_FAIL_CLOSED",
    "CAPACITY_INCOMPLETE",
    "CAPACITY_NOT_EVALUATED",
    "FIT8_SUBJECTS",
    "QUALIFIED",
    "SCHEMA_VERSION",
    "AxisCensusV1",
    "CohortSemanticPartitionCensusV1",
    "SemanticPartitionSubjectTruthV1",
    "SubjectSemanticPartitionCensusV1",
    "census_fit8_semantic_partitions_v1",
    "census_subject_semantic_partition_v1",
]
