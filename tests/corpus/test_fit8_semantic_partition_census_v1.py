import pytest

from tools.corpus.fit8_semantic_partition_census_v1 import (
    ABSTAINED,
    CAPACITY_COVERED,
    CAPACITY_FAIL_CLOSED,
    CAPACITY_INCOMPLETE,
    EXPECTED_TEACHER_BUNDLE_SCHEMA_VERSION,
    FIT8_SUBJECTS,
    QUALIFIED,
    SemanticPartitionSubjectTruthV1,
    census_fit8_semantic_partitions_v1,
    census_subject_semantic_partition_v1,
)


def _manifest(count: int):
    return {
        "owner_partition_version": "synthetic-test-v1",
        "regions": {
            str(index): {"owner_region_id": index, "label": f"owner-{index}"}
            for index in range(count)
        },
    }


def _record(
    subject: str,
    *,
    owner_ids=(0, 0, 1, 1),
    owner_count=2,
    material_regions=(0, 1, 2, 2),
    material_valid=(True, True, True, True),
    material_ids=(7, 7, 7, 7),
    components=(0, 1, 2, 3),
    joints=(3, 3, 4, 4),
    qualification_state=QUALIFIED,
    teacher_bundle_schema_version=EXPECTED_TEACHER_BUNDLE_SCHEMA_VERSION,
):
    return SemanticPartitionSubjectTruthV1(
        subject_key=subject,
        source_ref=f"synthetic://{subject}",
        teacher_bundle_schema_version=teacher_bundle_schema_version,
        teacher_bundle_manifest_sha256="0" * 64,
        owner_partition_manifest=_manifest(owner_count),
        surface_owner_region_id=owner_ids,
        surface_material_region_id=material_regions,
        surface_material_region_valid=material_valid,
        surface_material_id=material_ids,
        surface_geometric_component_id=components,
        surface_mechanical_owner_joint_id=joints,
        expected_surface_sample_count=len(owner_ids),
        qualification_state=qualification_state,
    )


def test_owner_cardinality_is_independent_from_material_regions_components_and_joints():
    result = census_subject_semantic_partition_v1(_record("Snake"))
    assert result.owner.status == QUALIFIED
    assert result.owner.distinct_count == 2
    assert result.material_region.distinct_count == 3
    assert result.material_identity.distinct_count == 1
    assert result.geometric_component.distinct_count == 4
    assert result.mechanical_owner_joint.distinct_count == 2
    assert result.substitution_for_owner_forbidden


def test_same_authored_material_may_form_multiple_material_regions():
    result = census_subject_semantic_partition_v1(
        _record(
            "Bat",
            material_regions=(0, 1, 2, 3),
            material_ids=(11, 11, 11, 11),
        )
    )
    assert result.material_identity.distinct_count == 1
    assert result.material_region.distinct_count == 4


def test_missing_owner_truth_abstains_instead_of_substituting_other_axes():
    base = _record("T-Rex")
    record = SemanticPartitionSubjectTruthV1(
        subject_key=base.subject_key,
        source_ref=base.source_ref,
        teacher_bundle_schema_version=base.teacher_bundle_schema_version,
        teacher_bundle_manifest_sha256=base.teacher_bundle_manifest_sha256,
        owner_partition_manifest=None,
        surface_owner_region_id=None,
        surface_material_region_id=base.surface_material_region_id,
        surface_material_region_valid=base.surface_material_region_valid,
        surface_material_id=base.surface_material_id,
        surface_geometric_component_id=base.surface_geometric_component_id,
        surface_mechanical_owner_joint_id=base.surface_mechanical_owner_joint_id,
        expected_surface_sample_count=4,
    )
    result = census_subject_semantic_partition_v1(record)
    assert result.owner.status == ABSTAINED
    assert result.owner.distinct_count is None
    assert result.material_region.distinct_count == 3
    assert result.geometric_component.distinct_count == 4
    assert result.mechanical_owner_joint.distinct_count == 2


def test_owner_id_not_present_in_manifest_abstains():
    result = census_subject_semantic_partition_v1(
        _record("Horse", owner_ids=(0, 0, 2, 2), owner_count=2)
    )
    assert result.owner.status == ABSTAINED
    assert result.owner.reason == "OWNER_REGION_ID_NOT_IN_MANIFEST"


def test_manifest_cardinality_is_authoritative_even_when_sampling_misses_a_region():
    result = census_subject_semantic_partition_v1(
        _record("Spider", owner_ids=(0, 0, 1, 1), owner_count=3)
    )
    assert result.owner.status == QUALIFIED
    assert result.owner.distinct_count == 3
    assert result.owner.assigned_sample_count == 4


def test_material_region_requires_explicit_validity():
    result = census_subject_semantic_partition_v1(
        _record("GreenBlob", material_valid=None)
    )
    assert result.material_region.status == ABSTAINED
    assert result.material_region.reason == "MATERIAL_REGION_VALIDITY_MISSING"
    assert result.owner.status == QUALIFIED


def test_unqualified_bundle_abstains_qualified_derived_axes_without_relabeling_material_identity():
    result = census_subject_semantic_partition_v1(
        _record("Shark", qualification_state="ABSTAINED")
    )
    assert result.owner.status == ABSTAINED
    assert result.owner.reason == "TEACHER_BUNDLE_NOT_QUALIFIED"
    assert result.material_region.status == ABSTAINED
    assert result.geometric_component.status == ABSTAINED
    assert result.mechanical_owner_joint.status == ABSTAINED
    assert result.material_identity.status == QUALIFIED


def test_wrong_teacher_schema_abstains_owner_authority():
    result = census_subject_semantic_partition_v1(
        _record("George", teacher_bundle_schema_version="wrong-schema")
    )
    assert result.owner.status == ABSTAINED
    assert result.owner.reason == "TEACHER_BUNDLE_SCHEMA_VERSION_INVALID"
    assert result.material_region.status == ABSTAINED


def test_incomplete_fit8_cannot_mint_full_cohort_capacity_or_freeze_query_cap():
    result = census_fit8_semantic_partitions_v1(
        [_record("Snake"), _record("Bat")],
        supplied_query_capacity=64,
    )
    assert not result.complete_owner_cohort
    assert result.minimum_query_capacity_for_observed_fit8 is None
    assert result.supplied_query_capacity_state == CAPACITY_INCOMPLETE
    assert not result.global_query_cap_freeze_authority
    assert not result.unseen_generalization_authority
    assert not result.silent_region_merge_permitted


def test_complete_fit8_reports_observed_minimum_without_generalization_authority():
    owner_counts = {
        "Snake": 2,
        "Bat": 5,
        "T-Rex": 4,
        "Horse": 3,
        "Spider": 8,
        "GreenBlob": 1,
        "Shark": 4,
        "George": 6,
    }
    records = []
    for subject in FIT8_SUBJECTS:
        count = owner_counts[subject]
        # Deliberately sample only one owner when possible. Query-cap evidence
        # must come from the qualified partition manifest, not sampling coverage.
        sampled_count = 1 if count > 1 else count
        owner_ids = tuple(range(sampled_count))
        records.append(
            SemanticPartitionSubjectTruthV1(
                subject_key=subject,
                source_ref=f"synthetic://{subject}",
                teacher_bundle_schema_version=EXPECTED_TEACHER_BUNDLE_SCHEMA_VERSION,
                teacher_bundle_manifest_sha256="0" * 64,
                owner_partition_manifest=_manifest(count),
                surface_owner_region_id=owner_ids,
                surface_material_region_id=owner_ids,
                surface_material_region_valid=(True,) * sampled_count,
                expected_surface_sample_count=sampled_count,
            )
        )

    result = census_fit8_semantic_partitions_v1(records, supplied_query_capacity=8)
    assert result.complete_owner_cohort
    assert result.qualified_owner_subject_count == 8
    assert result.observed_owner_count_max == 8
    assert result.minimum_query_capacity_for_observed_fit8 == 8
    assert result.supplied_query_capacity_state == CAPACITY_COVERED
    assert not result.global_query_cap_freeze_authority
    assert not result.unseen_generalization_authority


def test_query_capacity_below_any_observed_qualified_owner_count_fails_closed():
    result = census_fit8_semantic_partitions_v1(
        [_record("Snake", owner_ids=(0,), owner_count=4)],
        supplied_query_capacity=3,
    )
    assert result.observed_owner_count_max == 4
    assert result.supplied_query_capacity_state == CAPACITY_FAIL_CLOSED


def test_unexpected_subject_fails_strict_fit8_census():
    with pytest.raises(ValueError, match="SEMANTIC_CENSUS_UNEXPECTED_SUBJECTS"):
        census_fit8_semantic_partitions_v1([_record("Knight")])
