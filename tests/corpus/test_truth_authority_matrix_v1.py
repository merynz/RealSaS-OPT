from __future__ import annotations

import pytest

from tools.corpus.truth_authority_matrix_v1 import (
    derive_consumer_eligibility,
    new_pending_matrix,
    validate_subject_matrix,
    with_family_authority,
)


H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64


def qualified_all_core():
    matrix = new_pending_matrix(subject_id="subject", source_sha256=H1)
    families = [row["family"] for row in matrix["families"]]
    for family in families:
        matrix = with_family_authority(
            matrix,
            family=family,
            authority="QUALIFIED_CORE",
            coverage_fraction=1.0,
            producer_evidence_sha256=H2,
            court_evidence_sha256=H3,
            reason="INDEPENDENT_COURT_PASS",
        )
    return matrix


def test_pending_matrix_is_explicit_and_training_inactive():
    matrix = new_pending_matrix(subject_id="subject", source_sha256=H1)
    report = validate_subject_matrix(matrix)
    assert report["subject_disposition"] == "PENDING_AUTHORITY_AUDIT"
    assert report["training_authorized"] is False
    assert all(row["status"] == "PENDING" for row in report["consumer_eligibility"].values())


def test_all_core_truth_feeds_every_consumer():
    matrix = qualified_all_core()
    report = validate_subject_matrix(matrix)
    assert report["subject_disposition"] == "ADMIT_CORE"
    assert all(
        row["status"] == "ELIGIBLE_CORE"
        for row in report["consumer_eligibility"].values()
    )


def test_signed_distance_can_be_unavailable_without_erasing_surface_geometry():
    matrix = qualified_all_core()
    matrix = with_family_authority(
        matrix,
        family="SIGNED_DISTANCE_OCCUPANCY",
        authority="UNAVAILABLE",
        reason="NO_QUALIFIED_SOLID_SIGN_AUTHORITY",
    )
    report = validate_subject_matrix(matrix)
    iris = report["consumer_eligibility"]["IRIS_SCENE_GEOMETRY"]
    assert iris["status"] == "ELIGIBLE_CORE"
    assert iris["optional_absent_families"] == ["SIGNED_DISTANCE_OCCUPANCY"]
    assert report["subject_disposition"] == "ADMIT_MASKED"


def test_partial_surface_geometry_uses_mask_and_does_not_become_negative_truth():
    matrix = qualified_all_core()
    matrix = with_family_authority(
        matrix,
        family="SURFACE_GEOMETRY",
        authority="QUALIFIED_MASKED",
        coverage_fraction=0.75,
        validity_mask_sha256=H1,
        producer_evidence_sha256=H2,
        court_evidence_sha256=H3,
        reason="QUALIFIED_VALIDITY_MASK",
    )
    report = validate_subject_matrix(matrix)
    assert report["subject_disposition"] == "ADMIT_MASKED"
    for consumer in ("IRIS_SCENE_GEOMETRY", "TESSA_TOPOLOGY", "AXIS_RIG", "MIRA_SKIN"):
        assert report["consumer_eligibility"][consumer]["status"] == "ELIGIBLE_MASKED"


def test_missing_skin_blocks_only_skin_and_motion_consumers():
    matrix = qualified_all_core()
    matrix = with_family_authority(
        matrix,
        family="SKIN_WEIGHTS",
        authority="UNAVAILABLE",
        reason="SOURCE_HAS_NO_SKIN_TRUTH",
    )
    report = validate_subject_matrix(matrix)
    assert report["subject_disposition"] == "ADMIT_MASKED"
    assert report["consumer_eligibility"]["MIRA_SKIN"]["status"] == "INELIGIBLE_MISSING_TRUTH"
    assert report["consumer_eligibility"]["MOTION_COMPILER_COURT"]["status"] == "INELIGIBLE_MISSING_TRUTH"
    assert report["consumer_eligibility"]["IRIS_SCENE_GEOMETRY"]["status"] == "ELIGIBLE_CORE"
    assert report["consumer_eligibility"]["PRESENTATION_COMPILER_COURT"]["status"] == "ELIGIBLE_CORE"


def test_mask_forbidden_on_rig_does_not_silently_admit_axis_or_mira():
    matrix = qualified_all_core()
    matrix = with_family_authority(
        matrix,
        family="RIG_HIERARCHY",
        authority="QUALIFIED_MASKED",
        coverage_fraction=0.9,
        validity_mask_sha256=H1,
        producer_evidence_sha256=H2,
        court_evidence_sha256=H3,
        reason="PARTIAL_RIG_ONLY",
    )
    eligibility = derive_consumer_eligibility(matrix)
    assert eligibility["AXIS_RIG"]["status"] == "INELIGIBLE_MASK_FORBIDDEN"
    assert eligibility["MIRA_SKIN"]["status"] == "INELIGIBLE_MASK_FORBIDDEN"


def test_masked_truth_without_mask_is_rejected():
    matrix = qualified_all_core()
    family = next(row for row in matrix["families"] if row["family"] == "SURFACE_GEOMETRY")
    family["authority"] = "QUALIFIED_MASKED"
    family["coverage_fraction"] = 0.5
    family["validity_mask_sha256"] = None
    matrix["subject_disposition"] = "ADMIT_MASKED"
    with pytest.raises(ValueError, match="CORPUS_TRUTH_MASKED_EVIDENCE"):
        validate_subject_matrix(matrix)


def test_pre_admission_split_is_fail_closed():
    matrix = qualified_all_core()
    matrix["split_membership"] = "TRAIN"
    with pytest.raises(ValueError, match="CORPUS_TRUTH_PREMATURE_SPLIT"):
        validate_subject_matrix(matrix)


def test_quarantine_and_drop_take_precedence_over_partial_good_truth():
    matrix = qualified_all_core()
    matrix = with_family_authority(
        matrix,
        family="SURFACE_TOPOLOGY",
        authority="QUARANTINE_REPAIRABLE",
        reason="TOPOLOGY_REPAIR_REQUIRED",
    )
    assert validate_subject_matrix(matrix)["subject_disposition"] == "QUARANTINE_REPAIRABLE"

    matrix = with_family_authority(
        matrix,
        family="SOURCE_IMPORT",
        authority="DROP_UNSUITABLE",
        reason="SOURCE_IMPORT_NOT_TRUSTWORTHY",
    )
    assert validate_subject_matrix(matrix)["subject_disposition"] == "DROP_UNSUITABLE"
