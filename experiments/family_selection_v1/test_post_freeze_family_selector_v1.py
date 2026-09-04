from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pytest

import experiments.family_selection_v1.post_freeze_family_selector_v1 as selector
from experiments.family_selection_v1.post_freeze_family_selector_v1 import (
    POLICY_ID,
    PrefitFamilyCandidateV1,
    candidate_from_mapping_v1,
    select_prefit_families_v1,
)
from experiments.family_selection_v1.prefit_family_truth_eligibility_v1 import (
    POLICY_ID as TRUTH_POLICY_ID,
    PrefitFamilyTruthEligibilityV1,
)


FREEZE_SHA = "a" * 64


def candidate(i: int, *, eligible: bool = True) -> PrefitFamilyCandidateV1:
    h = f"{i + 1:064x}"[-64:]
    return PrefitFamilyCandidateV1(
        asset_id=f"asset_blind_{i:04d}",
        source_family_id=f"family_blind_{i:04d}",
        master_native_1024=True,
        master_fit_admit=True,
        iris_truth_capable=True,
        geppetto_truth_capable=True,
        arachne_truth_capable=True,
        single_pose_core_eligible=True,
        exact_camera_raster_authority=True,
        explicit_source_textured_rgba=True,
        prefit_truth_eligibility_pass=True,
        prefit_truth_eligibility_sha256=h,
        visual_character_only_pass=eligible,
        visual_no_render_artifact_pass=True,
        visual_no_dominant_geometric_block_pass=True,
        visual_clear_character_silhouette_pass=True,
        visual_audit_id="visual-audit-prefit-v1",
        visual_audit_sha256=h,
        candidate_authority_sha256=h,
    )


def truth_report(row: PrefitFamilyCandidateV1, *, passed: bool = True) -> PrefitFamilyTruthEligibilityV1:
    return PrefitFamilyTruthEligibilityV1(
        schema="RealSaS.PrefitFamilyTruthEligibility.v1",
        policy_id=TRUTH_POLICY_ID,
        asset_id=row.asset_id,
        master_manifest_sha256="b" * 64,
        observation_authority_sha256="c" * 64,
        image_filename="source_textured_rgba.png",
        pass_prefit_truth_eligibility=passed,
        observed_triangle_count=12,
        observed_degenerate_triangle_count=0,
        stable_triangle_view_pair_count=80,
        raster_unstable_excluded_triangle_view_pair_count=16,
        hull_miss_stable_triangle_view_pair_count=0,
        fully_raster_unstable_observed_triangle_count=0,
        minimum_triangle_weighted_normal_polarity_purity=1.0,
        minimum_pixel_weighted_normal_polarity_purity=1.0,
        views=(),
        policy={"frozen": True},
        scientific_fit_steps=0,
        postfit_information_consumed=False,
        source_mesh_used_for_model_input=False,
        teacher_truth_used_for_model_input=False,
        eligibility_sha256=row.prefit_truth_eligibility_sha256,
    )


def reports(rows):
    return {row.asset_id: truth_report(row) for row in rows if row.prefit_truth_eligibility_pass}


@pytest.fixture(autouse=True)
def frozen_authority(monkeypatch):
    monkeypatch.setattr(selector, "require_family_selection_authority", lambda root: {"generic_source_fingerprint_sha256": FREEZE_SHA})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_selection_is_deterministic_prefit_only_with_typed_truth_authority():
    rows = tuple(candidate(i) for i in range(12))
    a = select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=reports(rows), count=8)
    b = select_prefit_families_v1(repo_root(), tuple(reversed(rows)), truth_eligibility_reports=reports(rows), count=8)
    assert a.policy_id == POLICY_ID
    assert a.fit_metrics_consumed is False
    assert a.scientific_fit_steps_before_selection == 0
    assert a.architecture_freeze_fingerprint_sha256 == FREEZE_SHA
    assert [x.asset_id for x in a.selected] == [x.asset_id for x in b.selected]
    assert a.selection_sha256 == b.selection_sha256
    assert len(a.selected) == 8
    for selected in a.selected:
        original = next(row for row in rows if row.asset_id == selected.asset_id)
        assert selected.prefit_truth_eligibility_sha256 == original.prefit_truth_eligibility_sha256


def test_ineligible_visual_candidate_cannot_enter_panel():
    rows = [candidate(i) for i in range(9)]
    rows[2] = replace(rows[2], visual_character_only_pass=False)
    result = select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=reports(rows), count=8)
    assert rows[2].asset_id not in {x.asset_id for x in result.selected}


def test_truth_pass_boolean_cannot_bypass_missing_typed_report():
    rows = tuple(candidate(i) for i in range(8))
    report_map = reports(rows)
    del report_map[rows[3].asset_id]
    with pytest.raises(ValueError, match="PREFIT_TRUTH_ELIGIBILITY_REPORT_REQUIRED"):
        select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=report_map, count=8)


def test_truth_report_sha_mismatch_fails_closed():
    rows = tuple(candidate(i) for i in range(8))
    report_map = reports(rows)
    report_map[rows[0].asset_id] = replace(report_map[rows[0].asset_id], eligibility_sha256="f" * 64)
    with pytest.raises(ValueError, match="PREFIT_TRUTH_ELIGIBILITY_SHA_MISMATCH"):
        select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=report_map, count=8)


def test_failed_or_postfit_contaminated_truth_report_fails_closed():
    rows = tuple(candidate(i) for i in range(8))
    failed = reports(rows)
    failed[rows[0].asset_id] = replace(failed[rows[0].asset_id], pass_prefit_truth_eligibility=False)
    with pytest.raises(ValueError, match="PREFIT_TRUTH_ELIGIBILITY_NOT_PASS"):
        select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=failed, count=8)
    contaminated = reports(rows)
    contaminated[rows[0].asset_id] = replace(contaminated[rows[0].asset_id], scientific_fit_steps=1, postfit_information_consumed=True)
    with pytest.raises(ValueError, match="PREFIT_TRUTH_ELIGIBILITY_POSTFIT_CONTAMINATION"):
        select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=contaminated, count=8)


def test_postfit_fields_are_rejected_not_ignored():
    raw = candidate(1).__dict__.copy()
    raw["fit_loss"] = 0.00001
    with pytest.raises(ValueError, match="POSTFIT_INFORMATION_FORBIDDEN"):
        candidate_from_mapping_v1(raw)


def test_unknown_fields_are_rejected_to_keep_selection_schema_closed():
    raw = candidate(1).__dict__.copy()
    raw["pretty_score"] = 0.99
    with pytest.raises(ValueError, match="UNKNOWN_PREFIT_CANDIDATE_FIELDS"):
        candidate_from_mapping_v1(raw)


def test_insufficient_eligible_population_fails_closed():
    rows = tuple(candidate(i, eligible=(i < 7)) for i in range(12))
    with pytest.raises(RuntimeError, match="INSUFFICIENT_PREFIT_ELIGIBLE_FAMILIES"):
        select_prefit_families_v1(repo_root(), rows, truth_eligibility_reports=reports(rows), count=8)
