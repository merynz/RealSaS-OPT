from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import pytest

from experiments.family_selection_v1.post_freeze_family_selector_v1 import (
    POLICY_ID,
    PrefitFamilyCandidateV1,
    candidate_from_mapping_v1,
    select_prefit_families_v1,
)


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
        visual_character_only_pass=eligible,
        visual_no_render_artifact_pass=True,
        visual_no_dominant_geometric_block_pass=True,
        visual_clear_character_silhouette_pass=True,
        visual_audit_id="visual-audit-prefit-v1",
        visual_audit_sha256=h,
        candidate_authority_sha256=h,
    )


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_freeze_fingerprint() -> str:
    seal = json.loads((repo_root() / "canonical/ARCHITECTURE_FREEZE_V1.json").read_text(encoding="utf-8"))
    assert seal["status"] == "PASS_ARCHITECTURE_FROZEN"
    assert seal["family_selection_authorized"] is True
    return str(seal["generic_source_fingerprint_sha256"])


def test_selection_requires_frozen_architecture_and_is_deterministic_prefit_only():
    rows = tuple(candidate(i) for i in range(12))
    a = select_prefit_families_v1(repo_root(), rows, count=8)
    b = select_prefit_families_v1(repo_root(), tuple(reversed(rows)), count=8)
    assert a.policy_id == POLICY_ID
    assert a.fit_metrics_consumed is False
    assert a.scientific_fit_steps_before_selection == 0
    assert a.architecture_freeze_fingerprint_sha256 == current_freeze_fingerprint()
    assert [x.asset_id for x in a.selected] == [x.asset_id for x in b.selected]
    assert a.selection_sha256 == b.selection_sha256
    assert len(a.selected) == 8


def test_ineligible_visual_candidate_cannot_enter_panel():
    rows = [candidate(i) for i in range(9)]
    rows[2] = replace(rows[2], visual_character_only_pass=False)
    result = select_prefit_families_v1(repo_root(), rows, count=8)
    assert rows[2].asset_id not in {x.asset_id for x in result.selected}


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
        select_prefit_families_v1(repo_root(), rows, count=8)
