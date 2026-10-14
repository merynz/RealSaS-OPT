from compiler.realsas_compiler_core.cao_authority_v1 import (
    CAOAppearanceSampleV1,
    CAOVisibleSurfaceCoherenceReportV1,
    resolve_cao_sample_v1,
)


def test_cao_source_exact_wins_over_high_confidence_learned_completion():
    source = CAOAppearanceSampleV1(
        surface_address="face:12:bary:0.2,0.3,0.5",
        rgba=(0.1, 0.2, 0.3, 1.0),
        provenance="SOURCE_EXACT",
        origin="OBSERVED_SOURCE",
        confidence=0.8,
        source_view=0,
    )
    learned = CAOAppearanceSampleV1(
        surface_address=source.surface_address,
        rgba=(0.9, 0.8, 0.7, 1.0),
        provenance="LEARNED_SOURCE_CONSISTENT",
        origin="LEARNED_FIELD",
        confidence=0.999,
    )
    assert resolve_cao_sample_v1([learned, source]) == source


def test_cao_generated_or_learned_evidence_cannot_claim_source_provenance():
    row = CAOAppearanceSampleV1(
        surface_address="face:1:bary:1,0,0",
        rgba=(0.1, 0.1, 0.1, 1.0),
        provenance="SOURCE_EXACT",
        origin="LEARNED_FIELD",
        confidence=1.0,
    )
    try:
        row.validate()
    except ValueError as exc:
        assert "PROVENANCE_ORIGIN_MISMATCH" in str(exc)
    else:
        raise AssertionError("learned evidence must never masquerade as exact source")


def test_visible_surface_coherence_fails_on_any_unexplained_gap_or_fragment():
    clean = CAOVisibleSurfaceCoherenceReportV1(
        geometric_visible_pixels=100,
        appearance_visible_pixels=100,
        unexplained_missing_pixels=0,
        fragmented_island_pixels=0,
        maximum_unexplained_hole_area_px=0,
    )
    assert clean.pass_fail_closed

    hole = CAOVisibleSurfaceCoherenceReportV1(
        geometric_visible_pixels=100,
        appearance_visible_pixels=99,
        unexplained_missing_pixels=1,
        fragmented_island_pixels=0,
        maximum_unexplained_hole_area_px=1,
    )
    assert not hole.pass_fail_closed

    pepper = CAOVisibleSurfaceCoherenceReportV1(
        geometric_visible_pixels=100,
        appearance_visible_pixels=100,
        unexplained_missing_pixels=0,
        fragmented_island_pixels=2,
        maximum_unexplained_hole_area_px=0,
    )
    assert not pepper.pass_fail_closed
