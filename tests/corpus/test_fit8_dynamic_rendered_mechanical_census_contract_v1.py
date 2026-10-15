import pytest

from tools.corpus.fit8_dynamic_rendered_mechanical_census_contract_v1 import (
    DynamicPoseSampleV1,
    DynamicSubjectCensusV1,
)


SHA = "1" * 64


def sample(*, presentation=True):
    kwargs = {}
    if presentation:
        kwargs = dict(
            directed_cycle_view_count=8,
            mixed_pair_view_count=7,
            mixed_pair_count_sum=12,
            ordered_relation_count_sum=40,
            maximum_distinct_component_stack=4,
            new_directed_edges_vs_static=9,
            reversed_directed_edges_vs_static=3,
            strict_pair_flips_vs_static=2,
        )
    return DynamicPoseSampleV1(
        clip_name="Run",
        clip_index=0,
        normalized_time=0.5,
        authored_time=0.5,
        quotient_worst_separation=0.0,
        edge_stretch_p99=1.2,
        face_condition_p99=1.8,
        face_area_ratio_p01=0.7,
        face_area_ratio_p99=1.4,
        face_area_ratio_min=0.5,
        face_area_ratio_max=1.8,
        **kwargs,
    )


def subject(samples, *, presentation_authority="QUALIFIED", raw_maxima_diagnostic_only=True):
    return DynamicSubjectCensusV1(
        subject_key="fit8_fixture_v1",
        source_sha256=SHA,
        carrier_sha256=SHA,
        animation_evaluator_sha256=SHA,
        static_ray_producer_sha256=SHA,
        sampling_policy_sha256=SHA,
        mechanics_authority_state="DIAGNOSTIC",
        presentation_authority_state=presentation_authority,
        samples=tuple(samples),
        all_native_clips_covered=True,
        raw_maxima_diagnostic_only=raw_maxima_diagnostic_only,
    )


def test_complete_qualified_presentation_is_valid():
    row = subject([sample(), sample()])
    row.validate()
    assert len(row.sha256()) == 64


def test_qualified_presentation_rejects_missing_sample_graph():
    row = subject([sample(), sample(presentation=False)])
    with pytest.raises(ValueError, match="QUALIFIED_PRESENTATION_REQUIRES_ALL_SAMPLES"):
        row.validate()


def test_partial_presentation_fields_are_rejected():
    row = sample()
    broken = DynamicPoseSampleV1(**{**row.__dict__, "strict_pair_flips_vs_static": None})
    with pytest.raises(ValueError, match="PRESENTATION_PARTIAL"):
        broken.validate()


def test_raw_maxima_cannot_be_promoted_to_scientific_gate():
    row = subject([sample()], raw_maxima_diagnostic_only=False)
    with pytest.raises(ValueError, match="RAW_MAXIMA_MUST_REMAIN_DIAGNOSTIC"):
        row.validate()


def test_pass_claims_are_forbidden_at_census_layer():
    row = subject([sample()])
    broken = DynamicSubjectCensusV1(**{**row.__dict__, "fit8_pass_claimed": True})
    with pytest.raises(ValueError, match="UNSUPPORTED_PASS_CLAIM"):
        broken.validate()
