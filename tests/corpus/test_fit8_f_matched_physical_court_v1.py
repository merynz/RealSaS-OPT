from dataclasses import replace

from tools.corpus.fit8_f_matched_physical_court_v1 import (
    FMatchedPhysicalPolicyV1,
    FPhysicalCandidateMeasurementV1,
    evaluate_f_matched_physical_court_v1,
)


H = "a" * 64
H2 = "b" * 64


def _policy():
    return FMatchedPhysicalPolicyV1(
        max_symmetric_surface_p95_pitch=0.05,
        max_normal_p95_deg=10.0,
        max_depth_p95_pitch=0.05,
        min_silhouette_iou=0.99,
        min_thin_structure_recall=0.99,
        max_self_intersections=0,
        max_degenerate_faces=0,
        require_exact_topology=True,
        require_zero_false_bridges=True,
        require_zero_missing_sheets=True,
        require_zero_residual_false_negatives=True,
        provenance="synthetic_preregistered_guard",
    )


def _row(arm, *, payload, topology=True, residual_required=0, residual_activated=0):
    return FPhysicalCandidateMeasurementV1(
        subject_key="Synthetic",
        arm=arm,
        teacher_bundle_sha256=H,
        extraction_truth_sha256=H,
        canonical_camera_contract_sha256=H,
        extraction_scale_manifest_sha256=H,
        measurement_protocol_sha256=H,
        exact_topology=topology,
        symmetric_surface_p95_pitch=0.01,
        normal_p95_deg=2.0,
        depth_p95_pitch=0.01,
        silhouette_iou=0.999,
        false_bridge_count=0,
        missing_sheet_count=0,
        self_intersection_count=0,
        degenerate_face_count=0,
        thin_structure_recall=1.0,
        residual_required_entity_count=residual_required,
        residual_activated_entity_count=residual_activated,
        residual_false_negative_count=residual_required - residual_activated,
        explicit_variable_payload_count=payload,
        candidate_vertex_count=100,
        candidate_face_count=200,
        peak_working_bytes=1000,
    )


def test_mismatched_teacher_or_measurement_authority_is_rejected():
    a = _row("F_B", payload=0)
    b = replace(_row("F_R", payload=1), canonical_camera_contract_sha256=H2)
    try:
        evaluate_f_matched_physical_court_v1([a, b], policy=_policy())
    except ValueError as exc:
        assert "AUTHORITY_NOT_MATCHED" in str(exc)
    else:
        raise AssertionError("expected mismatched authority rejection")


def test_lower_payload_cannot_beat_a_quality_pass_if_topology_fails():
    cheap_but_wrong = _row("F_B", payload=0, topology=False)
    correct = _row("F_R", payload=4)
    report = evaluate_f_matched_physical_court_v1(
        [cheap_but_wrong, correct], policy=_policy()
    )
    assert report.admitted_arms == ("F_R",)
    assert report.lowest_payload_admitted_arm == "F_R"
    assert "TOPOLOGY_NOT_EXACT" in report.decisions[0].failure_reasons


def test_residual_false_negative_is_fail_closed_even_when_payload_is_small():
    missed = _row("F_R", payload=1, residual_required=3, residual_activated=2)
    explicit = _row("F_C", payload=10, residual_required=3, residual_activated=3)
    report = evaluate_f_matched_physical_court_v1([missed, explicit], policy=_policy())
    assert report.admitted_arms == ("F_C",)
    assert "RESIDUAL_FALSE_NEGATIVE" in report.decisions[0].failure_reasons


def test_complexity_breaks_ties_only_after_both_arms_pass_quality():
    f_r = _row("F_R", payload=2)
    f_c = _row("F_C", payload=20)
    report = evaluate_f_matched_physical_court_v1([f_r, f_c], policy=_policy())
    assert report.admitted_arms == ("F_R", "F_C")
    assert report.lowest_payload_admitted_arm == "F_R"
