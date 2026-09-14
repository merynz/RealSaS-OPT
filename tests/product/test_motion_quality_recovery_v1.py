from __future__ import annotations

from compiler.realsas_compiler_core.motion_quality import (
    build_mage_quality_profiles,
    compile_motion_quality,
    evaluate_named_curve,
    hermite_sample,
    impact_spring_damper_response,
    minimum_jerk,
    phase_retime,
)
from compiler.realsas_compiler_core.motion_rotation_only import build_mage_rotation_only_qualified_motion
from compiler.realsas_compiler_core.motion_deformation import verified_motion_deformation_report
from compiler.realsas_compiler_services.proof.directional_motion_evaluator import _assert_rotation_only_scope
from tests.product.test_mage_topology_motion_deformation_v1 import _fixture


def _transform(key):
    return (key.translation_xy, key.rotation_deg, key.scale_xy, key.depth_offset)


def test_recovered_curve_vocabulary_is_bounded_endpoint_exact_and_minimum_jerk_flat():
    names = (
        "LINEAR", "SMOOTH", "EASE_OUT", "SINE_IN_OUT", "HEAVY", "SNAP",
        "HOLD_THEN_SNAP", "RECOIL", "SETTLE", "ELASTIC_LITE",
    )
    for name in names:
        values = [evaluate_named_curve(name, i / 100.0) for i in range(101)]
        assert abs(values[0]) < 1e-12
        assert abs(values[-1] - 1.0) < 1e-12
        assert all(-1e-12 <= value <= 1.0 + 1e-12 for value in values)
        assert all(b + 1e-10 >= a for a, b in zip(values, values[1:]))

    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == 1.0
    eps = 1e-5
    assert minimum_jerk(eps) < 1e-12
    assert 1.0 - minimum_jerk(1.0 - eps) < 1e-12


def test_hermite_and_phase_retime_are_deterministic_and_nontrivial():
    assert hermite_sample((0.0, 1.0, 2.0), (0.0, 10.0, 0.0), 1.0) == 10.0
    profiles = build_mage_quality_profiles()
    run = profiles["mage_fit1_run_v2"]
    a = phase_retime(run, 0.08)
    b = phase_retime(run, 0.08)
    assert a == b
    assert 0.0 <= a <= 0.14
    assert abs(a - 0.08) > 1e-4
    assert abs(impact_spring_damper_response(0.3)) > 1e-5


def test_historical_quality_recovery_is_deterministic_rotation_only_loop_exact_and_articulated():
    mechanical, mesh, mesh_skin = _fixture()
    source = build_mage_rotation_only_qualified_motion(mechanical)
    first = compile_motion_quality(source, mechanical)
    second = compile_motion_quality(source, mechanical)

    assert first.motion_state_hash == second.motion_state_hash
    assert first.motion_state_hash != source.motion_state_hash
    assert first.metadata["historical_motion_engine_recovered"] is True
    assert first.metadata["contact_lock_claimed"] is False
    assert first.metadata["xpbd_secondary_claimed"] is False
    assert first.metadata["corrective_deformation_claimed"] is False
    assert {c.clip_id for c in first.clips} == {"mage_fit1_idle_v2", "mage_fit1_run_v2"}
    assert all(len(t.keys) == 33 for t in first.joint_tracks)
    assert all(key.translation_xy == (0.0, 0.0) for track in first.joint_tracks for key in track.keys)

    for clip in first.clips:
        clip_tracks = [t for t in first.joint_tracks if t.clip_id == clip.clip_id]
        assert clip_tracks
        for track in clip_tracks:
            assert _transform(track.keys[0]) == _transform(track.keys[-1])

    product = type("Product", (), {"motion_state": first})()
    for clip in first.clips:
        _assert_rotation_only_scope(product, clip.clip_id)

    idle = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, first, "mage_fit1_idle_v2")
    run = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, first, "mage_fit1_run_v2")
    for report in (idle, run):
        assert report["status"] == "PASS"
        assert report["frame0_exact_identity_pass"]
        assert report["loop_closure_identity_pass"]
        assert report["dynamic_nonzero_pass"]
    assert run["peak_max_displacement"] > idle["peak_max_displacement"]


def test_contact_markers_are_explicitly_intent_not_false_proof():
    profiles = build_mage_quality_profiles()
    run = profiles["mage_fit1_run_v2"]
    contact = [p for p in run.phases if p.requires_contact_lock]
    assert len(contact) >= 2
    assert run.contact_schedule_is_intent_only is True
