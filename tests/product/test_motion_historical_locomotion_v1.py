from __future__ import annotations

from compiler.realsas_compiler_core.motion_locomotion import build_mage_historical_phase_motion
from compiler.realsas_compiler_core.motion_quality import compile_motion_quality
from compiler.realsas_compiler_services.proof.directional_motion_evaluator import _assert_rotation_only_scope
from tests.product.test_mage_topology_motion_deformation_v1 import _fixture


def _transform(key):
    return (key.translation_xy, key.rotation_deg, key.scale_xy, key.depth_offset)


def test_phase_authored_locomotion_is_deterministic_rotation_only_and_preserves_clip_ids():
    mechanical, _mesh, _mesh_skin = _fixture()
    a=build_mage_historical_phase_motion(mechanical)
    b=build_mage_historical_phase_motion(mechanical)
    assert a.motion_state_hash == b.motion_state_hash
    assert {clip.clip_id for clip in a.clips} == {"mage_fit1_idle_v2", "mage_fit1_run_v2"}
    assert all(key.translation_xy == (0.0,0.0) for track in a.joint_tracks for key in track.keys)
    assert len(a.joint_tracks) >= 8


def test_idle_is_rest_cyclic_but_run_is_exact_nonrest_planted_cycle():
    mechanical, _mesh, _mesh_skin = _fixture()
    motion=build_mage_historical_phase_motion(mechanical)
    by_clip={clip.clip_id:clip for clip in motion.clips}
    idle_tracks=[t for t in motion.joint_tracks if t.clip_id=="mage_fit1_idle_v2"]
    run_tracks=[t for t in motion.joint_tracks if t.clip_id=="mage_fit1_run_v2"]

    assert idle_tracks and run_tracks
    assert all(abs(float(t.keys[0].rotation_deg)) <= 1e-12 for t in idle_tracks)
    assert all(_transform(t.keys[0]) == _transform(t.keys[-1]) for t in idle_tracks)
    assert all(_transform(t.keys[0]) == _transform(t.keys[-1]) for t in run_tracks)
    assert any(abs(float(t.keys[0].rotation_deg)) > 1e-6 for t in run_tracks)
    assert by_clip["mage_fit1_idle_v2"].metadata["return_to_rest_required"] is True
    assert by_clip["mage_fit1_run_v2"].metadata["return_to_rest_required"] is False
    assert by_clip["mage_fit1_run_v2"].metadata["loop_cycle_semantics"] == "CYCLIC_POSE_NOT_REST_RETURN"
    assert "PLANT_RELEASE_FLIGHT_CONTACT" in by_clip["mage_fit1_run_v2"].metadata["phase_model"]


def test_quality_compiler_polishes_phase_locomotion_without_losing_cyclic_semantics():
    mechanical, _mesh, _mesh_skin = _fixture()
    base=build_mage_historical_phase_motion(mechanical)
    polished=compile_motion_quality(base, mechanical)
    polished2=compile_motion_quality(base, mechanical)
    assert polished.motion_state_hash == polished2.motion_state_hash
    assert polished.motion_state_hash != base.motion_state_hash
    by_clip={clip.clip_id:clip for clip in polished.clips}
    assert by_clip["mage_fit1_run_v2"].metadata["return_to_rest_required"] is False
    assert by_clip["mage_fit1_run_v2"].metadata["contact_lock_claimed"] is False
    assert by_clip["mage_fit1_run_v2"].metadata["phase_aware_retime"] is True
    assert all(len(track.keys)==33 for track in polished.joint_tracks)
    product=type("Product",(),{"motion_state":polished})()
    for clip in polished.clips:
        _assert_rotation_only_scope(product,clip.clip_id)
