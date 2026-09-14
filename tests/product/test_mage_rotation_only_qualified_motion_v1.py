from __future__ import annotations

from compiler.realsas_compiler_core.motion_rotation_only import (
    OPERATION_SCOPE,
    build_mage_rotation_only_qualified_motion,
)
from compiler.realsas_compiler_core.motion_deformation import verified_motion_deformation_report
from compiler.realsas_compiler_services.proof.directional_motion_evaluator import _assert_rotation_only_scope
from tests.product.test_mage_topology_motion_deformation_v1 import _fixture, _is_identity_key


def test_rotation_only_projection_is_deterministic_evaluator_scoped_and_articulated():
    mechanical, _mesh, _mesh_skin = _fixture()
    first = build_mage_rotation_only_qualified_motion(mechanical)
    second = build_mage_rotation_only_qualified_motion(mechanical)

    assert first.motion_state_hash == second.motion_state_hash
    assert {clip.clip_id for clip in first.clips} == {"mage_fit1_idle_v2", "mage_fit1_run_v2"}
    assert first.metadata["qualified_operation_scope"] == OPERATION_SCOPE
    assert first.metadata["nonzero_translation_authored"] is False
    assert len(first.joint_tracks) >= 8
    assert all(_is_identity_key(track.keys[0]) for track in first.joint_tracks)
    assert all(_is_identity_key(track.keys[-1]) for track in first.joint_tracks)
    assert all(key.translation_xy == (0.0, 0.0) for track in first.joint_tracks for key in track.keys)
    assert any(abs(key.rotation_deg) > 1e-9 for track in first.joint_tracks for key in track.keys)

    product = type("Product", (), {"motion_state": first})()
    for clip in first.clips:
        _assert_rotation_only_scope(product, clip.clip_id)


def test_rotation_only_projection_still_causes_real_multijoint_deformation_and_exact_loop():
    mechanical, mesh, mesh_skin = _fixture()
    motion = build_mage_rotation_only_qualified_motion(mechanical)
    idle = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, motion, "mage_fit1_idle_v2")
    run = verified_motion_deformation_report(mesh, mesh_skin, mechanical.skeleton, motion, "mage_fit1_run_v2")

    for report in (idle, run):
        assert report["status"] == "PASS"
        assert report["frame0_exact_identity_pass"]
        assert report["loop_closure_identity_pass"]
        assert report["dynamic_nonzero_pass"]
    assert run["peak_max_displacement"] > idle["peak_max_displacement"]
