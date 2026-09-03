from pathlib import Path

import pytest

from compiler.realsas_compiler_services.proof.motion_bake import (
    assert_motion_bake_binding,
    bind_qualification_owned_motion_bake,
)


def _bake():
    return bind_qualification_owned_motion_bake(
        source_product_state_hash="a" * 64,
        proof_plan_hash="b" * 64,
        clip_id="idle",
        duration_seconds=1.0,
        fps=1.0,
        loop=True,
        evaluator_semantic_version="test-qualified-evaluator.v1",
        evaluator_binding_hash="c" * 64,
        sampling_policy="TEST_EXACT_ENDPOINTS",
        rest_mesh_vertices_by_id={"V0:body": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
        triangles_by_mesh_id={"V0:body": ((0, 1, 2),)},
        frame_rows=(
            {
                "time_seconds": 0.0,
                "mesh_vertices_by_id": {"V0:body": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
                "render_order_by_view": {"V0": ("V0:body",)},
            },
            {
                "time_seconds": 1.0,
                "mesh_vertices_by_id": {"V0:body": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))},
                "render_order_by_view": {"V0": ("V0:body",)},
            },
        ),
    )


def test_qualification_owned_bake_binds_exact_product_and_plan():
    bake = _bake()
    assert_motion_bake_binding(bake, source_product_state_hash="a" * 64, proof_plan_hash="b" * 64)
    assert bake.metadata["frames_generated_by_this_binder"] is False
    assert bake.metadata["same_frames_required_for_export"] is True
    assert bake.metadata["export_solver_replay_forbidden"] is True
    with pytest.raises(ValueError, match="STALE_MOTION_BAKE_PRODUCT_BINDING"):
        assert_motion_bake_binding(bake, source_product_state_hash="d" * 64, proof_plan_hash="b" * 64)
    with pytest.raises(ValueError, match="STALE_MOTION_BAKE_PROOF_PLAN_BINDING"):
        assert_motion_bake_binding(bake, source_product_state_hash="a" * 64, proof_plan_hash="d" * 64)


def test_proof_engine_is_fail_closed_without_directional_frame_authority():
    root = Path(__file__).resolve().parents[2]
    source = (root / "compiler/realsas_compiler_core/proof_engine.py").read_text(encoding="utf-8")
    assert "motion_bake_provider" in source
    assert "MISSING_QUALIFICATION_OWNED_BAKE" in source
    assert "Direct mechanical-joint/P.xy evaluation is forbidden" in source
    for forbidden in ("motion_probe", "measure_authored_motion_v1", "skinning_transforms"):
        assert forbidden not in source


def test_binder_cannot_manufacture_directional_deformation():
    root = Path(__file__).resolve().parents[2]
    source = (root / "compiler/realsas_compiler_services/proof/motion_bake.py").read_text(encoding="utf-8")
    for forbidden in ("apply_lbs_probe_v1", "skinning_transforms", "joint.position"):
        assert forbidden not in source
