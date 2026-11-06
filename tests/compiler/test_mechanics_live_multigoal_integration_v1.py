from __future__ import annotations

"""Integration: opt-in multi-goal Mechanics through full deform runtime."""

from dataclasses import replace
import math
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, CanonicalSkinRowIRV2,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import ControlTransformIRV1
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    MechanicsInterventionStepV1,
    evaluate_mechanics_intervention_frame_v1,
    evaluate_mechanics_intervention_sequence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

from tests.compiler.test_mechanics_control_solver_v1 import _mesh
from tests.compiler.test_mechanics_multigoal_solver_v1 import _branched_rig


def _objects():
    mesh = _mesh()
    rig = replace(_branched_rig(), mesh_lineage_hash=mesh.mesh_lineage_hash)
    skin = CanonicalSkinIRV2(
        rows=(
            CanonicalSkinRowIRV2("MV:0", (("BASE", 1.0),), "CV:0"),
            CanonicalSkinRowIRV2("MV:1", (("A1", 1.0),), "CV:1"),
            CanonicalSkinRowIRV2("MV:2", (("A2", 1.0),), "CV:2"),
        ),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        carrier_skin_qualification_hash="f"*64,
        skin_code_proposal_hash="1"*64,
        skin_codec_architecture_id="reference",
        skin_codec_checkpoint_hash="2"*64,
        skin_lineage_hash="3"*64,
        metadata={"authority": "COMPILER_CANONICAL_SKIN"},
    )
    mechanics = CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=("A_IK", "B_IK"),
        executable_operator_ids=("IK",),
        rest_pose_hash="4"*64,
        qualification_hash="5"*64,
        mechanics_lineage_hash="6"*64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )
    return mesh, rig, skin, mechanics


def test_full_live_deformer_drag_and_pin_at_same_time():
    mesh, rig, skin, mechanics = _objects()
    frame = evaluate_mechanics_intervention_frame_v1(
        mesh, rig, skin, mechanics,
        overrides=(ControlTransformIRV1("A_HANDLE", translation=(-0.4, 0.8, 0.)),),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        pinned_contact_control_ids=("B_HANDLE",),
        max_allowed_constraint_residual=1.0e-3,
    )
    assert frame.status == "PASS_MECHANICS_BEHAVIORAL_COURT", frame.failures
    assert frame.max_constraint_residual < 1.0e-3
    assert frame.max_segment_length_relative_error < 1.0e-9
    assert frame.metadata["solver_mode"] == "DECLARED_MULTI_GOAL_V1"
    assert frame.metadata["multi_goal_status"] == "PASS_MULTIGOAL_BEHAVIORAL_COURT"
    assert frame.metadata["graphics_authority_claimed"] is False
    assert np.allclose(frame.deformation.positions[0], (0., 0., 0.), atol=1e-6)


def test_unreachable_live_multi_goal_returns_failure_and_not_fake_success():
    mesh, rig, skin, mechanics = _objects()
    frame = evaluate_mechanics_intervention_frame_v1(
        mesh, rig, skin, mechanics,
        overrides=(ControlTransformIRV1("A_HANDLE", translation=(4., 0., 0.)),),
        solver_mode="DECLARED_MULTI_GOAL_V1",
        pinned_contact_control_ids=("B_HANDLE",),
    )
    assert frame.status == "FAIL_MECHANICS_BEHAVIORAL_COURT"
    assert "CONSTRAINT_RESIDUAL_EXCEEDED" in frame.failures
    assert "MULTIGOAL_SOLVER_NOT_QUALIFIED" in frame.failures


def test_sequence_supports_declared_pin_without_per_frame_neural_inference():
    mesh, rig, skin, mechanics = _objects()
    # Targets must lie on the reachable arc of an articulated two-link chain
    # whose opposite branch is pinned at full extension.
    angles = (0.0, 0.2, 0.4)
    steps = tuple(MechanicsInterventionStepV1(
        time_seconds=float(i) * 0.1,
        overrides=(ControlTransformIRV1(
            "A_HANDLE", translation=(math.cos(angle) - 1.0,
                                     math.sin(angle), 0.0),
        ),),
    ) for i, angle in enumerate(angles))
    result = evaluate_mechanics_intervention_sequence_v1(
        mesh, rig, skin, mechanics, steps,
        solver_mode="DECLARED_MULTI_GOAL_V1",
        pinned_contact_control_ids=("B_HANDLE",),
        max_allowed_constraint_residual=1e-3,
    )
    assert result.status == "PASS_MECHANICS_SEQUENCE_DIAGNOSTIC"
    assert result.metadata["training_authorized"] is False
    assert result.metadata["pinned_contact_control_ids"] == ("B_HANDLE",)
    assert result.max_vertex_speed >= 0.0


def test_live_solver_mode_and_pins_fail_closed():
    mesh, rig, skin, mechanics = _objects()
    with pytest.raises(QualificationError, match="SOLVER_MODE_UNSUPPORTED"):
        evaluate_mechanics_intervention_frame_v1(
            mesh, rig, skin, mechanics, solver_mode="NO_SUCH_SOLVER",
        )
    with pytest.raises(QualificationError, match="PINS_REQUIRE_MULTI_GOAL"):
        evaluate_mechanics_intervention_frame_v1(
            mesh, rig, skin, mechanics, pinned_contact_control_ids=("B_HANDLE",),
        )
