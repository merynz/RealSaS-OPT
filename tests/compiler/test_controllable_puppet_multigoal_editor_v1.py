from __future__ import annotations

"""Editor execution of the exact multi-goal Mechanics consumer, not a rig teacher."""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_controllable_puppet_runtime_v1 import (
    qualified_synthetic_bundle,
)


def _with_mechanics(*, keep_pole: bool = False):
    bundle = qualified_synthetic_bundle()
    if not keep_pole:
        # The solver supports an explicitly declared single-effector IK
        # court, but not the two-input FABRIK pole operator.
        rig = replace(bundle.rig, constraints=(replace(
            bundle.rig.constraints[0],
            input_canonical_control_ids=("C_IK",),
        ),))
        bundle = replace(bundle, rig=rig)
    rig, skin, mesh = bundle.rig, bundle.skin, bundle.mesh
    mechanics = CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=tuple(x.canonical_constraint_id for x in rig.constraints),
        executable_operator_ids=("IK",),
        rest_pose_hash="1" * 64,
        qualification_hash="2" * 64,
        mechanics_lineage_hash="3" * 64,
        metadata={"authority": "COMPILER_CANONICAL_MECHANICS"},
    )
    return replace(bundle, mechanics=mechanics)


def test_runtime_requires_canonical_mechanics_before_multi_goal_opt_in():
    bundle = qualified_synthetic_bundle()
    session = ControllablePuppetSessionV1(bundle)
    baseline = session.frame()
    with pytest.raises(QualificationError, match="MECHANICS_PARENT_REQUIRED"):
        session.enable_declared_multi_goal()
    assert session.frame() == baseline


def test_editor_multi_goal_runs_full_rig_skin_and_keeps_default_path_available():
    session = ControllablePuppetSessionV1(_with_mechanics())
    baseline = session.frame()
    session.enable_declared_multi_goal()
    solved = session.frame()
    assert solved.product_revision_id == baseline.product_revision_id
    assert tuple(v.canonical_mesh_vertex_id for v in session.bundle.mesh.vertices) == solved.deformed.canonical_mesh_vertex_ids
    assert len(solved.deformed.positions) == len(baseline.deformed.positions)
    assert solved.view_id == baseline.view_id
    # Both consumers must preserve the canonical rest pose.
    assert np.allclose(solved.deformed.positions, baseline.deformed.positions, atol=1e-6)
    session.disable_declared_multi_goal()
    assert session.frame() == baseline


def test_existing_fabrik_pole_is_never_downgraded_or_discarded():
    session = ControllablePuppetSessionV1(_with_mechanics(keep_pole=True))
    baseline = session.frame()
    with pytest.raises(QualificationError, match="POLE_OR_OPERATOR_UNSUPPORTED"):
        session.enable_declared_multi_goal()
    assert session.frame() == baseline


def test_pin_selection_and_target_policy_cannot_be_inferred_or_forged():
    session = ControllablePuppetSessionV1(_with_mechanics())
    for pins in (("UNKNOWN",), ("C_IK", "C_IK")):
        with pytest.raises(QualificationError, match="PINS_INVALID"):
            session.enable_declared_multi_goal(pinned_contact_control_ids=pins)
    with pytest.raises(QualificationError, match="PIN_TARGET_DOMAIN"):
        session.enable_declared_multi_goal(
            pinned_contact_control_ids=("C_IK",),
            pinned_world_targets={"C_ROOT": (0., 0., 0.)},
        )
    with pytest.raises(QualificationError, match="PIN_TARGET_INVALID"):
        session.enable_declared_multi_goal(
            pinned_contact_control_ids=("C_IK",),
            pinned_world_targets={"C_IK": (float("nan"), 0., 0.)},
        )
    # Invalid configuration must not modify the selected runtime mode.
    assert session.frame() == ControllablePuppetSessionV1(_with_mechanics()).frame()


def test_foreign_mechanics_parent_is_rejected_at_editor_open():
    bundle = _with_mechanics()
    forged = replace(bundle, mechanics=replace(
        bundle.mechanics, control_program_hash="foreign",
    ))
    with pytest.raises(QualificationError, match="AUTHORITY_LINEAGE_DRIFT"):
        ControllablePuppetSessionV1(forged)
