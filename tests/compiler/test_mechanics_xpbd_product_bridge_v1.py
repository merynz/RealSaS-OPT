from __future__ import annotations

"""Real consumer: primary Mechanics → XPBD pose → SAME MIRA skin/Graphics mesh."""

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import (
    CanonicalMechanicsIRV1,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    ControllablePuppetSessionV1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    MechanicsInterventionStepV1, evaluate_mechanics_intervention_frame_v1,
    evaluate_mechanics_intervention_sequence_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_domain_v1 import (
    DeclaredXPBDSecondaryRequestV1, prepare_declared_xpbd_secondary_domain_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_secondary_v1 import (
    XPBDSecondaryPolicyV1, XPBDSecondaryChainPolicyV1,
    XPBDSphereColliderV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_controllable_puppet_runtime_v1 import (
    qualified_synthetic_bundle,
)
from tests.compiler.test_mechanics_xpbd_domain_v1 import _fixture


def _domain_policy(mesh,rig,skin,mechanics,**kwargs):
    domain=prepare_declared_xpbd_secondary_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSecondaryRequestV1(("K0",)),
    )
    policy=XPBDSecondaryPolicyV1(
        domain_hash=domain.domain_hash,
        chains=(XPBDSecondaryChainPolicyV1("K0"),),
        **kwargs,
    )
    return domain,policy


def test_live_mechanics_intervention_executes_xpbd_before_skinning():
    mesh,rig,skin,mechanics=_fixture()
    domain,policy=_domain_policy(
        mesh,rig,skin,mechanics,gravity_xyz=(0.,0.,0.),
        spheres=(XPBDSphereColliderV1("S",(.1,2.,0.),.18),),
    )
    first=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_secondary_domain=domain,xpbd_secondary_policy=policy,
    )
    assert first.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert first.metadata["xpbd_unsimulated_initialization"] is True
    assert first.xpbd_secondary_state is not None
    second=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_secondary_domain=domain,xpbd_secondary_policy=policy,
        xpbd_secondary_previous_state=first.xpbd_secondary_state,
        xpbd_secondary_dt_seconds=1./60.,
        max_allowed_segment_strain=.02,
    )
    assert second.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert second.metadata["secondary_motion_mode"]=="DECLARED_XPBD_SECONDARY_V1"
    assert second.metadata["xpbd_status"]=="PASS_XPBD_SECONDARY_RUNTIME"
    assert second.metadata["xpbd_contact_projections"]>0
    assert second.pose != first.pose
    assert second.deformation != first.deformation
    assert second.deformation.canonical_mesh_vertex_ids == first.deformation.canonical_mesh_vertex_ids
    assert second.xpbd_secondary_state.simulation_tick_count==2


def test_live_mechanics_rejects_duplicate_solver_owner_and_domain_drift():
    mesh,rig,skin,mechanics=_fixture()
    domain,policy=_domain_policy(mesh,rig,skin,mechanics)
    with pytest.raises(QualificationError,match="POLICY_REQUIRED"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,xpbd_secondary_domain=domain,
        )
    with pytest.raises(QualificationError,match="LINEAGE_DRIFT"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,
            xpbd_secondary_domain=replace(domain,skin_lineage_hash="alien"),
            xpbd_secondary_policy=policy,
        )


def test_live_sequence_transfers_temporal_state_not_pose_heuristics():
    mesh,rig,skin,mechanics=_fixture()
    domain,policy=_domain_policy(mesh,rig,skin,mechanics)
    steps=(
        MechanicsInterventionStepV1(0.),
        MechanicsInterventionStepV1(1./60.),
        MechanicsInterventionStepV1(2./60.),
    )
    result=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,steps,
        xpbd_secondary_domain=domain,xpbd_secondary_policy=policy,
        max_allowed_segment_strain=.02,
    )
    assert len(result.frames)==3
    assert [f.xpbd_secondary_state.simulation_tick_count for f in result.frames] == [0,2,4]
    assert all(f.metadata["secondary_motion_mode"]=="DECLARED_XPBD_SECONDARY_V1"
               for f in result.frames)


def _editor_bundle():
    """Canonical-addressed editor fixture with explicit secondary chain ownership."""
    original=qualified_synthetic_bundle()
    provisional=replace(
        original.mesh,metadata={"authority":"COMPILER_CANONICAL_MESH"},
        mesh_lineage_hash="",
    )
    mesh=replace(provisional,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(provisional))
    rig=replace(
        original.rig,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        # The IK chain cannot also be secondary. Retain the control objects,
        # but remove their primary IK output and declare root-only controls.
        constraints=(),
        controls=tuple(
            replace(c,target_canonical_joint_ids=())
            if c.canonical_control_id in ("C_IK","C_POLE") else c
            for c in original.rig.controls
        ),
        metadata={"authority":"COMPILER_CANONICAL_RIG"},
    )
    skin=replace(
        original.skin,
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        metadata={"authority":"COMPILER_CANONICAL_SKIN"},
    )
    mechanics=CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=(),executable_operator_ids=(),
        rest_pose_hash="1"*64,qualification_hash="2"*64,
        mechanics_lineage_hash="3"*64,
        metadata={"authority":"COMPILER_CANONICAL_MECHANICS"},
    )
    bundle=replace(original,mesh=mesh,rig=rig,skin=skin,mechanics=mechanics)
    domain,policy=_domain_policy(mesh,rig,skin,mechanics,gravity_xyz=(0.,0.,0.))
    return bundle,domain,policy


def test_editor_xpbd_frame_snapshots_do_not_advance_simulation():
    bundle,domain,policy=_editor_bundle()
    session=ControllablePuppetSessionV1(bundle)
    session.enable_declared_xpbd_secondary(domain,policy)
    first=session.frame()
    assert first.xpbd_secondary_state.simulation_tick_count==0
    session.set_view_index(2)
    snapshot=session.frame()
    assert snapshot.xpbd_secondary_state==first.xpbd_secondary_state
    session.set_view_index(0)
    assert session.frame().xpbd_secondary_state==first.xpbd_secondary_state
    session.set_control(ControlTransformIRV1(
        "C_ROOT",translation=(.3,0.,0.),
    ))
    with pytest.raises(QualificationError,match="SNAPSHOT_PRIMARY_CHANGED_REQUIRES_DT"):
        session.frame()
    advanced=session.frame(dt_seconds=1./60.)
    assert advanced.xpbd_secondary_state.simulation_tick_count==2
    assert advanced.pose != first.pose
    assert advanced.deformed.positions != first.deformed.positions
    assert advanced.product_revision_id==first.product_revision_id
    assert tuple(advanced.deformed.canonical_mesh_vertex_ids)==tuple(
        v.canonical_mesh_vertex_id for v in bundle.mesh.vertices
    )
    assert session.frame().xpbd_secondary_state==advanced.xpbd_secondary_state
    session.disable_declared_xpbd_secondary()
    assert session.frame().xpbd_secondary_state is None


def test_editor_refuses_foreign_domain_and_disabled_dt():
    bundle,domain,policy=_editor_bundle()
    session=ControllablePuppetSessionV1(bundle)
    with pytest.raises(QualificationError,match="DT_WITHOUT_POLICY"):
        session.frame(dt_seconds=1./60.)
    with pytest.raises(QualificationError,match="DOMAIN_LINEAGE_DRIFT"):
        session.enable_declared_xpbd_secondary(
            replace(domain,mesh_lineage_hash="foreign"),policy,
        )
    assert session.frame().xpbd_secondary_state is None
