from __future__ import annotations

"""Canonical TESSA cape surface XPBD court, without fake source material truth."""

from dataclasses import replace
import json
import numpy as np
import pytest

from compiler.realsas_compiler_core.canonical_mechanics_ir_v1 import CanonicalMechanicsIRV1
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_lineage_hash_v1
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import canonical_skin_lineage_hash_v2
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, solve_control_pose_v1, deform_mesh_lbs_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    MechanicsInterventionStepV1, evaluate_mechanics_intervention_frame_v1,
    evaluate_mechanics_intervention_sequence_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1, XPBDSurfacePolicyV1,
    prepare_declared_xpbd_surface_domain_v1,
    requalify_declared_xpbd_surface_domain_v1,
    step_declared_xpbd_surface_v1, xpbd_surface_state_from_dict_v1,
    _distance_pair,
)
from compiler.realsas_compiler_core.mechanics_xpbd_secondary_v1 import (
    XPBDPlaneColliderV1, XPBDSphereColliderV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_control_solver_v1 import _mesh,_rig,_skin,_support


F0=("MV:0","MV:2","MV:1")
F1=("MV:1","MV:2","MV:3")


def _parents():
    base=_mesh()
    vertices=(
        replace(base.vertices[0],P=(0.,1.,0.),component_id="cape"),
        replace(base.vertices[1],P=(1.,1.,0.),component_id="cape"),
        replace(base.vertices[2],P=(0.,0.,0.),component_id="cape"),
        replace(base.vertices[0],
            canonical_mesh_vertex_id="MV:3",
            source_candidate_vertex_id="CV:3",
            support_binding=_support(3),
            P=(1.,0.,0.),component_id="cape"),
    )
    mesh=replace(base,
        vertices=vertices,faces=(F0,F1),
        edges=(("MV:0","MV:2"),("MV:2","MV:1"),
               ("MV:1","MV:0"),("MV:2","MV:3"),
               ("MV:3","MV:1")),
        mesh_lineage_hash="",
    )
    mesh=replace(mesh,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))
    rig=_rig(mesh)
    old=_skin(mesh,rig)
    rows=tuple(
        replace(row, influences=(("J0",1.),))
        for row in old.rows
    ) + (replace(old.rows[0],
                  canonical_mesh_vertex_id="MV:3",
                  source_carrier_vertex_id="CV:3",
                  influences=(("J0",1.),)),)
    skin=replace(old,rows=rows,skin_lineage_hash="")
    skin=replace(skin,skin_lineage_hash=canonical_skin_lineage_hash_v2(skin))
    mechanics=CanonicalMechanicsIRV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        constraint_schedule=("Q0",),
        executable_operator_ids=("IK",),
        rest_pose_hash="1"*64,qualification_hash="2"*64,
        mechanics_lineage_hash="3"*64,
        metadata={"authority":"COMPILER_CANONICAL_MECHANICS"},
    )
    return mesh,rig,skin,mechanics


def _setup(**options):
    mesh,rig,skin,mechanics=_parents()
    domain=prepare_declared_xpbd_surface_domain_v1(
        mesh,rig,skin,mechanics,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=(F0,F1),pinned_vertex_ids=("MV:0","MV:1"),
        ),
    )
    policy=XPBDSurfacePolicyV1(domain_hash=domain.domain_hash,**options)
    primary=deform_mesh_lbs_v1(mesh,rig,skin,solve_control_pose_v1(rig,()))
    return (mesh,rig,skin,mechanics),domain,policy,primary


def _simulate(data,domain,policy,primary,**kw):
    return step_declared_xpbd_surface_v1(
        data[0],primary,domain,policy,**kw,
    )


def test_tessa_surface_face_patch_and_pinned_boundary_admitted_without_new_mesh():
    parents,domain,policy,primary=_setup()
    assert domain.selected_faces==(F0,F1)
    assert domain.vertex_ids==("MV:0","MV:1","MV:2","MV:3")
    assert domain.pinned_vertex_ids==("MV:0","MV:1")
    assert domain.dynamic_vertex_ids==("MV:2","MV:3")
    assert len(domain.stretch_edges)==5
    assert len(domain.bend_edges)==1
    assert len(domain.triangle_rest_areas)==2
    assert domain.metadata["partverse_label_is_not_motion_truth"] is True
    requalify_declared_xpbd_surface_domain_v1(*parents,domain)
    result=_simulate(parents,domain,policy,primary)
    assert result.status=="INITIALIZED_XPBD_SURFACE_UNSIMULATED"
    assert result.deformation.canonical_mesh_vertex_ids==primary.canonical_mesh_vertex_ids
    assert result.deformation.positions==primary.positions
    assert result.state.ticks==0 and result.state.frames==0


def test_surface_gravity_deforms_free_vertices_but_pins_and_other_mesh_stay_owned():
    parents,domain,policy,primary=_setup()
    start=_simulate(parents,domain,policy,primary)
    moved=_simulate(parents,domain,policy,primary,
                    previous_state=start.state,dt_seconds=1./60.)
    assert moved.status=="PASS_XPBD_SURFACE_RUNTIME"
    p={jid:np.asarray(x) for jid,x in zip(
        moved.deformation.canonical_mesh_vertex_ids,moved.deformation.positions)}
    original={jid:np.asarray(x) for jid,x in zip(
        primary.canonical_mesh_vertex_ids,primary.positions)}
    for jid in ("MV:0","MV:1"):
        assert np.array_equal(p[jid],original[jid])
    assert p["MV:2"][1]<original["MV:2"][1]
    assert p["MV:3"][1]<original["MV:3"][1]
    assert moved.max_stretch_relative_error<.025
    assert moved.max_area_relative_error<.08
    assert moved.deformation.mesh_lineage_hash==primary.mesh_lineage_hash
    assert moved.deformation.skin_lineage_hash==primary.skin_lineage_hash
    assert moved.metadata["surface_self_triangle_collision_qualified"] is False


def test_exact_time_partition_and_state_json_roundtrip():
    parents,domain,policy,primary=_setup(gravity_xyz=(0.,0.,0.))
    start=_simulate(parents,domain,policy,primary)
    once=_simulate(parents,domain,policy,primary,
                   previous_state=start.state,dt_seconds=1./60.)
    half=_simulate(parents,domain,policy,primary,
                   previous_state=start.state,dt_seconds=1./120.)
    twice=_simulate(parents,domain,policy,primary,
                    previous_state=half.state,dt_seconds=1./120.)
    assert once.deformation.positions==twice.deformation.positions
    assert once.state.particles==twice.state.particles
    assert once.state.ticks==twice.state.ticks==2
    loaded=xpbd_surface_state_from_dict_v1(
        json.loads(json.dumps(once.state.to_dict())))
    assert loaded==once.state
    still=_simulate(parents,domain,policy,primary,previous_state=loaded)
    assert still.status=="SNAPSHOT_XPBD_SURFACE_NO_STEP"
    assert still.state==loaded and still.deformation.positions==once.deformation.positions
    altered=json.loads(json.dumps(once.state.to_dict()))
    altered["particles"][0]["world_P"][0]+=0.02
    with pytest.raises(QualificationError,match="STATE_HASH_DRIFT"):
        xpbd_surface_state_from_dict_v1(altered)


def test_surface_plane_contact_projected_without_fake_contact_truth():
    plane=XPBDPlaneColliderV1("FLOOR",(0.,0.,0.),(0.,1.,0.),margin=.002)
    parents,domain,policy,primary=_setup(planes=(plane,))
    start=_simulate(parents,domain,policy,primary)
    result=_simulate(parents,domain,policy,primary,
                     previous_state=start.state,dt_seconds=1./60.)
    assert result.status=="PASS_XPBD_SURFACE_RUNTIME"
    bottom=[dict(zip(result.deformation.canonical_mesh_vertex_ids,
                     result.deformation.positions))[x][1]
            for x in domain.dynamic_vertex_ids]
    assert min(bottom)>=.002-1e-4
    assert result.contact_projections>0
    assert result.max_contact_penetration<=1e-4
    assert result.metadata["friction_qualified"] is False


def test_unreachable_world_plane_cannot_be_called_success():
    plane=XPBDPlaneColliderV1("IMPOSSIBLE",(0.,4.,0.),(0.,1.,0.))
    parents,domain,policy,primary=_setup(planes=(plane,))
    start=_simulate(parents,domain,policy,primary)
    out=_simulate(parents,domain,policy,primary,
                  previous_state=start.state,dt_seconds=1./60.)
    assert out.status=="FAIL_XPBD_SURFACE_CONSTRAINT_RESIDUAL"
    assert out.max_stretch_relative_error>.025


def test_source_surface_label_does_not_authorize_automatic_animation():
    mesh,rig,skin,mechanics=_parents()
    with pytest.raises(QualificationError,match="EXPLICIT_SELECTION_REQUIRED"):
        prepare_declared_xpbd_surface_domain_v1(mesh,rig,skin,mechanics,
            DeclaredXPBDSurfaceRequestV1(
                selected_faces=(F0,F1),pinned_vertex_ids=("MV:0","MV:1"),
                activation_authority="PARTVERSE_CAPE_LABEL",
            ))
    with pytest.raises(QualificationError,match="CUT_BOUNDARY_MUST_BE_PINNED"):
        prepare_declared_xpbd_surface_domain_v1(mesh,rig,skin,mechanics,
            DeclaredXPBDSurfaceRequestV1(
                selected_faces=(F0,),pinned_vertex_ids=("MV:0","MV:1"),
            ))


def test_rehashed_but_wrong_surface_ownership_fails_exact_source_requalification():
    parents,domain,policy,_=_setup()
    altered=replace(domain,stretch_edges=())
    forged=replace(altered,domain_hash=content_sha256({
        k:v for k,v in altered.to_dict().items() if k!="domain_hash"
    }))
    with pytest.raises(QualificationError,match="REQUALIFICATION_DRIFT"):
        requalify_declared_xpbd_surface_domain_v1(*parents,forged)


def test_surface_policy_and_state_fail_closed():
    parents,domain,policy,primary=_setup()
    start=_simulate(parents,domain,policy,primary)
    with pytest.raises(QualificationError,match="INITIALIZE_WITHOUT_DT"):
        _simulate(parents,domain,policy,primary,dt_seconds=1./60.)
    with pytest.raises(QualificationError,match="FIXED_STEP_PARTITION_DRIFT"):
        _simulate(parents,domain,policy,primary,
                  previous_state=start.state,dt_seconds=.01)
    with pytest.raises(QualificationError,match="STATE_POLICY_OR_DOMAIN_DRIFT"):
        _simulate(parents,domain,replace(policy,damping_per_second=2.),primary,
                  previous_state=start.state,dt_seconds=1./60.)
    with pytest.raises(QualificationError,match="SURFACE_PARAMETER_INVALID"):
        _simulate(parents,domain,replace(policy,stretch_compliance=-.01),primary)
    with pytest.raises(QualificationError,match="PLANE_NORMAL_NOT_UNIT"):
        _simulate(parents,domain,replace(policy,planes=(
            XPBDPlaneColliderV1("bad",(0.,0.,0.),(0.,3.,0.)),
        )),primary)
    with pytest.raises(QualificationError,match="SURFACE_COLLIDER_DUPLICATE"):
        _simulate(parents,domain,replace(policy,planes=(
            XPBDPlaneColliderV1("x",(0.,0.,0.),(0.,1.,0.)),
            XPBDPlaneColliderV1("x",(0.,0.,0.),(0.,1.,0.)),
        )),primary)


def test_two_fixed_particles_do_not_divide_by_zero_or_fake_compliant_motion():
    positions={"a":np.array((0.,0.,0.)), "b":np.array((1.,0.,0.))}
    inverse={"a":0., "b":0.}
    assert _distance_pair(positions,"a","b",1.,0.,0.,1./120.,inverse)==0.
    assert np.array_equal(positions["a"],(0.,0.,0.))
    assert np.array_equal(positions["b"],(1.,0.,0.))
    positions["b"]=np.array((1.1,0.,0.))
    with pytest.raises(QualificationError,match="FIXED_CONSTRAINT_CONFLICT"):
        _distance_pair(positions,"a","b",1.,0.,0.,1./120.,inverse)


def test_live_mechanics_consumes_surface_after_canonical_lbs():
    parents,domain,policy,primary=_setup()
    mesh,rig,skin,mechanics=parents
    first=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_surface_domain=domain,xpbd_surface_policy=policy,
    )
    assert first.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert first.metadata["xpbd_surface_status"]=="INITIALIZED_XPBD_SURFACE_UNSIMULATED"
    assert first.xpbd_surface_state is not None
    second=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_surface_domain=domain,xpbd_surface_policy=policy,
        xpbd_surface_previous_state=first.xpbd_surface_state,
        xpbd_surface_dt_seconds=1./60.,
    )
    assert second.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert second.metadata["xpbd_surface_status"]=="PASS_XPBD_SURFACE_RUNTIME"
    assert second.pose.joints==first.pose.joints
    assert second.deformation.positions!=first.deformation.positions
    assert second.deformation.canonical_mesh_vertex_ids==first.deformation.canonical_mesh_vertex_ids
    assert second.xpbd_surface_state.ticks==2


def test_live_surface_sequence_and_policy_safety():
    parents,domain,policy,primary=_setup(gravity_xyz=(0.,0.,0.))
    mesh,rig,skin,mechanics=parents
    sequence=evaluate_mechanics_intervention_sequence_v1(
        mesh,rig,skin,mechanics,
        (MechanicsInterventionStepV1(0.),
         MechanicsInterventionStepV1(1./60.),
         MechanicsInterventionStepV1(2./60.)),
        xpbd_surface_domain=domain,xpbd_surface_policy=policy,
    )
    assert sequence.status=="PASS_MECHANICS_SEQUENCE_DIAGNOSTIC"
    assert tuple(f.xpbd_surface_state.ticks for f in sequence.frames)==(0,2,4)
    with pytest.raises(QualificationError,match="SURFACE_POLICY_REQUIRED"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,xpbd_surface_domain=domain,
        )
    forged=replace(domain,mesh_lineage_hash="foreign")
    with pytest.raises(QualificationError,match="PARENT_LINEAGE_DRIFT"):
        evaluate_mechanics_intervention_frame_v1(
            mesh,rig,skin,mechanics,
            xpbd_surface_domain=forged,xpbd_surface_policy=policy,
        )


def test_live_unsatisfiable_surface_contact_fails_product_court():
    parents,domain,policy,_=_setup(planes=(
        XPBDPlaneColliderV1("WALL",(0.,4.,0.),(0.,1.,0.)),
    ))
    mesh,rig,skin,mechanics=parents
    first=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,xpbd_surface_domain=domain,
        xpbd_surface_policy=policy,
    )
    second=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,xpbd_surface_domain=domain,
        xpbd_surface_policy=policy,
        xpbd_surface_previous_state=first.xpbd_surface_state,
        xpbd_surface_dt_seconds=1./60.,
    )
    assert second.status=="FAIL_MECHANICS_BEHAVIORAL_COURT"
    assert "XPBD_SURFACE_CONTACT_OR_MATERIAL_RESIDUAL" in second.failures
