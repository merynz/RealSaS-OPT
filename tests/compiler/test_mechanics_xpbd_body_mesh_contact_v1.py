from __future__ import annotations

"""Canonical body-mesh surface contact and linear-trajectory CCD regressions."""

from dataclasses import replace
import json
import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_xpbd_body_mesh_contact_v1 import (
    XPBDBodyMeshColliderV1,
    qualify_xpbd_body_mesh_colliders_v1,
    resolve_xpbd_body_mesh_contact_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    DeclaredXPBDSurfaceRequestV1, XPBDSurfacePolicyV1,
    prepare_declared_xpbd_surface_domain_v1,
    step_declared_xpbd_surface_v1,
    xpbd_surface_state_from_dict_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import canonical_mesh_lineage_hash_v1
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import canonical_skin_lineage_hash_v2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    solve_control_pose_v1, deform_mesh_lbs_v1,
)
from compiler.realsas_compiler_core.mechanics_live_runtime_v1 import (
    evaluate_mechanics_intervention_frame_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_control_solver_v1 import _rig,_skin,_support
from tests.compiler.test_mechanics_xpbd_surface_v1 import _parents,F0,F1


FACE=("MV:4","MV:5","MV:6")


def _xyz(*values):
    return np.asarray(values,dtype=np.float64)


def _contact(positions, previous, start, end, *, project=True, **kwargs):
    return resolve_xpbd_body_mesh_contact_v1(
        positions,previous,start,end,tuple(positions),
        (XPBDBodyMeshColliderV1("body",(FACE,),thickness=.025),),
        project=project,max_candidates=100,**kwargs,
    )


def _triangle(z):
    return {
        "MV:4":_xyz(-1.,-1.,z),
        "MV:5":_xyz(2.,-1.,z),
        "MV:6":_xyz(.5,2.,z),
    }


def test_static_canonical_body_triangle_projects_cloth_without_moving_body():
    p={"cloth":_xyz(0.,0.,.01)}
    old={"cloth":_xyz(0.,0.,.01)}
    body=_triangle(0.)
    out=_contact(p,old,body,body)
    assert out.projections==1
    assert out.swept_hits==0
    assert p["cloth"][2]==pytest.approx(.025)
    assert body["MV:4"][2]==0.
    final=_contact(p,old,body,body,project=False)
    assert final.max_penetration<1e-9


def test_fast_body_triangle_swept_ccd_blocks_tunneling_beyond_final_thickness():
    # Cloth and triangle start and finish well apart; discrete end-point
    # collision would miss a complete crossing. Cubic coplanarity does not.
    previous={"cloth":_xyz(0.,0.,.3)}
    positions={"cloth":_xyz(0.,0.,-.3)}
    body=_triangle(0.)
    out=_contact(positions,previous,body,body)
    assert out.swept_hits==1
    assert out.projections==1
    assert positions["cloth"][2]==pytest.approx(.025)
    check=_contact(positions,previous,body,body,project=False)
    assert check.swept_hits==0
    assert check.unresolved_sweeps==0


def test_deforming_mira_body_triangle_sweeps_across_stationary_cloth():
    previous={"cloth":_xyz(0.,0.,0.)}
    positions={"cloth":_xyz(0.,0.,0.)}
    old=_triangle(-.2)
    current=_triangle(.2)
    out=_contact(positions,previous,old,current)
    assert out.projections==1 and out.swept_hits>=1
    # The incoming particle is on the positive side of the old plane.
    assert positions["cloth"][2]>=.225-1e-7
    assert _contact(positions,previous,old,current,project=False).unresolved_sweeps==0


def test_coplanar_continuous_trajectories_fail_closed_instead_of_claiming_ccd():
    p={"cloth":_xyz(0.,0.,0.)}
    with pytest.raises(QualificationError,match="COPLANAR_SWEEP_UNSUPPORTED"):
        _contact(p,{"cloth":_xyz(0.,0.,0.)},_triangle(0.),_triangle(0.))


def test_body_selection_cannot_relabel_cloth_vertices_or_foreign_faces():
    faces=(F0,F1,FACE)
    assert qualify_xpbd_body_mesh_colliders_v1(
        faces,("MV:0","MV:1","MV:2","MV:3"),
        (XPBDBodyMeshColliderV1("body",(FACE,)),),
    )==FACE
    with pytest.raises(QualificationError,match="FACE_NOT_CANONICAL"):
        qualify_xpbd_body_mesh_colliders_v1(
            faces,("MV:0",),
            (XPBDBodyMeshColliderV1("body",(("other","face","vertex"),)),),
        )
    with pytest.raises(QualificationError,match="DOUBLE_VERTEX_OWNERSHIP"):
        qualify_xpbd_body_mesh_colliders_v1(
            faces,("MV:0",),
            (XPBDBodyMeshColliderV1("body",(F0,)),),
        )
    with pytest.raises(QualificationError,match="FACE_DOUBLE_OWNERSHIP"):
        qualify_xpbd_body_mesh_colliders_v1(
            faces,(),
            (XPBDBodyMeshColliderV1("body",(FACE,)),
             XPBDBodyMeshColliderV1("other",(FACE,))),
        )
    with pytest.raises(QualificationError,match="EXCLUSION_INVALID"):
        qualify_xpbd_body_mesh_colliders_v1(
            faces,("MV:0",),
            (XPBDBodyMeshColliderV1(
                "body",(FACE,),excluded_cloth_vertex_ids=("foreign",),
            ),),
        )


def _source(*, include_mechanics=False):
    cape,_,_,old_mechanics=_parents()
    vertices=list(cape.vertices)
    for i,xyz in enumerate(((-1.,-1.,.05),(2.,-1.,.05),(.5,2.,.05)),4):
        vertices.append(replace(
            cape.vertices[0],
            canonical_mesh_vertex_id=f"MV:{i}",
            source_candidate_vertex_id=f"CV:{i}",
            support_binding=_support(i),
            component_id="body",P=xyz,
        ))
    m=replace(cape,
        vertices=tuple(vertices),
        faces=cape.faces+(FACE,),
        edges=cape.edges+(("MV:4","MV:5"),("MV:5","MV:6"),("MV:6","MV:4")),
        mesh_lineage_hash="",
    )
    m=replace(m,mesh_lineage_hash=canonical_mesh_lineage_hash_v1(m))
    rig=_rig(m)
    old=_skin(m,rig)
    rows=old.rows+tuple(replace(
        old.rows[0],canonical_mesh_vertex_id=f"MV:{i}",
        source_carrier_vertex_id=f"CV:{i}",influences=(("J0",1.),)
    ) for i in range(3,7))
    skin=replace(old,rows=rows,skin_lineage_hash="")
    skin=replace(skin,skin_lineage_hash=canonical_skin_lineage_hash_v2(skin))
    mech=replace(old_mechanics,
        mesh_lineage_hash=m.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        skin_lineage_hash=skin.skin_lineage_hash,
        deform_structure_hash=rig.deform_structure_hash,
        control_program_hash=rig.control_program_hash,
    )
    domain=prepare_declared_xpbd_surface_domain_v1(
        m,rig,skin,mech,
        DeclaredXPBDSurfaceRequestV1(
            selected_faces=(F0,F1),pinned_vertex_ids=("MV:0","MV:1"),
        ),
    )
    primary=deform_mesh_lbs_v1(m,rig,skin,solve_control_pose_v1(rig,()))
    return (m,domain,primary,rig,skin,mech) if include_mechanics else (m,domain,primary)


def test_live_product_surface_consumes_canonical_body_addresses_and_json_replay():
    mesh,domain,primary=_source()
    policy=XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        body_mesh_colliders=(XPBDBodyMeshColliderV1(
            "arm",(FACE,),thickness=.01,
        ),),
        gravity_xyz=(0.,0.,0.),
        stretch_compliance=.005,bend_compliance=.005,
        area_compliance=.005,follow_compliance=1.,
        max_stretch_relative_error=.5,max_area_relative_error=.5,
    )
    initial=step_declared_xpbd_surface_v1(mesh,primary,domain,policy)
    assert initial.status=="INITIALIZED_XPBD_SURFACE_UNSIMULATED"
    assert [row[0] for row in initial.state.metadata[
        "kinematic_body_mesh_vertices"
    ]]==["MV:4","MV:5","MV:6"]
    advanced=step_declared_xpbd_surface_v1(
        mesh,primary,domain,policy,
        previous_state=initial.state,dt_seconds=1./120.,
    )
    assert advanced.status=="PASS_XPBD_SURFACE_RUNTIME"
    assert advanced.deformation.canonical_mesh_vertex_ids==primary.canonical_mesh_vertex_ids
    assert advanced.deformation.positions[4:]==primary.positions[4:]
    reopened=xpbd_surface_state_from_dict_v1(json.loads(
        json.dumps(advanced.state.to_dict())
    ))
    assert step_declared_xpbd_surface_v1(
        mesh,primary,domain,policy,previous_state=reopened,
    ).state==reopened
    tampered=json.loads(json.dumps(advanced.state.to_dict()))
    tampered["metadata"]["kinematic_body_mesh_vertices"][0][1][2]+=.1
    with pytest.raises(QualificationError,match="STATE_HASH_DRIFT"):
        xpbd_surface_state_from_dict_v1(tampered)
    moved_positions=list(primary.positions)
    moved_positions[4]=(-1.,-1.,.15)
    moved_primary=replace(primary,positions=tuple(moved_positions))
    with pytest.raises(QualificationError,match="BODY_CHANGED_PRIMARY_COLLIDER_REQUIRES_DT"):
        step_declared_xpbd_surface_v1(
            mesh,moved_primary,domain,policy,previous_state=advanced.state,
        )


def test_body_mesh_contact_explicit_candidate_resource_bound():
    p={"cloth":_xyz(0.,0.,.01),"cloth2":_xyz(.1,0.,.01)}
    with pytest.raises(QualificationError,match="CANDIDATE_BOUND"):
        resolve_xpbd_body_mesh_contact_v1(
            p,{k:v.copy() for k,v in p.items()},
            _triangle(0.),_triangle(0.),tuple(p),
            (XPBDBodyMeshColliderV1("body",(FACE,),thickness=.02,
             swept_vertex_triangle_ccd=False),),
            max_candidates=1,project=True,
        )


def test_pinned_cloth_body_intersection_must_be_explicitly_excluded():
    mesh,domain,primary=_source()
    intruding=XPBDBodyMeshColliderV1(
        "body",(FACE,),thickness=.1,
    )
    policy=XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        body_mesh_colliders=(intruding,),
    )
    with pytest.raises(QualificationError,match="BODY_PINNED_CONTACT_UNSATISFIABLE"):
        step_declared_xpbd_surface_v1(mesh,primary,domain,policy)
    excluded=replace(intruding,excluded_cloth_vertex_ids=domain.pinned_vertex_ids)
    admitted=step_declared_xpbd_surface_v1(
        mesh,primary,domain,replace(policy,body_mesh_colliders=(excluded,)),
    )
    assert admitted.status=="INITIALIZED_XPBD_SURFACE_UNSIMULATED"


def test_moving_body_linear_time_partition_preserves_physical_tick_state():
    mesh,domain,primary=_source()
    body=XPBDBodyMeshColliderV1("body",(FACE,),thickness=.01)
    policy=XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,body_mesh_colliders=(body,),
        gravity_xyz=(0.,0.,0.),follow_compliance=1.,
    )
    start=step_declared_xpbd_surface_v1(mesh,primary,domain,policy)
    def shifted(dz):
        values=list(primary.positions)
        values[4:]=[tuple((x,y,z+dz)) for x,y,z in values[4:]]
        return replace(primary,positions=tuple(values))
    middle=shifted(.01)
    final=shifted(.02)
    one=step_declared_xpbd_surface_v1(
        mesh,final,domain,policy,
        previous_state=start.state,dt_seconds=1./60.,
    )
    half=step_declared_xpbd_surface_v1(
        mesh,middle,domain,policy,
        previous_state=start.state,dt_seconds=1./120.,
    )
    twice=step_declared_xpbd_surface_v1(
        mesh,final,domain,policy,
        previous_state=half.state,dt_seconds=1./120.,
    )
    assert one.state.ticks==twice.state.ticks==2
    assert one.state.particles==twice.state.particles
    assert one.deformation.positions==twice.deformation.positions
    assert one.state.frames==1 and twice.state.frames==2
    assert one.state.metadata["kinematic_body_mesh_vertices"]==twice.state.metadata[
        "kinematic_body_mesh_vertices"
    ]


def test_actual_live_mechanics_frame_consumes_mira_deformed_body_collider():
    mesh,domain,primary,rig,skin,mechanics=_source(include_mechanics=True)
    policy=XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        body_mesh_colliders=(XPBDBodyMeshColliderV1(
            "body",(FACE,),thickness=.01,
        ),),
        gravity_xyz=(0.,0.,0.),follow_compliance=1.,
    )
    first=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_surface_domain=domain,xpbd_surface_policy=policy,
    )
    assert first.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert first.xpbd_surface_state is not None
    assert first.xpbd_surface_state.metadata[
        "kinematic_body_mesh_vertices"
    ]==[[vid,list(primary.positions[i])] for i,vid in enumerate(
        primary.canonical_mesh_vertex_ids
    ) if vid in FACE]
    second=evaluate_mechanics_intervention_frame_v1(
        mesh,rig,skin,mechanics,
        xpbd_surface_domain=domain,xpbd_surface_policy=policy,
        xpbd_surface_previous_state=first.xpbd_surface_state,
        xpbd_surface_dt_seconds=1./120.,
    )
    assert second.status=="PASS_MECHANICS_BEHAVIORAL_COURT"
    assert second.metadata["xpbd_surface_status"]=="PASS_XPBD_SURFACE_RUNTIME"
    assert second.xpbd_surface_state.ticks==1
    assert second.deformation.canonical_mesh_vertex_ids==primary.canonical_mesh_vertex_ids
