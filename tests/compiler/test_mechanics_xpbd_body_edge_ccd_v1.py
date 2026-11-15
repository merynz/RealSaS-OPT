from __future__ import annotations

"""Motion-aware body/cloth edge CCD, BVH and live XPBD integration courts."""

from dataclasses import replace
import json

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_xpbd_body_edge_ccd_v1 import (
    evaluate_swept_body_edge_pair_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_body_mesh_contact_v1 import (
    XPBDBodyMeshColliderV1,
    prepare_xpbd_body_mesh_swept_bvh_v1,
    resolve_xpbd_body_mesh_edge_contact_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfacePolicyV1, step_declared_xpbd_surface_v1,
    xpbd_surface_state_from_dict_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_body_mesh_contact_v1 import (
    FACE, _source,
)


def v(x,y,z):
    return np.asarray((x,y,z),dtype=np.float64)


def _edge(z):
    return (v(-1.,0.,z),v(1.,0.,z))


def _body(z):
    return (v(0.,-1.,z),v(0.,1.,z))


def test_fast_cloth_edge_crossing_body_edge_cannot_tunnel():
    collision=evaluate_swept_body_edge_pair_v1(
        _edge(.3),_edge(-.3),_body(0.),_body(0.),
        thickness=.02,
    )
    assert collision is not None
    assert collision.swept_hits==1
    assert collision.required_separation==pytest.approx(.32)
    assert collision.cloth_edge_parameter==pytest.approx(.5)
    assert collision.body_edge_parameter==pytest.approx(.5)
    assert collision.normal==pytest.approx((0.,0.,1.))


def test_moving_mira_body_edge_crossing_stationary_cloth_is_detected():
    collision=evaluate_swept_body_edge_pair_v1(
        _edge(0.),_edge(0.),_body(-.3),_body(.3),
        thickness=.02,
    )
    assert collision is not None and collision.swept_hits==1
    assert collision.required_separation==pytest.approx(.32)
    assert collision.normal==pytest.approx((0.,0.,1.))


def test_static_nearby_edges_have_discrete_nonpenetration_when_sweeps_disabled():
    collision=evaluate_swept_body_edge_pair_v1(
        _edge(.01),_edge(.01),_body(0.),_body(0.),
        thickness=.03,check_sweeps=False,
    )
    assert collision is not None
    assert collision.swept_hits==0
    assert collision.endpoint_penetration==pytest.approx(.02)
    assert collision.required_separation==pytest.approx(.02)


def test_parallel_and_coplanar_unresolved_sweeps_fail_closed():
    with pytest.raises(QualificationError,match="COPLANAR_SWEEP_UNSUPPORTED"):
        evaluate_swept_body_edge_pair_v1(
            _edge(0.),_edge(0.),_body(0.),_body(0.),
            thickness=.01,
        )
    with pytest.raises(QualificationError,match="PARALLEL_NORMAL_UNSUPPORTED"):
        evaluate_swept_body_edge_pair_v1(
            (v(-1.,0.,.01),v(1.,0.,.01)),
            (v(-1.,0.,.01),v(1.,0.,.01)),
            (v(-1.,0.,0.),v(1.,0.,0.)),
            (v(-1.,0.,0.),v(1.,0.,0.)),
            thickness=.03,check_sweeps=False,
        )


def _direct_edge_solver(cloth_start_z,cloth_end_z,
                        body_start_z=0.,body_end_z=0.,*,pinned=False):
    collider=XPBDBodyMeshColliderV1(
        "arm",(("X","Y","Z"),),thickness=.02,
        swept_edge_edge_ccd=True,
    )
    # The triangle's Y--X edge runs along the body Y axis and intersects
    # the middle of the moving cloth edge at the contact instant.
    start={
        "X":v(0.,-1.,body_start_z),
        "Y":v(0.,1.,body_start_z),
        "Z":v(1.,-1.,body_start_z),
    }
    end={k:v(x[0],x[1],body_end_z) for k,x in start.items()}
    before={"a":v(-1.,0.,cloth_start_z),"b":v(1.,0.,cloth_start_z)}
    now={"a":v(-1.,0.,cloth_end_z),"b":v(1.,0.,cloth_end_z)}
    inv={"a":0. if pinned else 1.,"b":1.}
    bvh=prepare_xpbd_body_mesh_swept_bvh_v1((collider,),start,end)
    result=resolve_xpbd_body_mesh_edge_contact_v1(
        now,before,start,end,(("a","b",2.),),
        inv,(collider,),max_candidates=10,project=True,
        prepared_bvh=bvh,
    )
    return result,now,before,start,end,bvh,collider,inv


def test_body_edge_xpbd_separates_high_speed_crossing_and_admits_final_residual():
    result,now,before,start,end,bvh,collider,mass=_direct_edge_solver(.3,-.3)
    assert result.projections>=1 and result.swept_hits>=1
    assert result.broadphase_examined_primitives>0
    assert now["a"][2]==pytest.approx(.02)
    assert now["b"][2]==pytest.approx(.02)
    verified=resolve_xpbd_body_mesh_edge_contact_v1(
        now,before,start,end,(("a","b",2.),),mass,
        (collider,),max_candidates=10,project=False,
        prepared_bvh=bvh,
    )
    assert verified.unresolved_sweeps==0
    assert verified.max_penetration<1e-7


def test_one_pinned_endpoint_stays_immutable_but_other_endpoint_resolves():
    result,now,before,*_=_direct_edge_solver(.3,-.3,pinned=True)
    assert result.projections>=1
    assert now["a"]==pytest.approx((-1.,0.,-.3))
    assert now["b"][2]>.02


def test_direct_edge_ccd_rejects_stale_bvh_and_candidate_budget():
    result,now,before,start,end,bvh,collider,mass=_direct_edge_solver(.3,-.3)
    with pytest.raises(QualificationError,match="BINDING_DRIFT"):
        resolve_xpbd_body_mesh_edge_contact_v1(
            now,before,start,{**end,"X":v(0.,-1.,.1)},
            (("a","b",2.),),mass,(collider,),
            max_candidates=10,project=False,prepared_bvh=bvh,
        )
    with pytest.raises(QualificationError,match="CANDIDATE_BOUND"):
        resolve_xpbd_body_mesh_edge_contact_v1(
            now,before,start,end,(("a","b",2.),),mass,(collider,),
            max_candidates=1,project=False,prepared_bvh=bvh,
        )


def test_live_canonical_surface_edge_ccd_opt_in_and_replay_roundtrip():
    mesh,domain,primary=_source()
    collider=XPBDBodyMeshColliderV1(
        "arm",(FACE,),thickness=.01,
        swept_edge_edge_ccd=True,
    )
    policy=XPBDSurfacePolicyV1(
        domain_hash=domain.domain_hash,
        body_mesh_colliders=(collider,),
        gravity_xyz=(0.,0.,0.),
        follow_compliance=1.,
    )
    start=step_declared_xpbd_surface_v1(mesh,primary,domain,policy)
    tick=step_declared_xpbd_surface_v1(
        mesh,primary,domain,policy,previous_state=start.state,
        dt_seconds=1./120.,
    )
    assert tick.status=="PASS_XPBD_SURFACE_RUNTIME"
    assert tick.metadata["body_mesh_edge_linear_ccd_qualified"] is True
    assert tick.metadata["body_mesh_edge_bvh_examined_primitives"]>=0
    assert tick.deformation.positions[4:]==primary.positions[4:]
    reopened=xpbd_surface_state_from_dict_v1(
        json.loads(json.dumps(tick.state.to_dict()))
    )
    step1=step_declared_xpbd_surface_v1(
        mesh,primary,domain,policy,previous_state=tick.state,
        dt_seconds=1./120.,
    )
    step2=step_declared_xpbd_surface_v1(
        mesh,primary,domain,policy,previous_state=reopened,
        dt_seconds=1./120.,
    )
    assert step1.state==step2.state


def test_edge_contact_flag_off_preserves_original_authority_without_extra_work():
    collider=XPBDBodyMeshColliderV1("arm",(FACE,),swept_edge_edge_ccd=False)
    result=resolve_xpbd_body_mesh_edge_contact_v1(
        {"a":v(0.,0.,0.),"b":v(1.,0.,0.)},
        {"a":v(0.,0.,0.),"b":v(1.,0.,0.)},
        {},{},(("a","b",1.),),{"a":1.,"b":1.},(collider,),
        max_candidates=10,project=True,
    )
    assert result.projections==result.candidates==result.swept_hits==0
