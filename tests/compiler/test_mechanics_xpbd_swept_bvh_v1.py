from __future__ import annotations

"""Canonical swept BVH broadphase exact-candidate equivalence and bounds."""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_xpbd_swept_bvh_v1 import (
    SweptAABBPrimitiveV1,
    prepare_swept_bvh_v1,
    query_swept_bvh_v1,
    swept_aabb_from_endpoints_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_body_mesh_contact_v1 import (
    XPBDBodyMeshColliderV1,
    prepare_xpbd_body_mesh_swept_bvh_v1,
    resolve_xpbd_body_mesh_contact_v1,
)
from compiler.realsas_compiler_core.types import QualificationError

FACE=("a","b","c")


def v(x,y,z):
    return np.asarray((x,y,z),dtype=np.float64)


def test_bvh_sorted_candidates_match_bruteforce_3d_overlap_and_prune_far_faces():
    primitives=tuple(
        swept_aabb_from_endpoints_v1(
            (v(i*5.,0.,0.),v(i*5.+1.,1.,0.)),
            (v(i*5.,0.,.02),v(i*5.+1.,1.,.02)),
            margin=.025,
        ) for i in range(512)
    )
    tree=prepare_swept_bvh_v1(primitives)
    for at in (0.,3.,20.,60.,200.,765.,2550.):
        q=swept_aabb_from_endpoints_v1(
            (v(at,0.2,-.1),),(v(at,0.2,.1),),margin=0.,
        )
        found=query_swept_bvh_v1(tree,q,max_candidates=128)
        direct=tuple(i for i,p in enumerate(primitives) if all(
            q.maximum[axis]>=p.minimum[axis] and
            q.minimum[axis]<=p.maximum[axis]
            for axis in range(3)
        ))
        assert found.candidate_indices==direct
        assert found.examined_leaf_primitives<32
        assert found.visited_nodes<len(tree.nodes)


def test_bvh_deterministic_order_and_reject_invalid_bounds_or_resource_limits():
    primitives=tuple(
        SweptAABBPrimitiveV1((float(i),0.,0.),(float(i+1),1.,1.))
        for i in range(17)
    )
    left=prepare_swept_bvh_v1(primitives)
    right=prepare_swept_bvh_v1(primitives)
    assert left==right
    q=SweptAABBPrimitiveV1((2.5,0.,0.),(3.5,1.,1.))
    assert query_swept_bvh_v1(left,q,max_candidates=10).candidate_indices==(2,3)
    with pytest.raises(QualificationError,match="CANDIDATE_BOUND"):
        query_swept_bvh_v1(left,q,max_candidates=1)
    with pytest.raises(QualificationError,match="BOUNDS_INVALID"):
        prepare_swept_bvh_v1((SweptAABBPrimitiveV1((2.,0.,0.),(1.,1.,1.)),))
    with pytest.raises(QualificationError,match="RESOURCE_BOUND"):
        prepare_swept_bvh_v1(primitives,max_primitives=16)


def test_prepared_mesh_bvh_binding_rejects_stale_body_transform_and_policy():
    collider=XPBDBodyMeshColliderV1("body",(FACE,),thickness=.02)
    before={"a":v(-1.,-1.,0.),"b":v(1.,-1.,0.),"c":v(0.,1.,0.)}
    after={k:x.copy() for k,x in before.items()}
    prepared=prepare_xpbd_body_mesh_swept_bvh_v1((collider,),before,after)
    positions={"cloth":v(0.,0.,.01)}
    previous={"cloth":v(0.,0.,.01)}
    result=resolve_xpbd_body_mesh_contact_v1(
        positions,previous,before,after,("cloth",),(collider,),
        max_candidates=100,project=True,prepared_bvh=prepared,
    )
    assert result.projections==1
    assert positions["cloth"][2]==pytest.approx(.02)
    assert result.broadphase_visited_nodes>0
    shifted={**after,"a":v(-1.,-1.,.01)}
    with pytest.raises(QualificationError,match="BINDING_DRIFT"):
        resolve_xpbd_body_mesh_contact_v1(
            positions,previous,before,shifted,("cloth",),(collider,),
            max_candidates=100,project=False,prepared_bvh=prepared,
        )
    with pytest.raises(QualificationError,match="BINDING_DRIFT"):
        resolve_xpbd_body_mesh_contact_v1(
            positions,previous,before,after,("cloth",),
            (replace(collider,thickness=.03),),
            max_candidates=100,project=False,prepared_bvh=prepared,
        )


def test_swept_bvh_never_culls_fast_moving_body_triangle_through_cloth():
    collider=XPBDBodyMeshColliderV1("body",(FACE,),thickness=.025)
    old={"a":v(-1.,-1.,-.3),"b":v(1.,-1.,-.3),"c":v(0.,1.,-.3)}
    now={k:np.asarray((xyz[0],xyz[1],.3)) for k,xyz in old.items()}
    prepared=prepare_xpbd_body_mesh_swept_bvh_v1((collider,),old,now)
    before={"cloth":v(0.,0.,0.)}
    after={"cloth":v(0.,0.,0.)}
    r=resolve_xpbd_body_mesh_contact_v1(
        after,before,old,now,("cloth",),(collider,),
        max_candidates=5,project=True,prepared_bvh=prepared,
    )
    assert r.swept_hits==1
    assert r.candidates==1
    assert after["cloth"][2]>=.325-1e-7
