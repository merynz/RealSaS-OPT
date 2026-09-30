from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.appearance_canonical_completion_v1 import (
    build_all_view_unseen_canonical_completion,
    prolongate_control_pm_to_adaptive_faces,
)
from compiler.realsas_compiler_core.appearance_completion_v2 import (
    SurfaceSampleGraph,
)


def _line_graph(count: int) -> SurfaceSampleGraph:
    edges=[(i,i+1) for i in range(count-1)]
    edge_a=np.asarray([a for a,_ in edges],dtype=np.int32)
    edge_b=np.asarray([b for _,b in edges],dtype=np.int32)
    directed=[]
    for a,b in edges:
        directed.extend(((a,b),(b,a)))
    directed.sort()
    offsets=np.zeros(count+1,dtype=np.int64)
    indices=[]
    cursor=0
    for node in range(count):
        while cursor<len(directed) and directed[cursor][0]==node:
            indices.append(directed[cursor][1])
            cursor+=1
        offsets[node+1]=len(indices)
    return SurfaceSampleGraph(
        offsets=offsets,
        indices=np.asarray(indices,dtype=np.int32),
        edge_a=edge_a,
        edge_b=edge_b,
    )


def test_all_view_unseen_completion_writes_only_unseen_samples():
    direct_valid=np.zeros((8,3),dtype=bool)
    direct_rgba=np.zeros((8,3,4),dtype=np.uint8)
    direct_valid[0,0]=True
    direct_rgba[0,0]=(255,0,0,255)
    direct_valid[1,2]=True
    direct_rgba[1,2]=(0,0,255,255)
    support=np.ones((8,1),dtype=np.float64)
    result=build_all_view_unseen_canonical_completion(
        direct_valid=direct_valid,
        direct_rgba=direct_rgba,
        face_support_by_view=support,
        sample_face_index=np.zeros(3,dtype=np.int32),
        sample_component_index=np.zeros(3,dtype=np.int32),
        sample_positions=np.asarray(
            ((0,0,0),(1,0,0),(2,0,0)),
            dtype=np.float64,
        ),
        surface_graph=_line_graph(3),
    )
    assert result.globally_unseen_mask.tolist()==[False,True,False]
    assert np.all(result.rgba[[0,2]]==0)
    assert int(result.rgba[1,3])==255
    assert int(result.rgba[1,0])>0
    assert int(result.rgba[1,2])>0
    assert result.solver_stats.source_constraints_exact is True


def test_anchor_donor_is_max_support_not_first_view():
    direct_valid=np.zeros((8,2),dtype=bool)
    direct_rgba=np.zeros((8,2,4),dtype=np.uint8)
    direct_valid[0,0]=True
    direct_rgba[0,0]=(255,0,0,255)
    direct_valid[3,0]=True
    direct_rgba[3,0]=(0,255,0,255)
    direct_valid[1,1]=True
    direct_rgba[1,1]=(0,0,255,255)
    support=np.zeros((8,1),dtype=np.float64)
    support[0,0]=0.2
    support[3,0]=0.9
    support[1,0]=0.8
    result=build_all_view_unseen_canonical_completion(
        direct_valid=direct_valid,
        direct_rgba=direct_rgba,
        face_support_by_view=support,
        sample_face_index=np.zeros(2,dtype=np.int32),
        sample_component_index=np.zeros(2,dtype=np.int32),
        sample_positions=np.asarray(
            ((0,0,0),(1,0,0)),
            dtype=np.float64,
        ),
        surface_graph=_line_graph(2),
    )
    assert result.globally_unseen_mask.tolist()==[False,False]
    assert result.anchor_source_view.tolist()==[3,1]
    assert np.all(result.rgba==0)


def test_directional_source_arrays_are_not_mutated():
    direct_valid=np.zeros((8,3),dtype=bool)
    direct_rgba=np.zeros((8,3,4),dtype=np.uint8)
    direct_valid[2,0]=True
    direct_valid[2,2]=True
    direct_rgba[2,0]=(20,40,60,255)
    direct_rgba[2,2]=(80,100,120,255)
    before=direct_rgba.copy()
    support=np.ones((8,1),dtype=np.float64)
    build_all_view_unseen_canonical_completion(
        direct_valid=direct_valid,
        direct_rgba=direct_rgba,
        face_support_by_view=support,
        sample_face_index=np.zeros(3,dtype=np.int32),
        sample_component_index=np.zeros(3,dtype=np.int32),
        sample_positions=np.asarray(
            ((0,0,0),(1,0,0),(2,0,0)),
            dtype=np.float64,
        ),
        surface_graph=_line_graph(3),
    )
    assert np.array_equal(direct_rgba,before)


def test_control_field_prolongation_is_affine_exact_and_face_local():
    control_resolution=4
    control=[]
    for face in range(2):
        denom=float(control_resolution-1)
        for j in range(control_resolution):
            for i in range(control_resolution-j):
                u=float(i)/denom
                v=float(j)/denom
                # Affine PM field; second face carries a distinct offset so any
                # accidental cross-face transfer is immediately visible.
                base=0.1*face
                control.append((base+0.2*u,base+0.3*v,base+0.1*u+0.1*v,0.8))
    control=np.asarray(control,dtype=np.float64)
    resolutions=np.asarray((7,8),dtype=np.int32)
    dense=prolongate_control_pm_to_adaptive_faces(
        control_pm_linear=control,
        face_tile_resolutions=resolutions,
        control_resolution=control_resolution,
    )
    cursor=0
    for face,resolution in enumerate(resolutions):
        denom=float(int(resolution)-1)
        for j in range(int(resolution)):
            for i in range(int(resolution)-j):
                u=float(i)/denom
                v=float(j)/denom
                base=0.1*face
                wanted=np.asarray(
                    (base+0.2*u,base+0.3*v,base+0.1*u+0.1*v,0.8),
                    dtype=np.float64,
                )
                assert np.allclose(dense[cursor],wanted,atol=1e-12)
                cursor+=1
    assert cursor==len(dense)
