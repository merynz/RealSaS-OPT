from __future__ import annotations

import numpy as np
import pytest

from compiler.realsas_compiler_core.appearance_completion_v2 import (
    SurfaceSampleGraph,
)
from compiler.realsas_compiler_core.appearance_variational_completion_v1 import (
    geodesic_source_guidance,
    solve_weighted_surface_dirichlet,
)
from compiler.realsas_compiler_core.types import QualificationError


def _graph(node_count, edges):
    edge_a=np.asarray([a for a,_ in edges],dtype=np.int32)
    edge_b=np.asarray([b for _,b in edges],dtype=np.int32)
    directed=[]
    for a,b in edges:
        directed.append((a,b)); directed.append((b,a))
    directed.sort()
    offsets=np.zeros(node_count+1,dtype=np.int64)
    indices=[]
    cursor=0
    for node in range(node_count):
        while cursor<len(directed) and directed[cursor][0]==node:
            indices.append(directed[cursor][1]); cursor+=1
        offsets[node+1]=len(indices)
    return SurfaceSampleGraph(
        offsets=offsets,
        indices=np.asarray(indices,dtype=np.int32),
        edge_a=edge_a,
        edge_b=edge_b,
    )


def test_variational_completion_keeps_source_exact_and_interpolates_line():
    graph=_graph(5,((0,1),(1,2),(2,3),(3,4)))
    positions=np.asarray(
        [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(4,0,0)],
        dtype=np.float64,
    )
    values=np.asarray(
        [[0.0],[0.0],[0.0],[0.0],[1.0]],
        dtype=np.float64,
    )
    known=np.asarray([True,False,False,False,True])
    out,stats=solve_weighted_surface_dirichlet(
        values=values,
        known_mask=known,
        sample_component=np.zeros(5,dtype=np.int32),
        positions=positions,
        graph=graph,
    )
    assert np.array_equal(out[known],values[known])
    assert np.allclose(out[:,0],[0.0,0.25,0.5,0.75,1.0],atol=1e-8)
    assert stats.unknown_count==3
    assert max(stats.channel_relative_residuals)<1e-8


def test_variational_completion_cannot_cross_components():
    graph=_graph(6,((0,1),(1,2),(2,3),(3,4),(4,5)))
    positions=np.asarray([(i,0,0) for i in range(6)],dtype=np.float64)
    values=np.asarray(
        [[1,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,1]],
        dtype=np.float64,
    )
    known=np.asarray([True,False,False,False,False,True])
    component=np.asarray([0,0,0,1,1,1],dtype=np.int32)
    out,_=solve_weighted_surface_dirichlet(
        values=values,
        known_mask=known,
        sample_component=component,
        positions=positions,
        graph=graph,
    )
    assert np.allclose(out[:3],[1,0,0])
    assert np.allclose(out[3:],[0,0,1])


def test_variational_completion_fails_component_without_source_constraint():
    graph=_graph(4,((0,1),(2,3)))
    positions=np.asarray([(0,0,0),(1,0,0),(0,1,0),(1,1,0)],dtype=np.float64)
    values=np.zeros((4,1),dtype=np.float64)
    known=np.asarray([True,False,False,False])
    component=np.asarray([0,0,1,1],dtype=np.int32)
    with pytest.raises(
        QualificationError,
        match="CAA_VARIATIONAL_COMPONENT_WITHOUT_SOURCE_CONSTRAINT",
    ):
        solve_weighted_surface_dirichlet(
            values=values,
            known_mask=known,
            sample_component=component,
            positions=positions,
            graph=graph,
        )


def test_zero_length_topology_edge_is_strong_but_finite():
    graph=_graph(4,((0,1),(1,2),(2,3)))
    positions=np.asarray(
        [(0,0,0),(1,0,0),(1,0,0),(2,0,0)],
        dtype=np.float64,
    )
    values=np.asarray([[0.0],[0.0],[0.0],[1.0]],dtype=np.float64)
    known=np.asarray([True,False,False,True])
    out,stats=solve_weighted_surface_dirichlet(
        values=values,
        known_mask=known,
        sample_component=np.zeros(4,dtype=np.int32),
        positions=positions,
        graph=graph,
    )
    assert stats.zero_length_topology_edge_count==1
    assert stats.maximum_edge_weight==32.0
    assert abs(float(out[1,0])-float(out[2,0]))<0.05


def test_geodesic_guidance_is_source_exact_and_tie_breaks_lower_seed():
    graph=_graph(5,((0,1),(1,2),(2,3),(3,4)))
    positions=np.asarray([(i,0,0) for i in range(5)],dtype=np.float64)
    values=np.asarray(
        [[1.0],[0.0],[0.0],[0.0],[0.0]],
        dtype=np.float64,
    )
    values[4,0]=0.25
    known=np.asarray([True,False,False,False,True])
    guidance,donor=geodesic_source_guidance(
        values=values,
        known_mask=known,
        sample_component=np.zeros(5,dtype=np.int32),
        positions=positions,
        graph=graph,
    )
    assert donor.tolist()==[0,0,0,4,4]
    assert np.array_equal(guidance[known],values[known])
    assert float(guidance[2,0])==1.0


def test_screened_poisson_preserves_sources_and_follows_geodesic_guide():
    graph=_graph(5,((0,1),(1,2),(2,3),(3,4)))
    positions=np.asarray([(i,0,0) for i in range(5)],dtype=np.float64)
    values=np.asarray(
        [[1.0],[0.0],[0.0],[0.0],[0.0]],
        dtype=np.float64,
    )
    values[4,0]=0.0
    known=np.asarray([True,False,False,False,True])
    component=np.zeros(5,dtype=np.int32)
    guidance,_=geodesic_source_guidance(
        values=values,
        known_mask=known,
        sample_component=component,
        positions=positions,
        graph=graph,
    )
    unscreened,_=solve_weighted_surface_dirichlet(
        values=values,
        known_mask=known,
        sample_component=component,
        positions=positions,
        graph=graph,
    )
    screened,stats=solve_weighted_surface_dirichlet(
        values=values,
        known_mask=known,
        sample_component=component,
        positions=positions,
        graph=graph,
        guide_values=guidance,
        guide_weight=4.0,
        guide_mode="GEODESIC_NEAREST_SOURCE_V1",
    )
    assert np.array_equal(screened[known],values[known])
    assert float(screened[1,0])>float(unscreened[1,0])
    assert float(screened[3,0])<float(unscreened[3,0])
    assert stats.guide_weight==4.0
    assert stats.guide_mode=="GEODESIC_NEAREST_SOURCE_V1"
