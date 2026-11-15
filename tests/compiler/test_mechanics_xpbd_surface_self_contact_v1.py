from __future__ import annotations

"""Discrete point/triangle and edge/edge XPBD self-contact behavioral courts."""

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_xpbd_surface_self_contact_v1 import (
    prepare_surface_self_contact_topology_v1,
    evaluate_surface_self_contacts_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    XPBDSurfacePolicyV1, step_declared_xpbd_surface_v1,
    xpbd_surface_state_from_dict_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_v1 import _setup, _simulate
import json


def _v(x,y,z):
    return np.asarray((x,y,z),dtype=np.float64)


def test_point_triangle_contact_moves_the_point_not_the_pinned_surface():
    positions={
        "a":_v(-1.,-1.,0.),
        "b":_v(1.,-1.,0.),
        "c":_v(0.,1.,0.),
        "p":_v(0.,0.,.01),
    }
    before={k:v.copy() for k,v in positions.items()}
    mass={"a":0.,"b":0.,"c":0.,"p":1.}
    topo=prepare_surface_self_contact_topology_v1(
        (("a","b","c"),),
        (("a","b",2.),("b","c",2.),("c","a",2.)),
    )
    result=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.05,cell_size=1.,
        project=True,
    )
    assert result.projections==1
    assert result.maximum_penetration==pytest.approx(.04)
    assert positions["p"][2]==pytest.approx(.05)
    assert all(np.array_equal(positions[k],before[k]) for k in ("a","b","c"))
    final=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.05,cell_size=1.,
        project=False,
    )
    assert final.maximum_penetration<1e-9


def test_coincident_vertex_triangle_uses_defined_deterministic_normal():
    positions={
        "a":_v(-1.,-1.,0.),
        "b":_v(1.,-1.,0.),
        "c":_v(0.,1.,0.),
        "p":_v(0.,0.,0.),
    }
    topo=prepare_surface_self_contact_topology_v1(
        (("a","b","c"),),
        (("a","b",2.),("b","c",2.),("c","a",2.)),
    )
    mass={v:(1. if v=="p" else 0.) for v in positions}
    result=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.03,cell_size=1.,
        project=True,
    )
    assert result.projections==1
    assert positions["p"][2]==pytest.approx(.03)


def test_crossed_non_adjacent_edges_resolve_by_two_sided_xpbd():
    positions={
        "a":_v(-1.,0.,0.),"b":_v(1.,0.,0.),
        "c":_v(0.,-1.,0.),"d":_v(0.,1.,0.),
    }
    topo=prepare_surface_self_contact_topology_v1(
        (), (("a","b",2.),("c","d",2.)),
    )
    mass={v:1. for v in positions}
    result=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.08,cell_size=1.,
        project=True,
    )
    assert result.projections>=1
    assert result.candidate_pairs>=1
    final=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.08,cell_size=1.,
        project=False,
    )
    assert final.maximum_penetration<1e-6


def test_canonical_one_ring_neighbors_never_self_collide():
    # The diagonal and four boundary edges belong to a single surface patch,
    # not two independent colliding objects.
    positions={
        "a":_v(0.,0.,0.),"b":_v(1.,0.,0.),
        "c":_v(0.,1.,0.),"d":_v(1.,1.,0.),
    }
    topo=prepare_surface_self_contact_topology_v1(
        (("a","b","c"),("b","d","c")),
        (("a","b",1.),("a","c",1.),("b","c",1.41),
         ("b","d",1.),("c","d",1.)),
    )
    mass={v:1. for v in positions}
    result=evaluate_surface_self_contacts_v1(
        positions,mass,topo,thickness=.1,cell_size=1.,project=True,
    )
    assert result.projections==0
    assert result.maximum_penetration==0.


def test_self_contact_resource_and_degenerate_surface_fail_closed():
    positions={
        "a":_v(-1.,-1.,0.),"b":_v(1.,-1.,0.),
        "c":_v(0.,1.,0.),
        "p":_v(0.,0.,.01),"q":_v(.1,0.,.01),
    }
    topo=prepare_surface_self_contact_topology_v1(
        (("a","b","c"),),
        (("a","b",2.),("b","c",2.),("c","a",2.)),
    )
    mass={v:1. for v in positions}
    with pytest.raises(QualificationError,match="CANDIDATE_RESOURCE_BOUND"):
        evaluate_surface_self_contacts_v1(
            positions,mass,topo,thickness=.05,cell_size=1.,
            max_pairs=1,project=True,
        )
    with pytest.raises(QualificationError,match="SPATIAL_RESOURCE_BOUND"):
        evaluate_surface_self_contacts_v1(
            positions,mass,topo,thickness=.05,cell_size=.0001,
            max_cells=10,project=True,
        )
    positions["c"]=_v(0.,-1.,0.)
    with pytest.raises(QualificationError,match="DEGENERATE_MOVING_FACE"):
        evaluate_surface_self_contacts_v1(
            positions,mass,topo,thickness=.05,cell_size=1.,
            project=True,
        )


def test_opt_in_surface_self_contact_state_replay_and_fail_closed_policy():
    parents,domain,policy,primary=_setup(
        self_collision_thickness=.01,gravity_xyz=(0.,0.,0.),
    )
    start=_simulate(parents,domain,policy,primary)
    out=_simulate(parents,domain,policy,primary,
                  previous_state=start.state,dt_seconds=1./60.)
    assert out.status=="PASS_XPBD_SURFACE_RUNTIME"
    assert out.metadata["surface_self_triangle_collision_qualified"] is True
    assert out.metadata["self_contact_vertex_triangle_and_edge_edge_enabled"] is True
    assert out.metadata["self_contact_max_penetration"]==0.
    assert out.deformation.canonical_mesh_vertex_ids==primary.canonical_mesh_vertex_ids
    saved=json.loads(json.dumps(out.state.to_dict()))
    reopened=xpbd_surface_state_from_dict_v1(saved)
    next1=_simulate(parents,domain,policy,primary,
                    previous_state=out.state,dt_seconds=1./60.)
    next2=_simulate(parents,domain,policy,primary,
                    previous_state=reopened,dt_seconds=1./60.)
    assert next1.state==next2.state
    assert next1.deformation==next2.deformation
    with pytest.raises(QualificationError,match="THICKNESS_INVALID"):
        _simulate(parents,domain,replace(policy,self_collision_thickness=-.1),primary)
    with pytest.raises(QualificationError,match="RESOURCE_POLICY_INVALID"):
        _simulate(parents,domain,replace(policy,max_self_collision_pairs=0),primary)
    with pytest.raises(QualificationError,match="INACTIVE_COMPLIANCE"):
        _simulate(parents,domain,replace(policy,
            self_collision_thickness=0.,self_collision_compliance=.1),primary)
