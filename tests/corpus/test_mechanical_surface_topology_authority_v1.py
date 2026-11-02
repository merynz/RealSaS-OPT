from __future__ import annotations

from pathlib import Path
import numpy as np

from tools.corpus.mechanical_surface_topology_authority_v1 import build_mechanical_surface_topology_authority_v1
from tools.corpus.build_mechanical_surface_topology_manifest_v1 import build as build_manifest
from tools.corpus.validate_mechanical_surface_topology_authority_v1 import validate as court_validate


def modular_pair(*, reverse=True, drift=False):
    vertices=np.asarray([
        [0.,0.,0.], [1.,0.,0.], [0.,1.,0.],
        [0.,1.,0.], [1.,0.,0.], [1.,1.,0.],
    ],np.float64)
    faces=np.asarray([[0,1,2],[3,4,5] if reverse else [4,3,5]],np.int32)
    owner=np.asarray([0,1],np.int32)
    posed=np.stack([vertices.copy(), vertices.copy()],axis=0)
    posed[1,:,2]=0.25
    # Keep duplicate seam endpoints mechanically coincident through the action.
    if drift:
        posed[1,4,0]+=1e-4
    return vertices,faces,owner,posed


def test_reverse_cross_owner_boundary_with_exact_trajectory_welds():
    v,f,o,p=modular_pair()
    a=build_mechanical_surface_topology_authority_v1(v,f,o,p)
    assert a.qualified_cross_owner_seam_count==1
    assert len(a.mechanical_vertices_source_world)==4
    assert len(np.unique(a.surface_component_id))==1
    assert a.raw_to_mechanical_vertex[1]==a.raw_to_mechanical_vertex[4]
    assert a.raw_to_mechanical_vertex[2]==a.raw_to_mechanical_vertex[3]
    seam_vertices={int(a.raw_to_mechanical_vertex[1]),int(a.raw_to_mechanical_vertex[2])}
    for mid in seam_vertices:
        start=int(a.mechanical_vertex_owner_offsets[mid]); end=int(a.mechanical_vertex_owner_offsets[mid+1])
        assert a.mechanical_vertex_owner_values[start:end].tolist()==[0,1]


def test_trajectory_drift_refuses_cross_owner_weld():
    v,f,o,p=modular_pair(drift=True)
    a=build_mechanical_surface_topology_authority_v1(v,f,o,p)
    assert a.qualified_cross_owner_seam_count==0
    assert len(a.mechanical_vertices_source_world)==6
    assert len(np.unique(a.surface_component_id))==2


def test_same_orientation_boundary_refuses_cross_owner_weld():
    v,f,o,p=modular_pair(reverse=False)
    a=build_mechanical_surface_topology_authority_v1(v,f,o,p)
    assert a.qualified_cross_owner_seam_count==0
    assert len(a.mechanical_vertices_source_world)==6


def test_point_only_cross_owner_contact_is_not_welded():
    v=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[2.,1.,0.],[1.,2.,0.]],np.float64)
    f=np.asarray([[0,1,2],[3,4,5]],np.int32); o=np.asarray([0,1],np.int32)
    p=np.stack([v.copy(),v.copy()],axis=0)
    a=build_mechanical_surface_topology_authority_v1(v,f,o,p)
    assert a.qualified_cross_owner_seam_count==0
    assert len(a.mechanical_vertices_source_world)==6
    assert len(np.unique(a.surface_component_id))==2


def test_manifest_to_independent_court_roundtrip(tmp_path: Path):
    v,f,o,p=modular_pair()
    source=tmp_path/"geometry_and_response.npz"
    np.savez_compressed(source,rest_vertices_source_world=v,faces=f,face_owner_privileged=o,isolated_position=p)
    out=tmp_path/"out"
    manifest=build_manifest("subject",source,out)
    court=court_validate(out/"MECHANICAL_SURFACE_TOPOLOGY_MANIFEST_V1.json")
    assert manifest["counts"]["qualified_cross_owner_seam_edge_count"]==1
    assert court["status"]=="PASS_MECHANICAL_SURFACE_TOPOLOGY_INDEPENDENT_COURT__FINAL_ADMISSION_SEPARATE"
    assert court["family_candidate_outcomes"]["SURFACE_TOPOLOGY"]=="QUALIFIED_CORE_CANDIDATE"
    assert court["authority"]["training_authorized"] is False
