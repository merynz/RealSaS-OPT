from __future__ import annotations

import numpy as np

from tools.corpus.structural_geometry_audit_v1 import audit_structural_geometry_v1


def split_tetra(owner=0, offset=(0.0, 0.0, 0.0)):
    base = np.asarray(
        [[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], dtype=np.float64
    ) + np.asarray(offset, dtype=np.float64)
    canonical = ((0,2,1),(0,1,3),(0,3,2),(1,2,3))
    vertices=[]; faces=[]
    for tri in canonical:
        start=len(vertices)
        vertices.extend(base[list(tri)])
        faces.append((start,start+1,start+2))
    return np.asarray(vertices), np.asarray(faces,np.int32), np.full(4,owner,np.int32)


def modular_split_tetra():
    base=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=np.float64)
    canonical=((0,2,1),(0,1,3),(0,3,2),(1,2,3))
    vertices=[]; faces=[]; owners=[]
    for fi, tri in enumerate(canonical):
        start=len(vertices)
        vertices.extend(base[list(tri)])
        faces.append((start,start+1,start+2))
        owners.append(0 if fi < 2 else 1)
    return np.asarray(vertices),np.asarray(faces,np.int32),np.asarray(owners,np.int32)


def test_within_owner_render_seams_collapse_without_losing_closed_surface():
    v,f,o=split_tetra()
    r=audit_structural_geometry_v1(v,f,o)
    assert r["counts"]["raw_vertex_count"] == 12
    assert r["counts"]["owner_preserving_topology_vertex_count"] == 4
    assert r["counts"]["within_owner_exact_duplicate_collapse_count"] == 8
    assert r["counts"]["edge_connected_face_component_count"] == 1
    assert r["counts"]["boundary_edge_count"] == 0
    assert r["edge_components"][0]["owner_preserving_closed_sign_candidate"] is True


def test_coincident_closed_objects_are_not_cross_owner_weld_candidates():
    av,af,ao=split_tetra(owner=0)
    bv,bf,bo=split_tetra(owner=1)
    bf=bf+len(av)
    v=np.concatenate([av,bv]); f=np.concatenate([af,bf]); o=np.concatenate([ao,bo])
    r=audit_structural_geometry_v1(v,f,o)
    assert r["counts"]["edge_connected_face_component_count"] == 2
    assert r["counts"]["boundary_edge_count"] == 0
    assert r["counts"]["cross_owner_exact_position_group_count"] == 4
    assert r["counts"]["cross_owner_matched_boundary_edge_count"] == 0


def test_modular_cross_owner_boundary_is_evidence_not_automatic_weld():
    v,f,o=modular_split_tetra()
    posed=np.stack([v, v + np.asarray([0.2,-0.1,0.05])],axis=0)
    r=audit_structural_geometry_v1(v,f,o,posed_vertices=posed)
    assert r["counts"]["edge_connected_face_component_count"] == 2
    assert r["counts"]["boundary_edge_count"] > 0
    assert r["counts"]["cross_owner_matched_boundary_edge_count"] > 0
    assert "CROSS_OWNER_BOUNDARY_SEAM_CANDIDATE" in r["failure_classes_observed"]
    assert r["authority"]["cross_owner_weld_performed"] is False


def test_cross_owner_same_rest_position_but_divergent_motion_is_detected():
    v,f,o=modular_split_tetra()
    posed0=v.copy()
    posed1=v.copy()
    owner1_raw=np.unique(f[o==1])
    posed1[owner1_raw] += np.asarray([0.25,0.,0.])
    posed=np.stack([posed0,posed1],axis=0)
    r=audit_structural_geometry_v1(v,f,o,posed_vertices=posed)
    assert r["cross_owner_boundary_evidence"]["trajectory_incompatible_count"] > 0


def test_point_only_contact_does_not_merge_edge_connected_components():
    av,af,ao=split_tetra(owner=0)
    bv,bf,bo=split_tetra(owner=0)
    for i,p in enumerate(bv):
        if not np.array_equal(p,np.zeros(3)):
            bv[i]=p+np.asarray([-3.,-3.,-3.])
    bf=bf+len(av)
    v=np.concatenate([av,bv]); f=np.concatenate([af,bf]); o=np.concatenate([ao,bo])
    r=audit_structural_geometry_v1(v,f,o)
    assert r["counts"]["edge_connected_face_component_count"] == 2
    assert r["counts"]["vertex_connected_face_component_count_diagnostic"] == 1
    assert r["diagnostics"]["point_only_contact_detected"] is True


def test_zero_area_face_is_reported_not_silently_removed():
    v=np.asarray([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]],dtype=np.float64)
    f=np.asarray([[0,1,2]],dtype=np.int32)
    o=np.asarray([0],dtype=np.int32)
    r=audit_structural_geometry_v1(v,f,o)
    assert r["counts"]["raw_zero_area_face_count"] == 1
    assert "DEGENERATE_FACE_EVIDENCE" in r["failure_classes_observed"]
