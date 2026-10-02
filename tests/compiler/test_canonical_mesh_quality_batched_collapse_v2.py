from __future__ import annotations
from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
)
import pytest
from compiler.realsas_compiler_core.types import QualificationError,SurfaceSupportBinding


def _policy():
    return build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.999,0.999,0.00025,0.0005),
            CarrierCoverageThresholdIR("PLANAR",0.99,0.995,0.0025,0.0025),
        ),
        g3_min_dynamic_area_ratio=0.05,
        g3_max_dynamic_area_ratio=20.0,
        g3_max_dynamic_condition_number=16.0,
    )


def _fixture():
    # Two disjoint closed manifold patches. Each contains a very short interior
    # edge (u,v); unlike the old boundary fixture, collapsing it cannot pinch
    # two boundary arcs into a bow-tie.
    base={
        "u":(0.0,0.0,1.0),
        "v":(0.02,0.0,0.99),
        "e1":(1.0,0.0,0.0),
        "e2":(0.0,1.0,0.0),
        "e3":(-1.0,0.0,0.0),
        "b":(0.0,0.0,-1.0),
    }
    pts={}
    faces=[]
    for k,dx in (("L",0.0),("R",10.0)):
        for name,p in base.items():
            pts[k+name]=(p[0]+dx,p[1],p[2])
        u,v,e1,e2,e3,b=(k+x for x in ("u","v","e1","e2","e3","b"))
        faces += [
            (u,v,e1),
            (u,e1,e2),
            (u,e2,e3),
            (u,e3,v),
            (b,e1,v),
            (b,e2,e1),
            (b,e3,e2),
            (b,v,e3),
        ]
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            vid,SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{vid}",1.0),)),
            "c0",pts[vid]
        )
        for vid in sorted(pts)
    )
    edges=tuple(sorted({
        tuple(sorted((face[i],face[j])))
        for face in faces for i,j in ((0,1),(1,2),(2,0))
    }))
    value=CanonicalMeshCandidateIR(
        vertices,tuple(faces),edges,
        "surface","partition","carrier","fixture","fixture-policy","",
    )
    return replace(value,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(value))


def test_batched_collapse_repairs_two_disjoint_stars_in_one_batch():
    candidate=_fixture()
    repaired,report=repair_candidate_endpoint_collapses_batched_v2(
        candidate,_policy(),max_batches=2,max_collapses=8
    )
    assert report["accepted_collapse_count"]==2
    assert report["batch_count"]==1
    assert report["after"]["policy_violating_face_count"] < report["before"]["policy_violating_face_count"]
    assert len(repaired.vertices)==10
    assert all(count<=2 for count in _edge_incidence(repaired.faces).values())


def _edge_incidence(faces):
    out={}
    for face in faces:
        a,b,c=map(str,face)
        for u,v in ((a,b),(b,c),(c,a)):
            e=tuple(sorted((u,v)))
            out[e]=out.get(e,0)+1
    return out


def test_batched_collapse_preserves_surviving_support_bindings():
    candidate=_fixture()
    repaired,report=repair_candidate_endpoint_collapses_batched_v2(
        candidate,_policy(),max_batches=2,max_collapses=8
    )
    original={v.candidate_vertex_id:v.support_binding for v in candidate.vertices}
    assert report["accepted_collapse_count"]==2
    assert all(original[v.candidate_vertex_id]==v.support_binding for v in repaired.vertices)

def test_batched_collapse_rejects_bow_tie_input_vertex_link():
    pts={
        "v":(0.0,0.0,0.0),
        "a":(1.0,0.0,0.0),"b":(0.0,1.0,0.0),"c":(-1.0,0.0,0.0),
        "d":(3.0,0.0,0.0),"e":(2.0,1.0,0.0),"f":(1.0,2.0,0.0),
    }
    faces=(("v","a","b"),("v","b","c"),("v","c","a"),
           ("v","d","e"),("v","e","f"),("v","f","d"))
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            vid,SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{vid}",1.0),)),
            "c0",pts[vid]
        ) for vid in sorted(pts)
    )
    edges=tuple(sorted({
        tuple(sorted((face[i],face[j])))
        for face in faces for i,j in ((0,1),(1,2),(2,0))
    }))
    value=CanonicalMeshCandidateIR(
        vertices,faces,edges,
        "surface","partition","carrier","fixture","fixture-policy","",
    )
    value=replace(value,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(value))
    with pytest.raises(QualificationError,match="INPUT_NONMANIFOLD"):
        repair_candidate_endpoint_collapses_batched_v2(value,_policy())