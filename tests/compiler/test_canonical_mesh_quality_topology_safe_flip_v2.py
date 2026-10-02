from __future__ import annotations
from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
)
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


def _candidate(offsets=(0.0,)):
    base={
        "a":(0.92923931,0.61113662,0.0),
        "b":(0.11881640,0.81884888,0.0),
        "c":(0.31157783,0.19427709,0.0),
        "d":(0.59387040,0.75528399,0.0),
    }
    pts={}
    faces=[]
    for idx,dx in enumerate(offsets):
        pfx=f"{idx}:"
        for name,p in base.items():
            pts[pfx+name]=(p[0]+dx,p[1],p[2])
        a,b,c,d=(pfx+x for x in ("a","b","c","d"))
        faces += [(a,b,c),(b,a,d)]
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


def test_topology_safe_flip_repairs_single_quad():
    candidate=_candidate()
    repaired,report=repair_candidate_fixed_vertex_flips_topology_safe_v2(candidate,_policy())
    assert report["accepted_flip_count"]==1
    assert report["after"]["policy_violating_face_count"]==0
    assert report["final_topology"]["passed"] is True
    assert tuple(v.P for v in repaired.vertices)==tuple(v.P for v in candidate.vertices)


def test_topology_safe_flip_batches_two_disjoint_quads():
    candidate=_candidate((0.0,10.0))
    repaired,report=repair_candidate_fixed_vertex_flips_topology_safe_v2(candidate,_policy())
    assert report["accepted_flip_count"]==2
    assert report["passes"][0]["accepted_flip_count"]==2
    assert report["after"]["policy_violating_face_count"]==0
    assert report["final_topology"]["passed"] is True


def test_topology_safe_flip_rejects_bow_tie_input_vertex_link():
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
        repair_candidate_fixed_vertex_flips_topology_safe_v2(value,_policy())
