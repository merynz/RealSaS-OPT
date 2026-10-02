from __future__ import annotations
from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_quality_batched_relaxation_v2 import (
    repair_candidate_projected_relaxation_batched_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.types import SurfaceSupportBinding


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


def _two_stars():
    pts={}
    faces=[]
    for k,dx in (("L",0.0),("R",10.0)):
        local={
            "v":(0.9+dx,0.0,0.0),
            "a":(1.0+dx,0.0,0.0),
            "b":(0.0+dx,1.0,0.0),
            "c":(-1.0+dx,0.0,0.0),
            "d":(0.0+dx,-1.0,0.0),
        }
        for n,p in local.items(): pts[k+n]=p
        v,a,b,c,d=(k+x for x in ("v","a","b","c","d"))
        faces += [(v,a,b),(v,b,c),(v,c,d),(v,d,a)]
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            vid,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{vid}",1.0),)),
            "c0",
            pts[vid],
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


def test_batched_projected_relaxation_moves_two_disjoint_stars_in_one_batch():
    candidate=_two_stars()
    repaired,report=repair_candidate_projected_relaxation_batched_v2(
        candidate,candidate,_policy(),max_batches=4,max_moves=8
    )
    assert report["accepted_move_count"]>=2
    assert report["batch_count"]>=1
    assert report["batches"][0]["accepted_move_count"]==2
    assert report["after"]["policy_violating_face_count"] < report["before"]["policy_violating_face_count"]
    assert report["final_topology"]["passed"] is True
    assert repaired.faces==candidate.faces


def test_batched_projected_relaxation_preserves_convex_support_and_topology():
    candidate=_two_stars()
    repaired,report=repair_candidate_projected_relaxation_batched_v2(
        candidate,candidate,_policy(),max_batches=4,max_moves=8
    )
    assert report["accepted_move_count"]>=2
    original={v.candidate_vertex_id:v for v in candidate.vertices}
    moved=[v for v in repaired.vertices if v.P!=original[v.candidate_vertex_id].P]
    assert moved
    for v in moved:
        assert v.support_binding.mode in {"IDENTITY_SURFACE_NODE","LOCAL_CONVEX_INTERPOLATION"}
        assert abs(sum(c for _,c in v.support_binding.coefficients)-1.0)<1e-9
    assert report["final_topology"]["nonmanifold_edge_count"]==0
    assert report["final_topology"]["illegal_vertex_link_count"]==0


def test_batched_projected_relaxation_respects_protected_support():
    candidate=_two_stars()
    repaired,report=repair_candidate_projected_relaxation_batched_v2(
        candidate,candidate,_policy(),
        protected_surface_ids={"sLv","sRv"},
        max_batches=4,max_moves=8
    )
    before={v.candidate_vertex_id:v.P for v in candidate.vertices}
    after={v.candidate_vertex_id:v.P for v in repaired.vertices}
    assert after["Lv"]==before["Lv"]
    assert after["Rv"]==before["Rv"]
    assert report["rejected_protected_count"]>0
