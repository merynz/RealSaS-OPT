from __future__ import annotations
from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_quality_patch_purge_v1 import (
    repair_candidate_patch_interior_purge_v1,
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


def _patch(prefix,dx):
    pts={
        prefix+"a":(dx+0.0,0.0,0.0),
        prefix+"b":(dx+1.0,0.0,0.0),
        prefix+"c":(dx+1.0,1.0,0.0),
        prefix+"d":(dx+0.0,1.0,0.0),
        prefix+"u":(dx+0.49,0.50,0.0),
        prefix+"v":(dx+0.51,0.50,0.0),
    }
    a,b,c,d,u,v=(prefix+x for x in ("a","b","c","d","u","v"))
    faces=(
        (a,b,u),
        (b,v,u),
        (b,c,v),
        (c,d,v),
        (d,u,v),
        (d,a,u),
    )
    return pts,faces


def _candidate(two=False):
    pts0,faces0=_patch("L",0.0)
    pts=dict(pts0);faces=list(faces0)
    if two:
        pts1,faces1=_patch("R",10.0)
        pts.update(pts1);faces.extend(faces1)
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


def test_patch_purge_removes_multiple_bad_interior_vertices():
    candidate=_candidate()
    repaired,report=repair_candidate_patch_interior_purge_v1(
        candidate,_policy(),max_hops=3,max_patch_faces=32,max_batches=8,max_patches=8
    )
    assert report["accepted_patch_count"]>=1
    assert report["after"]["policy_violating_face_count"]==0
    assert report["final_topology"]["passed"] is True
    assert len(repaired.vertices)==4
    assert len(repaired.faces)==2
    assert all(v.candidate_vertex_id not in {"Lu","Lv"} for v in repaired.vertices)


def test_patch_purge_batches_disjoint_patches():
    candidate=_candidate(two=True)
    repaired,report=repair_candidate_patch_interior_purge_v1(
        candidate,_policy(),max_hops=3,max_patch_faces=32,max_batches=8,max_patches=8
    )
    assert report["accepted_patch_count"]>=2
    assert report["after"]["policy_violating_face_count"]==0
    assert report["final_topology"]["nonmanifold_edge_count"]==0
    assert report["final_topology"]["illegal_vertex_link_count"]==0
    assert len(repaired.vertices)==8
    assert len(repaired.faces)==4


def test_patch_purge_does_not_mint_product_triangle_authority():
    candidate=_candidate()
    repaired,report=repair_candidate_patch_interior_purge_v1(
        candidate,_policy(),max_hops=3,max_patch_faces=32,max_batches=8,max_patches=8
    )
    meta=repaired.metadata["patch_interior_purge_quality_repair"]
    assert report["accepted_patch_count"]>=1
    assert meta["audit_only"] is True
    assert meta["product_triangle_authority_minted"] is False
    assert meta["stage14_surface_remesh_authority_required_for_product_promotion"] is True
