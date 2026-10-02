from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_quality_source_surface_optimize_v1 import (
    repair_candidate_source_surface_quality_v1,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    build_mesh_qualification_policy,
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


def _candidate(center_x: float):
    pts={
        "v":(center_x,0.0,0.0),
        "a":(-1.0,-1.0,0.0),
        "b":(1.0,-1.0,0.0),
        "c":(1.0,1.0,0.0),
        "d":(-1.0,1.0,0.0),
    }
    faces=(("v","a","b"),("v","b","c"),("v","c","d"),("v","d","a"))
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
        vertices,faces,edges,
        "surface","partition","carrier",
        "fixture","fixture-policy","",
    )
    return replace(
        value,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(value),
    )


def test_source_surface_quality_search_closes_interior_angle_without_topology_change():
    reference=_candidate(0.0)
    candidate=_candidate(0.9)
    repaired,report=repair_candidate_source_surface_quality_v1(
        candidate,
        reference,
        _policy(),
        grid_schedule=(8,16,32),
        max_batches=4,
        max_moves=8,
    )
    assert report["before"]["policy_violating_face_count"]>0
    assert report["accepted_move_count"]==1
    assert report["after"]["policy_violating_face_count"]==0
    assert report["final_topology"]["passed"] is True
    assert repaired.faces==candidate.faces
    before={v.candidate_vertex_id:v.P for v in candidate.vertices}
    after={v.candidate_vertex_id:v.P for v in repaired.vertices}
    assert after["v"]!=before["v"]
    for vid in ("a","b","c","d"):
        assert after[vid]==before[vid]


def test_source_surface_quality_search_rebinds_to_convex_reference_support():
    reference=_candidate(0.0)
    candidate=_candidate(0.9)
    repaired,report=repair_candidate_source_surface_quality_v1(
        candidate,
        reference,
        _policy(),
        grid_schedule=(8,16,32),
        max_batches=4,
        max_moves=8,
    )
    row={v.candidate_vertex_id:v for v in repaired.vertices}["v"]
    assert report["accepted_move_count"]==1
    assert row.support_binding.mode in {"IDENTITY_SURFACE_NODE","LOCAL_CONVEX_INTERPOLATION"}
    assert abs(sum(float(w) for _,w in row.support_binding.coefficients)-1.0)<1e-9
    assert all(float(w)>=0.0 for _,w in row.support_binding.coefficients)


def test_source_surface_quality_search_respects_protected_support():
    reference=_candidate(0.0)
    candidate=_candidate(0.9)
    repaired,report=repair_candidate_source_surface_quality_v1(
        candidate,
        reference,
        _policy(),
        protected_surface_ids={"sv"},
        grid_schedule=(8,16,32),
        max_batches=4,
        max_moves=8,
    )
    assert report["accepted_move_count"]==0
    assert repaired.candidate_lineage_hash!=candidate.candidate_lineage_hash
    before={v.candidate_vertex_id:v.P for v in candidate.vertices}
    after={v.candidate_vertex_id:v.P for v in repaired.vertices}
    assert after==before
