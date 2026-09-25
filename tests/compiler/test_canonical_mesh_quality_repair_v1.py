from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    repair_candidate_endpoint_collapses_v1,
    repair_candidate_fixed_vertex_flips_v1,
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


def _candidate(*, nonplanar=False):
    pts={
        "a":(0.92923931,0.61113662,0.0),
        "b":(0.11881640,0.81884888,0.0),
        "c":(0.31157783,0.19427709,0.0),
        "d":(0.59387040,0.75528399,1.5 if nonplanar else 0.0),
    }
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            vid,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{vid}",1.0),)),
            "c0",
            pts[vid],
        )
        for vid in ("a","b","c","d")
    )
    faces=(("a","b","c"),("b","a","d"))
    edges=(("a","b"),("a","c"),("b","c"),("a","d"),("b","d"))
    value=CanonicalMeshCandidateIR(
        vertices,faces,edges,
        "surface","partition","carrier",
        "fixture","fixture-policy","",
    )
    return replace(value,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(value))


def test_fixed_vertex_flip_repairs_bad_diagonal_without_moving_vertices():
    candidate=_candidate()
    repaired,report=repair_candidate_fixed_vertex_flips_v1(candidate,_policy())
    assert report["accepted_flip_count"]==1
    assert report["before"]["below_min_angle_face_count"]==1
    assert report["after"]["below_min_angle_face_count"]==0
    assert report["after"]["policy_violating_face_count"]==0
    assert tuple(v.P for v in repaired.vertices)==tuple(v.P for v in candidate.vertices)
    assert tuple(v.support_binding for v in repaired.vertices)==tuple(
        v.support_binding for v in candidate.vertices
    )


def test_nonplanar_flip_is_rejected_by_frozen_g1_deviation_budget():
    candidate=_candidate(nonplanar=True)
    repaired,report=repair_candidate_fixed_vertex_flips_v1(candidate,_policy())
    assert report["accepted_flip_count"]==0
    assert repaired.faces==candidate.faces


def test_endpoint_collapse_removes_short_edge_slivers_with_link_condition():
    pts={
        "u":(0.0,0.0,0.0),
        "v":(0.02,0.02,0.0),
        "a":(0.0,1.0,0.0),
        "b":(0.0,-1.0,0.0),
        "c":(1.0,1.0,0.0),
        "d":(1.0,-1.0,0.0),
    }
    vertices=tuple(
        CanonicalMeshVertexCandidateIR(
            vid,
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{vid}",1.0),)),
            "c0",
            pts[vid],
        )
        for vid in ("u","v","a","b","c","d")
    )
    faces=(
        ("u","v","a"),
        ("v","u","b"),
        ("v","c","a"),
        ("v","b","d"),
    )
    edges=tuple(sorted({
        tuple(sorted((face[i],face[j])))
        for face in faces for i,j in ((0,1),(1,2),(2,0))
    }))
    candidate=CanonicalMeshCandidateIR(
        vertices,faces,edges,
        "surface","partition","carrier",
        "fixture","fixture-policy","",
    )
    candidate=replace(
        candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate),
    )
    repaired,report=repair_candidate_endpoint_collapses_v1(
        candidate,_policy(),max_collapses=4
    )
    assert report["accepted_collapse_count"]==1
    assert report["before"]["policy_violating_face_count"]==2
    assert report["after"]["policy_violating_face_count"]==0
    assert len(repaired.vertices)==5
    assert len(repaired.faces)==2
    assert all(
        vertex.support_binding in tuple(v.support_binding for v in candidate.vertices)
        for vertex in repaired.vertices
    )
