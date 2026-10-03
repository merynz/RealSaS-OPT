from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    build_mesh_qualification_policy,
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_core.types import SurfaceSupportBinding
from tools.audit_knight_static_quality_split_cycle_v8 import synchronized_long_edge_split


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


def test_seam_touching_split_preserves_mechanical_support_separately_from_geometry():
    vertices=(
        CanonicalMeshVertexCandidateIR(
            "u",
            SurfaceSupportBinding(
                "SEAM_GEOMETRY_INTERPOLATION",
                (("s0",0.5),("sx",0.5)),
                metadata={
                    "mechanical_component_id":"c0",
                    "skin_support_coefficients":(("s0",1.0),),
                    "seam_geometry":"FIXTURE",
                },
            ),
            "c0",(0.0,0.0,0.0),
        ),
        CanonicalMeshVertexCandidateIR(
            "v",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s2",1.0),)),
            "c0",(4.0,0.0,0.0),
        ),
        CanonicalMeshVertexCandidateIR(
            "a",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s3",1.0),)),
            "c0",(0.0,0.1,0.0),
        ),
        CanonicalMeshVertexCandidateIR(
            "b",SurfaceSupportBinding("IDENTITY_SURFACE_NODE",(("s4",1.0),)),
            "c0",(4.0,0.1,0.0),
        ),
    )
    faces=(("u","v","a"),("v","u","b"))
    edges=tuple(sorted({
        tuple(sorted((face[i],face[j])))
        for face in faces for i,j in ((0,1),(1,2),(2,0))
    }))
    candidate=CanonicalMeshCandidateIR(
        vertices,faces,edges,"surface","partition","carrier",
        "fixture","fixture-policy","",
    )
    candidate=replace(
        candidate,candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate)
    )

    repaired,report=synchronized_long_edge_split(candidate,_policy(),max_splits=1)
    assert report["accepted_split_count"]==1
    created=[v for v in repaired.vertices if v.candidate_vertex_id.startswith("SPLITV:")]
    assert len(created)==1
    row=created[0]

    assert row.component_id=="c0"
    assert row.support_binding.mode=="SEAM_GEOMETRY_INTERPOLATION"
    assert tuple(row.support_binding.coefficients)==(
        ("s0",0.25),("s2",0.5),("sx",0.25),
    )
    assert tuple(_skin_support_coefficients(row))==(
        ("s0",0.5),("s2",0.5),
    )
    assert row.support_binding.metadata["mechanical_component_id"]=="c0"
    assert row.support_binding.metadata["seam_geometry"]=="DERIVED_EDGE_MIDPOINT"
