from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    build_component_carrier_policy,
    canonical_mesh_candidate_lineage_hash,
)
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
    SurfaceSupportBinding,
)


def test_stage35_repair_proposes_repartition_and_never_deletes_faces():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("a",(0.0,0.0,0.0),(0,),("p",),("o0",)),
            SurfaceNode("b",(1.0,0.0,0.0),(0,),("p",),("o1",)),
            SurfaceNode("c",(0.0,1.0,0.0),(0,),("p",),("o2",)),
        ),
        (
            SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("bc","b","c","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("ca","c","a","LOCAL_NEIGHBOR",1.0),
        ),
        "surface-stage35",
    )
    partition=build_structural_partition(surface)
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(x.component_id,"MESH",("automatic",))
            for x in partition.components
        ),
    )
    candidate=build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash="stage35-test"
    )
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("j0",(0.0,0.0,0.0),None,("a","b"),"p0"),
            QualifiedJoint("j1",(0.0,1.0,0.0),"j0",("c",),"p1"),
        ),
        "j0",{"status":"PASS"},"skeleton-stage35",
    )
    skin=QualifiedSkinIR(
        (
            QualifiedSkinRow("a",(("j0",1.0),),0.0,0.0),
            QualifiedSkinRow("b",(("j0",1.0),),0.0,0.0),
            QualifiedSkinRow("c",(("j1",1.0),),0.0,0.0),
        ),
        surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,{"status":"PASS"},"skin-stage35",
    )
    compatibility={
        "unsafe_face_indices":(0,),
        "report_hash":"compatibility-report",
    }
    directive=propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report=compatibility,
    )
    assert directive["status"]=="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
    assert directive["repair_operation"]=="STAGE17_REPARTITION_THEN_STAGE18_HOLELESS_DENSE_SUBDIVISION"
    assert directive["restart_from"]=="17_MECHANICAL_PARTITION_QUALIFIED"
    assert directive["face_deletion_count"]==0
    assert directive["weight_mutation"] is False
    assert directive["auto_apply_allowed"] is False
    assert directive["requires_trustworthy_skin_reliability_authority"] is True
    assert directive["candidate_separate_pair_count"]==1
    row=directive["proposed_boundary_overrides"][0]
    assert {row["a_surface_id"],row["b_surface_id"]} in ({"a","c"},{"b","c"})
    assert row["decision"]=="SEPARATE"
    assert row["max_pairwise_skin_l1"]==2.0



def test_stage35_repair_accepts_harmonic_multi_support_seam_owner():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("a",(0.0,0.0,0.0),(0,),("p",),("o0",)),
            SurfaceNode("b",(1.0,0.0,0.0),(0,),("p",),("o1",)),
            SurfaceNode("c",(0.0,1.0,0.0),(0,),("p",),("o2",)),
        ),
        (
            SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("bc","b","c","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("ca","c","a","LOCAL_NEIGHBOR",1.0),
        ),
        "surface-stage35-multisupport",
    )
    partition=build_structural_partition(surface)
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(x.component_id,"MESH",("automatic",))
            for x in partition.components
        ),
    )
    candidate=build_canonical_relation_candidate(
        surface,partition,carrier,producer_policy_hash="stage35-multisupport-test"
    )
    first=candidate.vertices[0]
    seam=replace(
        first,
        support_binding=SurfaceSupportBinding(
            "SEAM_GEOMETRY_INTERPOLATION",
            (("a",0.5),("b",0.5)),
            metadata={
                "mechanical_component_id":first.component_id,
                "skin_support_coefficients":(("a",0.6),("b",0.4)),
                "seam_geometry":"HARMONIC_FIXTURE",
            },
        ),
        P=(0.5,0.0,0.0),
    )
    candidate=replace(
        candidate,
        vertices=(seam,)+candidate.vertices[1:],
        candidate_lineage_hash="",
    )
    candidate=replace(
        candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate),
    )
    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("j0",(0.0,0.0,0.0),None,("a","b"),"p0"),
            QualifiedJoint("j1",(0.0,1.0,0.0),"j0",("c",),"p1"),
        ),
        "j0",{"status":"PASS"},"skeleton-stage35-multisupport",
    )
    skin=QualifiedSkinIR(
        (
            QualifiedSkinRow("a",(("j0",1.0),),0.0,0.0),
            QualifiedSkinRow("b",(("j0",1.0),),0.0,0.0),
            QualifiedSkinRow("c",(("j1",1.0),),0.0,0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status":"PASS"},
        "skin-stage35-multisupport",
    )
    directive=propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report={
            "unsafe_face_indices":(0,),
            "report_hash":"compatibility-report-multisupport",
        },
    )
    assert directive["candidate_separate_pair_count"]==1
    assert directive["unresolved_unsafe_face_count"]==0
    row=directive["proposed_boundary_overrides"][0]
    assert {row["a_surface_id"],row["b_surface_id"]}=={"a","c"}
    assert row["metadata"]["multi_support_owner_resolution"]=="CARTESIAN_SUPPORT_MASS_V1"
    assert row["metadata"]["max_support_pair_coeff_product"]==0.6
