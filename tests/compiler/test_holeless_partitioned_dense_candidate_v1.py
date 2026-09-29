from __future__ import annotations

from dataclasses import replace

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_holeless_partitioned_dense_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    propose_mechanical_repartition_directive_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
    build_component_carrier_policy,
    build_mesh_qualification_policy,
    deformation_envelope_lineage_hash,
    qualified_mesh_intrinsic_audit,
    qualified_mesh_lineage_hash,
    validate_canonical_mesh_candidate,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import bind_product_mesh_skin
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _fixture():
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("a",(0.0,0.0,0.0),(0,),("p",),("o0",)),
            SurfaceNode("b",(2.0,0.0,0.0),(0,),("p",),("o1",)),
            SurfaceNode("c",(0.0,2.0,0.0),(0,),("p",),("o2",)),
        ),
        (
            SurfaceRelation("ab","a","b","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("ac","a","c","LOCAL_NEIGHBOR",1.0),
            SurfaceRelation("bc","b","c","LOCAL_NEIGHBOR",1.0),
        ),
        "surface-holeless",
    )
    partition = build_structural_partition(
        surface,
        boundary_overrides=(
            ComponentBoundaryConstraintIR("cut-ac","a","c","SEPARATE",("dynamic-ac",),1.0),
            ComponentBoundaryConstraintIR("cut-bc","b","c","SEPARATE",("dynamic-bc",),1.0),
        ),
    )
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(c.component_id,"MESH",("automatic",))
            for c in partition.components
        ),
    )
    candidate = build_holeless_partitioned_dense_candidate(
        surface,
        partition,
        carrier,
        producer_policy_hash="holeless-test-policy",
        explicit_face_provenance=(("a","b","c"),),
    )
    return surface, partition, carrier, candidate


def _qualified_mesh(surface, partition, carrier, candidate):
    ids={v.candidate_vertex_id:"MV:"+v.candidate_vertex_id for v in candidate.vertices}
    vertices=tuple(
        QualifiedMeshVertexIR(
            ids[v.candidate_vertex_id],
            v.support_binding,
            v.component_id,
            v.P,
            v.candidate_vertex_id,
            v.refinement,
            dict(v.metadata or {}),
        )
        for v in candidate.vertices
    )
    faces=tuple(tuple(ids[x] for x in face) for face in candidate.faces)
    edges=tuple(tuple(ids[x] for x in edge) for edge in candidate.edges)

    skeleton=QualifiedSkeletonIR(
        (
            QualifiedJoint("ja",(0.0,0.0,0.0),None,("a","b"),"p0"),
            QualifiedJoint("jb",(0.0,2.0,0.0),"ja",("c",),"p1"),
        ),
        "ja",{"status":"PASS"},"skeleton-holeless",
    )
    skin=QualifiedSkinIR(
        (
            QualifiedSkinRow("a",(("ja",1.0),),0.0,0.0),
            QualifiedSkinRow("b",(("ja",1.0),),0.0,0.0),
            QualifiedSkinRow("c",(("jb",1.0),),0.0,0.0),
        ),
        surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,{"status":"PASS"},"skin-holeless",
    )
    envelope=DeformationCapabilityEnvelopeIR(
        skeleton.skeleton_lineage_hash,
        (JointCapabilityRangeIR("ja",0.0,0.0),JointCapabilityRangeIR("jb",0.0,0.0)),
        tuple(f"cam{i}" for i in range(8)),(),"axis","probe","",
    )
    envelope=replace(envelope,envelope_lineage_hash=deformation_envelope_lineage_hash(envelope))
    policy=build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH",0.0,0.0,1.0,1.0),
            CarrierCoverageThresholdIR("PLANAR",0.0,0.0,1.0,1.0),
        ),
    )
    report={
        "gates":{
            "G1_SUPPORT_LINEAGE":"PASS","G2_TOPOLOGY":"PASS","G3_DEFORMATION":"PASS",
            "G3B_SKIN_TOPOLOGY_COMPATIBILITY":"PASS",
            "G4_COMPONENT_BOUNDARY":"PASS","G5_MULTIVIEW_COVERAGE":"PASS",
        },
        "single_aggregate_score_authority":False,
        "view_component_coverage_matrix_complete":True,
        "consequential_unknown_boundary_count":0,
        "g3_envelope_binding_hash":envelope.envelope_lineage_hash,
        "g3_stress_probe_hash":"g3",
        "g3_stress_probe_status":"PASS",
        "skin_topology_compatibility_report_hash":"skin-topology-test",
        "skin_topology_compatibility_status":"PASS",
        "skin_topology_weight_mutation":False,
        "carrier_policy_hash":carrier.carrier_policy_lineage_hash,
        "view_component_coverage":tuple(
            {
                "view_index":vi,"component_id":c.component_id,"carrier_class":"MESH",
                "recall":1.0,"precision":1.0,"largest_coherent_hole_fraction":0.0,
                "interior_uncovered_fraction":0.0,"status":"PASS",
            }
            for vi in range(8) for c in partition.components
        ),
    }
    mesh=QualifiedMeshIR(
        vertices,faces,edges,surface.geometry_lineage_hash,partition.partition_lineage_hash,
        carrier.carrier_policy_lineage_hash,envelope.envelope_lineage_hash,
        policy.qualification_policy_lineage_hash,report,"",
    )
    audit=qualified_mesh_intrinsic_audit(mesh,surface=surface,partition=partition)
    mesh=replace(mesh,qualification_report={**report,"intrinsic_audit_hash":content_sha256(audit)})
    mesh=replace(mesh,mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    return skeleton,skin,envelope,policy,mesh


def test_holeless_dense_split_preserves_face_area_and_component_purity():
    surface,partition,carrier,candidate=_fixture()
    validate_canonical_mesh_candidate(candidate,surface=surface,partition=partition,carrier_policy=carrier)
    assert candidate.metadata["face_deletion_count"] == 0
    assert candidate.metadata["source_face_count"] == 1
    assert candidate.metadata["mixed_source_face_count"] == 1
    assert candidate.metadata["output_face_count"] == 3
    assert candidate.metadata["rest_area_relative_error"] <= 1e-12
    by_id={v.candidate_vertex_id:v for v in candidate.vertices}
    assert all(len({by_id[x].component_id for x in face})==1 for face in candidate.faces)
    seam=[v for v in candidate.vertices if v.support_binding.mode=="SEAM_GEOMETRY_INTERPOLATION"]
    assert len(seam)==4
    # Two geometric seam locations, duplicated once per mechanical side.
    positions=[tuple(round(float(x),12) for x in v.P) for v in seam]
    assert len(set(positions))==2
    assert all(positions.count(p)==2 for p in set(positions))


def test_holeless_seam_geometry_does_not_mix_skin_across_components():
    surface,partition,carrier,candidate=_fixture()
    skeleton,skin,envelope,policy,mesh=_qualified_mesh(surface,partition,carrier,candidate)
    bound=bind_product_mesh_skin(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,
        partition=partition,carrier_policy=carrier,envelope=envelope,policy=policy,
    )
    mesh_by_id={v.canonical_mesh_vertex_id:v for v in mesh.vertices}
    rows={r.canonical_mesh_vertex_id:r for r in bound.rows}
    seam_vertices=[
        v for v in mesh.vertices
        if v.support_binding.mode=="SEAM_GEOMETRY_INTERPOLATION"
    ]
    assert seam_vertices
    for v in seam_vertices:
        support=tuple(v.support_binding.metadata["skin_support_coefficients"])
        assert rows[v.canonical_mesh_vertex_id].source_support_coefficients==support
        owned_surface=str(support[0][0])
        expected="jb" if owned_surface=="c" else "ja"
        assert rows[v.canonical_mesh_vertex_id].influences==((expected,1.0),)
        # Geometry support intentionally spans the seam while skin support does not.
        assert len(v.support_binding.coefficients)==2
        assert tuple(v.support_binding.coefficients)!=support


def test_stage35_holeless_seam_transfer_uses_mechanical_skin_support():
    surface,partition,carrier,candidate=_fixture()
    skeleton,skin,_,_,_=_qualified_mesh(surface,partition,carrier,candidate)
    _,weights,_=_candidate_skin_matrix(
        candidate,surface=surface,skeleton=skeleton,skin=skin
    )
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints)
    ji={jid:i for i,jid in enumerate(joint_ids)}
    for vi,v in enumerate(candidate.vertices):
        if v.support_binding.mode!="SEAM_GEOMETRY_INTERPOLATION":
            continue
        support=tuple(v.support_binding.metadata["skin_support_coefficients"])
        assert len(support)==1
        owned_surface=str(support[0][0])
        expected="jb" if owned_surface=="c" else "ja"
        assert abs(float(weights[vi,ji[expected]])-1.0)<=1e-12
        assert sum(abs(float(weights[vi,j])) for jid,j in ji.items() if jid!=expected)<=1e-12
        # Geometry interpolation crosses the seam; Stage35 mechanical skin must not.
        assert len(v.support_binding.coefficients)==2
        assert tuple(v.support_binding.coefficients)!=support


def test_stage35_repartition_can_recover_mechanical_owner_from_holeless_seam_vertex():
    surface,partition,carrier,candidate=_fixture()
    skeleton,skin,_,_,_=_qualified_mesh(surface,partition,carrier,candidate)
    by_id={v.candidate_vertex_id:v for v in candidate.vertices}

    def owner(v):
        if v.support_binding.mode=="IDENTITY_SURFACE_NODE":
            return str(v.support_binding.coefficients[0][0])
        if v.support_binding.mode=="SEAM_GEOMETRY_INTERPOLATION":
            return str(v.support_binding.metadata["skin_support_coefficients"][0][0])
        return None

    target=None
    for fi,face in enumerate(candidate.faces):
        vs=[by_id[x] for x in face]
        owners=[owner(v) for v in vs]
        if (
            any(v.support_binding.mode=="SEAM_GEOMETRY_INTERPOLATION" for v in vs)
            and None not in owners
            and len(set(owners))>=2
        ):
            target=fi
            break
    assert target is not None

    directive=propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report={
            "unsafe_face_indices":(int(target),),
            "report_hash":"holeless-seam-owner-compat",
        },
    )
    assert directive["status"]=="REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
    assert directive["candidate_separate_pair_count"]==1
    assert directive["unresolved_unsafe_face_count"]==0
    row=directive["proposed_boundary_overrides"][0]
    assert row["metadata"]["mechanical_owner_surface_support_required"] is True
    assert row["metadata"]["identity_or_holeless_seam_owner_supported"] is True
