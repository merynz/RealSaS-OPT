from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_puppet_state_v1 import (
    build_canonical_puppet_state,
    canonical_puppet_state_hash,
    validate_canonical_puppet_state,
)
from compiler.realsas_compiler_core.product_mesh_skin_v1 import bind_product_mesh_skin
from compiler.realsas_compiler_core.product_authority_v1 import (
    CarrierCoverageThresholdIR,
    ComponentCarrierDecisionIR,
    ComponentRegionIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
    build_component_carrier_policy,
    build_mechanical_partition,
    build_mesh_qualification_policy,
    deformation_envelope_lineage_hash,
    qualified_mesh_intrinsic_audit,
    qualified_mesh_lineage_hash,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import (
    QualifiedJoint,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedSkinRow,
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceSupportBinding,
)


def _fixture():
    surface=RiggingSurfaceIR(
        (
            SurfaceNode("s0",(0.0,0.0,0.0),(0,),("p",),("o0",)),
            SurfaceNode("s1",(1.0,0.0,0.0),(0,),("p",),("o1",)),
            SurfaceNode("s2",(0.0,1.0,0.0),(0,),("p",),("o2",)),
        ),(),"surface-hash"
    )
    skeleton=QualifiedSkeletonIR(
        (QualifiedJoint("j0",(0.0,0.0,0.0),None,("s0","s1","s2"),"p0"),),
        "j0",{"status":"PASS"},"skeleton-hash"
    )
    skin=QualifiedSkinIR(
        tuple(QualifiedSkinRow(sid,(("j0",1.0),),0.0,0.0) for sid in ("s0","s1","s2")),
        surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,{"status":"PASS"},"skin-hash"
    )
    partition=build_mechanical_partition(
        surface=surface,
        components=(ComponentRegionIR("c0",("s0","s1","s2")),),
        boundary_constraints=(),
    )
    carrier=build_component_carrier_policy(
        partition=partition,
        decisions=(ComponentCarrierDecisionIR("c0","MESH",("evidence",)),),
    )
    envelope=DeformationCapabilityEnvelopeIR(
        skeleton.skeleton_lineage_hash,
        (JointCapabilityRangeIR("j0",0.0,0.0),),
        tuple(f"cam{i}" for i in range(8)),
        (),
        "axis-hash",
        "probe-hash",
        "",
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
    vertices=tuple(
        QualifiedMeshVertexIR(
            f"v{i}",
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE",((f"s{i}",1.0),)),
            "c0",surface.surface_nodes[i].P,f"cv{i}"
        )
        for i in range(3)
    )
    report={
        "gates":{
            "G1_SUPPORT_LINEAGE":"PASS","G2_TOPOLOGY":"PASS","G3_DEFORMATION":"PASS",
            "G4_COMPONENT_BOUNDARY":"PASS","G5_MULTIVIEW_COVERAGE":"PASS",
        },
        "single_aggregate_score_authority":False,
        "view_component_coverage_matrix_complete":True,
        "consequential_unknown_boundary_count":0,
        "g3_envelope_binding_hash":envelope.envelope_lineage_hash,
        "g3_stress_probe_hash":"g3-hash",
        "g3_stress_probe_status":"PASS",
        "carrier_policy_hash":carrier.carrier_policy_lineage_hash,
        "view_component_coverage":tuple(
            {"view_index":vi,"component_id":"c0","carrier_class":"MESH","recall":1.0,"precision":1.0,
             "largest_coherent_hole_fraction":0.0,"interior_uncovered_fraction":0.0,"status":"PASS"}
            for vi in range(8)
        ),
    }
    mesh=QualifiedMeshIR(
        vertices,(("v0","v1","v2"),),(("v0","v1"),("v1","v2"),("v0","v2")),
        surface.geometry_lineage_hash,partition.partition_lineage_hash,carrier.carrier_policy_lineage_hash,
        envelope.envelope_lineage_hash,policy.qualification_policy_lineage_hash,report,""
    )
    audit=qualified_mesh_intrinsic_audit(mesh,surface=surface,partition=partition)
    mesh=replace(mesh,qualification_report={**report,"intrinsic_audit_hash":content_sha256(audit)})
    mesh=replace(mesh,mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    mesh_skin=bind_product_mesh_skin(
        surface=surface,skeleton=skeleton,skin=skin,mesh=mesh,
        partition=partition,carrier_policy=carrier,envelope=envelope,policy=policy,
    )
    return surface,skeleton,skin,partition,carrier,envelope,policy,mesh,mesh_skin


def test_canonical_puppet_state_binds_complete_mechanical_product_identity():
    args=_fixture()
    state=build_canonical_puppet_state(
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],envelope=args[5],policy=args[6],mesh=args[7],mesh_skin=args[8],
    )
    assert len(state.product_state_hash)==64
    assert state.mesh_lineage_hash==args[7].mesh_lineage_hash
    assert state.mesh_skin_lineage_hash==args[8].mesh_skin_lineage_hash
    assert state.metadata["single_product_mesh_authority"] is True
    assert state.metadata["presentation_bound"] is False
    assert len(state.qualification_ledger)==9
    validate_canonical_puppet_state(
        state,
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],envelope=args[5],policy=args[6],mesh=args[7],mesh_skin=args[8],
    )


def test_state_hash_changes_if_policy_identity_changes_even_when_geometry_is_same():
    args=_fixture()
    state=build_canonical_puppet_state(
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],envelope=args[5],policy=args[6],mesh=args[7],mesh_skin=args[8],
    )
    changed_policy=replace(args[6],metadata={**args[6].metadata,"revision":"different"},qualification_policy_lineage_hash="")
    from compiler.realsas_compiler_core.product_authority_v1 import mesh_qualification_policy_lineage_hash
    changed_policy=replace(changed_policy,qualification_policy_lineage_hash=mesh_qualification_policy_lineage_hash(changed_policy))
    stale=replace(state,mesh_policy_lineage_hash=changed_policy.qualification_policy_lineage_hash,product_state_hash="")
    stale=replace(stale,product_state_hash=canonical_puppet_state_hash(stale))
    with pytest.raises(QualificationError):
        validate_canonical_puppet_state(
            stale,
            surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
            carrier_policy=args[4],envelope=args[5],policy=changed_policy,mesh=args[7],mesh_skin=args[8],
        )


def test_state_rejects_mixed_mesh_skin_lineage():
    args=_fixture()
    state=build_canonical_puppet_state(
        surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
        carrier_policy=args[4],envelope=args[5],policy=args[6],mesh=args[7],mesh_skin=args[8],
    )
    bad=replace(state,mesh_skin_lineage_hash="wrong",product_state_hash="")
    bad=replace(bad,product_state_hash=canonical_puppet_state_hash(bad))
    with pytest.raises(QualificationError,match="mesh_skin_lineage_hash"):
        validate_canonical_puppet_state(
            bad,
            surface=args[0],skeleton=args[1],skin=args[2],partition=args[3],
            carrier_policy=args[4],envelope=args[5],policy=args[6],mesh=args[7],mesh_skin=args[8],
        )
