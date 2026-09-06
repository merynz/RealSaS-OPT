from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.rig import (
    compile_scene_first_rigging_v1,
    qualify_skeleton_v2,
)
from compiler.realsas_compiler_core.skeleton_admission_v1 import (
    PRODUCT_SKELETON_ADMISSION_POLICY_V1,
    admit_skeleton_proposal_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SkeletonProposalEdge,
    SkeletonProposalIR,
    SkeletonProposalJoint,
    SurfaceNode,
    SurfaceRelation,
)
from models.geppetto.v2.geppetto_conditioning_v2 import (
    C0_LOCALITY_POLICY_V1,
    DIAGNOSTIC_CONDITIONING_AUTHORITY_V1,
    PRODUCT_CONDITIONING_AUTHORITY_V1,
    GeppettoConditioningAdapterV2,
    GeppettoProductConditioningAdapterV2,
    topology_neighbor_index_v1,
)


def _surface() -> RiggingSurfaceIR:
    nodes = (
        SurfaceNode(
            "S0",
            (0.0, 0.0, 0.0),
            (0, 1),
            ("IRIS:TEST",),
            (),
            raster_bindings=((0, (12.0, 24.0)), (1, (32.0, 28.0))),
            persistence_group_id="PG0",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
        SurfaceNode(
            "S1",
            (0.4, 0.0, 0.0),
            (0,),
            ("IRIS:TEST",),
            (),
            raster_bindings=((0, (52.0, 24.0)),),
            persistence_group_id="PG1",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("OBSERVED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
        SurfaceNode(
            "S2",
            (0.8, 0.0, 0.0),
            (),
            ("IRIS:TEST",),
            (),
            raster_bindings=(),
            persistence_group_id="PG2",
            derived_normal=(0.0, 0.0, 1.0),
            validity_flags=("MODEL_COMPLETED_SIGNED_ZERO_SURFACE",),
            metadata={"teacher_truth_used": False},
        ),
    )
    relations = (
        SurfaceRelation(
            "R0",
            "S0",
            "S1",
            "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
            1.0,
            metadata={"teacher_truth_used": False},
        ),
        SurfaceRelation(
            "R1",
            "S1",
            "S2",
            "SIGNED_ZERO_SURFACE_TOPOLOGY_NEIGHBOR",
            1.0,
            metadata={"teacher_truth_used": False},
        ),
    )
    return RiggingSurfaceIR(
        nodes,
        relations,
        geometry_lineage_hash="SURFACE_LINEAGE_TEST_V1",
        builder_id="RealSaS.GeometricSubstrateAssembler.SceneFirstSigned.v1",
        schema_version="RealSaS.RiggingSurfaceIR.v1",
        metadata={
            "scene_first_signed_geometry": True,
            "teacher_truth_used": False,
            "character_gen_runtime_used": False,
            "raster_coordinate_system": "PIXEL_CENTER_XY",
            "resolution": 128,
            "source_run_id": "TEST_RUN",
            "source_checkpoint_sha256": "a" * 64,
            "source_zero_surface_sha256": "b" * 64,
            "Nd_operator_sha256": "c" * 64,
            "compact_surface_node_count": 3,
            "observed_node_count": 2,
            "completed_node_count": 1,
        },
    )


def _proposal(surface: RiggingSurfaceIR, *, unsupported: bool = False) -> SkeletonProposalIR:
    support0 = () if unsupported else ("S0",)
    joints = (
        SkeletonProposalJoint(
            "P0",
            (0.2, 0.0, 0.0),
            root_score=0.95,
            confidence=0.9,
            support_surface_ids=support0,
        ),
        SkeletonProposalJoint(
            "P1",
            (0.2, 0.0, 0.0),
            root_score=0.1,
            confidence=0.85,
            support_surface_ids=("S1",),
        ),
    )
    edges = (
        SkeletonProposalEdge(
            "E0",
            "P0",
            "P1",
            score=0.9,
            confidence=0.9,
            reason="TEST_PARENT_EVIDENCE",
        ),
        SkeletonProposalEdge(
            "E1",
            "P1",
            "P0",
            score=0.2,
            confidence=0.8,
            reason="TEST_PARENT_EVIDENCE",
        ),
    )
    return SkeletonProposalIR(
        joints,
        edges,
        surface.geometry_lineage_hash,
        model_provenance="TEST_GEPPETTO",
        metadata={"teacher_truth_used": False},
    )


def test_product_conditioning_is_fail_closed_and_preserves_raster_and_topology():
    surface = _surface()
    conditioning = GeppettoProductConditioningAdapterV2()([surface])

    assert conditioning.conditioning_authorities == (
        PRODUCT_CONDITIONING_AUTHORITY_V1,
    )
    assert conditioning.surface_boundary_audit_hashes[0]
    assert conditioning.local_relation_pairs[0] == ((0, 1), (1, 2))
    assert conditioning.local_relation_hashes[0]
    assert conditioning.current_c0_locality_policy == C0_LOCALITY_POLICY_V1

    raster = np.asarray(conditioning.features[0, :, 20:24], np.float64)
    assert np.isfinite(raster).all()
    assert np.any(np.abs(raster) > 1e-6)

    idx, valid = topology_neighbor_index_v1(conditioning, k=2)
    assert idx.shape == (1, 3, 2)
    assert valid.shape == (1, 3, 2)
    assert set(idx[0, 1, valid[0, 1]].tolist()) == {0, 2}


def test_product_conditioning_rejects_stripped_scene_first_raster():
    surface = _surface()
    bad0 = replace(surface.surface_nodes[0], raster_bindings=())
    bad = replace(surface, surface_nodes=(bad0, *surface.surface_nodes[1:]))
    with pytest.raises(QualificationError, match="RIGGING_SURFACE_IR_AUDIT_FAIL"):
        GeppettoProductConditioningAdapterV2()([bad])


def test_diagnostic_adapter_never_mints_product_conditioning_authority():
    conditioning = GeppettoConditioningAdapterV2()([_surface()])
    assert conditioning.conditioning_authorities == (
        DIAGNOSTIC_CONDITIONING_AUTHORITY_V1,
    )
    assert conditioning.surface_boundary_audit_hashes == ("",)


def test_product_admission_preserves_coincident_controls_and_native_count():
    surface = _surface()
    admitted = admit_skeleton_proposal_v1(
        surface,
        _proposal(surface),
        policy=PRODUCT_SKELETON_ADMISSION_POLICY_V1,
    )
    report = admitted.admission_report
    assert report["input_joint_count"] == 2
    assert report["admitted_joint_count"] == 2
    assert report["coincident_locus_group_sizes"] == (2,)
    assert report["geometry_only_fused_joint_count"] == 0
    assert report["synthesized_deform_node_count"] == 0
    assert report["native_count_preserved"] is True


def test_product_admission_rejects_unsupported_joint_instead_of_hiding_it():
    surface = _surface()
    with pytest.raises(QualificationError, match="SKELETON_ADMISSION_UNSUPPORTED_JOINTS"):
        admit_skeleton_proposal_v1(
            surface,
            _proposal(surface, unsupported=True),
            policy=PRODUCT_SKELETON_ADMISSION_POLICY_V1,
        )


def test_scene_first_product_route_requires_product_conditioning_certificate():
    surface = _surface()
    proposal = _proposal(surface)
    diagnostic = GeppettoConditioningAdapterV2()([surface])
    with pytest.raises(
        QualificationError,
        match="RIGGING_PRODUCT_CONDITIONING_NOT_PRODUCT_VALIDATED",
    ):
        compile_scene_first_rigging_v1(
            surface,
            proposal,
            conditioning_certificate=diagnostic.product_certificate(0),
        )


def test_scene_first_product_route_closes_to_single_canonical_deform_tree():
    surface = _surface()
    proposal = _proposal(surface)
    conditioning = GeppettoProductConditioningAdapterV2()([surface])
    qualified = compile_scene_first_rigging_v1(
        surface,
        proposal,
        conditioning_certificate=conditioning.product_certificate(0),
    )

    assert len(qualified.joints) == 2
    assert len(qualified.deform_root_ids) == 1
    canonical = {j.canonical_joint_id for j in qualified.joints}
    assert canonical.isdisjoint({"P0", "P1"})
    assert all(
        j.parent_canonical_id is None or j.parent_canonical_id in canonical
        for j in qualified.joints
    )

    report = qualified.qualification_report
    assert report["promotion_authority"] is True
    assert report["product_route"] == "compile_scene_first_rigging_v1"
    assert report["product_deform_graph_policy"] == "SINGLE_CONNECTED_DEFORM_TREE_V1"
    assert report["geometry_only_duplicate_fusion"] == "FORBIDDEN"
    assert report["deform_node_completion"] == "FORBIDDEN_BUDGET_0"
    assert report["admission_report"]["admitted_joint_count"] == 2
    assert report["admission_report"]["geometry_only_fused_joint_count"] == 0


def test_legacy_v2_qualifier_is_explicitly_non_promoting():
    surface = _surface()
    qualified = qualify_skeleton_v2(surface, _proposal(surface))
    assert qualified.qualification_report["promotion_authority"] is False
    assert qualified.qualification_report["product_route"] == (
        "USE_compile_scene_first_rigging_v1"
    )
