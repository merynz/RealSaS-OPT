from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_authority_v1 import (
    CanonicalMeshCandidateIR,
    CanonicalMeshVertexCandidateIR,
    CarrierCoverageThresholdIR,
    ComponentBoundaryConstraintIR,
    ComponentCarrierDecisionIR,
    ComponentRegionIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    PresentationAttachmentIR,
    PresentationDecisionEvidenceIR,
    PresentationSlotIR,
    PresentationViewOverlayIR,
    QualifiedMeshIR,
    QualifiedMeshVertexIR,
    QualifiedPresentationGraphIR,
    build_component_carrier_policy,
    build_mechanical_partition,
    build_mesh_qualification_policy,
    deformation_envelope_lineage_hash,
    canonical_mesh_candidate_lineage_hash,
    qualified_mesh_intrinsic_audit,
    qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
    qualify_canonical_mesh_candidate,
    validate_deformation_capability_envelope,
    validate_qualified_mesh,
    validate_qualified_presentation_graph,
)
from compiler.realsas_compiler_core.types import (
    QualificationError, RiggingSurfaceIR, SurfaceNode, SurfaceSupportBinding,
)


def _surface():
    points = (
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.9, 0.0),
        (3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.5, 0.9, 0.0),
    )
    nodes = tuple(
        SurfaceNode(f"s{i}", p, (0,), ("src",), (f"o{i}",))
        for i, p in enumerate(points)
    )
    return RiggingSurfaceIR(nodes, (), "surface-hash")


def _partition(surface):
    return build_mechanical_partition(
        surface=surface,
        components=(
            ComponentRegionIR("c0", ("s0", "s1", "s2")),
            ComponentRegionIR("c1", ("s3", "s4", "s5")),
        ),
        boundary_constraints=(
            ComponentBoundaryConstraintIR("b0", "s2", "s3", "SEPARATE", ("mechanical",)),
            ComponentBoundaryConstraintIR("b1", "s0", "s1", "PRESERVE_CONTINUITY", ("mechanical",)),
            ComponentBoundaryConstraintIR("b2", "s4", "s5", "UNKNOWN", ("ambiguous",)),
        ),
    )


def _carrier_policy(partition, *, c0="MESH", c1="MESH"):
    return build_component_carrier_policy(
        partition=partition,
        decisions=(
            ComponentCarrierDecisionIR("c0", c0, ("structural-evidence-c0",)),
            ComponentCarrierDecisionIR("c1", c1, ("structural-evidence-c1",)),
        ),
    )


def _mesh_policy(*, mesh_recall=0.0):
    return build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.25,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH", mesh_recall, 0.0, 1.0, 1.0),
            CarrierCoverageThresholdIR("PLANAR", 0.0, 0.0, 1.0, 1.0),
        ),
        metadata={"test_only_thresholds": True},
    )


def _envelope():
    value = DeformationCapabilityEnvelopeIR(
        "skeleton-hash",
        (JointCapabilityRangeIR("j0", -45.0, 45.0),),
        tuple(f"cam{i}" for i in range(8)),
        (),
        "axis-contract-hash",
        "probe-plan-hash",
        "",
    )
    value = replace(value, envelope_lineage_hash=deformation_envelope_lineage_hash(value))
    validate_deformation_capability_envelope(value, known_joint_ids={"j0"})
    return value


def _identity_vertex(vid, sid, component_id, point):
    return QualifiedMeshVertexIR(
        vid,
        SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((sid, 1.0),)),
        component_id,
        point,
        f"candidate:{vid}",
    )


def _valid_mesh(surface, partition, carrier_policy, envelope, policy):
    vertices = tuple(
        _identity_vertex(f"v{i}", f"s{i}", "c0" if i < 3 else "c1", surface.surface_nodes[i].P)
        for i in range(6)
    )
    faces = (("v0", "v1", "v2"), ("v3", "v4", "v5"))
    edges = (
        ("v0", "v1"), ("v1", "v2"), ("v0", "v2"),
        ("v3", "v4"), ("v4", "v5"), ("v3", "v5"),
    )
    carrier_by_component = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    base_report = {
        "gates": {
            "G1_SUPPORT_LINEAGE": "PASS",
            "G2_TOPOLOGY": "PASS",
            "G3_DEFORMATION": "PASS",
            "G4_COMPONENT_BOUNDARY": "PASS",
            "G5_MULTIVIEW_COVERAGE": "PASS",
        },
        "single_aggregate_score_authority": False,
        "view_component_coverage_matrix_complete": True,
        "consequential_unknown_boundary_count": 0,
        "unknown_boundary_analysis_hash": "unknown-analysis-hash",
        "g3_envelope_binding_hash": envelope.envelope_lineage_hash,
        "g3_stress_probe_hash": "stress-probe-hash",
        "g3_stress_probe_status": "PASS",
        "carrier_policy_hash": carrier_policy.carrier_policy_lineage_hash,
        "view_component_coverage": tuple(
            {
                "view_index": view,
                "component_id": component_id,
                "carrier_class": carrier_by_component[component_id],
                "recall": 1.0,
                "precision": 1.0,
                "largest_coherent_hole_fraction": 0.0,
                "interior_uncovered_fraction": 0.0,
                "status": "PASS",
            }
            for view in range(8)
            for component_id in ("c0", "c1")
        ),
    }
    mesh = QualifiedMeshIR(
        vertices, faces, edges,
        surface.geometry_lineage_hash,
        partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash,
        envelope.envelope_lineage_hash,
        policy.qualification_policy_lineage_hash,
        base_report,
        "",
    )
    audit = qualified_mesh_intrinsic_audit(mesh, surface=surface, partition=partition)
    report = {**base_report, "intrinsic_audit_hash": content_sha256(audit)}
    mesh = replace(mesh, qualification_report=report)
    return replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))


def _context():
    surface = _surface()
    partition = _partition(surface)
    carrier_policy = _carrier_policy(partition)
    envelope = _envelope()
    policy = _mesh_policy()
    return surface, partition, carrier_policy, envelope, policy


def test_canonical_qualified_mesh_has_no_view_or_camera_authority_fields():
    assert "view_index" not in QualifiedMeshIR.__dataclass_fields__
    assert "camera_binding_hash" not in QualifiedMeshIR.__dataclass_fields__


def test_partition_is_bound_to_immutable_surface_and_declares_both_boundary_directions():
    surface = _surface()
    partition = _partition(surface)
    assert partition.surface_lineage_hash == surface.geometry_lineage_hash
    assert {x.decision for x in partition.boundary_constraints} == {
        "SEPARATE", "PRESERVE_CONTINUITY", "UNKNOWN"
    }


def test_qualified_mesh_recomputes_intrinsics_and_binds_exact_carrier_and_mesh_policy():
    surface, partition, carrier_policy, envelope, policy = _context()
    mesh = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    validate_qualified_mesh(
        mesh, surface=surface, partition=partition, carrier_policy=carrier_policy,
        envelope=envelope, policy=policy,
    )


def test_empty_mesh_cannot_pass_by_claiming_all_five_gates():
    surface, partition, carrier_policy, envelope, policy = _context()
    good = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    bad = replace(good, vertices=(), faces=(), edges=(), mesh_lineage_hash="")
    with pytest.raises(QualificationError, match="EMPTY_PRODUCT_GEOMETRY"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_face_cannot_cross_mechanical_component_boundary():
    surface, partition, carrier_policy, envelope, policy = _context()
    good = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    bad = replace(good, faces=(("v0", "v1", "v3"), ("v3", "v4", "v5")), mesh_lineage_hash="")
    with pytest.raises(QualificationError, match="FACE_CROSSES_COMPONENT_BOUNDARY"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_g5_cannot_reclassify_component_after_carrier_policy_freeze():
    surface, partition, carrier_policy, envelope, policy = _context()
    good = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    rows = list(good.qualification_report["view_component_coverage"])
    rows[0] = {**rows[0], "carrier_class": "PLANAR"}
    report = {**good.qualification_report, "view_component_coverage": tuple(rows)}
    bad = replace(good, qualification_report=report, mesh_lineage_hash="")
    bad = replace(bad, mesh_lineage_hash=qualified_mesh_lineage_hash(bad))
    with pytest.raises(QualificationError, match="CARRIER_CLASS_DRIFT"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_g5_threshold_is_enforced_from_typed_policy_not_report_status():
    surface, partition, carrier_policy, envelope, _ = _context()
    policy = _mesh_policy(mesh_recall=0.99)
    good = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    rows = list(good.qualification_report["view_component_coverage"])
    rows[0] = {**rows[0], "recall": 0.98, "status": "PASS"}
    report = {**good.qualification_report, "view_component_coverage": tuple(rows)}
    bad = replace(good, qualification_report=report, mesh_lineage_hash="")
    bad = replace(bad, mesh_lineage_hash=qualified_mesh_lineage_hash(bad))
    with pytest.raises(QualificationError, match="G5_RECALL_FAIL"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_g3_policy_cannot_be_weaker_than_subject_free_calibration_floor():
    with pytest.raises(QualificationError, match="WEAKER_THAN_G3_ANGLE_FLOOR"):
        build_mesh_qualification_policy(
            g1_max_normal_refinement_ratio=0.25,
            g1_max_tangential_to_normal_ratio=0.25,
            g3_min_angle_deg=5.0,
            g3_max_aspect_longest_over_min_altitude=16.0,
            coverage_thresholds=(
                CarrierCoverageThresholdIR("MESH", 0.0, 0.0, 1.0, 1.0),
                CarrierCoverageThresholdIR("PLANAR", 0.0, 0.0, 1.0, 1.0),
            ),
        )


def test_qualified_mesh_requires_zero_consequential_unknown():
    surface, partition, carrier_policy, envelope, policy = _context()
    mesh = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    report = {**mesh.qualification_report, "consequential_unknown_boundary_count": 1}
    bad = replace(mesh, qualification_report=report, mesh_lineage_hash="")
    bad = replace(bad, mesh_lineage_hash=qualified_mesh_lineage_hash(bad))
    with pytest.raises(QualificationError, match="CONSEQUENTIAL_UNKNOWN"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_presentation_carrier_class_is_orthogonal_to_mechanics_but_cannot_drift_from_frozen_policy():
    surface = _surface()
    partition = _partition(surface)
    carrier_policy = _carrier_policy(partition, c0="PLANAR", c1="MESH")
    slots = (PresentationSlotIR("slot0", "j0", 0, "a0", ("ATTACHMENT", "ORDER")),)
    attachments = (PresentationAttachmentIR("a0", "slot0", ("c0",), "RIGID", "PLANAR", "plane-hash"),)
    overlays = tuple(PresentationViewOverlayIR(i, f"cam{i}", f"app{i}", f"comp{i}") for i in range(8))
    decisions = (
        PresentationDecisionEvidenceIR("d0", "SLOT_BINDING", "MECHANICAL", ("partition-hash",)),
        PresentationDecisionEvidenceIR("d1", "SETUP_COMPOSITION", "OBSERVATION_REST", ("obs-hash",)),
    )
    graph = QualifiedPresentationGraphIR(
        slots, attachments, overlays, decisions,
        "skeleton-hash", "mesh-hash", partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash, "product-state-hash",
        {"status": "PASS"}, "",
    )
    graph = replace(graph, presentation_lineage_hash=qualified_presentation_lineage_hash(graph))
    validate_qualified_presentation_graph(graph, carrier_policy=carrier_policy)
    assert graph.attachments[0].mechanical_class == "RIGID"
    assert graph.attachments[0].carrier_class == "PLANAR"


def test_candidate_promotion_is_deterministic_and_compiler_mints_intrinsic_audit():
    surface, partition, carrier_policy, envelope, policy = _context()
    candidate_vertices = tuple(
        CanonicalMeshVertexCandidateIR(
            f"cv{i}",
            SurfaceSupportBinding("IDENTITY_SURFACE_NODE", ((f"s{i}", 1.0),)),
            "c0" if i < 3 else "c1",
            surface.surface_nodes[i].P,
        )
        for i in range(6)
    )
    candidate = CanonicalMeshCandidateIR(
        candidate_vertices,
        (("cv0", "cv1", "cv2"), ("cv3", "cv4", "cv5")),
        (
            ("cv0", "cv1"), ("cv1", "cv2"), ("cv0", "cv2"),
            ("cv3", "cv4"), ("cv4", "cv5"), ("cv3", "cv5"),
        ),
        surface.geometry_lineage_hash,
        partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash,
        "TEST_CANONICAL_MESH_PRODUCER",
        "producer-policy-hash",
        "",
    )
    candidate = replace(candidate, candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(candidate))
    carrier_by_component = {row.component_id: row.carrier_class for row in carrier_policy.decisions}
    report = {
        "gates": {
            "G1_SUPPORT_LINEAGE": "PASS",
            "G2_TOPOLOGY": "PASS",
            "G3_DEFORMATION": "PASS",
            "G4_COMPONENT_BOUNDARY": "PASS",
            "G5_MULTIVIEW_COVERAGE": "PASS",
        },
        "single_aggregate_score_authority": False,
        "view_component_coverage_matrix_complete": True,
        "consequential_unknown_boundary_count": 0,
        "unknown_boundary_analysis_hash": "unknown-analysis-hash",
        "g3_envelope_binding_hash": envelope.envelope_lineage_hash,
        "g3_stress_probe_hash": "stress-probe-hash",
        "g3_stress_probe_status": "PASS",
        "carrier_policy_hash": carrier_policy.carrier_policy_lineage_hash,
        "view_component_coverage": tuple(
            {
                "view_index": view,
                "component_id": component_id,
                "carrier_class": carrier_by_component[component_id],
                "recall": 1.0,
                "precision": 1.0,
                "largest_coherent_hole_fraction": 0.0,
                "interior_uncovered_fraction": 0.0,
                "status": "PASS",
            }
            for view in range(8)
            for component_id in ("c0", "c1")
        ),
        "intrinsic_audit_hash": "CALLER_MUST_NOT_CONTROL_THIS",
    }
    mesh_a = qualify_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        qualification_report=report,
    )
    mesh_b = qualify_canonical_mesh_candidate(
        candidate,
        surface=surface,
        partition=partition,
        carrier_policy=carrier_policy,
        envelope=envelope,
        policy=policy,
        qualification_report=report,
    )
    assert mesh_a.mesh_lineage_hash == mesh_b.mesh_lineage_hash
    assert mesh_a.qualification_report["intrinsic_audit_hash"] != "CALLER_MUST_NOT_CONTROL_THIS"
    assert all(v.canonical_mesh_vertex_id.startswith("MV:") for v in mesh_a.vertices)


def test_mechanical_carrier_policy_cannot_use_presentation_only_clip_class():
    surface = _surface()
    partition = _partition(surface)
    with pytest.raises(QualificationError, match="CARRIER_POLICY_DECISION_INVALID"):
        _carrier_policy(partition, c0="CLIP", c1="MESH")


def test_g5_interior_uncovered_metric_is_a_hard_gate_against_peppering():
    surface, partition, carrier_policy, envelope, _ = _context()
    policy = build_mesh_qualification_policy(
        g1_max_normal_refinement_ratio=0.125,
        g1_max_tangential_to_normal_ratio=0.25,
        g3_min_angle_deg=7.5,
        g3_max_aspect_longest_over_min_altitude=16.0,
        coverage_thresholds=(
            CarrierCoverageThresholdIR("MESH", 0.97, 0.995, 0.005, 0.005),
            CarrierCoverageThresholdIR("PLANAR", 0.99, 0.995, 0.0025, 0.0025),
        ),
        metadata={"policy_id": "QUALIFIED_MESH_PRODUCT_POLICY_V1_TEST"},
    )
    good = _valid_mesh(surface, partition, carrier_policy, envelope, policy)
    rows = list(good.qualification_report["view_component_coverage"])
    rows[0] = {**rows[0], "interior_uncovered_fraction": 0.006, "status": "PASS"}
    bad = replace(good, qualification_report={**good.qualification_report, "view_component_coverage": tuple(rows)}, mesh_lineage_hash="")
    bad = replace(bad, mesh_lineage_hash=qualified_mesh_lineage_hash(bad))
    with pytest.raises(QualificationError, match="INTERIOR_UNCOVERED_FAIL"):
        validate_qualified_mesh(
            bad, surface=surface, partition=partition, carrier_policy=carrier_policy,
            envelope=envelope, policy=policy,
        )


def test_deformation_envelope_requires_explicit_axis_contract_authority():
    value = DeformationCapabilityEnvelopeIR(
        "skeleton-hash",
        (JointCapabilityRangeIR("j0", -10.0, 10.0),),
        tuple(f"cam{i}" for i in range(8)),
        (),
        "",
        "probe-plan-hash",
        "",
    )
    value = replace(value, envelope_lineage_hash=deformation_envelope_lineage_hash(value))
    with pytest.raises(QualificationError, match="DEFORMATION_ENVELOPE_AUTHORITY_MISSING"):
        validate_deformation_capability_envelope(value, known_joint_ids={"j0"})


def test_presentation_graph_rejects_slot_bone_and_camera_drift_when_exact_authorities_are_supplied():
    surface = _surface()
    partition = _partition(surface)
    carrier_policy = _carrier_policy(partition, c0="MESH", c1="MESH")
    skeleton = type("Skel", (), {
        "skeleton_lineage_hash": "skeleton-hash",
        "joints": (type("J", (), {"canonical_joint_id": "root"})(),),
    })()
    envelope = type("Envelope", (), {
        "camera_binding_hashes": tuple(f"cam{i}" for i in range(8)),
    })()
    mesh = type("Mesh", (), {"mesh_lineage_hash": "mesh-hash"})()
    state = type("State", (), {
        "product_state_hash": "product-state-hash",
        "skeleton_lineage_hash": "skeleton-hash",
        "mesh_lineage_hash": "mesh-hash",
        "partition_lineage_hash": partition.partition_lineage_hash,
        "carrier_policy_lineage_hash": carrier_policy.carrier_policy_lineage_hash,
    })()
    slots = (
        PresentationSlotIR("slot0", "root", 0, "a0", ("ATTACHMENT","ORDER")),
        PresentationSlotIR("slot1", "root", 1, "a1", ("ATTACHMENT","ORDER")),
    )
    attachments = (
        PresentationAttachmentIR("a0","slot0",("c0",),"DEFORMABLE","MESH","mesh-hash"),
        PresentationAttachmentIR("a1","slot1",("c1",),"RIGID","MESH","mesh-hash"),
    )
    overlays = tuple(PresentationViewOverlayIR(i,f"cam{i}",f"app{i}",f"comp{i}") for i in range(8))
    decisions = (
        PresentationDecisionEvidenceIR("d0","SLOT_BINDING","MECHANICAL",("partition-evidence",)),
    )
    graph = QualifiedPresentationGraphIR(
        slots,attachments,overlays,decisions,
        "skeleton-hash","mesh-hash",partition.partition_lineage_hash,
        carrier_policy.carrier_policy_lineage_hash,"product-state-hash",
        {"status":"PASS"},"",
    )
    graph = replace(graph,presentation_lineage_hash=qualified_presentation_lineage_hash(graph))
    validate_qualified_presentation_graph(
        graph,carrier_policy=carrier_policy,puppet_state=state,skeleton=skeleton,
        partition=partition,envelope=envelope,mesh=mesh,
    )

    bad_slots=(replace(slots[0],bone_id="missing"),slots[1])
    bad=replace(graph,slots=bad_slots,presentation_lineage_hash="")
    bad=replace(bad,presentation_lineage_hash=qualified_presentation_lineage_hash(bad))
    with pytest.raises(QualificationError,match="UNKNOWN_BONE"):
        validate_qualified_presentation_graph(
            bad,carrier_policy=carrier_policy,puppet_state=state,skeleton=skeleton,
            partition=partition,envelope=envelope,mesh=mesh,
        )

    bad_overlays=(replace(overlays[0],camera_binding_hash="other"),*overlays[1:])
    bad=replace(graph,view_overlays=bad_overlays,presentation_lineage_hash="")
    bad=replace(bad,presentation_lineage_hash=qualified_presentation_lineage_hash(bad))
    with pytest.raises(QualificationError,match="CAMERA_SET_MISMATCH"):
        validate_qualified_presentation_graph(
            bad,carrier_policy=carrier_policy,puppet_state=state,skeleton=skeleton,
            partition=partition,envelope=envelope,mesh=mesh,
        )
