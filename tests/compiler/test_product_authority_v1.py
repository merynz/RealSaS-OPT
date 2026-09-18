from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentBoundaryConstraintIR,
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
    build_mechanical_partition,
    deformation_envelope_lineage_hash,
    qualified_mesh_intrinsic_audit,
    qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
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


def _envelope():
    value = DeformationCapabilityEnvelopeIR(
        "skeleton-hash",
        (JointCapabilityRangeIR("j0", -45.0, 45.0),),
        tuple(f"cam{i}" for i in range(8)),
        (),
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


def _valid_mesh(surface, partition, envelope):
    vertices = tuple(
        _identity_vertex(f"v{i}", f"s{i}", "c0" if i < 3 else "c1", surface.surface_nodes[i].P)
        for i in range(6)
    )
    faces = (("v0", "v1", "v2"), ("v3", "v4", "v5"))
    edges = (
        ("v0", "v1"), ("v1", "v2"), ("v0", "v2"),
        ("v3", "v4"), ("v4", "v5"), ("v3", "v5"),
    )
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
        "carrier_policy_hash": "carrier-policy-hash",
        "view_component_coverage": tuple(
            {
                "view_index": view,
                "component_id": component_id,
                "carrier_class": "MESH",
                "recall": 1.0,
                "precision": 1.0,
                "largest_coherent_hole_fraction": 0.0,
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
        envelope.envelope_lineage_hash,
        "policy-hash",
        base_report,
        "",
    )
    audit = qualified_mesh_intrinsic_audit(mesh, surface=surface, partition=partition)
    report = {**base_report, "intrinsic_audit_hash": content_sha256(audit)}
    mesh = replace(mesh, qualification_report=report)
    return replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))


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


def test_qualified_mesh_recomputes_intrinsics_and_requires_external_g3_g5_evidence():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    mesh = _valid_mesh(surface, partition, envelope)
    validate_qualified_mesh(mesh, surface=surface, partition=partition, envelope=envelope)


def test_empty_mesh_cannot_pass_by_claiming_all_five_gates():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    good = _valid_mesh(surface, partition, envelope)
    bad = replace(good, vertices=(), faces=(), edges=(), mesh_lineage_hash="")
    with pytest.raises(QualificationError, match="EMPTY_PRODUCT_GEOMETRY"):
        validate_qualified_mesh(bad, surface=surface, partition=partition, envelope=envelope)


def test_face_cannot_cross_mechanical_component_boundary():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    good = _valid_mesh(surface, partition, envelope)
    bad = replace(good, faces=(("v0", "v1", "v3"), ("v3", "v4", "v5")), mesh_lineage_hash="")
    with pytest.raises(QualificationError, match="FACE_CROSSES_COMPONENT_BOUNDARY"):
        validate_qualified_mesh(bad, surface=surface, partition=partition, envelope=envelope)


def test_preserve_continuity_requires_both_boundary_supports_to_exist_in_product_mesh():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    good = _valid_mesh(surface, partition, envelope)
    vertices = tuple(v for v in good.vertices if v.canonical_mesh_vertex_id != "v1")
    bad = replace(
        good,
        vertices=vertices,
        faces=(("v0", "v2", "v2"), ("v3", "v4", "v5")),
        edges=(),
        mesh_lineage_hash="",
    )
    with pytest.raises(QualificationError):
        validate_qualified_mesh(bad, surface=surface, partition=partition, envelope=envelope)


def test_g3_numerical_floor_rejects_skinny_rest_triangle_even_with_pass_report():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    good = _valid_mesh(surface, partition, envelope)
    skinny = replace(good.vertices[2], P=(0.5, 0.01, 0.0))
    bad = replace(good, vertices=(good.vertices[0], good.vertices[1], skinny, *good.vertices[3:]), mesh_lineage_hash="")
    with pytest.raises(QualificationError):
        validate_qualified_mesh(bad, surface=surface, partition=partition, envelope=envelope)


def test_qualified_mesh_requires_zero_consequential_unknown():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
    mesh = _valid_mesh(surface, partition, envelope)
    report = {**mesh.qualification_report, "consequential_unknown_boundary_count": 1}
    bad = replace(mesh, qualification_report=report, mesh_lineage_hash="")
    bad = replace(bad, mesh_lineage_hash=qualified_mesh_lineage_hash(bad))
    with pytest.raises(QualificationError, match="CONSEQUENTIAL_UNKNOWN"):
        validate_qualified_mesh(bad, surface=surface, partition=partition, envelope=envelope)


def test_presentation_keeps_mechanical_class_and_carrier_class_orthogonal():
    slots = (PresentationSlotIR("slot0", "j0", 0, "a0", ("ATTACHMENT", "ORDER")),)
    attachments = (PresentationAttachmentIR("a0", "slot0", ("c0",), "RIGID", "PLANAR", "plane-hash"),)
    overlays = tuple(PresentationViewOverlayIR(i, f"cam{i}", f"app{i}", f"comp{i}") for i in range(8))
    decisions = (
        PresentationDecisionEvidenceIR("d0", "SLOT_BINDING", "MECHANICAL", ("partition-hash",)),
        PresentationDecisionEvidenceIR("d1", "SETUP_COMPOSITION", "OBSERVATION_REST", ("obs-hash",)),
    )
    graph = QualifiedPresentationGraphIR(
        slots, attachments, overlays, decisions,
        "skeleton-hash", "mesh-hash", "partition-hash",
        {"status": "PASS"}, "",
    )
    graph = replace(graph, presentation_lineage_hash=qualified_presentation_lineage_hash(graph))
    validate_qualified_presentation_graph(graph)
    assert graph.attachments[0].mechanical_class == "RIGID"
    assert graph.attachments[0].carrier_class == "PLANAR"
