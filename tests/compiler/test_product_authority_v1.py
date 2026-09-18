from dataclasses import replace

import pytest

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
    QualifiedPresentationGraphIR,
    build_mechanical_partition,
    deformation_envelope_lineage_hash,
    qualified_mesh_lineage_hash,
    qualified_presentation_lineage_hash,
    validate_deformation_capability_envelope,
    validate_qualified_mesh,
    validate_qualified_presentation_graph,
)
from compiler.realsas_compiler_core.types import (
    QualificationError, RiggingSurfaceIR, SurfaceNode,
)


def _surface():
    nodes = tuple(
        SurfaceNode(f"s{i}", (float(i), 0.0, 0.0), (0,), ("src",), (f"o{i}",))
        for i in range(4)
    )
    return RiggingSurfaceIR(nodes, (), "surface-hash")


def _partition(surface):
    return build_mechanical_partition(
        surface=surface,
        components=(
            ComponentRegionIR("c0", ("s0", "s1")),
            ComponentRegionIR("c1", ("s2", "s3")),
        ),
        boundary_constraints=(
            ComponentBoundaryConstraintIR("b0", "s1", "s2", "SEPARATE", ("mechanical",)),
            ComponentBoundaryConstraintIR("b1", "s0", "s1", "PRESERVE_CONTINUITY", ("mechanical",)),
            ComponentBoundaryConstraintIR("b2", "s2", "s3", "UNKNOWN", ("ambiguous",)),
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


def test_qualified_mesh_requires_five_independent_gates_and_zero_consequential_unknown():
    surface = _surface()
    partition = _partition(surface)
    envelope = _envelope()
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
    }
    mesh = QualifiedMeshIR(
        (), (), (),
        surface.geometry_lineage_hash,
        partition.partition_lineage_hash,
        envelope.envelope_lineage_hash,
        "policy-hash",
        report,
        "",
    )
    mesh = replace(mesh, mesh_lineage_hash=qualified_mesh_lineage_hash(mesh))
    validate_qualified_mesh(mesh, surface=surface, partition=partition, envelope=envelope)
    bad = replace(mesh, qualification_report={**report, "consequential_unknown_boundary_count": 1}, mesh_lineage_hash="")
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
