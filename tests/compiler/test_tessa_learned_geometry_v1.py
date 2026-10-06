from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    GeometricRefinementIR,
    _support_position,
    build_component_carrier_policy,
)
from compiler.realsas_compiler_core.surface_addressing_v1 import (
    StaticCanonicalMeshQualificationIR,
    static_mesh_qualification_hash,
)
from compiler.realsas_compiler_core.tessa_learned_geometry_v1 import (
    TESSA_FIELD_SUPPORT_MODE_V1,
    TESSA_GEOMETRY_AUTHORITY_QUALIFIED_V1,
    TESSA_GEOMETRY_AUTHORITY_PENDING_V1,
    build_tessa_learned_geometry_admission_v1,
    build_tessa_learned_geometry_candidate_v1,
    validate_tessa_learned_geometry_admission_v1,
)
from compiler.realsas_compiler_core.tessa_surface_field_v1 import (
    build_tessa_material_support_field_v1,
)
from compiler.realsas_compiler_core.types import (
    QualificationError,
    RiggingSurfaceIR,
    SurfaceNode,
    SurfaceRelation,
)


def _surface() -> RiggingSurfaceIR:
    nodes = (
        SurfaceNode("S0", (0.0, 0.0, 0.0), (0,), ("P0",), ("O0",)),
        SurfaceNode("S1", (1.0, 0.0, 0.0), (0,), ("P1",), ("O1",)),
        SurfaceNode("S2", (0.0, 1.0, 0.0), (0,), ("P2",), ("O2",)),
        SurfaceNode("S3", (1.0, 1.0, 0.0), (0,), ("P3",), ("O3",)),
    )
    relations = (
        SurfaceRelation("R01", "S0", "S1", "LOCAL", 1.0),
        SurfaceRelation("R02", "S0", "S2", "LOCAL", 1.0),
        SurfaceRelation("R13", "S1", "S3", "LOCAL", 1.0),
        SurfaceRelation("R23", "S2", "S3", "LOCAL", 1.0),
    )
    lineage = content_sha256(
        {
            "schema": "Test.RiggingSurface.v1",
            "nodes": [row.to_dict() for row in nodes],
            "relations": [row.to_dict() for row in relations],
        }
    )
    return RiggingSurfaceIR(
        surface_nodes=nodes,
        local_relations=relations,
        geometry_lineage_hash=lineage,
        builder_id="TEST",
    )


def _fixture():
    surface = _surface()
    partition = build_structural_partition(surface)
    assert len(partition.components) == 1
    component_id = partition.components[0].component_id
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=(
            ComponentCarrierDecisionIR(
                component_id=component_id,
                carrier_class="MESH",
                evidence_refs=("TEST",),
            ),
        ),
    )

    # Deliberately not a convex lift of the z=0 GSA plane.  This is exactly the
    # authority separation the TESSA contract must preserve.
    vertices = np.asarray(
        [
            [0.10, 0.10, 0.25],
            [0.90, 0.10, 0.25],
            [0.10, 0.90, 0.25],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    vertex_ids = tuple(f"TESSA_V1:{index:08d}" for index in range(3))
    topology_hash = content_sha256(
        {"schema": "Test.TESSATopology.v1", "faces": faces.tolist()}
    )
    field = build_tessa_material_support_field_v1(
        vertices_world=vertices,
        decoded_component_indices=np.zeros(3, dtype=np.int64),
        surface=surface,
        partition=partition,
        topology_sequence_hash=topology_hash,
        proposal_vertex_ids=vertex_ids,
    )
    candidate = build_tessa_learned_geometry_candidate_v1(
        vertices_world=vertices,
        faces=faces,
        material_support_field=field,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
        model_provenance="TEST_TESSA_CHECKPOINT",
    )
    return surface, partition, carrier, field, candidate


def _static_qualification(candidate, partition, **report_overrides):
    report = {
        "status": "PASS_STATIC_CANONICAL_CARRIER",
        "static_quality_policy_passed": True,
        "actual_candidate_source_fidelity_passed": True,
        "source_fidelity_qualification_passed": True,
        "demo_geometry_lineage": False,
        "demo_only_source_fidelity_admission": False,
    }
    report.update(report_overrides)
    value = StaticCanonicalMeshQualificationIR(
        candidate_mesh_binding_hash=candidate.candidate_lineage_hash,
        surface_addressing_binding_hash="A" * 64,
        appearance_domain_binding_hash="B" * 64,
        geometry_gate_binding_hash="C" * 64,
        partition_binding_hash=partition.partition_lineage_hash,
        qualification_report=report,
        qualification_hash="",
    )
    return replace(value, qualification_hash=static_mesh_qualification_hash(value))


def test_tessa_candidate_keeps_learned_position_separate_from_field_support():
    surface, partition, carrier, field, candidate = _fixture()
    assert candidate.metadata["geometry_position_authority"] == TESSA_GEOMETRY_AUTHORITY_PENDING_V1
    assert candidate.metadata["legacy_g1_positional_support_claimed"] is False
    assert {row.support_binding.mode for row in candidate.vertices} == {TESSA_FIELD_SUPPORT_MODE_V1}
    assert all(
        row.support_binding.metadata["geometry_position_derived_from_support"] is False
        for row in candidate.vertices
    )

    # Safety catch: the legacy positional-support lift does not know this mode.
    # TESSA therefore cannot accidentally pass old G1 before Stage19-bound
    # learned-geometry admission is explicitly consumed.
    node_by_id = {row.surface_id: row for row in surface.surface_nodes}
    with pytest.raises(QualificationError, match="QUALIFIED_MESH_G1_SUPPORT_MODE_INVALID"):
        _support_position(node_by_id, candidate.vertices[0].support_binding)

    static = _static_qualification(candidate, partition)
    admission = build_tessa_learned_geometry_admission_v1(
        candidate=candidate,
        material_support_field=field,
        static_qualification=static,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
    )
    assert admission.geometry_authority == TESSA_GEOMETRY_AUTHORITY_QUALIFIED_V1
    assert admission.candidate_mesh_binding_hash == candidate.candidate_lineage_hash
    assert admission.static_qualification_binding_hash == static.qualification_hash
    assert admission.material_support_field_binding_hash == field.field_lineage_hash
    assert admission.metadata["threshold_relaxation_performed"] is False


def test_tessa_admission_rejects_stage19_candidate_hash_drift():
    surface, partition, carrier, field, candidate = _fixture()
    static = _static_qualification(candidate, partition)
    bad = replace(static, candidate_mesh_binding_hash="D" * 64, qualification_hash="")
    bad = replace(bad, qualification_hash=static_mesh_qualification_hash(bad))
    with pytest.raises(QualificationError, match="STATIC_CANDIDATE_DRIFT"):
        build_tessa_learned_geometry_admission_v1(
            candidate=candidate,
            material_support_field=field,
            static_qualification=bad,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
        )


def test_tessa_admission_rejects_demo_or_failed_source_fidelity():
    surface, partition, carrier, field, candidate = _fixture()
    demo = _static_qualification(
        candidate,
        partition,
        status="DEMO_ONLY_MEASURED_STATIC_CANONICAL_CARRIER__P999_FAIL",
        actual_candidate_source_fidelity_passed=False,
        source_fidelity_qualification_passed=False,
        demo_geometry_lineage=True,
        demo_only_source_fidelity_admission=True,
    )
    with pytest.raises(QualificationError, match="STATIC_STATUS_NOT_PASS"):
        build_tessa_learned_geometry_admission_v1(
            candidate=candidate,
            material_support_field=field,
            static_qualification=demo,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
        )


def test_tessa_admission_rejects_support_field_drift():
    surface, partition, carrier, field, candidate = _fixture()
    static = _static_qualification(candidate, partition)
    admission = build_tessa_learned_geometry_admission_v1(
        candidate=candidate,
        material_support_field=field,
        static_qualification=static,
        surface=surface,
        partition=partition,
        carrier_policy=carrier,
    )
    vertex0 = candidate.vertices[0]
    bad_support = replace(
        vertex0.support_binding,
        metadata={
            **dict(vertex0.support_binding.metadata),
            "geometry_position_derived_from_support": True,
        },
    )
    bad_vertex = replace(vertex0, support_binding=bad_support)
    bad_candidate = replace(
        candidate,
        vertices=(bad_vertex,) + candidate.vertices[1:],
        candidate_lineage_hash="",
    )
    from compiler.realsas_compiler_core.product_authority_v1 import (
        canonical_mesh_candidate_lineage_hash,
    )

    bad_candidate = replace(
        bad_candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(bad_candidate),
    )
    bad_static = _static_qualification(bad_candidate, partition)
    with pytest.raises(QualificationError, match="POSITION_SUPPORT_CLAIM_FORBIDDEN"):
        validate_tessa_learned_geometry_admission_v1(
            replace(
                admission,
                candidate_mesh_binding_hash=bad_candidate.candidate_lineage_hash,
                static_qualification_binding_hash=bad_static.qualification_hash,
                admission_hash="",
            ),
            candidate=bad_candidate,
            material_support_field=field,
            static_qualification=bad_static,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
        )


def test_tessa_admission_rejects_fake_geometric_refinement():
    surface, partition, carrier, field, candidate = _fixture()
    first = candidate.vertices[0]
    fake = GeometricRefinementIR(
        dense_lineage_hash="FAKE",
        base_position=(0.0, 0.0, 0.0),
        refined_position=first.P,
        local_scale=1.0,
        normal_component=0.0,
        tangential_component=0.0,
        method="FAKE_G1_ESCAPE",
    )
    bad_first = replace(first, refinement=fake)
    bad_candidate = replace(
        candidate,
        vertices=(bad_first,) + candidate.vertices[1:],
        candidate_lineage_hash="",
    )
    from compiler.realsas_compiler_core.product_authority_v1 import (
        canonical_mesh_candidate_lineage_hash,
    )

    bad_candidate = replace(
        bad_candidate,
        candidate_lineage_hash=canonical_mesh_candidate_lineage_hash(bad_candidate),
    )
    bad_static = _static_qualification(bad_candidate, partition)
    with pytest.raises(QualificationError, match="FAKE_REFINEMENT_FORBIDDEN"):
        build_tessa_learned_geometry_admission_v1(
            candidate=bad_candidate,
            material_support_field=field,
            static_qualification=bad_static,
            surface=surface,
            partition=partition,
            carrier_policy=carrier,
        )
