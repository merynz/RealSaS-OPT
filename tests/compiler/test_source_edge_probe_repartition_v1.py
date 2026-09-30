from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import (
    build_canonical_relation_candidate,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import (
    build_structural_partition,
)
from compiler.realsas_compiler_core.mechanical_repartition_v2 import (
    build_repartitioned_partition_v2,
    repartition_authorization_hash_v1,
)
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import (
    _stress_angle,
    propose_mechanical_repartition_directive_v2,
)
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,
    DeformationCapabilityEnvelopeIR,
    JointCapabilityRangeIR,
    build_component_carrier_policy,
    deformation_envelope_lineage_hash,
)
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
    # a-b is intentionally very short.  a follows root, b follows child.
    # Independent child rotation therefore creates a large measured source-edge
    # stretch ratio without any unsafe-face-local edge heuristic.
    surface = RiggingSurfaceIR(
        (
            SurfaceNode("a", (0.49, 0.0, 0.0), (0,), ("src",), ("oa",)),
            SurfaceNode("b", (0.51, 0.0, 0.0), (0,), ("src",), ("ob",)),
            SurfaceNode("c", (0.50, 1.0, 0.0), (0,), ("src",), ("oc",)),
        ),
        (
            SurfaceRelation("ab", "a", "b", "LOCAL_NEIGHBOR", 1.0),
            SurfaceRelation("bc", "b", "c", "LOCAL_NEIGHBOR", 0.9),
            SurfaceRelation("ca", "c", "a", "LOCAL_NEIGHBOR", 0.8),
        ),
        "surface-source-edge-probe",
    )
    skeleton = QualifiedSkeletonIR(
        (
            QualifiedJoint("j0", (0.0, 0.0, 0.0), None, ("a", "c"), "root"),
            QualifiedJoint("j1", (1.0, 0.0, 0.0), "j0", ("b",), "child"),
        ),
        "j0",
        {"status": "PASS"},
        "skeleton-source-edge-probe",
    )
    skin = QualifiedSkinIR(
        (
            QualifiedSkinRow("a", (("j0", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("b", (("j1", 1.0),), 0.0, 0.0),
            QualifiedSkinRow("c", (("j0", 1.0),), 0.0, 0.0),
        ),
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        {"status": "PASS"},
        "skin-source-edge-probe",
    )
    partition = build_structural_partition(surface)
    carrier = build_component_carrier_policy(
        partition=partition,
        decisions=tuple(
            ComponentCarrierDecisionIR(component.component_id, "MESH", ("test",))
            for component in partition.components
        ),
    )
    candidate = build_canonical_relation_candidate(
        surface,
        partition,
        carrier,
        producer_policy_hash="source-edge-probe-test-policy",
        explicit_face_provenance=(("a", "b", "c"),),
    )

    envelope = DeformationCapabilityEnvelopeIR(
        skeleton_lineage_hash=skeleton.skeleton_lineage_hash,
        joint_ranges=(
            JointCapabilityRangeIR("j0", -120.0, 120.0),
            JointCapabilityRangeIR("j1", -120.0, 120.0),
        ),
        camera_binding_hashes=tuple(f"cam-{i}" for i in range(8)),
        allowed_attachment_state_hashes=(),
        axis_contract_hash="axis-source-edge-probe",
        probe_plan_hash="probe-source-edge-probe",
        envelope_lineage_hash="",
    )
    envelope = replace(
        envelope,
        envelope_lineage_hash=deformation_envelope_lineage_hash(envelope),
    )
    cameras = tuple(
        SimpleNamespace(
            view_index=i,
            right=(1.0, 0.0, 0.0),
            screen_up=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, 1.0),
        )
        for i in range(8)
    )
    compatibility = {
        "schema": "RealSaS.SkinTopologyCompatibilityReport.v1",
        "report_hash": "c" * 64,
        "unsafe_face_count": 1,
        "unsafe_face_indices": [0],
    }
    return (
        surface,
        skeleton,
        skin,
        partition,
        candidate,
        envelope,
        cameras,
        compatibility,
    )


def _directive():
    (
        surface,
        skeleton,
        skin,
        partition,
        candidate,
        envelope,
        cameras,
        compatibility,
    ) = _fixture()
    directive = propose_mechanical_repartition_directive_v2(
        candidate,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        partition=partition,
        compatibility_report=compatibility,
        envelope=envelope,
        cameras=cameras,
        seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1",
    )
    return surface, skeleton, skin, partition, directive


def test_source_edge_probe_strategy_selects_measured_short_edge_not_face_local_proxy():
    surface, skeleton, skin, partition, directive = _directive()

    assert directive["seed_strategy"] == "SOURCE_EDGE_PROBE_RATIO_V1"
    assert directive["status"] == "REPARTITION_PROPOSED__AWAIT_TRUSTWORTHY_SKIN_AUTHORITY"
    assert directive["source_edge_probe_count"] == 3
    assert directive["unsafe_source_edge_count"] >= 1
    assert directive["weight_mutation"] is False
    assert directive["face_deletion_count"] == 0

    rows = {
        tuple(sorted((row["a_surface_id"], row["b_surface_id"]))): row
        for row in directive["proposed_boundary_overrides"]
    }
    assert ("a", "b") in rows
    ab = rows[("a", "b")]
    assert ab["decision"] == "SEPARATE"
    assert ab["unsafe_face_indices"] == ()
    assert ab["metadata"]["direct_probe_seed"] is True
    assert ab["metadata"]["source_edge_probe_max_ratio"] > 4.0

    local_pairs = {
        tuple(sorted((r.a_surface_id, r.b_surface_id)))
        for r in surface.local_relations
    }
    assert set(rows).issubset(local_pairs)


def test_source_edge_probe_directive_is_consumable_by_existing_stage17_mutex_closure():
    surface, skeleton, skin, parent, directive = _directive()

    auth = {
        "schema": "RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status": "PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash": directive["directive_hash"],
        "source_surface_lineage_hash": surface.geometry_lineage_hash,
        "source_partition_lineage_hash": parent.partition_lineage_hash,
        "source_skeleton_lineage_hash": skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash": skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor": False,
        "weight_reliability_closure_passed": True,
        "weight_reliability_evidence_hash": content_sha256(
            {"fixture": "source-edge-probe-reliability"}
        ),
        "authorization_hash": "",
    }
    auth["authorization_hash"] = repartition_authorization_hash_v1(auth)

    child = build_repartitioned_partition_v2(
        surface=surface,
        parent_partition=parent,
        directive=directive,
        authorization=auth,
    )

    owner = {
        sid: component.component_id
        for component in child.components
        for sid in component.surface_ids
    }
    assert owner["a"] != owner["b"]
    audit = child.metadata["partition_cut_closure"]
    assert audit["direct_seed_count"] >= 1
    assert audit["closure_added_separate_count"] >= 1
    assert audit["seed_constraint_violation_count"] == 0


def test_g3b_stress_angle_is_bound_to_sealed_envelope_not_metadata_fallback():
    envelope = SimpleNamespace(
        metadata={"skin_topology_compatibility_stress_angle_deg": 120.0},
        joint_ranges=(
            SimpleNamespace(min_rotation_deg=-10.0, max_rotation_deg=8.0),
            SimpleNamespace(min_rotation_deg=-6.0, max_rotation_deg=7.0),
        ),
    )
    assert _stress_angle(envelope) == 10.0
