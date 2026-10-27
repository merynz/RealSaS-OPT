from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR,
    CanonicalMeshVertexIR,
    canonical_mesh_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v1 import (
    CanonicalRigIR,
    CanonicalRigJointIR,
    canonical_rig_lineage_hash_v1,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v1 import (
    CanonicalSkinIR,
    CanonicalSkinRowIR,
    canonical_skin_lineage_hash_v1,
)
from compiler.realsas_compiler_core.mechanics_rigid_candidate_discovery_v1 import (
    MechanicsRigidCandidatePolicyV1,
    discover_rigid_surface_candidates_v1,
    mechanics_rigid_candidate_discovery_hash_v1,
)
from compiler.realsas_compiler_core.types import QualificationError, SurfaceSupportBinding


def _support(name: str) -> SurfaceSupportBinding:
    return SurfaceSupportBinding(mode="EXACT", coefficients=((name, 1.0),))


def _fixture(
    *,
    held_influences: tuple[tuple[str, float], ...] = (("joint-child", 1.0),),
    holder_influences: tuple[tuple[str, float], ...] = (("joint-root", 1.0),),
    component_ids: tuple[str, ...] = ("body", "body", "body", "piece", "piece", "piece"),
):
    positions = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.1),
        (1.0, 0.0, 0.1),
        (0.0, 1.0, 0.1),
    )
    vertex_ids = tuple(f"mv-{index}" for index in range(6))
    mesh = CanonicalMeshIR(
        vertices=tuple(
            CanonicalMeshVertexIR(
                canonical_mesh_vertex_id=vertex_ids[index],
                support_binding=_support(f"source-{index}"),
                component_id=component_ids[index],
                P=positions[index],
                source_candidate_vertex_id=f"source-{index}",
            )
            for index in range(6)
        ),
        faces=((vertex_ids[0], vertex_ids[1], vertex_ids[2]), (vertex_ids[3], vertex_ids[4], vertex_ids[5])),
        edges=(
            (vertex_ids[0], vertex_ids[1]),
            (vertex_ids[1], vertex_ids[2]),
            (vertex_ids[2], vertex_ids[0]),
            (vertex_ids[3], vertex_ids[4]),
            (vertex_ids[4], vertex_ids[5]),
            (vertex_ids[5], vertex_ids[3]),
        ),
        source_evidence_binding_hash="surface-binding",
        partition_binding_hash="partition-binding",
        carrier_policy_binding_hash="carrier-policy",
        static_qualification_hash="static-qualification",
        carrier_evidence_hash="carrier-evidence",
        carrier_topology_hash="carrier-topology",
        carrier_geometry_hash="carrier-geometry",
        proposal_evidence_refs=(),
        source_candidate_lineage_hash="candidate-lineage",
        mesh_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_MESH"},
    )
    mesh = replace(mesh, mesh_lineage_hash=canonical_mesh_lineage_hash_v1(mesh))

    rig = CanonicalRigIR(
        joints=(
            CanonicalRigJointIR(
                canonical_joint_id="joint-root",
                P=(0.0, 0.0, 0.0),
                parent_canonical_joint_id=None,
                source_proposal_id="proposal-root",
            ),
            CanonicalRigJointIR(
                canonical_joint_id="joint-child",
                P=(0.0, 0.0, 0.1),
                parent_canonical_joint_id="joint-root",
                source_proposal_id="proposal-child",
            ),
        ),
        root_joint_ids=("joint-root",),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        feature_evidence_binding_hash="surface-binding",
        proposal_evidence_hash="rig-proposal",
        structure_qualification_hash="rig-qualification",
        rig_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_RIG"},
    )
    rig = replace(rig, rig_lineage_hash=canonical_rig_lineage_hash_v1(rig))

    skin_rows = []
    for index, vertex_id in enumerate(vertex_ids):
        influences = holder_influences if index < 3 else held_influences
        skin_rows.append(
            CanonicalSkinRowIR(
                canonical_mesh_vertex_id=vertex_id,
                influences=influences,
                source_carrier_vertex_id=f"source-{index}",
            )
        )
    skin = CanonicalSkinIR(
        rows=tuple(skin_rows),
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        rig_lineage_hash=rig.rig_lineage_hash,
        mesh_carrier_evidence_hash=mesh.carrier_evidence_hash,
        mesh_carrier_topology_hash=mesh.carrier_topology_hash,
        mesh_carrier_geometry_hash=mesh.carrier_geometry_hash,
        structure_qualification_hash=rig.structure_qualification_hash,
        carrier_skin_qualification_hash="skin-qualification",
        proposal_evidence_refs=(),
        skin_lineage_hash="",
        metadata={"authority": "COMPILER_CANONICAL_SKIN"},
    )
    skin = replace(skin, skin_lineage_hash=canonical_skin_lineage_hash_v1(skin))
    return mesh, rig, skin


def _rows_by_component(result):
    return {row.component_id: row for row in result.rows}


def test_rigid_child_component_mints_exact_candidate_but_not_constraint_authority():
    mesh, rig, skin = _fixture()
    result = discover_rigid_surface_candidates_v1(mesh, rig, skin)
    rows = _rows_by_component(result)

    assert tuple(sorted(rows)) == ("body", "piece")
    assert rows["body"].status == "NO_ANCESTOR_HOLDER_DOMAIN"
    piece = rows["piece"]
    assert piece.status == "CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION"
    assert piece.rigid_owner_joint_id == "joint-child"
    assert piece.holder_ancestor_joint_ids == ("joint-root",)
    assert piece.held is not None and piece.held.face_index == 1
    assert piece.holder is not None and piece.holder.face_index == 0
    assert piece.held.barycentric == pytest.approx((1.0, 0.0, 0.0))
    assert piece.holder.barycentric == pytest.approx((1.0, 0.0, 0.0))
    assert piece.setup_surface_gap == pytest.approx(0.1)
    assert piece.metadata["attachment_authority_claimed"] is False
    assert piece.metadata["detachability_authority_claimed"] is False
    assert piece.metadata["constraint_authority_claimed"] is False
    assert result.metadata["physical_constraint_discovery_complete"] is False
    assert result.metadata["causal_authorization_required_for_candidates"] is True
    assert result.discovery_lineage_hash == mechanics_rigid_candidate_discovery_hash_v1(result)


def test_candidate_discovery_is_deterministic_and_subject_names_are_not_inputs():
    mesh, rig, skin = _fixture()
    first = discover_rigid_surface_candidates_v1(mesh, rig, skin)
    second = discover_rigid_surface_candidates_v1(mesh, rig, skin)
    assert first == second
    assert first.discovery_lineage_hash == second.discovery_lineage_hash
    assert "knight" not in repr(first).lower()
    assert "subject" not in repr(first).lower()
    assert first.metadata["categorical_or_subject_identity_used"] is False


def test_multicontrol_component_is_classified_without_attachment_claim():
    mesh, rig, skin = _fixture(
        held_influences=(("joint-child", 0.6), ("joint-root", 0.4)),
    )
    result = discover_rigid_surface_candidates_v1(mesh, rig, skin)
    piece = _rows_by_component(result)["piece"]
    assert piece.status == "DEFORMABLE_OR_MULTICONTROL"
    assert piece.held is None
    assert piece.holder is None
    assert piece.setup_surface_gap is None
    assert piece.metadata["absence_is_not_negative_physical_truth"] is True


def test_rigid_component_without_ancestor_holder_surface_abstains_from_candidate():
    mesh, rig, skin = _fixture(holder_influences=(("joint-child", 1.0),))
    result = discover_rigid_surface_candidates_v1(mesh, rig, skin)
    piece = _rows_by_component(result)["piece"]
    assert piece.status == "NO_HOLDER_SURFACE_DOMAIN"
    assert piece.rigid_owner_joint_id == "joint-child"
    assert piece.holder_ancestor_joint_ids == ("joint-root",)
    assert piece.held is None and piece.holder is None


def test_parent_lineage_drift_fails_closed():
    mesh, rig, skin = _fixture()
    drifted = replace(skin, rig_lineage_hash="not-the-rig")
    drifted = replace(drifted, skin_lineage_hash=canonical_skin_lineage_hash_v1(drifted))
    with pytest.raises(QualificationError, match="SKIN_PARENT_DRIFT"):
        discover_rigid_surface_candidates_v1(mesh, rig, drifted)


def test_face_crossing_mechanical_components_fails_closed():
    mesh, rig, skin = _fixture(
        component_ids=("body", "body", "piece", "piece", "piece", "piece"),
    )
    with pytest.raises(QualificationError, match="FACE_CROSSES_COMPONENT"):
        discover_rigid_surface_candidates_v1(mesh, rig, skin)


def test_policy_is_explicit_and_fail_closed():
    mesh, rig, skin = _fixture()
    with pytest.raises(QualificationError, match="OWNER_THRESHOLD_INVALID"):
        discover_rigid_surface_candidates_v1(
            mesh,
            rig,
            skin,
            policy=MechanicsRigidCandidatePolicyV1(minimum_rigid_owner_weight=0.0),
        )
