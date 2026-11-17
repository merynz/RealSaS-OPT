from __future__ import annotations

"""SAM3-free garment court on the canonical mesh and actual XPBD implementation.

The authored tiny source is a regression fixture, NOT evidence of learned
garment segmentation, source material physics or real Knight motion quality.
"""
from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    deform_mesh_lbs_v1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    step_declared_xpbd_surface_v1,
)
from compiler.realsas_compiler_core.mechanics_semantic_garment_v1 import (
    ROLE_SCHEMA, GarmentDynamicsRequestV1, SemanticGarmentEvidenceV1,
    prepare_semantic_garment_xpbd_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_v1 import _parents, F0, F1


def _inputs():
    mesh, rig, skin, mechanics = _parents()
    evidence = SemanticGarmentEvidenceV1(
        mesh_lineage_hash=mesh.mesh_lineage_hash,
        scene_sha256="a" * 64,
        semantic_role_schema_id=ROLE_SCHEMA,
        role_probabilities_by_face=(
            (0.01, 0.01, 0.02, 0.96),
            (0.01, 0.01, 0.02, 0.96),
        ),
    )
    policy = GarmentDynamicsRequestV1(
        material_profile="LIGHT_DRAPE",
        anchor_joint_ids=("J0",),
        scene_units_per_meter=10.0,
        explicitly_approved_flexible_surface=True,
    )
    return (mesh, rig, skin, mechanics), evidence, policy


def test_same_canonical_faces_stable_anchors_and_actual_xpbd_steps():
    parents, evidence, request = _inputs()
    proposal = prepare_semantic_garment_xpbd_v1(*parents, evidence, request)
    assert proposal.domain.selected_faces == (F0, F1)
    assert len(proposal.domain.pinned_vertex_ids) == 2
    assert len(proposal.domain.dynamic_vertex_ids) == 2
    assert proposal.domain.mesh_lineage_hash == parents[0].mesh_lineage_hash
    assert proposal.domain.skin_lineage_hash == parents[2].skin_lineage_hash
    assert proposal.audit["sam3_used"] is False
    assert proposal.audit["activation_automatic"] is False
    assert proposal.audit["source_physics_truth_claimed"] is False
    assert proposal.audit["source_geometry_role_is_unqualified_neural_candidate"]
    before = deform_mesh_lbs_v1(
        parents[0], parents[1], parents[2], solve_control_pose_v1(parents[1], ())
    )
    init = step_declared_xpbd_surface_v1(
        parents[0], before, proposal.domain, proposal.policy
    )
    assert init.status == "INITIALIZED_XPBD_SURFACE_UNSIMULATED"
    moved = step_declared_xpbd_surface_v1(
        parents[0], before, proposal.domain, proposal.policy,
        previous_state=init.state, dt_seconds=1.0 / 60.0,
    )
    assert moved.status == "PASS_XPBD_SURFACE_RUNTIME"
    assert moved.state.ticks == 2
    assert moved.deformation.canonical_mesh_vertex_ids == before.canonical_mesh_vertex_ids
    origin = dict(zip(before.canonical_mesh_vertex_ids, before.positions))
    changed = dict(zip(moved.deformation.canonical_mesh_vertex_ids, moved.deformation.positions))
    for pin in proposal.domain.pinned_vertex_ids:
        assert np.allclose(changed[pin], origin[pin], rtol=0, atol=1e-9)
    assert any(np.linalg.norm(np.asarray(changed[v]) - np.asarray(origin[v])) > 1e-6
               for v in proposal.domain.dynamic_vertex_ids)


def test_semantic_covering_does_not_automatically_mean_flexible_cloth():
    parents, evidence, request = _inputs()
    with pytest.raises(QualificationError, match="EXPLICIT_FLEXIBLE_GARMENT_POLICY"):
        prepare_semantic_garment_xpbd_v1(
            *parents, evidence,
            replace(request, explicitly_approved_flexible_surface=False),
        )
    with pytest.raises(QualificationError, match="EXPLICIT_FLEXIBLE_GARMENT_POLICY"):
        prepare_semantic_garment_xpbd_v1(
            *parents, evidence, replace(request, material_profile="ARMOR"),
        )


def test_neural_role_uncertainty_is_abstention_not_negative_physics_truth():
    parents, evidence, request = _inputs()
    unknown = replace(evidence, role_probabilities_by_face=((0.25,) * 4,) * 2)
    with pytest.raises(QualificationError, match="NO_CONFIDENT_GARMENT_PATCH"):
        prepare_semantic_garment_xpbd_v1(*parents, unknown, request)


def test_all_3d_mesh_rig_skin_and_scene_addresses_must_match():
    parents, evidence, request = _inputs()
    bad = replace(evidence, mesh_lineage_hash="f" * 64)
    with pytest.raises(QualificationError, match="UNQUALIFIED_SEMANTIC_MESH_BINDING"):
        prepare_semantic_garment_xpbd_v1(*parents, bad, request)
    bad = replace(evidence, scene_sha256="")
    with pytest.raises(QualificationError, match="UNQUALIFIED_SEMANTIC_MESH_BINDING"):
        prepare_semantic_garment_xpbd_v1(*parents, bad, request)


def test_no_guessing_new_material_or_missing_pins_and_full_surface_must_stay_valid():
    parents, evidence, request = _inputs()
    with pytest.raises(QualificationError, match="MISSING_CANONICAL_SKIN_OR_JOINT"):
        prepare_semantic_garment_xpbd_v1(
            *parents, evidence, replace(request, anchor_joint_ids=("NO_SUCH_JOINT",)),
        )
    with pytest.raises(QualificationError, match="OVERPINNED_OR_RIGID_GARMENT"):
        prepare_semantic_garment_xpbd_v1(
            *parents, evidence, replace(request, pins_per_joint=4),
        )
