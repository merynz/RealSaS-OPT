from __future__ import annotations

"""Actual SAM 3 V4 contract, original mesh binding, pin and XPBD courts.

Uses a tiny authored source fixture; this is NOT a validation of Knight
source-to-canonical face reindex or empirical physical material fitting.
"""

from dataclasses import replace
from copy import deepcopy

import pytest

from compiler.realsas_compiler_core.mechanics_sam3_xpbd_bridge_v1 import (
    MATERIAL_PROFILES_V1,
    SAM3CanonicalFaceBindingV1,
    SAM3XPBDAuthoringRequestV1,
    prepare_sam3_assisted_xpbd_surface_v1,
    _pin_vertices,
)
from compiler.realsas_compiler_core.mechanics_xpbd_surface_v1 import (
    requalify_declared_xpbd_surface_domain_v1,
    step_declared_xpbd_surface_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import CanonicalAttachmentAnchorIRV2
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    deform_mesh_lbs_v1, solve_control_pose_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.compiler.test_mechanics_xpbd_surface_v1 import (
    _parents, F0, F1,
)


SOURCE_SHA = "a" * 64


def _report():
    return {
        "schema": "BUSTED.KnightSAM3.MultiviewFaceEvidence.v4",
        "status": "REAL_GPU_SAM3_8VIEW_FACE_FUSION_DONE",
        "source_mesh_sha256": SOURCE_SHA,
        "camera_source": "FROZEN_8_OBSERVED_KNIGHT_VIEWS_NOT_MODEL_CONSTRUCTION",
        "source_camera_count": 8,
        "compiler_or_training_mutation": False,
        "per_concept": {
            "cape": {
                "only_original_front_back": {
                    "independent_views_with_masks": [0, 4],
                    "confirmed_2view_faces": [0, 1],
                    "contradicted_face_ids": [],
                },
                "all_eight_views": {
                    "independent_views_with_masks": [0, 4],
                    "confirmed_2view_faces": [0, 1],
                    "contradicted_face_ids": [],
                },
                "per_view": {
                    "0": {"status": "MASK_SELECTED", "positive_faces": [0, 1]},
                    "4": {"status": "MASK_SELECTED", "positive_faces": [0, 1]},
                },
            },
        },
    }


def _setup(*, scope="ORIGINAL_FRONT_BACK", profile="LIGHT_DRAPE"):
    mesh, rig, skin, mechanics = _parents()
    binding = SAM3CanonicalFaceBindingV1(
        source_mesh_sha256=SOURCE_SHA,
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
        indexed_faces=((0, F0), (1, F1)),
    )
    request = SAM3XPBDAuthoringRequestV1(
        concept="cape", observation_scope=scope,
        anchor_joint_ids=("J0",),
        material_profile=profile,
        scene_units_per_meter=1.0,
        user_approved_surface_dynamics=True,
    )
    return (mesh, rig, skin, mechanics), binding, request


def _prepare(parents, binding, request, report=None):
    return prepare_sam3_assisted_xpbd_surface_v1(
        *parents, report=report or _report(), binding=binding,
        request=request,
    )


def test_source_observed_mask_to_existing_canonical_xpbd_domain_and_policy():
    parents, binding, request = _setup()
    value = _prepare(parents, binding, request)
    assert value.domain.selected_faces == (F0, F1)
    assert len(value.domain.pinned_vertex_ids) == 2
    assert len(value.domain.dynamic_vertex_ids) == 2
    assert value.audit["safe_source_face_count"] == 2
    assert value.audit["source_material_parameters_qualified"] is False
    assert value.audit["source_face_reindex_verified_independently_by_bridge"] is False
    assert value.audit["xpbd_activated"] is False
    assert value.policy.domain_hash == value.domain.domain_hash
    assert value.policy.fixed_step_seconds == pytest.approx(1. / 120.)
    requalify_declared_xpbd_surface_domain_v1(*parents, value.domain)
    mesh, rig, skin, _ = parents
    primary = deform_mesh_lbs_v1(
        mesh, rig, skin, solve_control_pose_v1(rig, ()),
    )
    initial = step_declared_xpbd_surface_v1(
        mesh, primary, value.domain, value.policy,
    )
    assert initial.status == "INITIALIZED_XPBD_SURFACE_UNSIMULATED"


def test_source_2view_is_distinct_from_generated_6view_hypotheses():
    parents, binding, request = _setup(scope="ALL_EIGHT_OBSERVED")
    out = _prepare(parents, binding, request)
    assert out.audit["source_observation_scope"] == "ALL_EIGHT_OBSERVED"
    report = _report()
    report["camera_source"] = "PRODUCT_GENERATED_SIX_VIEW"
    with pytest.raises(QualificationError, match="UNQUALIFIED_OBSERVED_SOURCE_REPORT"):
        _prepare(parents, binding, request, report)
    report = _report()
    report["per_concept"]["cape"]["per_view"]["4"]["positive_faces"] = [0]
    with pytest.raises(QualificationError, match="FACE_VOTE_SUMMARY_DRIFT"):
        _prepare(parents, binding, request, report)


def test_no_mask_only_activation_and_no_foreign_mesh_reindex():
    parents, binding, request = _setup()
    with pytest.raises(QualificationError, match="USER_APPROVAL_REQUIRED"):
        _prepare(parents, binding, replace(request, user_approved_surface_dynamics=False))
    with pytest.raises(QualificationError, match="SOURCE_TO_CANONICAL_BINDING_REQUIRED"):
        _prepare(parents, replace(binding, source_mesh_sha256="b" * 64), request)
    with pytest.raises(QualificationError, match="REINDEX_MISSING_CONFIRMED_FACES"):
        _prepare(parents, replace(binding, indexed_faces=((0, F0),)), request)
    with pytest.raises(QualificationError, match="REINDEX_DUPLICATE_OR_FOREIGN_FACE"):
        _prepare(parents, replace(binding, indexed_faces=((0, F0), (1, F0))), request)


def test_contradictions_never_become_physics_surface_ground_truth():
    parents, binding, request = _setup()
    report = _report()
    report["per_concept"]["cape"]["all_eight_views"]["contradicted_face_ids"] = [1]
    with pytest.raises(QualificationError, match="INSUFFICIENT_CONNECTED_FACES"):
        _prepare(parents, binding, replace(request, observation_scope="ALL_EIGHT_OBSERVED"), report)
    # If no face remains, abstain before any pin/XPBD candidate is created.
    report["per_concept"]["cape"]["all_eight_views"]["contradicted_face_ids"] = [0, 1]
    with pytest.raises(QualificationError, match="NO_UNCONTRADICTED_SURFACE"):
        _prepare(parents, binding, replace(request, observation_scope="ALL_EIGHT_OBSERVED"), report)


def test_material_is_explicit_user_policy_and_not_semantic_identity():
    parents, binding, request = _setup()
    light = _prepare(parents, binding, request)
    heavy = _prepare(parents, binding, replace(request, material_profile="HEAVY_DRAPE"))
    stiff = _prepare(parents, binding, replace(request, material_profile="STRUCTURED_FABRIC"))
    assert light.domain == heavy.domain == stiff.domain
    assert light.policy.bend_compliance > heavy.policy.bend_compliance > stiff.policy.bend_compliance
    assert light.policy.damping_per_second < heavy.policy.damping_per_second
    assert set(MATERIAL_PROFILES_V1) == {"LIGHT_DRAPE", "HEAVY_DRAPE", "STRUCTURED_FABRIC"}
    with pytest.raises(QualificationError, match="EXPLICIT_MATERIAL_PROFILE_REQUIRED"):
        _prepare(parents, binding, replace(request, material_profile="CAPE"))
    with pytest.raises(QualificationError, match="SCENE_UNIT_CALIBRATION_REQUIRED"):
        _prepare(parents, binding, replace(request, scene_units_per_meter=0))


def test_no_hidden_full_boundary_pinning_or_unqualified_joint_attachment():
    parents, _, request = _setup()
    mesh, rig, skin, _ = parents
    # One face is a cut through the canonical body; two of three vertices
    # would have to be fixed by the current XPBD cut-seam contract.
    with pytest.raises(QualificationError, match="OVERCONSTRAINED_CUT_BOUNDARY"):
        _pin_vertices(mesh, rig, skin, (F0,), request)
    with pytest.raises(QualificationError, match="NO_SKIN_SUPPORTED_ATTACHMENT_PIN"):
        _pin_vertices(
            mesh, rig, skin, (F0, F1),
            replace(request, anchor_joint_ids=("J1",)),
        )
    with pytest.raises(QualificationError, match="UNKNOWN_ANCHOR_JOINT"):
        _pin_vertices(
            mesh, rig, skin, (F0, F1),
            replace(request, anchor_joint_ids=("NOT_A_REAL_JOINT",)),
        )


def test_same_source_exact_same_proposal_without_hidden_mutation():
    parents, binding, request = _setup()
    r = _report()
    before = deepcopy(r)
    one = _prepare(parents, binding, request, r)
    two = _prepare(parents, binding, request, r)
    assert r == before
    assert one == two
    assert one.audit["sam3_report_sha256"] == two.audit["sam3_report_sha256"]


def test_existing_rigprogram_socket_locates_pins_better_than_joint_head():
    parents, _, request = _setup()
    mesh, rig, skin, _ = parents
    root_pins, root_audit = _pin_vertices(mesh, rig, skin, (F0, F1), request)
    assert root_audit["pin_anchor_position_authority"] == "EXPLICIT_JOINT_REST_POSITION"

    # This extra anchor is an editor-authored RigProgram control location, not
    # an invented source-native shoulder attachment. It is geometrically
    # distinct from the J0 bone head, even though its parent joint is J0.
    anchor = CanonicalAttachmentAnchorIRV2(
        canonical_anchor_id="A_SHOULDER",
        parent_canonical_joint_id="J0",
        parent_canonical_control_id=None,
        local_P=(0.0, 1.0, 0.0),
        source_proposal_anchor_id="P:A_SHOULDER",
    )
    rig_with_anchor = replace(rig, attachment_anchors=(anchor,))
    anchored, audit = _pin_vertices(
        mesh, rig_with_anchor, skin, (F0, F1),
        replace(request, attachment_anchor_ids=("A_SHOULDER",)),
    )
    assert "MV:0" in anchored
    assert anchored != root_pins
    assert audit["pin_anchor_position_authority"] == "EXPLICIT_RIGPROGRAM_ATTACHMENT_ANCHOR"

    with pytest.raises(QualificationError, match="ATTACHMENT_ANCHOR_NOT_IN_RIGPROGRAM"):
        _pin_vertices(
            mesh, rig, skin, (F0, F1),
            replace(request, attachment_anchor_ids=("A_SHOULDER",)),
        )
