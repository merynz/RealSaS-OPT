from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.component_attachment import (
    ComponentEvidenceIR,
    qualify_component_evidence_v1,
    qualify_component_set_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


SURFACES={"S0","S1","S2","S3"}
JOINTS={"J_ROOT","J_HEAD","J_HAND_L","J_HAND_R","J_CHEST"}


def test_rigid_skinned_component_requires_explicit_binding_and_does_not_infer_detachability():
    q=qualify_component_evidence_v1(
        ComponentEvidenceIR(
            component_id="book",
            source_component_refs=("src:book",),
            observation_views=(0,1,2,3,4,5,6,7),
            surface_ids=("S1",),
            proposed_mechanical_class="RIGID_SKINNED_COMPONENT",
            proposed_parent_joint_id="J_HAND_L",
            detachability_class="DETACHABILITY_UNKNOWN",
            bind_state_authority_hash="BIND",
            source_geometry_hash="GEO",
            skin_or_deformer_lineage_hash="SKIN",
        ),
        known_surface_ids=SURFACES,
        known_joint_ids=JOINTS,
    )
    assert q.mechanical_class=="RIGID_SKINNED_COMPONENT"
    assert q.detachability_class=="DETACHABILITY_UNKNOWN"
    assert q.parent_joint_id=="J_HAND_L"
    assert q.component_lineage_hash


def test_rigid_component_without_joint_or_socket_fails_closed():
    with pytest.raises(QualificationError):
        qualify_component_evidence_v1(
            ComponentEvidenceIR(
                component_id="wand",
                source_component_refs=("src:wand",),
                observation_views=(0,),
                surface_ids=("S2",),
                proposed_mechanical_class="RIGID_SKINNED_COMPONENT",
                bind_state_authority_hash="BIND",
                skin_or_deformer_lineage_hash="SKIN",
            ),
            known_surface_ids=SURFACES,
            known_joint_ids=JOINTS,
        )


def test_visual_only_component_cannot_masquerade_as_mechanical_surface():
    with pytest.raises(QualificationError):
        qualify_component_evidence_v1(
            ComponentEvidenceIR(
                component_id="fx",
                source_component_refs=("src:fx",),
                observation_views=(0,),
                surface_ids=("S3",),
                proposed_mechanical_class="NON_MECHANICAL_VISUAL_COMPONENT",
            ),
            known_surface_ids=SURFACES,
            known_joint_ids=JOINTS,
        )


def test_required_visible_excluded_component_needs_reason():
    with pytest.raises(QualificationError):
        qualify_component_evidence_v1(
            ComponentEvidenceIR(
                component_id="mystery",
                source_component_refs=("src:mystery",),
                observation_views=(0,1),
                surface_ids=(),
                proposed_mechanical_class="EXCLUDED_SOURCE_COMPONENT",
                required_visible_component=True,
            ),
            known_surface_ids=SURFACES,
            known_joint_ids=JOINTS,
        )


def test_component_set_accounts_for_mixed_generic_classes():
    rows=(
        ComponentEvidenceIR(
            component_id="body",
            source_component_refs=("src:body",),
            observation_views=tuple(range(8)),
            surface_ids=("S0","S3"),
            proposed_mechanical_class="DEFORMABLE_COMPONENT",
            detachability_class="FIXED_COMPONENT",
            source_geometry_hash="GEO_BODY",
            skin_or_deformer_lineage_hash="SKIN",
        ),
        ComponentEvidenceIR(
            component_id="hat",
            source_component_refs=("src:hat",),
            observation_views=tuple(range(8)),
            surface_ids=("S1",),
            proposed_mechanical_class="RIGID_SKINNED_COMPONENT",
            proposed_parent_joint_id="J_HEAD",
            detachability_class="DETACHABILITY_UNKNOWN",
            bind_state_authority_hash="BIND_HAT",
            source_geometry_hash="GEO_HAT",
            skin_or_deformer_lineage_hash="SKIN",
        ),
        ComponentEvidenceIR(
            component_id="weapon",
            source_component_refs=("src:weapon",),
            observation_views=(0,1,2,3),
            surface_ids=("S2",),
            proposed_mechanical_class="RIGID_BONE_ATTACHMENT",
            proposed_socket_id="right_hand_slot",
            detachability_class="SWAPPABLE_SLOT_COMPONENT",
            bind_state_authority_hash="BIND_WEAPON",
            source_geometry_hash="GEO_WEAPON",
        ),
    )
    q=qualify_component_set_v1(
        rows,
        known_surface_ids=SURFACES,
        known_joint_ids=JOINTS,
        surface_lineage_hash="SURFACE",
        skeleton_lineage_hash="SKELETON",
        skin_lineage_hash="SKIN",
    )
    assert q.qualification_report["status"]=="PASS"
    assert q.qualification_report["required_visible_component_accounting_fraction"]==1.0
    assert len(q.components)==3
    assert q.component_set_lineage_hash
