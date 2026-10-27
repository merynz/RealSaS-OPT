from __future__ import annotations

import pytest

from compiler.realsas_compiler_core.mechanics_rigid_candidate_discovery_v1 import (
    CanonicalSurfaceAddressCandidateV1,
    MechanicsRigidCandidateRowV1,
)
from compiler.realsas_compiler_core.mechanics_rigid_candidate_evidence_v1 import (
    authorize_discovered_rigid_candidate_v1,
)
from compiler.realsas_compiler_core.types import QualificationError


def _candidate(status: str = "CANDIDATE_REQUIRES_CAUSAL_AUTHORIZATION") -> MechanicsRigidCandidateRowV1:
    return MechanicsRigidCandidateRowV1(
        component_id="component-a",
        status=status,
        canonical_vertex_ids=("mv-3", "mv-4", "mv-5"),
        rigid_owner_joint_id="joint-child",
        holder_ancestor_joint_ids=("joint-root",),
        held=CanonicalSurfaceAddressCandidateV1(face_index=7, barycentric=(0.2, 0.3, 0.5)),
        holder=CanonicalSurfaceAddressCandidateV1(face_index=2, barycentric=(0.6, 0.1, 0.3)),
        setup_surface_gap=0.125,
        candidate_hash="a" * 64,
        metadata={
            "candidate_only": True,
            "attachment_authority_claimed": False,
            "constraint_authority_claimed": False,
        },
    )


def test_discovered_candidate_authorizes_only_on_heldout_rigid_evidence():
    result = authorize_discovered_rigid_candidate_v1(
        candidate=_candidate(),
        evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
        heldout_confirmed=True,
        evidence_face_a=7,
        evidence_face_b=2,
    )
    assert result.decision == "AUTHORIZED"
    assert result.binding_held_face == 7
    assert result.binding_holder_face == 2


def test_discovered_candidate_rejects_on_confirmed_nonrigid_evidence():
    result = authorize_discovered_rigid_candidate_v1(
        candidate=_candidate(),
        evidence_status="NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
        heldout_confirmed=True,
        evidence_face_a=7,
        evidence_face_b=2,
    )
    assert result.decision == "REJECTED"


def test_unconfirmed_or_abstaining_evidence_cannot_authorize():
    unconfirmed = authorize_discovered_rigid_candidate_v1(
        candidate=_candidate(),
        evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
        heldout_confirmed=False,
        evidence_face_a=7,
        evidence_face_b=2,
    )
    abstained = authorize_discovered_rigid_candidate_v1(
        candidate=_candidate(),
        evidence_status="ABSTAIN",
        heldout_confirmed=True,
        evidence_face_a=7,
        evidence_face_b=2,
    )
    assert unconfirmed.decision == "ABSTAIN"
    assert abstained.decision == "ABSTAIN"


def test_directed_surface_address_mismatch_fails_closed():
    with pytest.raises(QualificationError, match="EVIDENCE_ADDRESS_MISMATCH"):
        authorize_discovered_rigid_candidate_v1(
            candidate=_candidate(),
            evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
            heldout_confirmed=True,
            evidence_face_a=2,
            evidence_face_b=7,
        )


def test_non_candidate_classification_cannot_enter_authorization_gate():
    with pytest.raises(QualificationError, match="NOT_AUTHORIZABLE"):
        authorize_discovered_rigid_candidate_v1(
            candidate=_candidate("DEFORMABLE_OR_MULTICONTROL"),
            evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
            heldout_confirmed=True,
            evidence_face_a=7,
            evidence_face_b=2,
        )
