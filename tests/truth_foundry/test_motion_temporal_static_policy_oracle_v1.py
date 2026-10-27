from __future__ import annotations

import inspect

import numpy as np
import pytest

from compiler.realsas_compiler_core.motion_temporal_static_policy_gate_v1 import (
    TemporalConditionEvidenceRefV1,
    evaluate_temporal_static_policy_v1,
)
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import (
    AttachmentSurfaceBindingV1,
    SurfaceAddressV1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.iris.truth.mechanics_oracle_v1 import SurfacePairQueryV1
from models.iris.truth.motion_temporal_oracle_v1 import (
    measure_temporal_relative_transport_v1,
)

REST = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
FACES = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
QUERY = SurfacePairQueryV1(0, 1)
TIMES = np.asarray([0.0, 0.1, 0.2, 0.3], dtype=np.float64)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _global_rigid(angle: float, translation) -> np.ndarray:
    return REST @ _rz(angle).T + np.asarray(translation, dtype=np.float64)


def _relative(angle: float) -> np.ndarray:
    out = REST.copy()
    tri = out[FACES[1]]
    centre = tri.mean(axis=0)
    out[FACES[1]] = (tri - centre) @ _rz(angle).T + centre
    return out


def _measure(sequence: np.ndarray, condition: str):
    return measure_temporal_relative_transport_v1(
        REST,
        FACES,
        sequence,
        TIMES,
        QUERY,
        condition_binding_hash=condition,
        discovery_frame_indices=(0, 1),
        validation_frame_indices=(2, 3),
        subject_scale=4.0,
        degeneracy_tolerance=1.0e-10,
    )


def _binding() -> AttachmentSurfaceBindingV1:
    return AttachmentSurfaceBindingV1(
        attachment_index=0,
        attachment_vertex_indices=np.asarray([0, 1, 2], dtype=np.int64),
        held=SurfaceAddressV1(0, (0.2, 0.3, 0.5)),
        holder=SurfaceAddressV1(1, (0.2, 0.3, 0.5)),
        setup_surface_gap=0.0,
    )


def _ref(evidence) -> TemporalConditionEvidenceRefV1:
    return TemporalConditionEvidenceRefV1(
        condition_binding_hash=evidence.condition_binding_hash,
        evidence_status=evidence.final_status,
        heldout_confirmed=evidence.heldout_confirmed,
        evidence_face_a=evidence.query.face_a,
        evidence_face_b=evidence.query.face_b,
    )


def test_temporal_global_rigid_pair_survives_future_heldout_frames():
    sequence = np.asarray(
        [
            _global_rigid(0.01, [0.0, 0.0, 0.0]),
            _global_rigid(0.02, [0.1, -0.1, 0.0]),
            _global_rigid(0.03, [0.2, -0.1, 0.1]),
            _global_rigid(0.04, [0.3, 0.0, 0.1]),
        ]
    )
    evidence = _measure(sequence, "a" * 64)
    assert evidence.final_status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.heldout_confirmed is True
    assert evidence.discovery_time_range_seconds == (0.0, 0.1)
    assert evidence.validation_time_range_seconds == (0.2, 0.3)


def test_temporal_relative_motion_contradicts_rigid_transport_in_both_splits():
    sequence = np.asarray([_relative(angle) for angle in (0.01, 0.02, 0.03, 0.04)])
    evidence = _measure(sequence, "b" * 64)
    assert evidence.final_status == "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.heldout_confirmed is True


def test_static_policy_is_rejected_by_one_confirmed_nonrigid_condition():
    rigid = _measure(
        np.asarray([_global_rigid(a, [0.0, 0.0, 0.0]) for a in (0.01, 0.02, 0.03, 0.04)]),
        "a" * 64,
    )
    nonrigid = _measure(np.asarray([_relative(a) for a in (0.01, 0.02, 0.03, 0.04)]), "b" * 64)
    decision = evaluate_temporal_static_policy_v1(binding=_binding(), evidence=(_ref(rigid), _ref(nonrigid)))
    assert decision.decision == "STATIC_POLICY_CONTRADICTED"
    assert decision.nonrigid_confirmed_count == 1
    assert decision.product_static_authority_minted is False
    assert decision.dynamic_generation_authorized is False


def test_all_rigid_finite_conditions_do_not_mint_global_static_authority():
    first = _measure(
        np.asarray([_global_rigid(a, [0.0, 0.0, 0.0]) for a in (0.01, 0.02, 0.03, 0.04)]),
        "a" * 64,
    )
    second = _measure(
        np.asarray([_global_rigid(a, [0.1, 0.0, 0.0]) for a in (0.02, 0.03, 0.04, 0.05)]),
        "c" * 64,
    )
    decision = evaluate_temporal_static_policy_v1(binding=_binding(), evidence=(_ref(first), _ref(second)))
    assert decision.decision == "NO_CONTRADICTION_ON_MEASURED_CONDITIONS"
    assert decision.rigid_confirmed_count == 2
    assert decision.product_static_authority_minted is False


def test_abstaining_condition_prevents_positive_measured_condition_statement():
    row = TemporalConditionEvidenceRefV1("d" * 64, "ABSTAIN", False, 0, 1)
    decision = evaluate_temporal_static_policy_v1(binding=_binding(), evidence=(row,))
    assert decision.decision == "ABSTAIN"


def test_temporal_split_must_be_future_heldout_and_nonoverlapping():
    sequence = np.asarray([_relative(a) for a in (0.01, 0.02, 0.03, 0.04)])
    with pytest.raises(ValueError, match="HELDOUT_MUST_FOLLOW_DISCOVERY"):
        measure_temporal_relative_transport_v1(
            REST,
            FACES,
            sequence,
            TIMES,
            QUERY,
            condition_binding_hash="e" * 64,
            discovery_frame_indices=(0, 2),
            validation_frame_indices=(1, 3),
            subject_scale=4.0,
            degeneracy_tolerance=1.0e-10,
        )


def test_gate_rejects_duplicate_conditions_and_address_mismatch():
    row = TemporalConditionEvidenceRefV1("f" * 64, "RIGID_RELATIVE_TRANSPORT_SUPPORTED", True, 0, 1)
    with pytest.raises(QualificationError, match="CONDITION_DUPLICATE"):
        evaluate_temporal_static_policy_v1(binding=_binding(), evidence=(row, row))
    bad = TemporalConditionEvidenceRefV1("1" * 64, "RIGID_RELATIVE_TRANSPORT_SUPPORTED", True, 1, 0)
    with pytest.raises(QualificationError, match="ADDRESS_MISMATCH"):
        evaluate_temporal_static_policy_v1(binding=_binding(), evidence=(bad,))


def test_temporal_evidence_api_cannot_accept_semantic_or_physical_teacher_labels():
    parameters = set(inspect.signature(measure_temporal_relative_transport_v1).parameters)
    forbidden = {
        "bone_id",
        "bone_name",
        "attachment_label",
        "semantic_part",
        "cape",
        "hair",
        "physical_dof",
        "joint_limit",
        "impulse",
        "force",
        "secondary_motion_class",
        "source_weights",
    }
    assert parameters.isdisjoint(forbidden)
