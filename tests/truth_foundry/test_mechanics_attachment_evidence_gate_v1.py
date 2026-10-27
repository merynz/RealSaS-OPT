import numpy as np
import pytest

from compiler.realsas_compiler_core.mechanics_attachment_evidence_gate_v1 import (
    authorize_rigid_surface_binding_v1,
)
from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import (
    AttachmentSurfaceBindingV1,
    SurfaceAddressV1,
    apply_surface_rigid_3d,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.iris.truth.mechanics_oracle_v1 import (
    SurfacePairQueryV1,
    measure_relative_transport_v1,
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
BARY = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
BINDING = AttachmentSurfaceBindingV1(
    attachment_index=1,
    attachment_vertex_indices=np.asarray([3, 4, 5], dtype=np.int64),
    held=SurfaceAddressV1(face_index=1, barycentric=BARY),
    holder=SurfaceAddressV1(face_index=0, barycentric=BARY),
    setup_surface_gap=2.0,
)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _common_rigid_sample(angle, translation):
    rotation = _rz(angle)
    return REST @ rotation.T + np.asarray(translation, dtype=np.float64)


def _mechanics_evidence():
    return measure_relative_transport_v1(
        REST,
        FACES,
        np.asarray(
            [
                _common_rigid_sample(0.002, [0.1, -0.2, 0.0]),
                _common_rigid_sample(-0.01, [-0.3, 0.2, 0.1]),
            ]
        ),
        np.asarray([_common_rigid_sample(0.05, [0.6, 0.2, -0.1])]),
        SurfacePairQueryV1(face_a=1, face_b=0),
        subject_scale=4.0,
        degeneracy_tolerance=1.0e-12,
    )


def test_heldout_confirmed_mechanics_evidence_authorizes_exact_candidate_and_fixed_consumer():
    evidence = _mechanics_evidence()
    gate = authorize_rigid_surface_binding_v1(
        binding=BINDING,
        evidence_status=evidence.final_status,
        heldout_confirmed=evidence.heldout_confirmed,
        evidence_face_a=evidence.query.face_a,
        evidence_face_b=evidence.query.face_b,
    )
    assert gate.decision == "AUTHORIZED"

    # Give the holder a new motion while deliberately leaving the held region at
    # its stale/rest location. The fixed compiler consequence must transport the
    # candidate without mutating canonical rest geometry/topology.
    posed = REST.copy()
    holder_ids = FACES[0]
    rotation = _rz(0.31)
    posed[holder_ids] = REST[holder_ids] @ rotation.T + np.asarray([0.7, -0.4, 0.2])
    rest_before = REST.copy()
    faces_before = FACES.copy()
    out, proof = apply_surface_rigid_3d(posed, rest=REST, faces=FACES, binding=BINDING)

    assert proof["anchor_transport_residual_world"] <= 1.0e-9
    assert proof["rigidity_residual_world"] <= 1.0e-9
    assert np.array_equal(REST, rest_before)
    assert np.array_equal(FACES, faces_before)
    assert not np.array_equal(out[BINDING.attachment_vertex_indices], posed[BINDING.attachment_vertex_indices])


def test_nonrigid_evidence_rejects_rigid_candidate():
    gate = authorize_rigid_surface_binding_v1(
        binding=BINDING,
        evidence_status="NONRIGID_RELATIVE_TRANSPORT_SUPPORTED",
        heldout_confirmed=True,
        evidence_face_a=1,
        evidence_face_b=0,
    )
    assert gate.decision == "REJECTED"


def test_abstain_or_unconfirmed_evidence_never_authorizes():
    abstain = authorize_rigid_surface_binding_v1(
        binding=BINDING,
        evidence_status="ABSTAIN",
        heldout_confirmed=False,
        evidence_face_a=1,
        evidence_face_b=0,
    )
    unconfirmed = authorize_rigid_surface_binding_v1(
        binding=BINDING,
        evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
        heldout_confirmed=False,
        evidence_face_a=1,
        evidence_face_b=0,
    )
    assert abstain.decision == "ABSTAIN"
    assert unconfirmed.decision == "ABSTAIN"


def test_address_mismatch_fails_closed_instead_of_remapping_evidence():
    with pytest.raises(QualificationError, match="EVIDENCE_ADDRESS_MISMATCH"):
        authorize_rigid_surface_binding_v1(
            binding=BINDING,
            evidence_status="RIGID_RELATIVE_TRANSPORT_SUPPORTED",
            heldout_confirmed=True,
            evidence_face_a=0,
            evidence_face_b=1,
        )


def test_unknown_mechanics_status_fails_closed():
    with pytest.raises(QualificationError, match="EVIDENCE_STATUS_INVALID"):
        authorize_rigid_surface_binding_v1(
            binding=BINDING,
            evidence_status="ATTACHMENT",
            heldout_confirmed=True,
            evidence_face_a=1,
            evidence_face_b=0,
        )
