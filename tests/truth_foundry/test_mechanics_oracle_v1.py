import inspect

import numpy as np
import pytest

from models.iris.truth.mechanics_oracle_v1 import (
    RelativeTransportPolicyV1,
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
QUERY = SurfacePairQueryV1(0, 1)


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rigid_sample(angle, translation):
    r = _rz(angle)
    return REST @ r.T + np.asarray(translation, dtype=np.float64)


def _relative_sample(angle):
    out = REST.copy()
    tri = out[FACES[1]]
    centre = tri.mean(axis=0)
    out[FACES[1]] = (tri - centre) @ _rz(angle).T + centre
    return out


def _court(discovery, validation, *, rest=REST, faces=FACES):
    return measure_relative_transport_v1(
        rest,
        faces,
        np.asarray(discovery, dtype=np.float64),
        np.asarray(validation, dtype=np.float64),
        QUERY,
        subject_scale=4.0,
        degeneracy_tolerance=1.0e-10,
    )


def test_rigid_relative_transport_survives_heldout_larger_motion():
    evidence = _court(
        [_rigid_sample(0.002, [0.1, -0.2, 0.3]), _rigid_sample(-0.01, [-0.4, 0.2, 0.1])],
        [_rigid_sample(0.05, [0.7, 0.3, -0.1])],
    )
    assert evidence.discovery.status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.validation.status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.heldout_confirmed is True
    assert evidence.final_status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.validation.max_relative_angle_rad < 1.0e-6
    assert evidence.validation.max_relative_translation_fraction < 1.0e-12
    assert evidence.validation.max_triangle_strain_fro < 1.0e-12


def test_articulated_relative_transport_is_detected_without_semantic_label():
    evidence = _court(
        [_relative_sample(0.006), _relative_sample(-0.01)],
        [_relative_sample(0.05)],
    )
    assert evidence.discovery.status == "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.validation.status == "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.heldout_confirmed is True
    assert evidence.final_status == "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.discovery.max_relative_angle_rad >= 0.009
    assert evidence.validation.max_relative_angle_rad >= 0.049


def test_uncertainty_band_abstains_instead_of_forcing_binary_mechanism():
    evidence = _court(
        [_relative_sample(0.0015), _relative_sample(-0.002)],
        [_relative_sample(0.0025)],
    )
    assert evidence.discovery.status == "ABSTAIN"
    assert evidence.final_status == "ABSTAIN"
    assert evidence.heldout_confirmed is False


def test_heldout_contradiction_demotes_discovery_rigid_candidate_to_abstain():
    evidence = _court(
        [_relative_sample(0.0002), _relative_sample(-0.0003)],
        [_relative_sample(0.05)],
    )
    assert evidence.discovery.status == "RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.validation.status == "NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert evidence.heldout_confirmed is False
    assert evidence.final_status == "ABSTAIN"


def test_surface_strain_prevents_false_rigid_admission():
    discovery = np.repeat(REST[None, ...], 2, axis=0)
    validation = REST[None, ...].copy()
    discovery[0, 4, 0] += 0.02
    discovery[1, 4, 0] += 0.03
    validation[0, 4, 0] += 0.04
    evidence = _court(discovery, validation)
    assert evidence.discovery.max_triangle_strain_fro > RelativeTransportPolicyV1().rigid_accept_strain_fro
    assert evidence.final_status != "RIGID_RELATIVE_TRANSPORT_SUPPORTED"


def test_degenerate_surface_fails_closed_to_abstain():
    rest = REST.copy()
    rest[2] = rest[1] * 2.0
    discovery = np.repeat(rest[None, ...], 2, axis=0)
    validation = rest[None, ...]
    evidence = _court(discovery, validation, rest=rest)
    assert evidence.discovery.valid is False
    assert evidence.validation.valid is False
    assert evidence.final_status == "ABSTAIN"


def test_api_cannot_accept_privileged_bone_owner_or_source_weight_truth():
    parameters = inspect.signature(measure_relative_transport_v1).parameters
    forbidden = {
        "bone",
        "bone_id",
        "control_index_privileged",
        "face_owner_privileged",
        "source_weights",
        "attachment_label",
    }
    assert forbidden.isdisjoint(parameters)


def test_policy_requires_a_nonempty_abstention_band():
    with pytest.raises(ValueError, match="ABSTENTION_BAND"):
        RelativeTransportPolicyV1(
            rigid_accept_angle_rad=0.01,
            nonrigid_reject_angle_rad=0.005,
        )
