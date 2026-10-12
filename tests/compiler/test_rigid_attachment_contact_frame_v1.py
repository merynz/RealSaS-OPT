import numpy as np

from compiler.realsas_compiler_core.rigid_attachment_contact_frame_v1 import (
    POLICY,
    apply_contact_frame_binding,
    derive_contact_frame_binding,
    rotation_distance_degrees,
)


def _fixture():
    # Carrier square + one rigid attachment triangle. No semantic labels or
    # subject-specific identifiers participate in the binding.
    rest = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.05, 0.05, 0.02],
            [0.25, 0.05, 0.02],
            [0.05, 0.25, 0.02],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]], dtype=int)
    owners = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    return rest, faces, owners


def _rigid(R, t, x):
    return np.asarray(x) @ np.asarray(R).T + np.asarray(t)


def test_setup_identity_and_anchor_gap_are_exact():
    rest, faces, owners = _fixture()
    binding = derive_contact_frame_binding(
        rest_xyz=rest,
        faces=faces,
        attachment_vertex_owner=owners,
        attachment_index=1,
        contact_band_edge_fraction=1.0,
    )
    out, proof = apply_contact_frame_binding(rest, rest_xyz=rest, binding=binding)
    assert np.array_equal(out, rest)
    assert proof["anchor_gap_pass"] is True
    assert proof["rigidity_pass"] is True
    assert proof["anchor_gap_residual"] <= POLICY["maximum_anchor_gap_drift"]


def test_contact_frame_transports_rigid_attachment_and_preserves_gap():
    rest, faces, owners = _fixture()
    binding = derive_contact_frame_binding(
        rest_xyz=rest,
        faces=faces,
        attachment_vertex_owner=owners,
        attachment_index=1,
        contact_band_edge_fraction=1.0,
    )
    angle = np.deg2rad(37.0)
    R = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle),  np.cos(angle), 0.0],
         [0.0,            0.0,           1.0]],
        dtype=float,
    )
    t = np.array([0.4, -0.2, 0.3])
    posed = rest.copy()
    holder = binding.holder_vertex_indices
    posed[holder] = _rigid(R, t, rest[holder])
    out, proof = apply_contact_frame_binding(posed, rest_xyz=rest, binding=binding)
    held = binding.attachment_vertex_indices
    expected = _rigid(R, t, rest[held])
    np.testing.assert_allclose(out[held], expected, atol=1e-10, rtol=0)
    assert proof["anchor_gap_pass"] is True
    assert proof["rigidity_pass"] is True


def test_contact_frame_is_independent_of_unrelated_carrier_motion():
    rest, faces, owners = _fixture()
    binding = derive_contact_frame_binding(
        rest_xyz=rest,
        faces=faces,
        attachment_vertex_owner=owners,
        attachment_index=1,
        contact_band_edge_fraction=1.0,
    )
    posed = rest.copy()
    # Vertex 3 is not in the local patch in this fixture; unrelated motion must
    # not move the rigid attachment authority.
    if 3 not in set(binding.holder_vertex_indices.tolist()):
        posed[3] += [10.0, 10.0, 10.0]
    out, proof = apply_contact_frame_binding(posed, rest_xyz=rest, binding=binding)
    np.testing.assert_allclose(out[binding.attachment_vertex_indices], rest[binding.attachment_vertex_indices], atol=1e-10, rtol=0)
    assert proof["anchor_gap_pass"] is True


def test_rotation_distance_reports_slot_equivalence_diagnostic():
    I = np.eye(3)
    angle = np.deg2rad(42.0)
    R = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, np.cos(angle), -np.sin(angle)],
         [0.0, np.sin(angle),  np.cos(angle)]],
        dtype=float,
    )
    assert abs(rotation_distance_degrees(I, R) - 42.0) < 1e-10
