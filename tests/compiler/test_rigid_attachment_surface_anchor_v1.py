import numpy as np

from compiler.realsas_compiler_core.rigid_attachment_surface_anchor_v1 import (
    apply_anchored_camera_twist_2d,
    derive_surface_binding,
)


def _fixture():
    # Holder square at z=0; disconnected rigid held square at z=.2.  The closest
    # surfaces overlap in XY, so the true contact addresses are face+barycentric
    # points rather than a privileged vertex pair.
    rest = np.array([
        [0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [1.,1.,0.],
        [.25,.25,.2], [.75,.25,.2], [.25,.75,.2], [.75,.75,.2],
    ])
    faces = np.array([
        [0,1,2], [1,3,2],
        [4,5,6], [5,7,6],
    ], dtype=int)
    owners = np.array([0,0,0,0,1,1,1,1], dtype=np.int32)
    return rest, faces, owners


def test_surface_binding_is_barycentric_and_replays_exact_surface_gap():
    rest, faces, owners = _fixture()
    b = derive_surface_binding(
        rest_xyz=rest, faces=faces, attachment_vertex_owner=owners, attachment_index=1
    )
    assert abs(b.setup_surface_gap - .2) < 1e-12
    assert b.held.face_index in (2, 3)
    assert b.holder.face_index in (0, 1)
    np.testing.assert_allclose(sum(b.held.barycentric), 1.0, atol=1e-12)
    np.testing.assert_allclose(sum(b.holder.barycentric), 1.0, atol=1e-12)


def test_identity_pose_is_exact_setup_identity_and_depth_is_untouched():
    rest, faces, owners = _fixture()
    b = derive_surface_binding(
        rest_xyz=rest, faces=faces, attachment_vertex_owner=owners, attachment_index=1
    )
    # Treat these as already projected camera coordinates for the unit theorem.
    projected = rest.copy()
    out, proof = apply_anchored_camera_twist_2d(
        projected, projected_rest=rest, faces=faces, binding=b, rotation_2d=np.eye(2)
    )
    np.testing.assert_array_equal(out, projected)
    assert proof["anchor_transport_pass"] is True
    assert proof["rigidity_pass"] is True
    assert proof["depth_unchanged"] is True


def test_slot_rotation_changes_orientation_while_surface_anchor_follows_holder():
    rest, faces, owners = _fixture()
    b = derive_surface_binding(
        rest_xyz=rest, faces=faces, attachment_vertex_owner=owners, attachment_index=1
    )
    posed = rest.copy()
    posed[:4, :2] += [3.0, -2.0]  # holder translation from mechanical pose
    R = np.array([[0., -1.], [1., 0.]])
    out, proof = apply_anchored_camera_twist_2d(
        posed, projected_rest=rest, faces=faces, binding=b, rotation_2d=R
    )
    assert proof["anchor_transport_pass"] is True
    assert proof["rigidity_pass"] is True
    np.testing.assert_array_equal(out[:, 2], posed[:, 2])
    # Every held edge undergoes exactly the same proper 2D rotation.
    held = b.attachment_vertex_indices
    before = rest[held[1], :2] - rest[held[0], :2]
    after = out[held[1], :2] - out[held[0], :2]
    np.testing.assert_allclose(after, R @ before, atol=1e-12)


def test_surface_address_is_stable_under_irrelevant_vertex_refinement():
    rest, faces, owners = _fixture()
    a = derive_surface_binding(
        rest_xyz=rest, faces=faces, attachment_vertex_owner=owners, attachment_index=1
    )
    # Add an unrelated remote holder triangle.  Vertex count/density changes, but
    # the closest physical held/holder surface pair and gap must not.
    extra = np.array([[10.,10.,0.], [11.,10.,0.], [10.,11.,0.]])
    rest2 = np.vstack([rest, extra])
    faces2 = np.vstack([faces, [8,9,10]])
    owners2 = np.r_[owners, [0,0,0]].astype(np.int32)
    b = derive_surface_binding(
        rest_xyz=rest2, faces=faces2, attachment_vertex_owner=owners2, attachment_index=1
    )
    assert abs(a.setup_surface_gap - b.setup_surface_gap) < 1e-12
    assert b.held.face_index == a.held.face_index
    assert b.holder.face_index == a.holder.face_index
