import numpy as np
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_attachment_motion_v1 import (
    evaluate_attachment_motion, slot_rigid_transform_2d, attachment_slot_carrier_residual,
)


def fixture():
    camera = qualify_camera_v3(dict(origin=[0, 0, -2], right=[1, 0, 0], screen_up=[0, -1, 0],
        forward=[0, 0, 1], half_extent=4, resolution=8), view_id="V0", view_index=0)
    rest = np.array([[5., 4], [6, 4], [5, 5], [7, 4], [8, 4], [7, 5], [2, 2]])
    field = np.c_[rest + [12, 0], np.arange(len(rest)) + 1.]
    matrices = np.eye(4)[None].copy()
    matrices[0, :2, :2] = [[0, -1], [1, 0]]
    contract = {"attachments": [{"attachment_index": 1, "target_slot_raw_index_fit_only": 0,
                                  "presentation_motion_model": "TARGET_SLOT_LOCAL_RIGID_2D_CAMERA_TWIST_V1",
                                  "vertex_indices": [0, 1, 2]}]}
    return camera, rest, field, matrices, contract


def test_disconnected_single_anchor_pieces_share_rotation_not_independent_translation():
    camera, rest, field, matrices, contract = fixture()
    actual = evaluate_attachment_motion(field, rest_source_xy=rest,
        vertex_attachment_owner=[1, 1, 1, 1, 1, 1, 0], attachments=contract,
        axis_positions_source=[[0, 0, 0]], skin_matrices_source=matrices, camera=camera)
    # Both disconnected triangles turn around the same pixel-space pivot (4,4).
    np.testing.assert_allclose(actual[:6, :2], [[4,5], [4,6], [3,5], [4,7], [4,8], [3,7]])
    np.testing.assert_array_equal(actual[:, 2], field[:, 2])
    np.testing.assert_array_equal(actual[6], field[6])
    np.testing.assert_array_equal(rest, fixture()[1])


def test_slot_local_art_offset_and_rest_are_preserved_under_translation():
    camera, rest, field, matrices, contract = fixture()
    matrices[0] = np.eye(4); matrices[0, :3, 3] = [2, 3, .5]
    actual = evaluate_attachment_motion(field, rest_source_xy=rest,
        vertex_attachment_owner=np.ones(len(rest)), attachments=contract,
        axis_positions_source=[[0, 0, 0]], skin_matrices_source=matrices, camera=camera)
    np.testing.assert_allclose(actual[:, :2], rest + [2, 3])


def test_carrier_wrong_slot_and_nonrigid_slot_cannot_hide_behind_valid_artwork():
    camera, rest, field, matrices, contract = fixture()
    xyz = np.array([[0.,0,0],[1,0,0],[0,1,0]])
    good = (np.c_[xyz,np.ones(3)] @ matrices[0].T)[:, :3]
    assert attachment_slot_carrier_residual(contract, rest_xyz=xyz, posed_xyz=good,
                                           skin_matrices_source=matrices) == 0
    assert attachment_slot_carrier_residual(contract, rest_xyz=xyz, posed_xyz=xyz,
                                           skin_matrices_source=matrices) > .1
    matrices[0, 0, 0] = 2
    with pytest.raises(QualificationError, match="SLOT_TRANSFORM_INVALID"):
        slot_rigid_transform_2d(slot_rest_xyz=[0,0,0],skin_matrix_source=matrices[0],camera=camera)


def test_unobservable_view_twist_fails_closed():
    camera, _, _, matrices, _ = fixture()
    matrices[0] = np.diag([1., -1, -1, 1])
    with pytest.raises(QualificationError, match="TWIST_UNOBSERVABLE"):
        slot_rigid_transform_2d(slot_rest_xyz=[0,0,0],skin_matrix_source=matrices[0],camera=camera)
