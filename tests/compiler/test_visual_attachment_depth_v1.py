import numpy as np
import pytest

from compiler.realsas_compiler_core.camera_geometry_v2 import qualify_camera_v3
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_attachment_depth_v1 import (
    evaluate_slot_owned_depth, attachment_owner_frame_metrics,
)


def fixture():
    camera = qualify_camera_v3(dict(origin=[0,0,-2], right=[1,0,0], screen_up=[0,-1,0],
        forward=[0,0,1], half_extent=4, resolution=8), view_id="V0", view_index=0)
    rest = np.array([[5.,4], [6,4], [5,5], [7,4], [8,4], [7,5], [2,2]])
    relief = np.array([2.,2.1,2.2,3.,3.1,3.2,4.])
    owners = np.array([1,1,1,1,1,1,0])
    matrices = np.eye(4)[None].copy(); matrices[0,:3,3]=[2,3,.5]
    contract = {"attachments":[{"attachment_index":1, "target_slot_raw_index_fit_only":0,
        "presentation_motion_model":"TARGET_SLOT_LOCAL_RIGID_2D_CAMERA_TWIST_V1"}]}
    args = dict(rest_depths=relief, vertex_attachment_owner=owners, attachments=contract,
                axis_positions_source=np.array([[0.,0,0]]), skin_matrices_source=matrices, camera=camera)
    return rest, relief, owners, args


def test_disconnected_art_pieces_share_slot_depth_and_frozen_rest_relief():
    rest, relief, owners, args = fixture()
    # Deliberately unrelated animated M depths must not survive on the prop.
    field = np.c_[rest+[2,3], np.arange(len(rest)) + 10.]
    got = evaluate_slot_owned_depth(field, **args)
    np.testing.assert_array_equal(got[:6,2],relief[:6]+.5)
    np.testing.assert_array_equal(got[:6,2]-got[0,2],relief[:6]-relief[0])
    np.testing.assert_array_equal(got[:,:2],field[:,:2])
    np.testing.assert_array_equal(got[owners==0],field[owners==0])
    assert attachment_owner_frame_metrics(got,rest_source_xy=rest,**args)["attachment_ownership_passed"]


@pytest.mark.parametrize("mutation", ["mechanical_depth", "piece_depth", "piece_xy", "wrong_slot", "wrong_rest"])
def test_owner_proof_rejects_semantic_depth_and_slot_tampering(mutation):
    rest, relief, owners, args = fixture()
    got = evaluate_slot_owned_depth(np.c_[rest+[2,3], relief], **args)
    if mutation == "mechanical_depth": got[:6,2] = np.arange(6) + 10.
    if mutation == "piece_depth": got[3:,2] += .1
    if mutation == "piece_xy": got[3,:2] += .1
    if mutation == "wrong_slot":
        args["skin_matrices_source"] = args["skin_matrices_source"].copy()
        args["skin_matrices_source"][0,2,3] += .1
    if mutation == "wrong_rest": args["rest_depths"] = relief + .1
    proof = attachment_owner_frame_metrics(got, rest_source_xy=rest, **args)
    assert proof["attachment_ownership_passed"] is False


def test_slot_depth_behind_camera_fails_closed():
    rest, relief, _, args = fixture()
    args["skin_matrices_source"][0,2,3] = -10
    with pytest.raises(QualificationError,match="BEHIND_CAMERA"):
        evaluate_slot_owned_depth(np.c_[rest,relief],**args)
