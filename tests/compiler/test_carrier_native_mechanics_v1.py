from types import SimpleNamespace
import pytest

from compiler.realsas_compiler_core.carrier_skin_v1 import QualifiedCarrierSkinIR,QualifiedCarrierSkinRow
from compiler.realsas_compiler_core.joint_frames_v2 import derive_joint_frames_post_bind_v2
from compiler.realsas_compiler_core.types import QualificationError


def _camera(i):
    return SimpleNamespace(view_index=i,right=(1.0,0.0,0.0),screen_up=(0.0,0.0,1.0),forward=(0.0,-1.0,0.0))


def _skeleton():
    joints=(
        SimpleNamespace(canonical_joint_id="J0",parent_canonical_id=None,position=(0.,0.,0.)),
        SimpleNamespace(canonical_joint_id="J1",parent_canonical_id="J0",position=(1.,0.,0.)),
        SimpleNamespace(canonical_joint_id="J2",parent_canonical_id="J1",position=(1.,0.,0.)),
    )
    return SimpleNamespace(joints=joints,root_id="J0",skeleton_lineage_hash="S")


def _skin(active_child=False,row_count=2):
    rows=[]
    for i in range(row_count):
        influence=(("J2",1.0),) if active_child and i==row_count-1 else (("J1",1.0),)
        rows.append(QualifiedCarrierSkinRow(f"V{i}",influence,0.0,0.0))
    return QualifiedCarrierSkinIR(tuple(rows),"CE","CT","CG","S",{},"W")


def test_inactive_coincident_subtree_inherits_parent_frame_without_epsilon():
    frames,report=derive_joint_frames_post_bind_v2(_skeleton(),carrier_skin=_skin(False),cameras=tuple(_camera(i) for i in range(8)))
    assert report["coincident_edge_count"]==1
    assert report["invented_epsilon"] is False
    assert frames["J2"].rotation_matrix==frames["J1"].rotation_matrix
    event=report["events"][0]
    assert event["subtree_weight_mean"]==0.0
    assert event["subtree_weight_max_vertex"]==0.0
    assert report["subject_specific_code_used"] is False
    assert report["joint_name_semantics_used"] is False
    assert report["source_index_semantics_used"] is False


def test_active_coincident_subtree_fails_closed_to_axis_extension():
    with pytest.raises(QualificationError,match="ACTIVE_COINCIDENT_CONTROL_REQUIRES_AXIS_ORIENTATION_EXTENSION"):
        derive_joint_frames_post_bind_v2(_skeleton(),carrier_skin=_skin(True),cameras=tuple(_camera(i) for i in range(8)))


def test_coincident_inactivity_is_not_carrier_cardinality_dependent():
    frames,report=derive_joint_frames_post_bind_v2(
        _skeleton(),carrier_skin=_skin(False,row_count=100),cameras=tuple(_camera(i) for i in range(8))
    )
    assert report["events"][0]["subtree_weight_mean"]==0.0
    assert report["events"][0]["subtree_weight_max_vertex"]==0.0
    assert frames["J2"].rotation_matrix==frames["J1"].rotation_matrix
