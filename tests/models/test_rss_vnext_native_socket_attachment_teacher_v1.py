from __future__ import annotations

"""Native socket truth candidate tests: synthetic authored source fixture ONLY.

No source-native corpus, signed authoring file, source rights, or independent
trainer admission is manufactured here. The self-contained interventions
exercise the same finite-source oracle as the existing mechanics source court.
"""
from dataclasses import replace
import numpy as np
import pytest

from tests.models.test_rss_vnext_native_tessa_face_reindex_v1 import _native,_run
from tools.truth_foundry.native_socket_attachment_teacher_v1 import (
    NativeSocketAttachmentRequestV1,
    build_native_socket_attachment_teacher_candidate_v1,
    compile_native_socket_training_targets_v1,
)

def _rigid_matrix(angle:float,translation:tuple[float,float,float])->np.ndarray:
    c,s=np.cos(angle),np.sin(angle)
    mat=np.eye(4,dtype=np.float64)
    mat[:3,:3]=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    mat[:3,3]=translation
    return mat


def _bank(rest:np.ndarray, transforms:tuple[np.ndarray,...]):
    return np.stack([rest@t[:3,:3].T+t[:3,3] for t in transforms])


def _source(*,relation="EXPLICIT_SOURCE_AUTHORED_RIGID_ATTACHMENT",
            motion="RIGID",exhaustive=False,independent=True):
    net,xyz,faces,comps,tokens=_native()
    _,reindex=_run()
    discovery=(
        _rigid_matrix(.15,(.05,.04,.02)),
        _rigid_matrix(-.24,(-.08,.07,-.01)),
    )
    validation=(_rigid_matrix(.36,(.01,-.06,.03)),)
    p1=_bank(xyz,discovery)
    p2=_bank(xyz,validation)
    if motion=="DECOUPLED":
        for bank in (p1,p2):
            for t in range(len(bank)):
                ids=faces[0]  # SOURCE held face moves separately from parent
                centre=bank[t,ids].mean(axis=0)
                extra=_rigid_matrix(.19+0.05*t,(.05,0.,.02))
                bank[t,ids]=(
                    (bank[t,ids]-centre)@extra[:3,:3].T
                    +centre+extra[:3,3]
                )
    request=NativeSocketAttachmentRequestV1(
        axis_anchor_event_index=10,
        socket_control_index=0,
        authoring_class=relation,
        native_held_face_index=None if relation=="UNKNOWN" else 0,
        native_holder_face_index=None if relation=="UNKNOWN" else 1,
        authored_native_attachment_record_sha256="c"*64,
        parent_control_native_frame_sha256="d"*64,
        independently_actuated_parent=independent,
    )
    root0=np.eye(4)
    root_disc=np.stack(
        discovery if motion=="GLOBAL_ONLY" else (root0,root0),
    )
    root_val=np.stack(
        validation if motion=="GLOBAL_ONLY" else (root0,),
    )
    args=dict(
        source_vertices_normalized=xyz,
        source_rest_vertices_world=xyz.copy(),
        source_faces=faces,
        discovery_positions_world=p1,
        validation_positions_world=p2,
        source_world_from_normalized=np.eye(4),
        native_tessa_reindex=reindex,
        request=request,
        parent_rest_world=root0,
        parent_discovery_world=np.stack(discovery),
        parent_validation_world=np.stack(validation),
        root_rest_world=root0,
        root_discovery_world=root_disc,
        root_validation_world=root_val,
        source_asset_sha256="a"*64,
        source_intervention_bank_sha256="b"*64,
        subject_scale=1.,
    )
    return args


def test_source_authored_rigid_heldout_parent_motion_maps_reordered_native_faces():
    args=_source()
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    assert out.relationship_state=="POSITIVE_CANDIDATE_NOT_YET_ADMITTED"
    assert out.relative_transport_status=="RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert out.heldout_confirmed
    assert out.source_held_face_index==0 and out.source_holder_face_index==1
    assert out.teacher_held_face_index==1 and out.teacher_holder_face_index==0
    assert (out.teacher_held_target,out.teacher_holder_target)==(1,0)
    assert out.held_target_mask and out.holder_target_mask
    assert out.discovery_parent_excitation>.005
    assert out.validation_parent_excitation>.005
    assert out.max_held_parent_angle_rad<1.e-3
    assert out.max_holder_parent_translation_fraction<1.e-4
    assert not out.trainer_admission_qualified
    assert not out.source_contact_qualified
    assert not out.executable_attachment_qualified
    labels,mask=compile_native_socket_training_targets_v1(
        candidates=(out,),axis_socket_anchor_events=(10,),tessa_face_count=2,
    )
    assert labels.tolist()==[[1,0]]
    assert mask.tolist()==[[True,True]]


def test_exhaustive_authored_empty_socket_and_separating_action_is_true_no_face_candidate():
    args=_source(
        relation="EXHAUSTIVE_SOURCE_AUTHORED_NO_ATTACHMENT",
        motion="DECOUPLED",
    )
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    assert out.relationship_state=="NEGATIVE_NO_FACE_CANDIDATE_NOT_YET_ADMITTED"
    assert out.relative_transport_status=="NONRIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert out.heldout_confirmed
    assert (out.teacher_held_target,out.teacher_holder_target)==(2,2)
    labels,mask=compile_native_socket_training_targets_v1(
        candidates=(out,),axis_socket_anchor_events=(10,),tessa_face_count=2,
    )
    assert labels.tolist()==[[2,2]]
    assert mask.tolist()==[[True,True]]
    assert out.trainer_admission_qualified is False


@pytest.mark.parametrize("relation,motion",[
    ("UNKNOWN","RIGID"),
    ("EXPLICIT_SOURCE_AUTHORED_RIGID_ATTACHMENT","DECOUPLED"),
    ("EXHAUSTIVE_SOURCE_AUTHORED_NO_ATTACHMENT","RIGID"),
    ("UNKNOWN","DECOUPLED"),
])
def test_unknown_or_contradictory_native_evidence_abstains_never_false_negative(relation,motion):
    args=_source(relation=relation,motion=motion)
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    assert out.relationship_state=="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"
    labels,mask=compile_native_socket_training_targets_v1(
        candidates=(out,),axis_socket_anchor_events=(10,),tessa_face_count=2,
    )
    assert labels.tolist()==[[-1,-1]]
    assert mask.tolist()==[[False,False]]
    assert not out.trainer_admission_qualified


def test_global_body_motion_is_not_an_independent_parent_control_intervention():
    args=_source(motion="GLOBAL_ONLY")
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    assert out.relative_transport_status=="RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert out.heldout_confirmed
    assert out.discovery_parent_excitation<1.e-9
    assert out.validation_parent_excitation<1.e-9
    assert out.relationship_state=="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"
    labels,mask=compile_native_socket_training_targets_v1(
        candidates=(out,),axis_socket_anchor_events=(10,),tessa_face_count=2,
    )
    assert labels.tolist()==[[-1,-1]]
    assert not mask.any()


def test_fake_independent_motion_flag_cannot_replace_parent_relative_excitation():
    args=_source(motion="GLOBAL_ONLY",independent=True)
    assert build_native_socket_attachment_teacher_candidate_v1(
        **args,
    ).relationship_state=="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"
    args=_source(motion="RIGID",independent=False)
    assert build_native_socket_attachment_teacher_candidate_v1(
        **args,
    ).relationship_state=="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"


def test_parent_frame_not_tracking_holder_abstains_despite_rigid_pair_transport():
    args=_source()
    args["parent_validation_world"]=np.stack((np.eye(4),))
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    assert out.relative_transport_status=="RIGID_RELATIVE_TRANSPORT_SUPPORTED"
    assert out.relationship_state=="ABSTAIN_SOURCE_ATTACHMENT_UNDETERMINED"


def test_wrong_source_world_normalization_or_changed_original_mesh_fails_closed():
    args=_source()
    wrong=np.eye(4);wrong[0,3]=.1
    with pytest.raises(ValueError,match="NATIVE_TO_TESSA_COORDINATE_FRAME_NOT_PROVEN"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"source_world_from_normalized":wrong},
        )
    altered=args["source_vertices_normalized"].copy()
    altered[0,0]+=0.001
    with pytest.raises(ValueError,match="SOURCE_MESH_REINDEX_BYTES_DRIFT"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"source_vertices_normalized":altered,
               "source_rest_vertices_world":altered},
        )


def test_source_intervention_parent_root_timebase_and_SO3_fail_closed():
    args=_source()
    with pytest.raises(ValueError,match="PARENT_ROOT_POSE_TIMEBASE_MISMATCH"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"root_discovery_world":args["root_discovery_world"][:1]},
        )
    bad=args["parent_discovery_world"].copy()
    bad[0,0,0]=2.
    with pytest.raises(ValueError,match="DISCOVERY_PARENT_ROTATION_NOT_SO3"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"parent_discovery_world":bad},
        )
    with pytest.raises(ValueError,match="EXTERNAL_SOURCE_OR_AUTHORING_DOMAIN"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"source_asset_sha256":"not-a-source-hash"},
        )


def test_native_address_reversal_and_unknown_with_held_face_are_rejected():
    args=_source()
    bad=replace(args["request"],native_held_face_index=1,
                native_holder_face_index=1)
    with pytest.raises(ValueError,match="NATIVE_AUTHORING_FACE_ADDRESS_INVALID"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"request":bad},
        )
    bad=replace(args["request"],authoring_class="UNKNOWN")
    with pytest.raises(ValueError,match="UNKNOWN_CANNOT_DECLARED_FACE_TRUTH"):
        build_native_socket_attachment_teacher_candidate_v1(
            **{**args,"request":bad},
        )


def test_axis_anchor_order_and_source_authority_cannot_be_promoted_by_label_helper():
    args=_source()
    out=build_native_socket_attachment_teacher_candidate_v1(**args)
    with pytest.raises(ValueError,match="TEACHER_TARGET_SOURCE_OR_AXIS_PROVENANCE_DRIFT"):
        compile_native_socket_training_targets_v1(
            candidates=(out,),axis_socket_anchor_events=(11,),tessa_face_count=2,
        )
    with pytest.raises(ValueError,match="TEACHER_TARGET_SOURCE_OR_AXIS_PROVENANCE_DRIFT"):
        compile_native_socket_training_targets_v1(
            candidates=(replace(out,trainer_admission_qualified=True),),
            axis_socket_anchor_events=(10,),tessa_face_count=2,
        )
    with pytest.raises(ValueError,match="TEACHER_TARGET_RESOURCE_OR_AXIS_ORDER"):
        compile_native_socket_training_targets_v1(
            candidates=(out,out),axis_socket_anchor_events=(10,10),
            tessa_face_count=2,
        )
