from __future__ import annotations

"""Native source anchor/local frame vs real AXIS causal program verification."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from models.axis.rig_program_model_v6 import RigTokenKindV6
from models.axis.rig_program_decode_v6 import _local_vector
from tests.models.test_rss_vnext_anchor_face_trainer_v1 import _training_court
from tests.models.test_rss_vnext_native_socket_attachment_teacher_v1 import _source
from tools.truth_foundry.native_socket_attachment_teacher_v1 import (
    build_native_socket_attachment_teacher_candidate_v1,
)
from tools.truth_foundry.native_axis_socket_address_v1 import (
    validate_native_axis_socket_anchor_teacher_v1,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(751)
    torch.set_num_threads(1)


def _setup():
    net,make,batch,receipt,axis=_training_court()
    authored=_source()
    candidate=build_native_socket_attachment_teacher_candidate_v1(**authored)
    assert candidate.relationship_state=="POSITIVE_CANDIDATE_NOT_YET_ADMITTED"
    seq=batch.axis_teacher_sequence
    cfg=net.cfg.axis_config()
    idx=10
    local=tuple(float(x) for x in _local_vector(
        cfg,tuple(int(i) for i in seq.coord_indices[0,idx].tolist()),
        scale=1.,
    ))
    args=dict(
        source_candidate=candidate,axis_teacher_sequence=seq,
        axis_config=cfg,tokens_per_joint=net.cfg.skin_tokens_per_joint,
        axis_normalization_scale=1.,
        authored_native_parent_local_xyz=local,
        authored_native_world_anchor_xyz=local,
        native_parent_rest_world=np.eye(4,dtype=np.float64),
        authored_attachment_record_sha256="c"*64,
        parent_native_frame_provenance_sha256="d"*64,
    )
    return args


def test_source_positive_face_pair_matches_exact_axis_anchor_control_event_and_local_codec():
    args=_setup()
    row=validate_native_axis_socket_anchor_teacher_v1(**args)
    assert row.axis_anchor_event_index==10
    assert row.axis_parent_control_event_index==5
    assert row.axis_parent_control_ordinal==0
    assert row.axis_anchor_coord_bins==(77,88,99)
    assert row.max_coord_quantization_residual<=1.e-10
    assert row.axis_teacher_sequence_sha256
    assert row.source_socket_face_evidence_sha256
    assert row.parent_source_independently_admitted is False
    assert row.source_contact_qualified is False
    assert row.source_attachment_executable is False
    assert row.product_revision_qualified is False


def test_source_local_roundtrip_accepts_real_quantization_residual_not_exact_float_equality():
    args=_setup()
    offset=1.9/255.
    near=tuple(v+offset*.45 for v in args["authored_native_parent_local_xyz"])
    row=validate_native_axis_socket_anchor_teacher_v1(
        **{**args,"authored_native_parent_local_xyz":near,
           "authored_native_world_anchor_xyz":near},
    )
    assert row.max_coord_quantization_residual>0
    assert row.max_coord_quantization_residual<2./255.


def test_wrong_control_pointer_cannot_borrow_correct_source_face_pair():
    args=_setup()
    seq=args["axis_teacher_sequence"]
    changed=seq.pointer_b.clone()
    changed[0,10]=1
    with pytest.raises(ValueError,match="ANCHOR_PARENT_CONTROL_POINTER_DRIFT"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"axis_teacher_sequence":replace(seq,pointer_b=changed)},
        )


def test_non_socket_control_cannot_borrow_valid_anchor_and_source_motion():
    args=_setup()
    seq=args["axis_teacher_sequence"]
    changed=seq.aux_type_ids.clone()
    changed[0,5]=2
    with pytest.raises(ValueError,match="SOCKET_KIND_OR_CAUSAL_ORDER_DRIFT"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"axis_teacher_sequence":replace(seq,aux_type_ids=changed)},
        )


def test_native_parent_local_xyz_does_not_match_other_axis_quantized_anchor():
    args=_setup()
    bad=tuple(v+.2 for v in args["authored_native_parent_local_xyz"])
    with pytest.raises(ValueError,match="AXIS_LOCAL_ANCHOR_NOT_SOURCE_QUANTIZED"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"authored_native_parent_local_xyz":bad,
               "authored_native_world_anchor_xyz":bad},
        )


def test_world_rest_anchor_cannot_be_substituted_for_parent_local_coordinate():
    args=_setup()
    wrong=tuple(v+.25 for v in args["authored_native_world_anchor_xyz"])
    with pytest.raises(ValueError,match="NATIVE_PARENT_LOCAL_VS_WORLD_ANCHOR_DRIFT"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"authored_native_world_anchor_xyz":wrong},
        )


def test_invalid_source_and_frame_provenance_fail_closed():
    args=_setup()
    with pytest.raises(ValueError,match="SOURCE_OR_AXIS_TEACHER_DOMAIN"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"authored_attachment_record_sha256":"0"*64},
        )
    bad=np.eye(4);bad[0,0]=2.
    with pytest.raises(ValueError,match="NATIVE_REST_PARENT_FRAME_NOT_SE3"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"native_parent_rest_world":bad},
        )


def test_invalid_seq_grammar_and_bos_cannot_become_a_native_source_teacher():
    args=_setup()
    seq=args["axis_teacher_sequence"]
    changed=seq.kind_ids.clone()
    changed[0,0]=int(RigTokenKindV6.FRAMES_START)
    with pytest.raises(ValueError,match="CAUSAL_PROGRAM_COMPLETE_UNPADDED_REQUIRED"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"axis_teacher_sequence":replace(seq,kind_ids=changed)},
        )
    changed=seq.kind_ids.clone()
    changed[0,9]=int(RigTokenKindV6.CONTROL)
    with pytest.raises(ValueError,match="CAUSAL_PROGRAM_GRAMMAR_DRIFT"):
        validate_native_axis_socket_anchor_teacher_v1(
            **{**args,"axis_teacher_sequence":replace(seq,kind_ids=changed)},
        )
