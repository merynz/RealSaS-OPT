from __future__ import annotations

"""Actual unified RSS neural graph -> source-receipted socket/face court.

Teacher mesh token prefixes and AXIS program are allowed only as declared
training inputs. Original FRONT/BACK are always the only student image inputs.
Source held/holder face labels never enter model.forward, only a separately
qualified, masked criterion.
"""
from dataclasses import replace
import pytest
import torch
import numpy as np

from models.axis.rig_program_model_v6 import AXISSequenceV6,RigTokenKindV6,RigProgramEventV6
from models.rss_vnext.attachment_face_correspondence_v1 import (
    RSSvNextSocketFaceCorrespondenceHeadV1,
    RSSvNextSocketFaceNeuralOutputV1,
    propose_socket_face_pairs_v1,
    source_qualified_socket_face_loss_v1,
    _digest_tensor,
)
from models.tessa.v1.model_v1 import TESSAConfigV1
from models.tessa.v1.tokenization_v1 import encode_adjacent_teacher_mesh_v1
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import _network,_images


@pytest.fixture(autouse=True)
def tiny_deterministic():
    torch.manual_seed(512)
    torch.set_num_threads(1)


def _seq(*,socket_kind=7,anchor_parent=1):
    events=(
        RigProgramEventV6(RigTokenKindV6.BOS),
        RigProgramEventV6(RigTokenKindV6.JOINT,coord_indices=(128,128,128)),
        RigProgramEventV6(RigTokenKindV6.FRAMES_START),
        RigProgramEventV6(RigTokenKindV6.CONTROLS_START),
        RigProgramEventV6(RigTokenKindV6.CONTROL,
                          aux_type_id=socket_kind,coord_indices=(50,70,90)),
        RigProgramEventV6(RigTokenKindV6.CHAINS_START),
        RigProgramEventV6(RigTokenKindV6.CONSTRAINTS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHORS_START),
        RigProgramEventV6(RigTokenKindV6.ANCHOR,coord_indices=(80,150,175),
                          anchor_parent_kind=anchor_parent,
                          pointer_a=0 if anchor_parent==0 else -1,
                          pointer_b=0 if anchor_parent==1 else -1),
        RigProgramEventV6(RigTokenKindV6.SKIN_START),
        RigProgramEventV6(RigTokenKindV6.EOS),
    )
    return AXISSequenceV6.from_events(events,device=torch.device("cpu"))


def _tokens(model):
    v=np.array([
        [-.35,-.35,0],[-.15,-.35,0],[-.35,-.15,0],
        [.15,.15,0],[.35,.15,0],[.15,.35,0],
    ],dtype=np.float64)
    cfg=TESSAConfigV1(coordinate_bins=model.cfg.tessa.coordinate_bins)
    toks=[]
    for i in (0,1):
        row=encode_adjacent_teacher_mesh_v1(
            v[i*3:(i+1)*3],np.array([[0,1,2]]),
            component_id=f"SOURCE_TEACHER_COMP_{i}",
            chart_id=f"SOURCE_TEACHER_CHART_{i}",cfg=cfg,
        ).token_ids
        toks.append(row)
    return torch.tensor(toks[0][:-1]+toks[1][1:],dtype=torch.long)


def _forward(*,socket_kind=7,anchor_parent=1):
    model=_network()
    model.train()
    rgba,cam=_images()
    seq=_seq(socket_kind=socket_kind,anchor_parent=anchor_parent)
    result=model(
        rgba,cam,
        mesh_teacher_tokens=_tokens(model),rig_sequence=seq,
    )
    assert result.mesh_candidate is not None
    return model,result,seq,rgba,cam


def _registry(out,labels,valid):
    return {"admitted-source-native-attachment":{
        "admission_status":"QUALIFIED_SOURCE_NATIVE_ATTACHMENT_FACE_REINDEX",
        "source_semantics":"NATIVE_RIG_ATTACHMENT_HELD_HOLDER_FACE_OR_VERIFIED_NO_FACE",
        "source_asset_sha256":"a"*64,
        "dataset_manifest_sha256":"b"*64,
        "tessa_mesh_topology_sha256":out.mesh_topology_digest,
        "axis_teacher_sequence_sha256":out.axis_sequence_digest,
        "source_face_targets_sha256":_digest_tensor(labels),
        "source_face_mask_sha256":_digest_tensor(valid),
        "source_reindex_proven":True,
    }}


def _loss(out,labels,valid,registry):
    return source_qualified_socket_face_loss_v1(
        out,source_face_targets=labels,source_face_mask=valid,
        source_asset_sha256="a"*64,dataset_manifest_sha256="b"*64,
        source_admission_key="admitted-source-native-attachment",
        trusted_registry=registry,
    )


def test_same_checkpoint_joint_tessa_axis_new_face_head_has_actual_logits():
    model,result,seq,rgba,cam=_forward()
    out=result.anchor_face_candidate
    assert isinstance(out,RSSvNextSocketFaceNeuralOutputV1)
    assert out.anchor_event_indices==(8,)
    assert out.socket_control_indices==(0,)
    assert out.held_face_logits.shape==(1,3)
    assert out.holder_face_logits.shape==(1,3)
    assert out.face_region_probabilities.shape==(2,model.cfg.region_slots+1)
    assert out.face_component_ids.tolist()==[0,1]
    assert out.source_correspondence_qualified is False
    assert out.executable_attachment_qualified is False
    assert result.state.observed_rgba.shape[1]==2
    assert result.mesh_candidate.topology_token_ids
    assert "anchor_face_correspondence.held_query.weight" in model.state_dict()


def test_registry_qualified_native_face_loss_grads_reach_iris_axis_and_new_head():
    model,result,seq,rgba,cam=_forward()
    out=result.anchor_face_candidate
    labels=torch.tensor([[0,1]],dtype=torch.long)
    mask=torch.tensor([[True,True]])
    loss=_loss(out,labels,mask,_registry(out,labels,mask))
    assert loss.eligible_held==1 and loss.eligible_holder==1
    assert loss.eligible_negative==0
    assert torch.isfinite(loss.loss)
    result.state.scene_triplanes.retain_grad()
    loss.loss.backward()
    for grad in (
        model.anchor_face_correspondence.held_query.weight.grad,
        model.anchor_face_correspondence.holder_query.weight.grad,
        model.anchor_face_correspondence.anchor_projection[1].weight.grad,
        model.anchor_face_correspondence.face_projection[1].weight.grad,
        model.axis.scene_cross_attention.in_proj_weight.grad,
        result.state.scene_triplanes.grad,
    ):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum()>0


def test_explicit_no_face_only_with_qualified_negative_source_teacher():
    model,result,_,_,_=_forward()
    out=result.anchor_face_candidate
    face_count=out.held_face_logits.shape[1]-1
    labels=torch.tensor([[face_count,face_count]])
    mask=torch.tensor([[True,True]])
    loss=_loss(out,labels,mask,_registry(out,labels,mask))
    assert loss.eligible_negative==2
    loss.loss.backward()
    assert model.anchor_face_correspondence.held_none.weight.grad.abs().sum()>0
    assert model.anchor_face_correspondence.holder_none.weight.grad.abs().sum()>0


def test_absent_or_masked_labels_are_not_source_negatives():
    _,result,_,_,_=_forward()
    out=result.anchor_face_candidate
    unknown=torch.tensor([[-1,-1]])
    missing=torch.tensor([[False,False]])
    with pytest.raises(ValueError,match="TEACHER_TARGET_DOMAIN"):
        _loss(out,unknown,missing,_registry(out,unknown,missing))
    # Only held labelled: unlabelled HOLDER does not silently become NO_FACE.
    labels=torch.tensor([[0,-1]])
    mask=torch.tensor([[True,False]])
    loss=_loss(out,labels,mask,_registry(out,labels,mask))
    assert loss.eligible_held==1 and loss.eligible_holder==0
    assert loss.eligible_negative==0


def test_label_reindex_and_axis_sequence_receipts_fail_closed():
    _,result,_,_,_=_forward()
    out=result.anchor_face_candidate
    labels=torch.tensor([[0,1]],dtype=torch.long)
    mask=torch.tensor([[True,True]])
    registry=_registry(out,labels,mask)
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _loss(out,labels[:,[1,0]],mask,registry)
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _loss(replace(out,mesh_topology_digest="c"*64),labels,mask,registry)
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _loss(replace(out,axis_sequence_digest="d"*64),labels,mask,registry)
    with pytest.raises(ValueError,match="SOURCE_ADMISSION_REQUIRED"):
        _loss(out,labels,mask,{})
    registry["admitted-source-native-attachment"]["source_reindex_proven"]=False
    with pytest.raises(ValueError,match="SOURCE_RECEIPT_DRIFT"):
        _loss(out,labels,mask,registry)


@pytest.mark.parametrize("kind,anchor",[(2,1),(7,0)])
def test_non_socket_or_joint_parent_has_no_fabricated_face_correspondence(kind,anchor):
    _,result,_,_,_=_forward(socket_kind=kind,anchor_parent=anchor)
    out=result.anchor_face_candidate
    assert out.held_face_logits.shape==(0,3)
    assert out.holder_face_logits.shape==(0,3)
    assert out.anchor_event_indices==()
    assert propose_socket_face_pairs_v1(out)==()


def test_none_argmax_abstains_and_cannot_be_replaced_by_nearest_surface():
    _,result,_,_,_=_forward()
    out=result.anchor_face_candidate
    pred=propose_socket_face_pairs_v1(replace(
        out,held_face_logits=torch.tensor([[-50.,-60.,50.]]),
        holder_face_logits=torch.tensor([[50.,-50.,-60.]]),
    ))
    assert pred[0].held_face_index is None
    assert pred[0].holder_face_index is None
    assert pred[0].state=="ABSTAIN_MODEL_NO_FACE_OR_PAIR"
    assert not pred[0].attachment_qualified


def test_directed_model_pairs_retain_semantic_direction_and_tessa_components():
    _,result,_,_,_=_forward()
    out=result.anchor_face_candidate
    a=propose_socket_face_pairs_v1(replace(
        out,held_face_logits=torch.tensor([[40.,-40.,-40.]]),
        holder_face_logits=torch.tensor([[-40.,40.,-40.]]),
    ))[0]
    b=propose_socket_face_pairs_v1(replace(
        out,held_face_logits=torch.tensor([[-40.,40.,-40.]]),
        holder_face_logits=torch.tensor([[40.,-40.,-40.]]),
    ))[0]
    assert (a.held_face_index,a.holder_face_index)==(0,1)
    assert (b.held_face_index,b.holder_face_index)==(1,0)
    assert a.same_tessa_component is False
    assert b.same_tessa_component is False
    assert a.known_cross_slot_support>0
    assert 0<=a.predicted_directed_relation<=1
    assert not a.attachment_qualified and not b.attachment_qualified
    assert a.authority.startswith("UNQUALIFIED_NEURAL")


def test_correspondence_forward_cannot_receive_source_teacher_mesh_or_labels():
    model,result,seq,rgba,cam=_forward()
    with pytest.raises(TypeError):
        model.anchor_face_correspondence(
            mesh=result.mesh_candidate,
            axis_hidden=result.control_logits["hidden"],
            axis_sequence=seq,
            region_relation_logits=result.state.region_relation_logits,
            source_teacher_face_indices=torch.tensor([[0,1]]),
        )


def test_full_model_inference_is_not_allowed_to_accept_teacher_topology():
    model,result,seq,rgba,cam=_forward()
    model.eval()
    with pytest.raises(ValueError,match="TEACHER_MESH_FORBIDDEN_AT_INFERENCE"):
        model(rgba,cam,mesh_teacher_tokens=_tokens(model),rig_sequence=seq)
