from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.native_face_layer_truth_v1 import (
    NativeFaceLayerSamplesV1,tensor_sha256_v1,
)
from models.rss_vnext.native_dynamic_tessa_importance_v1 import (
    SOURCE_RECEIPT_SCHEMA,SCHEMA,
    build_source_native_tessa_dynamic_importance_v1,
    verify_dynamic_importance_bound_to_teacher_v1,
)


def _source_fixture():
    xyz=torch.tensor([
        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.],
    ])
    faces=torch.tensor([[0,1,2],[1,3,2]],dtype=torch.long)
    witness=NativeFaceLayerSamplesV1(
        vertices=xyz,faces=faces,
        face_layer_ids=torch.zeros(2,dtype=torch.long),
        face_component_ids=torch.zeros(2,dtype=torch.long),
        sample_face_ids=torch.tensor([0,1],dtype=torch.long),
        sample_barycentric=torch.full((2,3),1./3.),
        source_asset_sha256="a"*64,
        layer_namespace_sha256="b"*64,
    )
    # One soft-anchored vertex on face 1. It collapses only under the
    # source-qualified second pose (joint 1 moves x=-5, 0.2 influence).
    skin=torch.tensor([
        [1.,0.],[1.,0.],[1.,0.],[.8,.2],
    ])
    matrices=torch.eye(4).repeat(1,2,2,1,1)
    matrices[0,1,1,0,3]=-5.
    mask=torch.tensor([[True,True]])
    key="qualified-source-knight-fixture"
    receipt={
        "schema":SOURCE_RECEIPT_SCHEMA,
        "admission_status":"QUALIFIED_INDEPENDENT_NATIVE_SKIN_AND_POSES",
        "source_asset_sha256":witness.source_asset_sha256,
        "source_native_face_witness_sha256":witness.receipt_sha256(),
        "source_skin_native_vertex_order":True,
        "generated_views_as_teacher":False,
        "source_rights_independently_verified":True,
        "native_skin_sha256":tensor_sha256_v1(skin),
        "native_pose_matrices_sha256":tensor_sha256_v1(matrices),
        "native_pose_mask_sha256":tensor_sha256_v1(mask),
    }
    return witness,skin,matrices,mask,key,receipt


def _run(witness,skin,matrices,mask,key,receipt):
    return build_source_native_tessa_dynamic_importance_v1(
        witness=witness,native_skin=skin,
        native_pose_matrices=matrices,qualified_pose_mask=mask,
        source_qualification_key=key,
        trusted_source_registry={key:receipt},
        face_chunk=1024,
    )


def test_dynamic_source_face_importance_finds_local_collapse_without_predicted_mesh():
    w,s,m,mask,key,receipt=_source_fixture()
    result=_run(w,s,m,mask,key,receipt)
    result.validate(w)
    assert result.schema==SCHEMA
    assert result.qualified_pose_count==2
    assert result.source_failure_face_count==1
    assert not result.source_absolute_g3_within_limits
    assert result.sample_mask.tolist()==[True,True]
    assert float(result.sample_importance[0])<.01
    assert float(result.sample_importance[1])>.99
    assert torch.equal(
        result.sample_importance,result.face_importance[w.sample_face_ids],
    )
    assert not result.animation_ready_qualified
    assert not result.source_generated_views_as_pose_truth


def test_missing_collapse_pose_preserves_g3b_skin_transition_risk_only():
    w,s,m,mask,key,receipt=_source_fixture()
    mask[0,1]=False
    receipt["native_pose_mask_sha256"]=tensor_sha256_v1(mask)
    risk=_run(w,s,m,mask,key,receipt)
    assert risk.source_absolute_g3_within_limits
    assert risk.source_failure_face_count==0
    assert risk.qualified_pose_count==1
    assert .19<float(risk.sample_importance[1])<.21


def test_digest_and_trusted_source_skin_pose_receipt_fail_closed():
    w,s,m,mask,key,receipt=_source_fixture()
    risk=_run(w,s,m,mask,key,receipt)
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        replace(risk,face_importance=torch.zeros_like(risk.face_importance),\n                sample_importance=torch.zeros_like(risk.sample_importance)).validate(w)
    bad=dict(receipt)
    bad["source_skin_native_vertex_order"]=False
    with pytest.raises(ValueError,match="TRUSTED_RECEIPT_REQUIRED"):
        _run(w,s,m,mask,key,bad)
    wrong_skin=s.clone()
    wrong_skin[3]=torch.tensor([1.,0.])
    with pytest.raises(ValueError,match="TRUSTED_RECEIPT_REQUIRED"):
        _run(w,wrong_skin,m,mask,key,receipt)


def test_training_teacher_cannot_erase_pose_risk_or_spoof_the_receipt():
    w,s,m,mask,key,receipt=_source_fixture()
    risk=_run(w,s,m,mask,key,receipt)
    tessa_receipt={
        "native_dynamic_importance_schema":SCHEMA,
        "native_dynamic_importance_receipt_sha256":risk.report_sha256,
    }
    importance=torch.maximum(
        risk.sample_importance,
        torch.tensor([.35,.2]),
    )
    source_mask=torch.ones(2,dtype=torch.bool)
    verify_dynamic_importance_bound_to_teacher_v1(
        risk,witness=w,
        source_asset_sha256=w.source_asset_sha256,
        source_importance=importance,
        source_importance_mask=source_mask,
        receipt=tessa_receipt,
    )
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        verify_dynamic_importance_bound_to_teacher_v1(
            risk,witness=w,
            source_asset_sha256=w.source_asset_sha256,
            source_importance=torch.zeros_like(importance),
            source_importance_mask=source_mask,
            receipt=tessa_receipt,
        )
    with pytest.raises(ValueError,match="RECEIPT_DRIFT"):
        verify_dynamic_importance_bound_to_teacher_v1(
            risk,witness=w,
            source_asset_sha256=w.source_asset_sha256,
            source_importance=importance,
            source_importance_mask=source_mask,
            receipt={},
        )


def test_native_source_invalid_pose_mask_fails():
    w,s,m,mask,key,receipt=_source_fixture()
    mask[:]=False
    with pytest.raises(ValueError,match="SOURCE_SKIN_POSE_DOMAIN"):
        _run(w,s,m,mask,key,receipt)
