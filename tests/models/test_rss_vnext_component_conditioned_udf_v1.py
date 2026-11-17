from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.component_conditioned_udf_v1 import (
    RSSvNextComponentConditionedUDFV1,
    SourceNativeComponentUDFFitV1,
    source_native_component_udf_objective_v1,
    KIND_CLOSED,KIND_OPEN,KIND_UNKNOWN,TRAIN_SCHEMA,
)
from models.rss_vnext.native_face_layer_truth_v1 import tensor_sha256_v1


def _teacher():
    x=torch.tensor([
        [0.,0.,0.],[0.,0.,1.],[1.,0.,0.],[1.,0.,1.],
    ])
    # Two native components intentionally share some same query coordinates,
    # but each has its OWN field truth. Global nearest UDF is insufficient.
    udf=torch.tensor([
        [0.,1.,1.,2.],
        [1.,0.,2.,1.],
    ])
    mask=torch.ones_like(udf,dtype=torch.bool)
    kinds=torch.tensor([KIND_OPEN,KIND_CLOSED],dtype=torch.long)
    teacher=SourceNativeComponentUDFFitV1(
        source_asset_sha256="a"*64,
        source_qualification_key="native-component-source-a",
        original_query_xyz=x,
        source_component_udf=udf,
        source_component_mask=mask,
        source_component_open_kind=kinds,
        native_component_inventory_sha256="b"*64,
        independently_complete_inventory=True,
    )
    receipt={
        "schema":TRAIN_SCHEMA,
        "admission_status":"QUALIFIED_NATIVE_COMPONENT_UDF_AND_OPENNESS",
        "source_asset_sha256":teacher.source_asset_sha256,
        "native_component_inventory_sha256":teacher.native_component_inventory_sha256,
        "original_query_xyz_sha256":tensor_sha256_v1(x),
        "source_component_udf_sha256":tensor_sha256_v1(udf),
        "source_component_mask_sha256":tensor_sha256_v1(mask),
        "source_component_open_kind_sha256":tensor_sha256_v1(kinds),
        "independently_complete_inventory":True,
        "source_rights_independently_verified":True,
        "generated_views_as_geometry_teacher":False,
        "student_slot_ids_as_source_truth":False,
    }
    return teacher,receipt


def _predictions():
    # Student slots: 0 is native CLOSED, 1 is extra UNKNOWN, 2 is OPEN.
    distances=torch.tensor([[
        [1.,9.,0.],
        [0.,9.,1.],
        [2.,9.,1.],
        [1.,9.,2.],
    ]],requires_grad=True)
    kind_logits=torch.tensor([[
        [9.,0.,0.],
        [0.,0.,9.],
        [0.,9.,0.],
    ]],requires_grad=True)
    return distances,kind_logits


def test_native_component_matching_is_permutation_invariant_and_differentiable():
    teacher,receipt=_teacher()
    dist,kind=_predictions()
    a=source_native_component_udf_objective_v1(
        predicted_distances=dist,predicted_kind_logits=kind,
        teacher=teacher,
        trusted_source_registry={teacher.source_qualification_key:receipt},
        source_query_xyz=teacher.original_query_xyz,
    )
    assert a["matched_predicted_slots"]==(2,0)
    assert torch.isfinite(a["total"])
    assert float(a["field"].detach())<1.e-9
    assert float(a["kind"].detach())<.001
    a["total"].backward()
    assert dist.grad is not None and torch.isfinite(dist.grad).all()
    assert kind.grad is not None and torch.isfinite(kind.grad).all()

    order=[2,1,0]
    b=source_native_component_udf_objective_v1(
        predicted_distances=dist.detach()[:,:,order],
        predicted_kind_logits=kind.detach()[:,order],
        teacher=teacher,
        trusted_source_registry={teacher.source_qualification_key:receipt},
        source_query_xyz=teacher.original_query_xyz,
    )
    assert b["matched_predicted_slots"]==(0,2)
    assert torch.allclose(a["total"].detach(),b["total"],atol=1.e-6)


def test_missing_native_component_receipt_and_fake_source_slot_match_fail():
    teacher,receipt=_teacher()
    dist,kind=_predictions()
    with pytest.raises(ValueError,match="TRUSTED_SOURCE_RECEIPT_REQUIRED"):
        source_native_component_udf_objective_v1(
            predicted_distances=dist,predicted_kind_logits=kind,
            teacher=teacher,trusted_source_registry={},
            source_query_xyz=teacher.original_query_xyz,
        )
    corrupt=dict(receipt)
    corrupt["student_slot_ids_as_source_truth"]=True
    with pytest.raises(ValueError,match="TRUSTED_SOURCE_RECEIPT_REQUIRED"):
        source_native_component_udf_objective_v1(
            predicted_distances=dist,predicted_kind_logits=kind,
            teacher=teacher,trusted_source_registry={
                teacher.source_qualification_key:corrupt,
            },
            source_query_xyz=teacher.original_query_xyz,
        )


def test_no_unqualified_negative_open_field_and_incomplete_component():
    teacher,receipt=_teacher()
    dist,kind=_predictions()
    bad=dist.detach().clone()
    bad[0,0,2]=-.01
    with pytest.raises(ValueError,match="TRAIN_DOMAIN"):
        source_native_component_udf_objective_v1(
            predicted_distances=bad,predicted_kind_logits=kind,
            teacher=teacher,trusted_source_registry={
                teacher.source_qualification_key:receipt,
            },
            source_query_xyz=teacher.original_query_xyz,
        )
    mask=teacher.source_component_mask.clone()
    mask[0]=False
    with pytest.raises(ValueError,match="NATIVE_TEACHER_DOMAIN"):
        replace(teacher,source_component_mask=mask).validate()


def test_shared_h_component_head_has_positive_distances_and_live_gradients():
    torch.manual_seed(13)
    head=RSSvNextComponentConditionedUDFV1(16,max_query_slot_pairs=128)
    h=torch.randn(1,9,16,requires_grad=True)
    slots=torch.randn(1,3,16,requires_grad=True)
    all_dist=head.query_all(h,slots)
    kinds=head.topology_kind_logits(slots)
    assert all_dist.shape==(1,9,3)
    assert kinds.shape==(1,3,3)
    assert bool((all_dist>=0).all())
    assert torch.allclose(all_dist[:,:,2],
                          head.query_slot(h,slots,slot_index=2),
                          atol=1.e-6)
    (all_dist.mean()+kinds.mean()).backward()
    assert h.grad is not None and h.grad.abs().sum()>0
    assert slots.grad is not None and slots.grad.abs().sum()>0


def test_source_incomplete_inventory_does_not_force_unmatched_slot_unknown():
    teacher,receipt=_teacher()
    teacher=replace(teacher,independently_complete_inventory=False)
    receipt["independently_complete_inventory"]=False
    dist,kind=_predictions()
    out=source_native_component_udf_objective_v1(
        predicted_distances=dist,predicted_kind_logits=kind,
        teacher=teacher,trusted_source_registry={
            teacher.source_qualification_key:receipt,
        },source_query_xyz=teacher.original_query_xyz,
    )
    assert float(out["unmatched_unknown"])==0.


def test_unified_model_component_udf_requeries_same_H_and_receives_gradients():
    from models.rss_vnext import (
        RSSvNextConfig,RSSvNextPatchEncoderForTests,
        RSSvNextUnifiedCharacterModel,
    )
    torch.manual_seed(17)
    torch.set_num_threads(1)
    cfg=RSSvNextConfig.tiny_for_test()
    net=RSSvNextUnifiedCharacterModel(
        cfg,vision_encoder=RSSvNextPatchEncoderForTests(
            cfg.iris.image_token_dim,
        ),
    )
    rgba=torch.rand(1,2,4,8,8)
    cams=torch.zeros((1,2,13))
    cams[...,3]=cams[...,7]=cams[...,11]=cams[...,12]=1.
    state,_,_=net.encode_observations(rgba,cams)
    pred=net.query_component_unsigned_distance(
        state,state.query_xyz,slot_index=1,
    )
    exact=net.component_udf.query_slot(
        state.local_features,state.region_slot_features,
        slot_index=1,
    )
    torch.testing.assert_close(pred,exact,atol=1.e-6,rtol=1.e-6)
    kinds=net.query_component_topology_kind_logits(state)
    assert kinds.shape==(1,cfg.region_slots,3)
    (pred.mean()+kinds.mean()).backward()
    assert net.component_udf.distance[0].weight.grad is not None
    assert net.component_udf.distance[0].weight.grad.abs().sum()>0
    assert net.semantic_structure.learned_region_queries.grad is not None
    assert net.semantic_structure.learned_region_queries.grad.abs().sum()>0
    assert net.iris.scene.scene_queries.grad is not None
    assert net.iris.scene.scene_queries.grad.abs().sum()>0
