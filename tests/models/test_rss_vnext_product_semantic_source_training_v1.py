from __future__ import annotations

"""Actual product Semantic training with distinct source evidence authorities.

The native labels here are controlled fixtures for integration tests only.
They are NOT proof of source corpus licensing, role coverage or G3 motion.
"""
from dataclasses import replace

import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,source_surface_tensor_sha256,
)
from models.rss_vnext.product_semantic_supervision_v1 import (
    ProductSemanticWeightsV1,
)
from models.rss_vnext.semantic_structure import (
    TYPED_REGION_RELATION_CHANNELS_V1,
)
from models.rss_vnext.training_seeded_tessa_v1 import SeededTESSALossWeightsV1
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import _scenario


def _fixture(monkeypatch):
    net,batch,make_seeded,native,generated=_scenario(monkeypatch)
    baseline=make_seeded()
    state=net.encode_generated_multiview(
        batch.rgba,None,
        generated_rgba=batch.product_generated_rgba,
        generated_cameras=batch.product_generated_cameras,
        generated_normals=batch.product_generated_normals,
        generation_checkpoint_sha256=batch.product_generation_checkpoint_sha256,
        source_binding_sha256=batch.product_source_binding_sha256,
    )
    digest=source_surface_tensor_sha256
    q=state.query_xyz.shape[1]
    assert q>=3
    patch=torch.zeros_like(state.source_semantic_patch_logits)
    patch[...,0]=1.
    patch_mask=torch.zeros_like(patch,dtype=torch.bool)
    patch_mask[...,0]=True
    roles=torch.zeros((1,q),dtype=torch.long)
    role_mask=torch.zeros((1,q),dtype=torch.bool)
    role_mask[0,:3]=True
    pairs=torch.tensor([[[0,1],[1,2]]],dtype=torch.long)
    pair_values=torch.tensor([[1.,0.]])
    pair_mask=torch.ones_like(pair_values,dtype=torch.bool)
    typed_pairs=torch.tensor([[[0,1]]],dtype=torch.long)
    typed_values=torch.tensor([[[1.,0.,0.]]])
    typed_mask=torch.tensor([[[True,False,False]]])
    common={
        "dataset_manifest_sha256":"f"*64,
        "source_qualification_key":batch.source_qualification_key,
        "source_asset_sha256":batch.seeded_tessa_source_teacher.source_asset_sha256,
        "observed_rgba_sha256":digest(batch.rgba),
        "original_camera_sha256":digest(batch.camera_features),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "generated_views_as_source_truth":False,
    }
    patch_key="product-source-patch-axes"
    role_key="product-source-3d-roles"
    region_key="product-source-native-region-pairs"
    typed_key="product-source-typed-relations"
    registers={
        "patch":{patch_key:{
            **common,
            "admission_status":"QUALIFIED_SOURCE_SEMANTIC_PATCH_AXES",
            "semantic_patch_target_sha256":digest(patch),
            "semantic_patch_mask_sha256":digest(patch_mask),
            "supervision_semantics":
                "ORIGINAL_TWO_VIEW_MASKED_PATCH_AXES_NOT_3D_DRAWABLE",
        }},
        "role":{role_key:{
            **common,"admission_status":"QUALIFIED_SOURCE_SURFACE_ROLE",
            "role_schema_id":net.cfg.semantic_role_schema_id,
            "query_xyz_sha256":digest(state.query_xyz),
            "role_indices_sha256":digest(roles),
            "role_mask_sha256":digest(role_mask),
        }},
        "region":{region_key:{
            **common,"admission_status":"QUALIFIED_SOURCE_REGION_PAIR",
            "query_xyz_sha256":digest(state.query_xyz),
            "pair_indices_sha256":digest(pairs),
            "pair_targets_sha256":digest(pair_values),
            "pair_mask_sha256":digest(pair_mask),
            "supervision_semantics":"SOURCE_REGION_CO_MEMBERSHIP_NOT_DRAWABLE",
        }},
        "typed":{typed_key:{
            **common,
            "admission_status":"QUALIFIED_SOURCE_TYPED_REGION_RELATIONS",
            "query_xyz_sha256":digest(state.query_xyz),
            "pair_indices_sha256":digest(typed_pairs),
            "typed_targets_sha256":digest(typed_values),
            "typed_mask_sha256":digest(typed_mask),
            "supervision_semantics":
                "SOURCE_QUERY_PAIR_TYPED_DRAWABLE_CUT_DIRECTED_ATTACHMENT_MASKED_V1",
            "channel_source_authority":{
                name:("QUALIFIED_SOURCE_EDIT_DRAWABLE_CONSEQUENCE"
                      if i==0 else "UNAVAILABLE")
                for i,name in enumerate(TYPED_REGION_RELATION_CHANNELS_V1)
            },
        }},
    }
    labelled=replace(
        batch,
        semantic_patch_target=patch,
        semantic_patch_mask=patch_mask,
        semantic_qualification_key=patch_key,
        source_surface_role_indices=roles,
        source_surface_role_mask=role_mask,
        source_surface_role_admission_key=role_key,
        source_region_pair_indices=pairs,
        source_region_pair_is_same=pair_values,
        source_region_pair_mask=pair_mask,
        source_region_pair_admission_key=region_key,
        source_typed_relation_pair_indices=typed_pairs,
        source_typed_relation_target=typed_values,
        source_typed_relation_mask=typed_mask,
        source_typed_relation_admission_key=typed_key,
    )
    weights=ProductSemanticWeightsV1(
        patch_axes=.3,patch_risk=.1,surface_roles=.4,
        region_pairs=.2,typed_relations=.1,
    )
    def make(*,w=weights,registries=registers):
        optimizer=torch.optim.AdamW(net.parameters(),lr=1.e-3)
        return RSSvNextStructuralTrainer(
            net,optimizer,
            dataset_manifest_sha256="f"*64,
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
            seeded_tessa_weights=replace(
                SeededTESSALossWeightsV1(),boundary=0.,
            ),
            trusted_seeded_tessa_registry=native,
            trusted_generated_conditioning_registry=generated,
            trusted_signed_source_registry={
                k:dict(v) for k,v in baseline.trusted_signed_source_registry.items()
            },
            trusted_semantic_patch_registry=registries["patch"],
            trusted_surface_role_registry=registries["role"],
            trusted_region_pair_registry=registries["region"],
            trusted_typed_relation_registry=registries["typed"],
            product_semantic_weights=w,
        )
    return net,labelled,make,registers,weights


def _has_gradient(module):
    return any(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and float(p.grad.abs().sum())>0
        for p in module.parameters() if p.requires_grad
    )


def test_product_semantic_all_source_targets_train_same_H_and_optimizer(monkeypatch,tmp_path):
    net,batch,make,_,_=_fixture(monkeypatch)
    trainer=make()
    result=trainer.train_step(batch)
    assert result.seeded_tessa_loss is not None
    assert result.product_semantic_loss is not None
    assert set(result.product_semantic_loss_parts)=={
        "patch","patch_risk","roles","region","typed",
    }
    assert result.eligible_semantic_patch_labels>0
    assert result.eligible_surface_role_points==3
    assert result.eligible_region_pairs==2
    assert result.eligible_typed_relation_channels[0]==1
    assert _has_gradient(net.early_semantics)
    assert _has_gradient(net.presentation_role_head)
    assert _has_gradient(net.semantic_structure)
    assert _has_gradient(net.semantic)
    assert _has_gradient(net.iris)
    assert _has_gradient(net.tessa_seeded_head)
    assert trainer.global_step==1
    path=tmp_path/"semantic.pt"
    trainer.save_checkpoint(path,sampler_cursor=7)
    assert trainer.load_checkpoint(path)==7


def test_semantic_patch_and_native_role_are_not_interchangeable_truth(monkeypatch):
    _,batch,make,*_=_fixture(monkeypatch)
    trainer=make()
    with pytest.raises(ValueError,match="ROLE_RECEIPT_DRIFT"):
        trainer.objective(replace(
            batch,source_surface_role_indices=torch.ones_like(
                batch.source_surface_role_indices,
            ),
        ))
    with pytest.raises(ValueError,match="PATCH_RECEIPT_DRIFT"):
        forged=batch.semantic_patch_target.clone()
        forged[...,0]=0.
        trainer.objective(replace(batch,semantic_patch_target=forged))


def test_region_and_typed_channels_cannot_be_trained_from_mismatched_authority(monkeypatch):
    _,batch,make,registries,_=_fixture(monkeypatch)
    changed={k:{key:dict(value) for key,value in entries.items()}
             for k,entries in registries.items()}
    row=next(iter(changed["typed"].values()))
    row["channel_source_authority"]={
        name:"SOURCE_PART_ID" for name in TYPED_REGION_RELATION_CHANNELS_V1
    }
    with pytest.raises(ValueError,match="TYPED_AUTHORITY_MISSING"):
        make(registries=changed).objective(batch)
    with pytest.raises(ValueError,match="REGION_RECEIPT_DRIFT"):
        make().objective(replace(
            batch,source_region_pair_is_same=torch.zeros_like(
                batch.source_region_pair_is_same,
            ),
        ))


def test_semantic_weight_family_is_opt_in_and_checkpoint_sealed(monkeypatch,tmp_path):
    _,batch,make,registries,weights=_fixture(monkeypatch)
    active=make()
    save=tmp_path/"semantic.pt"
    active.save_checkpoint(save,sampler_cursor=0)
    changed=make(w=replace(weights,patch_axes=.7))
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        changed.load_checkpoint(save)
    empty=replace(
        batch,semantic_patch_target=None,semantic_patch_mask=None,
        semantic_qualification_key="",
        source_surface_role_indices=None,source_surface_role_mask=None,
        source_surface_role_admission_key="",
        source_region_pair_indices=None,source_region_pair_is_same=None,
        source_region_pair_mask=None,source_region_pair_admission_key="",
        source_typed_relation_pair_indices=None,source_typed_relation_target=None,
        source_typed_relation_mask=None,source_typed_relation_admission_key="",
    )
    with pytest.raises(ValueError,match="SEMANTIC_NO_QUALIFIED_TARGET"):
        make().objective(empty)
    with pytest.raises(ValueError,match="SEMANTIC_WEIGHT_POLICY"):
        ProductSemanticWeightsV1(region_pairs=float("nan")).validate()
    with pytest.raises(ValueError,match="PRODUCT_SEMANTIC_PHASE_POLICY"):
        make(w=replace(weights,surface_roles=.4),
             registries={**registries,"role":{}})
