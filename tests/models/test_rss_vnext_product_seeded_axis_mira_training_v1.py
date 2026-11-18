from __future__ import annotations

"""Exact product H(x), seeded M*, source native AXIS causal rig and MIRA codec.

Tiny native fixtures test trainability and receipts only, not real-dataset
animation-ready generalization or qualified GPU source training.
"""

from dataclasses import replace
import pytest
import torch

from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,
    axis_teacher_sequence_sha256_v6, frozen_mira_codec_state_sha256_v6,
    source_surface_tensor_sha256,
)
from models.rss_vnext.product_axis_mira_seeded_training_v1 import (
    ProductSeededAXISMIRAWeightsV1,
)
from models.rss_vnext.training_seeded_tessa_v1 import SeededTESSALossWeightsV1
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import _scenario
from tests.models.test_rss_vnext_early_semantics import (
    _source_qualified_axis_teacher_fixture,
)


def _rig_case(monkeypatch):
    net,batch,make_base,native,generated=_scenario(monkeypatch)
    net.mira.codec.requires_grad_(False)
    seq,original=_source_qualified_axis_teacher_fixture(net)
    key="native-axis-product-mstar"
    receipt={
        **original,
        "dataset_manifest_sha256":"f"*64,
        "source_qualification_key":batch.source_qualification_key,
        "source_asset_sha256":
            batch.seeded_tessa_source_teacher.source_asset_sha256,
        "observed_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
        "observed_camera_sha256":
            source_surface_tensor_sha256(batch.camera_features),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "source_teacher_is_loss_only":True,
        "mesh_context":"SAME_SEEDED_TESSA_MSTAR_NOT_LEGACY_TOKEN_MESH",
    }
    batch=replace(batch,axis_teacher_sequence=seq,axis_teacher_admission_key=key)
    old=make_base()
    registry={key:receipt}
    def make(*,weights=None,registry_override=None):
        if weights is None:
            weights=ProductSeededAXISMIRAWeightsV1(axis_program=1.)
        return RSSvNextStructuralTrainer(
            net,torch.optim.AdamW(net.parameters(),lr=.001),
            dataset_manifest_sha256="f"*64,
            training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
            scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
            axis_training_policy="PRODUCT_MESH_MATCHED_REQUIRED",
            seeded_tessa_weights=replace(
                SeededTESSALossWeightsV1(),boundary=0.,
            ),
            trusted_seeded_tessa_registry=native,
            trusted_generated_conditioning_registry=generated,
            trusted_signed_source_registry={
                k:dict(v) for k,v in old.trusted_signed_source_registry.items()
            },
            trusted_axis_teacher_registry=(
                registry if registry_override is None else registry_override
            ),
            product_axis_mira_weights=weights,
        )
    return net,batch,make,receipt,key


def test_same_product_H_and_TESSA_mstar_backprop_real_axis_rig(monkeypatch,tmp_path):
    net,batch,make,*_=_rig_case(monkeypatch)
    trainer=make()
    codec_before=frozen_mira_codec_state_sha256_v6(net)
    axis_before=net.axis.kind_head.weight.detach().clone()
    output=trainer.train_step(batch)
    assert output.seeded_tessa_loss is not None
    assert output.product_axis_mira_loss is not None
    assert output.axis_rig_skin_loss is not None
    assert output.product_axis_mira_parts["axis"] is not None
    assert output.eligible_axis_sequences==1
    assert torch.isfinite(output.total_loss)
    assert not torch.equal(net.axis.kind_head.weight.detach(),axis_before)
    assert net.axis.skin_code_head.weight.grad is not None
    assert net.axis.skin_code_head.weight.grad.abs().sum()>0
    assert net.mesh_face_to_control[-1].weight.grad is not None
    assert net.mesh_face_to_control[-1].weight.grad.abs().sum()>0
    assert net.tessa_seeded_head[-1].weight.grad is not None
    assert frozen_mira_codec_state_sha256_v6(net)==codec_before
    path=tmp_path/"product-axis.pt"
    trainer.save_checkpoint(path,sampler_cursor=3)
    assert trainer.load_checkpoint(path)==3


def test_untrusted_axis_source_and_fake_teacher_frozen_codec_fail_closed(monkeypatch):
    net,batch,make,receipt,key=_rig_case(monkeypatch)
    trainer=make()
    with pytest.raises(ValueError,match="AXIS_RECEIPT_DRIFT"):
        forged=replace(batch,axis_teacher_admission_key="unregistered")
        trainer.objective(forged)
    corrupt=dict(receipt)
    corrupt["source_asset_sha256"]="0"*64
    with pytest.raises(ValueError,match="AXIS_RECEIPT_DRIFT"):
        make(registry_override={key:corrupt}).objective(batch)
    net.mira.codec.requires_grad_(True)
    with pytest.raises(ValueError,match="MIRA_CODEC_NOT_FROZEN"):
        trainer.objective(batch)


def test_source_skin_requires_real_face_support_not_merely_vertex_count(monkeypatch):
    _,batch,make,*_=_rig_case(monkeypatch)
    trainer=make(weights=ProductSeededAXISMIRAWeightsV1(
        axis_program=1.,source_skin=1.,
    ))
    with pytest.raises(ValueError,match="AXIS_SOURCE_FACE_REQUIRED"):
        trainer.objective(batch)


def test_optimizer_policy_and_checkpoint_pin_axis_phase(monkeypatch,tmp_path):
    net,batch,make,*_=_rig_case(monkeypatch)
    weights=ProductSeededAXISMIRAWeightsV1(axis_program=1.)
    trainer=make(weights=weights)
    path=tmp_path/"axis.pt"
    trainer.save_checkpoint(path,sampler_cursor=2)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        make(weights=ProductSeededAXISMIRAWeightsV1(
            axis_program=.5,
        )).load_checkpoint(path)
    with pytest.raises(ValueError,match="PHASE_ORDER"):
        ProductSeededAXISMIRAWeightsV1(
            axis_program=0.,source_skin=1.,
        ).validate()
    with pytest.raises(ValueError,match="PHASE_ORDER"):
        ProductSeededAXISMIRAWeightsV1(
            axis_program=1.,dynamic_surface=1.,
        ).validate()
    with pytest.raises(ValueError,match="WEIGHT_POLICY"):
        ProductSeededAXISMIRAWeightsV1(
            axis_program=float("nan"),
        ).validate()
