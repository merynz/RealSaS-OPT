from __future__ import annotations

"""Product six-view training with independently qualified source-native open UDF.

Synthetic native teacher + tiny H is a correctness regression ONLY. It is
not evidence of qualified real corpus, production CUDA UDF extraction or
an animation-ready open garment.
"""

from dataclasses import replace

import pytest
import torch

from models.rss_vnext.component_conditioned_udf_v1 import (
    SourceNativeComponentUDFFitV1,KIND_OPEN,KIND_CLOSED,TRAIN_SCHEMA,
)
from models.rss_vnext.native_face_layer_truth_v1 import tensor_sha256_v1
from models.rss_vnext.training_engine import (
    RSSvNextStructuralTrainer,RSSvNextTrainingWeights,
    source_surface_tensor_sha256,
)
from models.rss_vnext.training_seeded_tessa_v1 import SeededTESSALossWeightsV1
from tests.models.test_rss_vnext_product_seeded_training_route_v1 import (
    _scenario,
)


def _scenario_udf(monkeypatch):
    net,batch,make_seeded,native,generated=_scenario(monkeypatch)
    x=torch.tensor([
        [0.,0.,0.],[0.,.4,.3],[.3,-.2,0.],[.2,.2,.4],
    ],dtype=batch.rgba.dtype)
    udf=torch.tensor([
        [.0,.3,.45,.2],
        [.5,.2,.1,.4],
    ],dtype=batch.rgba.dtype)
    valid=torch.ones_like(udf,dtype=torch.bool)
    kind=torch.tensor([KIND_OPEN,KIND_CLOSED],dtype=torch.long)
    teacher=SourceNativeComponentUDFFitV1(
        source_asset_sha256=batch.seeded_tessa_source_teacher.source_asset_sha256,
        source_qualification_key="independent-open-component-teacher",
        original_query_xyz=x,
        source_component_udf=udf,
        source_component_mask=valid,
        source_component_open_kind=kind,
        native_component_inventory_sha256="5"*64,
        independently_complete_inventory=True,
    )
    manifest="f"*64
    receipt={
        "schema":TRAIN_SCHEMA,
        "admission_status":"QUALIFIED_NATIVE_COMPONENT_UDF_AND_OPENNESS",
        "source_asset_sha256":teacher.source_asset_sha256,
        "native_component_inventory_sha256":teacher.native_component_inventory_sha256,
        "original_query_xyz_sha256":tensor_sha256_v1(x),
        "source_component_udf_sha256":tensor_sha256_v1(udf),
        "source_component_mask_sha256":tensor_sha256_v1(valid),
        "source_component_open_kind_sha256":tensor_sha256_v1(kind),
        "independently_complete_inventory":True,
        "source_rights_independently_verified":True,
        "generated_views_as_geometry_teacher":False,
        "student_slot_ids_as_source_truth":False,
        "dataset_manifest_sha256":manifest,
        "source_qualification_key":batch.source_qualification_key,
        "observed_rgba_sha256":source_surface_tensor_sha256(batch.rgba),
        "observed_camera_sha256":source_surface_tensor_sha256(batch.camera_features),
        "source_binding_sha256":batch.product_source_binding_sha256,
        "combined_generation_checkpoint_sha256":
            batch.product_generation_checkpoint_sha256,
        "source_model_observation_count":2,
        "source_native_component_asset_sha256":teacher.source_asset_sha256,
    }
    component_registry={teacher.source_qualification_key:receipt}
    component_only=replace(
        batch,component_udf_source_teacher=teacher,
        seeded_tessa_source_teacher=None,
        seeded_tessa_source_face_budget=0,
        flexicubes_extractor=None,
        flexicubes_layer=None,
        flexicubes_reference_surface_xyz=None,
        flexicubes_signed_source_admission_key="",
    )
    optimizer=torch.optim.AdamW(net.parameters(),lr=1.e-3)
    def make(*,profile="PRODUCT_COMPONENT_UDF_V1",reg=component_registry,
             weight=.8,extra={}):
        args={
            "dataset_manifest_sha256":manifest,
            "scene_conditioning_policy":"PRODUCT_GENERATED_SIX_VIEW",
            "training_profile":profile,
            "trusted_generated_conditioning_registry":generated,
            "component_udf_weight":weight,
            "trusted_component_udf_registry":reg,
        }
        args.update(extra)
        return RSSvNextStructuralTrainer(net,optimizer,**args)
    return net,component_only,make,component_registry,batch,make_seeded,native,generated


def test_real_product_sixview_component_udf_phase_backprops_same_H(monkeypatch,tmp_path):
    net,batch,make,*_=_scenario_udf(monkeypatch)
    trainer=make()
    before=net.component_udf.distance[0].weight.detach().clone()
    output=trainer.train_step(batch)
    assert output.component_udf_loss is not None
    assert output.component_udf_loss.isfinite()
    assert output.eligible_component_udf_source_components==2
    assert output.seeded_tessa_loss is None
    assert output.gradient_norm.isfinite()
    assert not torch.equal(
        before,net.component_udf.distance[0].weight.detach(),
    )
    assert trainer.global_step==1
    where=tmp_path/"component.pt"
    trainer.save_checkpoint(where,sampler_cursor=9)
    assert trainer.load_checkpoint(where)==9
    assert trainer.global_step==1


def test_component_stage_rejects_student_truth_and_irrelevant_source(monkeypatch):
    _,batch,make,*_=_scenario_udf(monkeypatch)
    trainer=make()
    with pytest.raises(ValueError,match="UNCONSUMED_TEACHER"):
        trainer.objective(replace(
            batch,topology_teacher_tokens=torch.ones(1,3,dtype=torch.long),
        ))
    with pytest.raises(ValueError,match="UNCONSUMED_TEACHER"):
        trainer.objective(replace(
            batch,source_udf_target=torch.zeros((1,4)),
        ))
    with pytest.raises(ValueError,match="TRUSTED_SOURCE_RECEIPT_REQUIRED"):
        broken=replace(
            batch.component_udf_source_teacher,
            source_component_udf=torch.zeros_like(
                batch.component_udf_source_teacher.source_component_udf,
            ),
        )
        trainer.objective(replace(batch,component_udf_source_teacher=broken))


def test_component_product_checkpoint_pins_registry_source_and_weight(monkeypatch,tmp_path):
    _,batch,make,registry,*_=_scenario_udf(monkeypatch)
    trainer=make()
    checkpoint=tmp_path/"component.pt"
    trainer.save_checkpoint(checkpoint,sampler_cursor=0)
    changed={k:dict(v) for k,v in registry.items()}
    row=next(iter(changed.values()))
    row["source_rights_independently_verified"]=False
    other=make(reg=changed)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        other.load_checkpoint(checkpoint)
    weight_changed=make(weight=.2)
    with pytest.raises(ValueError,match="OBJECTIVE_POLICY_DRIFT"):
        weight_changed.load_checkpoint(checkpoint)
    with pytest.raises(ValueError,match="PRODUCT_COMPONENT_UDF_PHASE_POLICY"):
        make(weight=0.)
    with pytest.raises(ValueError,match="PRODUCT_COMPONENT_UDF_PHASE_POLICY"):
        make(weight=float("nan"))


def test_joint_seeded_tessa_and_component_udf_both_get_gradient(monkeypatch):
    net,component,make,reg,signed,make_seeded,native,generated=(
        _scenario_udf(monkeypatch)
    )
    seeded_ref=make_seeded()
    optimizer=torch.optim.AdamW(net.parameters(),lr=1.e-3)
    trainer=RSSvNextStructuralTrainer(
        net,optimizer,dataset_manifest_sha256="f"*64,
        scene_conditioning_policy="PRODUCT_GENERATED_SIX_VIEW",
        training_profile="PRODUCT_SEEDED_IRIS_TESSA_V1",
        seeded_tessa_weights=SeededTESSALossWeightsV1(),
        trusted_seeded_tessa_registry=native,
        trusted_signed_source_registry={
            k:dict(v) for k,v in
            seeded_ref.trusted_signed_source_registry.items()
        },
        trusted_generated_conditioning_registry=generated,
        component_udf_weight=.8,
        trusted_component_udf_registry=reg,
    )
    batch=replace(signed,component_udf_source_teacher=
                  component.component_udf_source_teacher)
    before_t=net.tessa_seeded_head[-1].weight.detach().clone()
    before_c=net.component_udf.distance[0].weight.detach().clone()
    result=trainer.train_step(batch)
    assert result.seeded_tessa_loss is not None
    assert result.component_udf_loss is not None
    assert torch.isfinite(result.total_loss)
    assert result.eligible_component_udf_source_components==2
    assert not torch.equal(before_t,net.tessa_seeded_head[-1].weight.detach())
    assert not torch.equal(before_c,net.component_udf.distance[0].weight.detach())
