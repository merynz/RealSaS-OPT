from __future__ import annotations

"""CPU smoke regression of real train/optimizer/checkpoint/sampler ABIs.

The 64px fixtures are NOT a product-quality or 300-source training claim.
"""

import torch
import pytest
from torch import nn
from types import SimpleNamespace

from tests.models.test_rss_vnext_sdxl_mvadapter_stageab_training_v2 import (
    VAE,Scheduler,FakeSDXLUnet,FakeMVUnet,FakeControl,
)
from models.rss_vnext.paired_sdxl_lora_prior_v1 import (
    RSSvNextTwoImageConditionerV1,
)
from models.rss_vnext.realsas_sdxl_stage_a_source_training_v2 import (
    SDXLCanonicalAlphaHeadV2,SDXLSourceStageABatchV2,
)
from models.rss_vnext.realsas_mvadapter_source_training_v2 import (
    MVAdapterSourceStageBBatchV2,
)
from models.rss_vnext.realsas_mvadapter_dual_reference_v2 import (
    MVAdapterRGBNormalTaskHeadsV2,
)
from models.rss_vnext.realsas_mvadapter_camera_bridge_v2 import (
    fixed_sdxl_mvadapter_two_plus_six_cameras_v2,
)
from models.rss_vnext.realsas_sdxl_mvadapter_stageab_trainer_v2 import (
    SourceSDXLStageABOptimizerV2,
)
from models.rss_vnext.realsas_sdxl_stage_a_sampler_v2 import (
    load_exact_stage_a_task_weights_v2,
    sample_trained_stage_a_canonical_v2,
)
from models.rss_vnext.realsas_mvadapter_stage_b_sampler_v2 import (
    load_exact_stage_b_task_checkpoint_v2,
    sample_stage_b_paired_rgb_normal_v2,
)


def _source_context():
    return torch.zeros(1,2,2048),torch.zeros(1,1280),torch.zeros(1,6)


def _source_registry(stage,batch):
    from models.rss_vnext.realsas_stageab_source_court_v2 import (
        SCHEMA,AUTHORITY,source_stageab_tensor_digests_v2,
    )
    key=batch.source_render_qualification_key
    return {key:{
        "schema":SCHEMA,
        "admission_status":AUTHORITY,
        "stage":stage,
        "dataset_manifest_sha256":"a"*64,
        "source_render_qualification_key":key,
        "generated_model_prediction_as_teacher":False,
        "source_3d_renders_independently_audited":True,
        "source_asset_sha256":"a"*64,
        "original_two_rgba_sha256":"b"*64,
        "native_source_receipt_sha256":"c"*64,
        **({"frozen_source_encoder_sha256":"e"*64}
           if stage=="A_CANONICAL_SDXL" else {}),
        **source_stageab_tensor_digests_v2(stage,batch),
    }}


def _stage_a():
    conditioner=RSSvNextTwoImageConditionerV1(
        source_dim=8,pose_dim=4,hidden_dim=16,output_dim=2048,
        queries=2,heads=4,
    )
    return FakeSDXLUnet(),conditioner,SDXLCanonicalAlphaHeadV2()


def _batch_a():
    canon,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    prompt,pooled,timeids=_source_context()
    return SDXLSourceStageABatchV2(
        source_tokens=torch.randn(1,2,4,8),
        source_token_valid=torch.ones(1,2,4,dtype=torch.bool),
        target_canonical_rgba=torch.rand(1,2,4,64,64),
        target_valid=torch.ones(1,2,1,64,64),
        target_cameras=canon,
        canonical_pose_queries=torch.zeros(1,2,4),
        prompt_embeds=prompt,pooled_text_embeds=pooled,time_ids=timeids,
        source_render_qualification_key="synthetic-regression",
    )


def _official_test_scheduler():
    # Synthetic regression only; production constructs the exact pinned
    # upstream DDPM interpolated ShiftSNR schedule (scale 8).
    from models.rss_vnext.realsas_mvadapter_shift_snr_training_v2 import (
        schedule_alphas_sha256_v1,
    )
    result=Scheduler()
    result._rss_vnext_training_shift_mode="interpolated"
    result._rss_vnext_training_shift_scale=8.0
    result._rss_vnext_shift_snr_schedule_sha256=(
        schedule_alphas_sha256_v1(result)
    )
    return result


def _stage_b():
    vae=VAE()
    vae._rss_vnext_vae_ancestor_sha256="d"*64
    return (
        SimpleNamespace(unet=FakeMVUnet(),cond_encoder=FakeControl(),
                        vae=vae,scheduler=_official_test_scheduler()),
        MVAdapterRGBNormalTaskHeadsV2(),
    )


def _batch_b():
    canon,six,control=fixed_sdxl_mvadapter_two_plus_six_cameras_v2(
        1,device="cpu",dtype=torch.float32,image_size=64,
    )
    prompt,pooled,timeids=_source_context()
    return MVAdapterSourceStageBBatchV2(
        canonical_rgba=torch.rand(1,2,4,64,64),
        rgb=torch.rand(1,6,3,64,64),
        normal_encoded_rgb=torch.rand(1,6,3,64,64),
        alpha=torch.rand(1,6,1,64,64),
        valid=torch.ones(1,6,1,64,64),
        target_cameras=six,plucker_controls=control,
        prompt_embeds=prompt,pooled_text_embeds=pooled,time_ids=timeids,
        source_render_qualification_key="synthetic-regression",
    )


def test_stage_a_task_fit_checkpoint_resume_and_two_canonical_sampler(tmp_path):
    torch.manual_seed(30)
    unet,conditioner,head=_stage_a()
    vae=VAE()
    vae._rss_vnext_vae_ancestor_sha256="d"*64
    batch=_batch_a()
    optimizer=SourceSDXLStageABOptimizerV2(
        stage="A_CANONICAL_SDXL",unet=unet,vae=vae,
        noise_scheduler=Scheduler(),
        source_manifest_sha256="a"*64,
        pretrained_source_sha256="b"*64,
        source_vae_sha256="d"*64,
        trusted_source_registry=_source_registry("A_CANONICAL_SDXL",batch),
        conditioner=conditioner,task_a_alpha=head,
        warmup_steps=2,
    )
    first=optimizer.train_step(batch)
    assert first.step==1
    assert set(first.gradient_norms)=={
        "sdxl_lora","paired_conditioner","canonical_alpha",
    }
    resume=tmp_path/"a-resume.pt"
    optimizer.save_resume_checkpoint(resume)
    ckpt=optimizer.export_trained_task_weights(tmp_path/"trained-a.pt")
    assert ckpt.trained_steps==1
    u2,c2,h2=_stage_a()
    load_exact_stage_a_task_weights_v2(
        unet=u2,conditioner=c2,alpha_head=h2,receipt=ckpt,
    )
    predicted=sample_trained_stage_a_canonical_v2(
        unet=u2,vae=vae,scheduler=Scheduler(),
        conditioner=c2,alpha_head=h2,receipt=ckpt,
        observed_tokens=batch.source_tokens,
        observed_token_valid=batch.source_token_valid,
        target_cameras=batch.target_cameras,
        canonical_pose_queries=batch.canonical_pose_queries,
        prompt_embeds=batch.prompt_embeds,
        pooled_text_embeds=batch.pooled_text_embeds,
        time_ids=batch.time_ids,height=64,width=64,steps=2,seed=5,
    )
    assert predicted.rgba.shape==(1,2,4,64,64)
    assert not predicted.original_observation_authority
    assert not predicted.source_teacher_authority
    with pytest.raises(RuntimeError,match="UNTRAINED_MODEL_NOT_ADMITTED"):
        sample_trained_stage_a_canonical_v2(
            unet=FakeSDXLUnet(),vae=vae,scheduler=Scheduler(),
            conditioner=c2,alpha_head=h2,receipt=ckpt,
            observed_tokens=batch.source_tokens,
            observed_token_valid=batch.source_token_valid,
            target_cameras=batch.target_cameras,
            canonical_pose_queries=batch.canonical_pose_queries,
            prompt_embeds=batch.prompt_embeds,
            pooled_text_embeds=batch.pooled_text_embeds,
            time_ids=batch.time_ids,height=64,width=64,steps=2,seed=5,
        )
    new_u,new_c,new_h=_stage_a()
    resumed=SourceSDXLStageABOptimizerV2(
        stage="A_CANONICAL_SDXL",unet=new_u,vae=vae,
        noise_scheduler=Scheduler(),
        source_manifest_sha256="a"*64,pretrained_source_sha256="b"*64,
        source_vae_sha256="d"*64,
        trusted_source_registry=_source_registry("A_CANONICAL_SDXL",batch),
        conditioner=new_c,task_a_alpha=new_h,warmup_steps=2,
    )
    resumed.load_resume_checkpoint(resume)
    assert resumed.step==1


def test_stage_b_task_fit_exact_native_weights_and_six_rgb_normals(tmp_path):
    torch.manual_seed(31)
    pipeline,heads=_stage_b()
    batch=_batch_b()
    optimizer=SourceSDXLStageABOptimizerV2(
        stage="B_PAIRED_RGB_NORMAL_MVADAPTER",
        unet=pipeline.unet,vae=pipeline.vae,
        noise_scheduler=_official_test_scheduler(),
        source_manifest_sha256="a"*64,
        pretrained_source_sha256="b"*64,
        source_vae_sha256="d"*64,
        pretrained_mv_adapter_sha256="c"*64,
        trusted_source_registry=_source_registry("B_PAIRED_RGB_NORMAL_MVADAPTER",batch),
        mv_pipeline=pipeline,task_b_heads=heads,
        warmup_steps=2,
    )
    result=optimizer.train_step(batch)
    assert result.step==1
    assert result.normal_epsilon_loss is not None
    assert set(result.gradient_norms)=={
        "mv_adapter_row","mv_adapter_reference",
        "native_camera_encoder","rgb_normal_task_heads",
    }
    resume=tmp_path/"b-resume.pt"
    optimizer.save_resume_checkpoint(resume)
    receipt=optimizer.export_trained_task_weights(tmp_path/"trained-b.pt")
    other,other_heads=_stage_b()
    load_exact_stage_b_task_checkpoint_v2(
        pipe=other,task_heads=other_heads,receipt=receipt,
    )
    sampled=sample_stage_b_paired_rgb_normal_v2(
        pipe=other,task_heads=other_heads,receipt=receipt,
        canonical_rgba=batch.canonical_rgba,
        plucker_controls=batch.plucker_controls,
        prompt_embeds=batch.prompt_embeds,
        pooled_text_embeds=batch.pooled_text_embeds,
        time_ids=batch.time_ids,steps=2,seed=7,
    )
    assert sampled.rgba.shape==(1,6,4,64,64)
    assert sampled.normals.shape==(1,6,3,64,64)
    assert sampled.canonical_reference_count==2
    assert not sampled.observation_authority
    assert not sampled.source_teacher_authority
    with pytest.raises(RuntimeError,match="EXACT_TASK_WEIGHTS_NOT_LOADED"):
        sample_stage_b_paired_rgb_normal_v2(
            pipe=pipeline,task_heads=heads,receipt=receipt,
            canonical_rgba=batch.canonical_rgba,
            plucker_controls=batch.plucker_controls,
            prompt_embeds=batch.prompt_embeds,
            pooled_text_embeds=batch.pooled_text_embeds,
            time_ids=batch.time_ids,steps=2,seed=7,
        )
    other2,heads2=_stage_b()
    restored=SourceSDXLStageABOptimizerV2(
        stage="B_PAIRED_RGB_NORMAL_MVADAPTER",
        unet=other2.unet,vae=other2.vae,
        noise_scheduler=_official_test_scheduler(),
        source_manifest_sha256="a"*64,
        pretrained_source_sha256="b"*64,
        source_vae_sha256="d"*64,
        pretrained_mv_adapter_sha256="c"*64,
        trusted_source_registry=_source_registry("B_PAIRED_RGB_NORMAL_MVADAPTER",batch),
        mv_pipeline=other2,task_b_heads=heads2,warmup_steps=2,
    )
    restored.load_resume_checkpoint(resume)
    assert restored.step==1
