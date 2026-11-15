from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from models.rss_vnext.realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    cleanroom_diffusion_noise_microstep_v1,
)
from models.rss_vnext.realsas_cleanroom_render_dataset_v1 import (
    RealSaSCleanroomRenderedSixViewDatasetV1,
    RealSaSCleanroomRenderedCanonicalPoseDatasetV1,
)
from models.rss_vnext.realsas_cleanroom_stage_batch_v1 import (
    prepare_rendered_two_source_diffusion_microstep_v1,
)
from tests.models.test_rss_vnext_realsas_cleanroom_render_dataset_v1 import (
    _materialize,
)
from tests.models.test_rss_vnext_realsas_cleanroom_diffusion_training_v1 import (
    _microcase,
)


class _FrozenRenderVision(nn.Module):
    """Tiny regression substitute, NOT the production DINOv2 backbone."""
    def __init__(self):
        super().__init__()
        self.patch=nn.Conv2d(4,8,8,stride=8)
        self.requires_grad_(False)
        self.eval()

    def forward(self, pair):
        assert pair.ndim==5 and pair.shape[1:3]==(2,4)
        b,v,c,h,w=pair.shape
        tokens=self.patch(pair.reshape(b*v,c,h,w))
        return tokens.flatten(2).transpose(1,2).reshape(b,2,4,8),None


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(2112)
    torch.set_num_threads(1)


def _case(tmp_path,stage):
    manifest=_materialize(tmp_path)
    reader=RealSaSCleanroomRenderedSixViewDatasetV1(
        manifest,stage=stage,
    )
    assert reader.stage==stage
    sample=reader[0]
    train=_microcase(stage)
    vision=_FrozenRenderVision()
    tensor_batch=prepare_rendered_two_source_diffusion_microstep_v1(
        sample,source_encoder=vision,conditioner=train["conditioner"],
    )
    return manifest,reader,sample,train,vision,tensor_batch


def test_canonical_stage_uses_two_arbitrary_pose_images_and_two_canonical_targets(tmp_path):
    manifest,reader,sample,train,vision,batch=_case(tmp_path,STAGE_CANONICAL)
    assert isinstance(reader,RealSaSCleanroomRenderedSixViewDatasetV1)
    assert sample["source_rgba"].shape==(2,4,16,16)
    assert sample["target_rgb"].shape==(2,3,16,16)
    assert sample["target_rgb_valid"].shape==(2,1,16,16)
    assert "target_normal_encoded_rgb" not in sample
    assert "reference_rgb" not in sample
    assert batch["source_tokens"].shape==(1,2,4,8)
    assert batch["rendered_target_rgb"].shape==(1,2,3,16,16)
    assert batch["target_cameras"].shape==(1,2,13)
    assert batch["target_modalities"].tolist()==[[0,0]]
    assert reader.corpus_report["canonical_pose_training_eligible_by_split"]["train"]==1

    # Source tokens must come from BOTH actual file-backed arbitrary-pose
    # renders; no canonical target may be recycled into the model input.
    changed={**sample,"source_rgba":sample["source_rgba"].clone()}
    changed["source_rgba"][1,0]+=0.3
    updated=prepare_rendered_two_source_diffusion_microstep_v1(
        changed,source_encoder=vision,conditioner=train["conditioner"],
    )
    assert torch.equal(batch["source_tokens"][:,0],updated["source_tokens"][:,0])
    assert not torch.allclose(
        batch["source_tokens"][:,1],updated["source_tokens"][:,1],
    )
    with torch.no_grad():
        first=train["conditioner"](
            batch["source_tokens"],batch["target_cameras"],batch["target_modalities"],
        )
        second=train["conditioner"](
            updated["source_tokens"],batch["target_cameras"],batch["target_modalities"],
        )
    assert not torch.allclose(first,second)
    train.update(batch)
    loss=cleanroom_diffusion_noise_microstep_v1(**train)
    assert torch.isfinite(loss)
    loss.backward()
    assert train["conditioner"].source_projection.weight.grad is not None
    assert train["denoiser"].to_q.lora_B.grad is not None
    assert all(p.grad is None for p in vision.parameters())


def test_multiview_stage_conditions_on_both_canonical_rgba_and_aligned_six_rgb_normal(tmp_path):
    manifest,reader,sample,train,vision,batch=_case(tmp_path,STAGE_MULTIVIEW)
    assert sample["source_pair_authority"]=="SOURCE_3D_RENDERED_TWO_CANONICAL_IMAGES"
    assert sample["reference_rgb"].shape==(3,16,16)
    assert sample["source_rgba"].shape==(2,4,16,16)
    assert sample["target_rgb"].shape==(6,3,16,16)
    assert sample["target_normal_encoded_rgb"].shape==(6,3,16,16)
    assert batch["source_tokens"].shape==(1,2,4,8)
    assert batch["rendered_target_rgb"].shape==(1,12,3,16,16)
    assert batch["rendered_target_validity_mask"].shape==(1,12,1,16,16)
    assert batch["target_cameras"].shape==(1,12,13)
    assert torch.equal(batch["target_cameras"][:,:6],batch["target_cameras"][:,6:])
    assert batch["target_modalities"].tolist()==[[0]*6+[1]*6]
    train.update(batch)
    loss=cleanroom_diffusion_noise_microstep_v1(**train)
    loss.backward()
    assert torch.isfinite(loss)
    assert train["rowwise_attention"].residual.weight.grad is not None
    assert all(p.grad is None for p in vision.parameters())


def test_canonical_stage_excludes_unrigged_static_source_but_multiview_can_use_it(tmp_path):
    manifest=_materialize(tmp_path)
    data=json.loads(manifest.read_text("utf8"))
    data["items"][0]["truth_record"]["source_pose_truth_available"]=False
    data["items"][0]["truth_record"]["input_pose_ids"]=["static","static"]
    manifest.write_text(json.dumps(data))
    assert len(RealSaSCleanroomRenderedSixViewDatasetV1(manifest))==1
    with pytest.raises(ValueError,match="RENDER_SPLIT_EMPTY"):
        RealSaSCleanroomRenderedCanonicalPoseDatasetV1(manifest)


def test_stage_reader_must_rehash_source_input_even_if_target_is_unchanged(tmp_path):
    manifest,reader,sample,train,vision,batch=_case(tmp_path,STAGE_CANONICAL)
    file=tmp_path/"a_input_scene_rgb_1.png"
    file.write_bytes(file.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match="HASH_CHANGED_DURING_TRAINING"):
        reader[0]


def test_stage_bridge_rejects_fake_observed_camera_and_unfrozen_vision(tmp_path):
    manifest,reader,sample,train,vision,batch=_case(tmp_path,STAGE_CANONICAL)
    with pytest.raises(ValueError,match="SOURCE_AUTHORITY_INVALID"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            {**sample,"original_source_camera_observed":True},
            source_encoder=vision,conditioner=train["conditioner"],
        )
    vision.train()
    with pytest.raises(ValueError,match="SOURCE_ENCODER_MUST_BE_FROZEN"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            sample,source_encoder=vision,conditioner=train["conditioner"],
        )
    vision.eval()
    vision.patch.weight.requires_grad_(True)
    with pytest.raises(ValueError,match="SOURCE_ENCODER_MUST_BE_FROZEN"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            sample,source_encoder=vision,conditioner=train["conditioner"],
        )


def test_stage_bridge_rejects_swapped_virtual_cameras_and_missing_normal_targets(tmp_path):
    manifest,reader,sample,train,vision,batch=_case(tmp_path,STAGE_MULTIVIEW)
    swapped={**sample,"target_camera_features":sample["target_camera_features"].clone()}
    swapped["target_camera_features"][[0,1]]=swapped["target_camera_features"][[1,0]]
    with pytest.raises(ValueError,match="VIRTUAL_CAMERA_DRIFT"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            swapped,source_encoder=vision,conditioner=train["conditioner"],
        )
    missing={**sample}
    missing.pop("target_normal_encoded_rgb")
    with pytest.raises(ValueError,match="TENSOR_MISSING:target_normal_encoded_rgb"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            missing,source_encoder=vision,conditioner=train["conditioner"],
        )
    lost={**sample,"target_normal_valid":sample["target_normal_valid"].clone()}
    lost["target_normal_valid"][0]=0.
    with pytest.raises(ValueError,match="NORMAL_TRUTH_DRIFT"):
        prepare_rendered_two_source_diffusion_microstep_v1(
            lost,source_encoder=vision,conditioner=train["conditioner"],
        )


def test_stage_B_batch_collation_retains_two_canonical_sources(tmp_path):
    manifest=_materialize(tmp_path)
    ds=RealSaSCleanroomRenderedSixViewDatasetV1(manifest)
    collated=next(iter(torch.utils.data.DataLoader(ds,batch_size=1)))
    train=_microcase(STAGE_MULTIVIEW)
    batch=prepare_rendered_two_source_diffusion_microstep_v1(
        collated,source_encoder=_FrozenRenderVision(),conditioner=train["conditioner"],
    )
    assert batch["source_tokens"].shape==(1,2,4,8)
    assert batch["rendered_target_rgb"].shape==(1,12,3,16,16)
