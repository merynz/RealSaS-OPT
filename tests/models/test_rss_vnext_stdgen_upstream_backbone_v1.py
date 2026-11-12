from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest
from torch import nn

from models.rss_vnext.stdgen_upstream_backbone_v1 import (
    CANONICAL,MULTIVIEW,RECONSTRUCTION,reference_ancestry_v1,
    StdGENBackbonePinV1, offline_snapshot_inventory_sha256_v1,
    verify_stdgen_backbone_pin_v1,stdgen_reference_recipe_v1,
    audit_stdgen_diffusion_training_architecture_v1,
)


def _snapshot(tmp_path,stage):
    root=tmp_path/("base-"+stage)
    root.mkdir()
    (root/"model_index.json").write_text(
        json.dumps({"_class_name":"StableUnCLIPImg2ImgPipeline"}))
    required=["unet","vae","scheduler","text_encoder","tokenizer"]
    if stage==MULTIVIEW:
        required+=["image_encoder","image_normalizer","image_noising_scheduler"]
    for path in required:
        folder=root/path
        folder.mkdir()
        (folder/"config.json").write_text("{}")
    (root/"unet"/"diffusion_pytorch_model.safetensors").write_bytes(b"test-only")
    return root


def _pin(tmp_path, stage):
    root=_snapshot(tmp_path,stage)
    return StdGENBackbonePinV1(
        stage=stage,
        upstream_repository_id=reference_ancestry_v1(stage).ancestral_base_repo,
        upstream_revision_sha="1"*40,
        snapshot_path=str(root),
        checkpoint_inventory_sha256=offline_snapshot_inventory_sha256_v1(str(root),stage),
        license_id="CreativeML OpenRAIL++-M",
        commercial_use_review_sha256="2"*64,
    )


def test_ancestry_exact_baselines_not_sdxl_or_stdgen_generated_weights():
    canonical=reference_ancestry_v1(CANONICAL)
    views=reference_ancestry_v1(MULTIVIEW)
    geom=reference_ancestry_v1(RECONSTRUCTION)
    assert canonical.ancestral_base_repo=="stabilityai/stable-diffusion-2-1"
    assert "INFERRED_NOT_PROVEN" in canonical.original_training_provenance
    assert views.ancestral_base_repo=="stabilityai/stable-diffusion-2-1-unclip"
    assert views.original_training_provenance=="ERA3D_TRAINING_CONFIG_DIRECT"
    assert geom.ancestral_base_repo=="TencentARC/InstantMesh:InstantNeRF"
    assert all(x.released_task_checkpoint_not_used for x in (canonical,views,geom))
    with pytest.raises(ValueError,match="UNKNOWN_REFERENCE_STAGE"):
        reference_ancestry_v1("SDXL")


@pytest.mark.parametrize("stage",[CANONICAL,MULTIVIEW])
def test_local_upstream_snapshot_is_hash_pinned_and_never_accepts_modified_model(tmp_path,stage):
    pin=_pin(tmp_path,stage)
    verify_stdgen_backbone_pin_v1(pin)
    with pytest.raises(ValueError,match="PRETRAINED_OR_RIGHTS_PIN_INVALID"):
        verify_stdgen_backbone_pin_v1(replace(
            pin,upstream_repository_id="hyz317/StdGEN",
        ))
    with pytest.raises(ValueError,match="PRETRAINED_OR_RIGHTS_PIN_INVALID"):
        verify_stdgen_backbone_pin_v1(replace(pin,license_id="RESEARCH_ONLY"))
    with pytest.raises(ValueError,match="UNVERIFIED_MIRROR_FORBIDDEN"):
        verify_stdgen_backbone_pin_v1(replace(pin,mirror_of_original=True))
    (tmp_path/("base-"+stage)/"unet"/"config.json").write_text('{"changed":1}')
    with pytest.raises(ValueError,match="SNAPSHOT_CONTENT_DRIFT"):
        verify_stdgen_backbone_pin_v1(pin)


def test_pickle_and_unattested_empty_snapshot_are_rejected(tmp_path):
    pin=_pin(tmp_path,CANONICAL)
    root=tmp_path/("base-"+CANONICAL)
    (root/"unet"/"danger.bin").write_bytes(b"pickle-placeholder")
    with pytest.raises(ValueError,match="PICKLE_CHECKPOINT_FORBIDDEN"):
        verify_stdgen_backbone_pin_v1(pin)
    (root/"unet"/"danger.bin").unlink()
    (root/"unet"/"diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(ValueError,match="BASE_SAFETENSORS_REQUIRED"):
        offline_snapshot_inventory_sha256_v1(str(root),CANONICAL)


def test_reference_era3d_train_policy_is_six_512_to_1024_full_unet_not_lora_only():
    r=stdgen_reference_recipe_v1()
    assert r.multiview_views==6
    assert r.multiview_base_resolution==512
    assert r.stdgen_progressive_target_resolution==1024
    assert r.denoiser_policy=="FULL_UNET_FINETUNE"
    assert r.row_attention_location=="UNET_FEATURE_TRANSFORMER"
    assert r.reconstruction_lora_location=="SLRM_ATTENTION_LINEAR_LAYERS"
    assert r.exact_stdgen_diffusion_training_recipe_public is False


def test_misplaced_attention_or_lora_only_denoiser_cannot_masquerade_as_stdgen():
    net=nn.Linear(8,8)
    blocks=[nn.Linear(8,8) for _ in range(3)]
    for block in blocks: block.requires_grad_(False)
    kwargs=dict(
        denoiser=net,vae=blocks[0],image_encoder=blocks[1],text_encoder=blocks[2],
        num_views=6,row_attention_location="UNET_FEATURE_TRANSFORMER",
        num_generated_modalities=2,
    )
    audit_stdgen_diffusion_training_architecture_v1(**kwargs)
    with pytest.raises(ValueError,match="ROW_ATTENTION_MUST_BE_IN_UNET_FEATURES"):
        audit_stdgen_diffusion_training_architecture_v1(
            **{**kwargs,"row_attention_location":"EXTERNAL_LATENT_MIXER"},
        )
    with pytest.raises(ValueError,match="SIX_VIEW_RGB_NORMAL_REQUIRED"):
        audit_stdgen_diffusion_training_architecture_v1(
            **{**kwargs,"num_views":8},
        )
    with pytest.raises(ValueError,match="LORA_ONLY_DENOISER_NOT_BASELINE"):
        net.weight.requires_grad_(False)
        audit_stdgen_diffusion_training_architecture_v1(**kwargs)
