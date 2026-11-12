from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from compiler.realsas_compiler_core.hashing import content_sha256
from models.rss_vnext.paired_sdxl_lora_prior_v1 import (
    BACKBONE_REPO, SDXLBackboneBindingV1, RSSvNextTwoImageConditionerV1,
    attach_pinned_sdxl_lora_unet_v1, seal_generated_diffusion_image_v1,
    validate_generated_diffusion_image_v1,
)


def _inputs():
    torch.manual_seed(12)
    source = torch.randn(2, 2, 5, 32)
    camera = torch.zeros(2, 2, 13)
    pose = torch.zeros(2, 2, 12)
    return dict(
        observed_tokens=source,
        observed_token_valid=torch.ones(2, 2, 5, dtype=torch.bool),
        original_cameras=camera,
        original_camera_known=torch.zeros(2, 2, dtype=torch.bool),
        original_poses=pose,
        original_pose_known=torch.zeros(2, 2, dtype=torch.bool),
        target_camera=torch.randn(2, 13),
        target_pose=torch.randn(2, 12),
    )


def test_joint_diffusion_conditioner_uses_both_originals_and_arbitrary_generic_pose():
    model = RSSvNextTwoImageConditionerV1(
        source_dim=32, pose_dim=12, hidden_dim=64, output_dim=2048,
        queries=6, heads=4,
    )
    inputs = _inputs()
    inputs["observed_tokens"].requires_grad_(True)
    result = model(**inputs)
    assert result.shape == (2, 6, 2048)
    assert torch.isfinite(result).all()
    result.square().mean().backward()
    gradient = inputs["observed_tokens"].grad
    assert gradient is not None
    assert gradient[:, 0].abs().sum() > 0
    assert gradient[:, 1].abs().sum() > 0
    assert not hasattr(model, "humanoid_joint_count")
    assert any(p.requires_grad for p in model.parameters())


def test_each_source_view_has_counterfactual_influence_on_same_target():
    model = RSSvNextTwoImageConditionerV1(
        source_dim=32, pose_dim=12, hidden_dim=64, output_dim=128,
        queries=4, heads=4,
    )
    args = _inputs()
    baseline = model(**args).detach()
    for view in (0, 1):
        changed = dict(args)
        changed["observed_tokens"] = args["observed_tokens"].clone()
        changed["observed_tokens"][:, view] += 6.
        rerun = model(**changed).detach()
        assert not torch.allclose(rerun, baseline)
    # Unknown original camera/pose remains legal and distinguishable.
    assert args["original_camera_known"].sum() == 0


def test_source_role_must_be_exactly_two_and_unknown_camera_cannot_be_fabricated():
    model = RSSvNextTwoImageConditionerV1(
        source_dim=32, pose_dim=12, hidden_dim=64, heads=4,
    )
    base = _inputs()
    with pytest.raises(ValueError, match="TWO_ORIGINAL_TOKEN_STREAMS_REQUIRED|PAIRED_CONDITIONING_SHAPE_OR_MASK"):
        model(**{**base, "observed_tokens": base["observed_tokens"][:, :1]})
    flags = base["observed_token_valid"].clone()
    flags[:, 1] = False
    with pytest.raises(ValueError, match="EACH_ORIGINAL_IMAGE_MUST_CONTRIBUTE"):
        model(**{**base, "observed_token_valid": flags})
    guessed = base["original_cameras"].clone()
    guessed[0, 0, 3] = 1.
    with pytest.raises(ValueError, match="UNKNOWN_ORIGINAL_CAMERA_CANNOT_BE_INVENTED"):
        model(**{**base, "original_cameras": guessed})
    guessed_pose = base["original_poses"].clone()
    guessed_pose[1, 1, 0] = 3.
    with pytest.raises(ValueError, match="UNKNOWN_ORIGINAL_POSE_CANNOT_BE_INVENTED"):
        model(**{**base, "original_poses": guessed_pose})


def _binding(*, trained=True):
    return SDXLBackboneBindingV1(
        unet_checkpoint_sha256="a" * 64,
        source_repository_revision="b" * 40,
        lora_adapter_sha256="c" * 64 if trained else None,
        lora_rank=16,
        is_trained_for_joint_two_image_conditioning=trained,
    )


def test_generated_output_cannot_be_claimed_before_actual_trained_paired_adapter():
    with pytest.raises(ValueError, match="GENERATED_OUTPUT_REQUIRES_TRAINED_TWO_IMAGE_ADAPTER"):
        seal_generated_diffusion_image_v1(
            original_image_payloads=(b"front", b"back"),
            original_source_binding_sha256="d"*64,
            generated_image_payload=b"canonical", target_id="canonical", task="CANONICAL_POSE_RGBA",
            binding=_binding(trained=False),
        )


def test_generated_view_is_lower_authority_than_two_unchanged_original_images():
    originals = (b"OBSERVED FRONT PIXELS", b"OBSERVED BACK PIXELS")
    model = _binding()
    generated = b"PREDICTED CANONICAL RGB"
    receipt = seal_generated_diffusion_image_v1(
        original_image_payloads=originals,
        original_source_binding_sha256="d"*64,
        generated_image_payload=generated, target_id="CANONICAL_REFERENCE",
        task="CANONICAL_POSE_RGBA", binding=model,
    )
    validate_generated_diffusion_image_v1(
        receipt, original_image_payloads=originals,
        generated_image_payload=generated,
        original_source_binding_sha256="d"*64,
    )
    assert receipt.original_image_sha256 != (receipt.generated_image_sha256,) * 2
    assert receipt.original_observed_view_count == 2
    assert receipt.conditioning_original_count == 2
    assert not receipt.constructed_view_is_observed
    assert not receipt.original_observation_overwritten
    assert not receipt.inferred_hidden_geometry_is_source_truth
    assert not receipt.canonical_mesh_authority
    assert not receipt.graphics_source_owner_authority
    assert originals == (b"OBSERVED FRONT PIXELS", b"OBSERVED BACK PIXELS")
    with pytest.raises(ValueError, match="GENERATED_RECEIPT_SOURCE_AUTHORITY_DRIFT"):
        validate_generated_diffusion_image_v1(
            replace(receipt, constructed_view_is_observed=True),
            original_image_payloads=originals,
            generated_image_payload=generated,
            original_source_binding_sha256="d"*64,
        )
    with pytest.raises(ValueError, match="GENERATED_RECEIPT_SOURCE_AUTHORITY_DRIFT"):
        validate_generated_diffusion_image_v1(
            receipt,
            original_image_payloads=(originals[1], originals[0]),
            generated_image_payload=generated,
            original_source_binding_sha256="d"*64,
        )
    changed = replace(receipt, target_id="DIFFERENT_TARGET", receipt_hash="")
    payload = asdict(changed)
    payload.pop("receipt_hash")
    forged = replace(changed, receipt_hash=content_sha256(payload))
    # This currently remains a predicted target, never source; only typed
    # trusted producer and later geometric consequence checks can qualify it.
    validate_generated_diffusion_image_v1(
        forged,
        original_image_payloads=originals,
        generated_image_payload=generated,
        original_source_binding_sha256="d"*64,
    )


def test_local_loader_abstains_before_external_pretrained_weights():
    with pytest.raises(ValueError, match="PINNED_LOCAL_SDXL_UNET_WEIGHTS_REQUIRED"):
        attach_pinned_sdxl_lora_unet_v1(
            local_model_directory="/nonexistent/realsas/sdxl",
            binding=_binding(),
        )
    bad = replace(_binding(), source_repository_revision="unversioned")
    with pytest.raises(ValueError, match="BACKBONE_PIN_OR_POLICY_INVALID"):
        bad.validate()
    assert _binding().model_repo == BACKBONE_REPO
