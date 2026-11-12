from __future__ import annotations

"""Clean-room paired-image SDXL/LoRA prior and immutable source authority.

Reference architecture: StdGEN (CVPR 2025): pretrained canonicalization
diffusion -> RGB/normal multiview diffusion -> semantic-aware 3D reconstruction.
This module reuses no StdGEN implementation. It provides a trainable TWO-IMAGE
conditioning interface, a local-only pinned SDXL UNet LoRA loader, and a typed
predicted-image receipt. It does NOT invent trained LoRA weights, a canonical
pose, a source camera, a source 3D point, or qualified Graphics.

The first baseline is SDXL OpenRAIL++ subject to commercial license review.
LoRA alone is not camera conditioning: a distinct joint-source conditioner
cross-attends to BOTH immutable source-image token streams, with explicitly
UNKNOWN original camera/pose channels. A generated view has lower authority
than an original observation even when its pixels happen to look convincing.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import math
from typing import Any

import torch
from torch import Tensor, nn

from compiler.realsas_compiler_core.hashing import content_sha256

BACKBONE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
BACKBONE_LICENSE = "creativeml-openrail++-m"
SCHEMA = "RealSaS.RSSvNext.PairedDiffusionCompletion.v1"
AUTHORITY = "GENERATED_COMPLETION_LOWER_THAN_ORIGINAL_OBSERVATION"
_GENERATION_TASKS = frozenset((
    "CANONICAL_POSE_RGBA",
    "NOVEL_VIEW_RGBA",
    "NOVEL_VIEW_NORMAL",
))
_LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def _fail(code: str) -> None:
    raise ValueError("RSS_VNEXT_DIFFUSION_" + code)


def _hex64(value: object) -> bool:
    return (type(value) is str and len(value) == 64
            and all(x in "0123456789abcdef" for x in value))


def _hash_bytes(raw: bytes) -> str:
    return sha256(raw).hexdigest()


@dataclass(frozen=True)
class SDXLBackboneBindingV1:
    """Pinned UNet weights; no arbitrary online download in a product worker."""
    unet_checkpoint_sha256: str
    source_repository_revision: str
    lora_adapter_sha256: str | None
    lora_rank: int
    model_repo: str = BACKBONE_REPO
    model_license: str = BACKBONE_LICENSE
    stage: str = "UNQUALIFIED_TWO_VIEW_CANONICALIZATION_PRIOR"
    is_trained_for_joint_two_image_conditioning: bool = False
    has_commercial_license_clearance: bool = False

    def validate(self) -> None:
        if (
            self.model_repo != BACKBONE_REPO
            or self.model_license != BACKBONE_LICENSE
            or not _hex64(self.unet_checkpoint_sha256)
            or type(self.source_repository_revision) is not str
            or len(self.source_repository_revision) != 40
            or any(c not in "0123456789abcdef" for c in self.source_repository_revision)
            or (self.lora_adapter_sha256 is not None
                and not _hex64(self.lora_adapter_sha256))
            or type(self.lora_rank) is not int
            or not 1 <= self.lora_rank <= 128
            or self.stage not in (
                "UNQUALIFIED_TWO_VIEW_CANONICALIZATION_PRIOR",
                "UNQUALIFIED_MULTIVIEW_RGB_NORMAL_PRIOR",
            )
            or type(self.is_trained_for_joint_two_image_conditioning) is not bool
            or type(self.has_commercial_license_clearance) is not bool
        ):
            _fail("BACKBONE_PIN_OR_POLICY_INVALID")


class RSSvNextTwoImageConditionerV1(nn.Module):
    """Joint original-image conditioner suitable for SDXL cross-attention.

    Input source_tokens must come from TWO source encodings. Camera and pose
    known flags distinguish unknown metadata from genuine zero parameters.
    The target_pose vector is a GENERIC articulation-space query, not human
    A-pose joints. Source tokens are never replaced by predicted views.

    This conditioner is a trainable module, NOT a released checkpoint or a
    replacement for camera-conditioned multi-view diffusion training.
    """

    def __init__(
        self, *, source_dim: int, pose_dim: int, hidden_dim: int = 256,
        output_dim: int = 2048, queries: int = 16, heads: int = 8,
    ):
        super().__init__()
        if (
            min(source_dim, pose_dim, hidden_dim, output_dim, queries, heads) < 1
            or hidden_dim % heads
            or queries > 256
        ):
            _fail("CONDITIONER_DIMENSIONS_INVALID")
        self.source_dim = source_dim
        self.pose_dim = pose_dim
        self.source_projection = nn.Linear(source_dim, hidden_dim)
        self.source_pose_camera_projection = nn.Linear(13 + pose_dim + 2, hidden_dim)
        self.view_slot_embedding = nn.Embedding(2, hidden_dim)
        self.target_projection = nn.Linear(13 + pose_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(queries, hidden_dim) * .02)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self, *,
        observed_tokens: Tensor,  # [B,2,T,D]
        observed_token_valid: Tensor,  # [B,2,T], boolean
        original_cameras: Tensor,  # [B,2,13], zeros if unknown
        original_camera_known: Tensor,  # [B,2], boolean
        original_poses: Tensor,  # [B,2,P], zeros if unknown
        original_pose_known: Tensor,  # [B,2], boolean
        target_camera: Tensor,  # [B,13], output target, not source observation
        target_pose: Tensor,  # [B,P], generic target articulation query
    ) -> Tensor:
        if observed_tokens.ndim != 4:
            _fail("TWO_ORIGINAL_TOKEN_STREAMS_REQUIRED")
        b, views, tokens, dim = observed_tokens.shape
        if (
            views != 2 or tokens < 1 or dim != self.source_dim
            or observed_token_valid.shape != (b, 2, tokens)
            or observed_token_valid.dtype != torch.bool
            or original_cameras.shape != (b, 2, 13)
            or original_camera_known.shape != (b, 2)
            or original_camera_known.dtype != torch.bool
            or original_poses.shape != (b, 2, self.pose_dim)
            or original_pose_known.shape != (b, 2)
            or original_pose_known.dtype != torch.bool
            or target_camera.shape != (b, 13)
            or target_pose.shape != (b, self.pose_dim)
        ):
            _fail("PAIRED_CONDITIONING_SHAPE_OR_MASK")
        if not bool(observed_token_valid.any(dim=2).all()):
            _fail("EACH_ORIGINAL_IMAGE_MUST_CONTRIBUTE")
        arrays = (observed_tokens, original_cameras, original_poses,
                  target_camera, target_pose)
        if any(not bool(torch.isfinite(x).all()) for x in arrays):
            _fail("PAIRED_CONDITIONING_NONFINITE")
        # Never accept guessed pose/camera numbers in the UNKNOWN slots.
        if bool((original_cameras[~original_camera_known] != 0).any()):
            _fail("UNKNOWN_ORIGINAL_CAMERA_CANNOT_BE_INVENTED")
        if bool((original_poses[~original_pose_known] != 0).any()):
            _fail("UNKNOWN_ORIGINAL_POSE_CANNOT_BE_INVENTED")
        flags = torch.stack(
            (original_camera_known, original_pose_known), dim=-1
        ).to(dtype=observed_tokens.dtype)
        meta = torch.cat((original_cameras, original_poses, flags), dim=-1)
        source = (
            self.source_projection(observed_tokens)
            + self.source_pose_camera_projection(meta)[:, :, None, :]
            + self.view_slot_embedding(
                torch.arange(2, device=observed_tokens.device)
            )[None, :, None, :]
        ).flatten(1, 2)
        key_padding_mask = ~observed_token_valid.flatten(1, 2)
        target = torch.cat((target_camera, target_pose), dim=-1)
        query = (
            self.queries[None, :, :]
            + self.target_projection(target)[:, None, :]
        )
        attended, _ = self.attention(
            query, source, source, key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        output = self.output(attended + query)
        if not bool(torch.isfinite(output).all()):
            _fail("CONDITIONED_GENERATION_NONFINITE")
        return output


def attach_pinned_sdxl_lora_unet_v1(
    *, local_model_directory: str | Path, binding: SDXLBackboneBindingV1,
) -> nn.Module:
    """Load EXISTING SDXL UNet + train LoRA only; never fetch remote weights.

    The text encoder, VAE, scheduler and IP-style projected conditioning
    remain separately pinned integration dependencies; loading this UNet is
    not a complete production diffusion pipeline. No fake adapter checkpoint.
    """
    binding.validate()
    root = Path(local_model_directory).expanduser().resolve()
    weights = root / "unet" / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        _fail("PINNED_LOCAL_SDXL_UNET_WEIGHTS_REQUIRED")
    digest = sha256()
    with weights.open("rb") as file:
        while True:
            chunk = file.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    if digest.hexdigest() != binding.unet_checkpoint_sha256:
        _fail("SDXL_UNET_CHECKPOINT_SHA256_DRIFT")
    try:
        from diffusers import UNet2DConditionModel
        from peft import LoraConfig
    except ImportError as exc:
        raise RuntimeError(
            "RSS_VNEXT_DIFFUSION_DEPENDENCIES_REQUIRED_DIFFUSERS_AND_PEFT"
        ) from exc
    unet = UNet2DConditionModel.from_pretrained(
        str(root), subfolder="unet", local_files_only=True,
        use_safetensors=True,
    )
    for param in unet.parameters():
        param.requires_grad_(False)
    unet.add_adapter(LoraConfig(
        r=binding.lora_rank, lora_alpha=binding.lora_rank,
        target_modules=list(_LORA_TARGETS), init_lora_weights=True,
    ))
    return unet


@dataclass(frozen=True)
class RSSvNextGeneratedImageV1:
    """A generated completion never gets the authority of source pixels."""
    original_image_sha256: tuple[str, str]
    original_source_binding_sha256: str
    target_id: str
    task: str
    generated_image_sha256: str
    model_unet_sha256: str
    model_revision: str
    adapter_sha256: str | None
    conditioning_original_count: int
    conditioned_from_both_originals: bool
    receipt_hash: str
    schema: str = SCHEMA
    authority: str = AUTHORITY
    original_observed_view_count: int = 2
    constructed_view_is_observed: bool = False
    original_observation_overwritten: bool = False
    source_camera_pose_qualified: bool = False
    inferred_hidden_geometry_is_source_truth: bool = False
    canonical_mesh_authority: bool = False
    graphics_source_owner_authority: bool = False


def seal_generated_diffusion_image_v1(
    *, original_image_payloads: tuple[bytes, bytes],
    original_source_binding_sha256: str,
    generated_image_payload: bytes, target_id: str, task: str,
    binding: SDXLBackboneBindingV1,
) -> RSSvNextGeneratedImageV1:
    binding.validate()
    if (
        type(original_image_payloads) is not tuple
        or len(original_image_payloads) != 2
        or any(type(x) is not bytes or len(x) == 0 for x in original_image_payloads)
        or type(generated_image_payload) is not bytes
        or not generated_image_payload
        or not _hex64(original_source_binding_sha256)
        or type(target_id) is not str or not 1 <= len(target_id) <= 64
        or task not in _GENERATION_TASKS
        or binding.lora_adapter_sha256 is None
        or not binding.is_trained_for_joint_two_image_conditioning
    ):
        _fail("GENERATED_OUTPUT_REQUIRES_TRAINED_TWO_IMAGE_ADAPTER")
    value = RSSvNextGeneratedImageV1(
        original_image_sha256=tuple(map(_hash_bytes, original_image_payloads)),
        original_source_binding_sha256=original_source_binding_sha256,
        target_id=target_id, task=task,
        generated_image_sha256=_hash_bytes(generated_image_payload),
        model_unet_sha256=binding.unet_checkpoint_sha256,
        model_revision=binding.source_repository_revision,
        adapter_sha256=binding.lora_adapter_sha256,
        conditioning_original_count=2,
        conditioned_from_both_originals=True,
        receipt_hash="",
    )
    fields = asdict(value)
    fields.pop("receipt_hash")
    return replace(value, receipt_hash=content_sha256(fields))


def validate_generated_diffusion_image_v1(
    receipt: RSSvNextGeneratedImageV1, *,
    original_image_payloads: tuple[bytes, bytes],
    generated_image_payload: bytes, original_source_binding_sha256: str,
) -> None:
    if (
        type(receipt) is not RSSvNextGeneratedImageV1
        or receipt.schema != SCHEMA or receipt.authority != AUTHORITY
        or receipt.task not in _GENERATION_TASKS
        or type(receipt.original_image_sha256) is not tuple
        or receipt.original_image_sha256 != tuple(map(_hash_bytes, original_image_payloads))
        or receipt.original_source_binding_sha256 != original_source_binding_sha256
        or not _hex64(original_source_binding_sha256)
        or receipt.generated_image_sha256 != _hash_bytes(generated_image_payload)
        or receipt.conditioning_original_count != 2
        or receipt.conditioned_from_both_originals is not True
        or receipt.original_observed_view_count != 2
        or any((
            receipt.constructed_view_is_observed,
            receipt.original_observation_overwritten,
            receipt.source_camera_pose_qualified,
            receipt.inferred_hidden_geometry_is_source_truth,
            receipt.canonical_mesh_authority,
            receipt.graphics_source_owner_authority,
        ))
    ):
        _fail("GENERATED_RECEIPT_SOURCE_AUTHORITY_DRIFT")
    payload = asdict(receipt)
    h = payload.pop("receipt_hash")
    if content_sha256(payload) != h:
        _fail("GENERATED_RECEIPT_HASH_DRIFT")


__all__ = [
    "BACKBONE_REPO", "BACKBONE_LICENSE", "SCHEMA", "AUTHORITY",
    "SDXLBackboneBindingV1", "RSSvNextTwoImageConditionerV1",
    "RSSvNextGeneratedImageV1", "attach_pinned_sdxl_lora_unet_v1",
    "seal_generated_diffusion_image_v1",
    "validate_generated_diffusion_image_v1",
]
