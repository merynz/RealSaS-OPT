from __future__ import annotations

"""Clean-room StdGEN task-training baseline, not a trained product model.

StdGEN CVPR 2025 §3.1: Stable Diffusion + ReferenceNet canonicalization,
Era3D-derived joint RGB+normal multi-view diffusion. §3.2 trains S-LRM
with attention LoRA, starting from InstantNeRF. We do NOT copy StdGEN source
or weights. This adapter can tune attention on an independently pre-trained
local denoiser; it cannot claim Era3D row-wise multiview parity or pose
canonicalization quality until measured against StdGEN in unseen families.

Task-specific difference: paired original source tokens and morphology-
conditional neutral poses. No fabricated human A-pose target for nonhumans.
"""

from dataclasses import dataclass
import math
from collections import defaultdict
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from compiler.realsas_compiler_core.hashing import content_sha256


STAGE_CANONICAL = "ARBITRARY_TWO_SCENES_TO_CANONICAL_POSE"
STAGE_MULTIVIEW = "CANONICAL_POSE_TO_CAMERA_CONDITIONED_RGBA_NORMAL"
SPLITS = frozenset({"train", "val", "unseen"})
_RENDER_OWNER = "INDEPENDENT_3D_ASSET_RENDERED_TRAINING_TRUTH_NOT_PRODUCT_SOURCE"
_TRAINING_SCHEMA = "RealSaS.RSSvNext.StdGENCleanroomTrainingRender.v1"


def _sha(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class CleanroomDiffusionRenderRecordV1:
    """Training examples come from real eligible 3D assets, not generated 2D.

    Geometry-derived supervision must be rendered from a source asset with
    rights admission, not reconstructed from diffusion predictions. The two
    *training* input views may have arbitrary camera and pose (not antipodal).
    They are rendered ground truth, never confused with customer uploads.
    """
    source_asset_lineage_sha256: str
    dedup_family_sha256: str
    rights_admission_sha256: str
    morphology_id: str
    neutral_pose_definition_sha256: str
    split: str
    input_pose_ids: tuple[str, str]
    input_scene_rgb_sha256: tuple[str, str]
    canonical_target_rgb_sha256: tuple[str, str]
    multiview_target_rgb_sha256: tuple[str, ...]
    multiview_target_normal_sha256: tuple[str, ...]
    multiview_target_alpha_sha256: tuple[str, ...]
    multiview_target_depth_sha256: tuple[str, ...]
    renderer_version_sha256: str
    source_pose_truth_available: bool
    schema: str = _TRAINING_SCHEMA
    training_truth_class: str = _RENDER_OWNER
    generated_diffusion_images_are_training_targets: bool = False
    source_asset_training_rights_independently_admitted: bool = False


def validate_render_record_v1(record: CleanroomDiffusionRenderRecordV1) -> None:
    if (
        type(record) is not CleanroomDiffusionRenderRecordV1
        or record.schema != _TRAINING_SCHEMA
        or record.training_truth_class != _RENDER_OWNER
        or record.generated_diffusion_images_are_training_targets is not False
        or record.source_asset_training_rights_independently_admitted is not True
        or type(record.source_pose_truth_available) is not bool
        or record.split not in SPLITS
        or type(record.morphology_id) is not str or not record.morphology_id
        or any(not _sha(s) for s in (
            record.source_asset_lineage_sha256, record.dedup_family_sha256,
            record.rights_admission_sha256, record.neutral_pose_definition_sha256,
            record.renderer_version_sha256,
        ))
        or type(record.input_pose_ids) is not tuple
        or len(record.input_pose_ids) != 2
        or any(type(x) is not str or not x for x in record.input_pose_ids)
    ):
        raise ValueError("RSS_VNEXT_DIFFUSION_RENDER_SOURCE_OR_RIGHTS_INVALID")
    fields = (
        (record.input_scene_rgb_sha256, (2,)),
        (record.canonical_target_rgb_sha256, (2,)),
        (record.multiview_target_rgb_sha256, (6, 8)),
        (record.multiview_target_normal_sha256, (6, 8)),
        (record.multiview_target_alpha_sha256, (6, 8)),
        (record.multiview_target_depth_sha256, (6, 8)),
    )
    v = len(record.multiview_target_rgb_sha256)
    for values, allowed in fields:
        if (
            type(values) is not tuple or len(values) not in allowed
            or any(not _sha(x) for x in values)
        ):
            raise ValueError("RSS_VNEXT_DIFFUSION_RENDER_TARGET_MISSING")
    if any(len(values) != v for values in (
        record.multiview_target_normal_sha256,
        record.multiview_target_alpha_sha256,
        record.multiview_target_depth_sha256,
    )):
        raise ValueError("RSS_VNEXT_DIFFUSION_MULTIVIEW_MODALITY_COUNT_DRIFT")
    if not record.source_pose_truth_available:
        # A static unrigged asset may supervise novel-view synthesis,
        # but not arbitrary->neutral pose canonicalization.
        if record.input_pose_ids[0] != record.input_pose_ids[1]:
            raise ValueError("RSS_VNEXT_DIFFUSION_UNRIGGED_POSE_TARGET_FABRICATED")


def validate_cleanroom_render_corpus_v1(
    records: Sequence[CleanroomDiffusionRenderRecordV1],
) -> dict[str, Any]:
    """Fail closed on lineage/near-duplicate cross-split leakage."""
    if not records:
        raise ValueError("RSS_VNEXT_DIFFUSION_EMPTY_CORPUS")
    split_by_family: dict[str,str] = {}
    split_by_asset: dict[str,str] = {}
    counts = defaultdict(int)
    morphologies = defaultdict(set)
    canonical_eligible = defaultdict(int)
    hashes = set()
    for record in records:
        validate_render_record_v1(record)
        identifier = content_sha256(record.__dict__)
        if identifier in hashes:
            raise ValueError("RSS_VNEXT_DIFFUSION_DUPLICATE_TRAINING_EXAMPLE")
        hashes.add(identifier)
        for index, key in (
            (split_by_family, record.dedup_family_sha256),
            (split_by_asset, record.source_asset_lineage_sha256),
        ):
            if key in index and index[key] != record.split:
                raise ValueError("RSS_VNEXT_DIFFUSION_TRAIN_UNSEEN_LINEAGE_LEAK")
            index[key] = record.split
        counts[record.split] += 1
        morphologies[record.split].add(record.morphology_id)
        canonical_eligible[record.split] += int(record.source_pose_truth_available)
    return {
        "schema": "RealSaS.RSSvNext.StdGENCleanroomCorpusValidation.v1",
        "counts_by_split": {s: counts[s] for s in sorted(SPLITS)},
        "canonical_pose_training_eligible_by_split": {
            s: canonical_eligible[s] for s in sorted(SPLITS)
        },
        "morphology_ids_by_split": {
            s: sorted(morphologies[s]) for s in sorted(SPLITS)
        },
        "cross_split_lineage_dedup_verified": True,
        "generated_2d_targets_accepted": False,
        "right_admission_receipts_cryptographically_verified_here": False,
        "product_quality_or_unseen_proof": False,
    }


class AttentionLoRALinearV1(nn.Module):
    """Own LoRA delta around an immutable pretrained Linear weight.

    Exact pretrained forward initially, learned low-rank delta after updates.
    No dependency on StdGEN custom code or checkpoint.
    """
    def __init__(self, linear: nn.Linear, *, rank: int = 8, alpha: float = 8.) -> None:
        super().__init__()
        if type(linear) is not nn.Linear or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("RSS_VNEXT_DIFFUSION_LORA_CONFIG")
        self.base = linear
        self.base.requires_grad_(False)
        # Train LoRA factors in FP32 while keeping the original SD UNet
        # projection and its half/bfloat16 dtype immutable.
        self.lora_A = nn.Parameter(torch.empty((rank, linear.in_features),
            device=linear.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((linear.out_features, rank),
            device=linear.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: Tensor) -> Tensor:
        base = self.base(x)
        delta = F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
        return base + (delta * self.scale).to(base.dtype)


def install_attention_lora_v1(
    denoiser: nn.Module, *, rank: int = 8, alpha: float = 8.
) -> tuple[str, ...]:
    """Freeze external UNet and inject trainable LoRA in self/cross attention.

    Diffusers UNet naming convention: to_q, to_k, to_v, to_out.0.
    A different backbone must show matching modules or explicitly abstain.
    """
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("RSS_VNEXT_DIFFUSION_LORA_CONFIG")
    candidates = [
        name for name, mod in denoiser.named_modules()
        if type(mod) is nn.Linear
        and name.split(".")[-1] in ("to_q","to_k","to_v")
        or (type(mod) is nn.Linear and name.endswith("to_out.0"))
    ]
    if not candidates:
        raise ValueError("RSS_VNEXT_DIFFUSION_LORA_NO_ATTENTION_TARGETS")
    denoiser.requires_grad_(False)
    for name in candidates:
        parent_name, _, leaf = name.rpartition(".")
        parent = denoiser.get_submodule(parent_name) if parent_name else denoiser
        original = getattr(parent,leaf)
        setattr(parent,leaf,AttentionLoRALinearV1(original,rank=rank,alpha=alpha))
    return tuple(candidates)


class PairedSourceDiffusionConditionerV1(nn.Module):
    """Two source token sets → per-target-camera cross-attention conditioning.

    Source tokens must come from a separately pinned image encoder such as
    the existing frozen DINO adapter. No text prompt or single-image averaging
    silently discards the second source. Does not yet implement Era3D's
    feature-row attention; that remains a measured training gap.
    """
    def __init__(self, source_width: int, conditioning_width: int, heads: int = 4):
        super().__init__()
        if source_width < 1 or conditioning_width < 1 or conditioning_width % heads:
            raise ValueError("RSS_VNEXT_DIFFUSION_CONDITIONER_WIDTH")
        self.source_projection = nn.Linear(source_width,conditioning_width)
        self.source_view_id = nn.Parameter(torch.zeros((2,conditioning_width)))
        self.target_camera_projection = nn.Sequential(
            nn.LayerNorm(13),nn.Linear(13,conditioning_width),
        )
        self.target_modality = nn.Embedding(2,conditioning_width)
        self.cross = nn.MultiheadAttention(
            conditioning_width,heads,batch_first=True,
        )

    def forward(
        self, source_tokens: Tensor, target_cameras: Tensor,
        target_modalities: Tensor,
    ) -> Tensor:
        if source_tokens.ndim != 4 or source_tokens.shape[1] != 2:
            raise ValueError("RSS_VNEXT_DIFFUSION_EXPECT_TWO_ORIGINAL_TOKEN_SETS")
        b,_,tokens,_=source_tokens.shape
        if (
            tokens < 1 or target_cameras.ndim != 3 or target_cameras.shape[0] != b
            or target_cameras.shape[2] != 13
            or target_modalities.shape != target_cameras.shape[:2]
            or target_modalities.dtype != torch.long
            or not bool(torch.isfinite(source_tokens).all())
            or not bool(torch.isfinite(target_cameras).all())
            or bool(((target_modalities < 0) | (target_modalities > 1)).any())
            or target_cameras.shape[1] not in (2,12,16)
        ):
            raise ValueError("RSS_VNEXT_DIFFUSION_CONDITIONING_DOMAIN")
        src = self.source_projection(source_tokens)
        src = src + self.source_view_id[None,:,None,:]
        src = src.reshape(b,2*tokens,-1)
        q = self.target_camera_projection(target_cameras) + self.target_modality(target_modalities)
        camera_fused,_ = self.cross(q,src,src,need_weights=False)
        v = target_cameras.shape[1]
        hidden = torch.cat((
            camera_fused[:,:,None,:],
            src[:,None,:,:].expand(-1,v,-1,-1),
        ),dim=2)
        return hidden.reshape(b*v,1+2*tokens,-1)


def cleanroom_diffusion_noise_microstep_v1(
    *, source_tokens: Tensor, rendered_target_rgb: Tensor,
    rendered_target_validity_mask: Tensor, target_cameras: Tensor,
    target_modalities: Tensor, vae: Any, scheduler: Any,
    denoiser: nn.Module, conditioner: PairedSourceDiffusionConditionerV1,
    stage: str, noise: Tensor | None = None, timesteps: Tensor | None = None,
) -> Tensor:
    """One real SD-family epsilon-loss forward/backward-capable training step.

    Inputs MUST be independently rendered GT asset targets, never model outputs
    or 'observed' customer images promoted to neutral-pose truth. The caller's
    dataset admission supplies the trust boundary. No model checkpoints are
    loaded or hidden teachers manufactured here.
    """
    if stage not in (STAGE_CANONICAL, STAGE_MULTIVIEW):
        raise ValueError("RSS_VNEXT_DIFFUSION_UNKNOWN_TRAINING_STAGE")
    if (
        rendered_target_rgb.ndim != 5 or rendered_target_rgb.shape[2] != 3
        or not torch.is_floating_point(rendered_target_rgb)
        or not bool(torch.isfinite(rendered_target_rgb).all())
        or bool((rendered_target_rgb < 0).any() or (rendered_target_rgb > 1).any())
    ):
        raise ValueError("RSS_VNEXT_DIFFUSION_RENDER_RGB_DOMAIN")
    b,v,_,h,w = rendered_target_rgb.shape
    if (
        v != (2 if stage==STAGE_CANONICAL else v)
        or (stage==STAGE_MULTIVIEW and v not in (12,16))
        or rendered_target_validity_mask.shape != (b,v,1,h,w)
        or not bool(torch.isfinite(rendered_target_validity_mask).all())
        or bool((rendered_target_validity_mask < 0).any()
                or (rendered_target_validity_mask > 1).any())
        or not bool((rendered_target_validity_mask > 0).any())
        or (stage==STAGE_CANONICAL and bool((target_modalities != 0).any()))
        or (stage==STAGE_MULTIVIEW and (
            int((target_modalities == 0).sum()) != b*v//2 or
            int((target_modalities == 1).sum()) != b*v//2
        ))
    ):
        raise ValueError("RSS_VNEXT_DIFFUSION_RENDER_SUPERVISION_DOMAIN")
    cond = conditioner(source_tokens,target_cameras,target_modalities)
    flat = rendered_target_rgb.reshape(b*v,3,h,w)
    with torch.no_grad():
        distribution = vae.encode(flat*2.-1.).latent_dist
        latent = distribution.mode() * float(vae.config.scaling_factor)
    if (
        latent.ndim != 4 or latent.shape[0] != b*v
        or not bool(torch.isfinite(latent).all())
    ):
        raise ValueError("RSS_VNEXT_DIFFUSION_VAE_LATENT_DOMAIN")
    if noise is None:
        noise=torch.randn_like(latent)
    if noise.shape != latent.shape or not bool(torch.isfinite(noise).all()):
        raise ValueError("RSS_VNEXT_DIFFUSION_NOISE_DOMAIN")
    if timesteps is None:
        timesteps=torch.randint(
            0,int(scheduler.config.num_train_timesteps),
            (b*v,),device=latent.device,
        )
    if timesteps.shape != (b*v,) or timesteps.dtype != torch.long:
        raise ValueError("RSS_VNEXT_DIFFUSION_TIMESTEP_DOMAIN")
    if getattr(scheduler.config,"prediction_type",None) != "epsilon":
        raise ValueError("RSS_VNEXT_DIFFUSION_EPSILON_SCHEDULER_REQUIRED")
    noisy=scheduler.add_noise(latent,noise,timesteps)
    predicted=denoiser(noisy,timesteps,encoder_hidden_states=cond).sample
    if predicted.shape != noise.shape or not bool(torch.isfinite(predicted).all()):
        raise ValueError("RSS_VNEXT_DIFFUSION_DENOISER_OUTPUT_DOMAIN")
    mask=F.interpolate(
        rendered_target_validity_mask.reshape(b*v,1,h,w),
        size=latent.shape[-2:],mode="nearest",
    ).to(dtype=predicted.dtype)
    denom=mask.sum()*predicted.shape[1]
    if float(denom.detach()) <= 0.:
        raise ValueError("RSS_VNEXT_DIFFUSION_NO_RENDER_TRUTH_PIXELS")
    return (((predicted.float()-noise.float())**2)*mask.float()).sum()/denom.float()


__all__=[
    "STAGE_CANONICAL","STAGE_MULTIVIEW",
    "CleanroomDiffusionRenderRecordV1","validate_render_record_v1",
    "validate_cleanroom_render_corpus_v1","AttentionLoRALinearV1",
    "install_attention_lora_v1","PairedSourceDiffusionConditionerV1",
    "cleanroom_diffusion_noise_microstep_v1",
]
