from __future__ import annotations

"""Pinned two-image diffusion inference using the SAME denoiser ABI as training.

Stage A generates two morphology-neutral canonical RGB/alpha images; Stage B
jointly generates six camera-aligned normals and six RGB/alpha images.
Both stages consume their own trained state, never source 3D teachers.
The generated images remain predictions; only original two views are inputs.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    PairedSourceDiffusionConditionerV1,
    RowWiseRGBNormalViewAttentionV1,
)

SCHEMA = "RealSaS.RSSvNext.PairedDiffusionStageCheckpoint.v1"
_MAX_CHECKPOINT_BYTES = 10 * 1024 * 1024 * 1024
_MAX_IMAGE_PIXELS = 1024 * 1024


def _digest_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scheduler_digest(value: Any) -> str:
    config = dict(value.config)
    return sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _module_state(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().contiguous().clone()
            for key, value in module.state_dict().items()}


@dataclass(frozen=True)
class RSSvNextStageGeneratedImagesV1:
    rgba: Tensor
    cameras: Tensor
    normals: Tensor | None
    stage: str
    checkpoint_sha256: str
    generation_seed: int
    observation_authority: bool = False


class RSSvNextPinnedDiffusionStageV1:
    """Sampler is not admissible until its exact task-weights file is loaded."""

    def __init__(
        self, *, stage: str, denoiser: nn.Module, vae: nn.Module,
        conditioner: PairedSourceDiffusionConditionerV1,
        alpha_head: nn.Module, scheduler: Any,
        vae_scale_factor: int = 8,
        rowwise_attention: RowWiseRGBNormalViewAttentionV1 | None = None,
    ) -> None:
        if stage not in (STAGE_CANONICAL, STAGE_MULTIVIEW):
            raise ValueError("RSS_VNEXT_GENERATOR_STAGE_INVALID")
        if (stage == STAGE_MULTIVIEW) != (rowwise_attention is not None):
            raise ValueError("RSS_VNEXT_GENERATOR_ROW_ATTENTION_STAGE_DRIFT")
        if type(conditioner) is not PairedSourceDiffusionConditionerV1:
            raise ValueError("RSS_VNEXT_GENERATOR_CONDITIONER_INVALID")
        if type(vae_scale_factor) is not int or vae_scale_factor not in (4, 8, 16):
            raise ValueError("RSS_VNEXT_GENERATOR_VAE_SCALE_INVALID")
        if getattr(scheduler.config, "prediction_type", None) != "epsilon":
            raise ValueError("RSS_VNEXT_GENERATOR_EPSILON_SCHEDULER_REQUIRED")
        self.stage = stage
        self.denoiser = denoiser
        self.vae = vae
        self.conditioner = conditioner
        self.alpha_head = alpha_head
        self.scheduler = scheduler
        self.rowwise_attention = rowwise_attention
        self.vae_scale_factor = vae_scale_factor
        self.checkpoint_sha256: str | None = None

    def _modules(self) -> dict[str, nn.Module]:
        result = {
            "denoiser": self.denoiser, "vae": self.vae,
            "conditioner": self.conditioner, "alpha_head": self.alpha_head,
        }
        if self.rowwise_attention is not None:
            result["rowwise_attention"] = self.rowwise_attention
        return result

    def save_training_weights(self, path: str | Path) -> str:
        """Export training task weights, including exact VAE and alpha readout."""
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "stage": self.stage,
            "vae_scale_factor": self.vae_scale_factor,
            "scheduler_config_sha256": _scheduler_digest(self.scheduler),
            "module_states": {
                name: _module_state(module) for name, module in self._modules().items()
            },
        }
        tmp = dest.with_name(dest.name + ".tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        return _digest_file(dest)

    def load_pinned_task_weights(self, path: str | Path, *, expected_sha256: str) -> None:
        source = Path(path).resolve()
        if (
            not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in expected_sha256)
            or not source.is_file() or not 0 < source.stat().st_size <= _MAX_CHECKPOINT_BYTES
            or _digest_file(source) != expected_sha256
        ):
            raise RuntimeError("RSS_VNEXT_GENERATOR_CHECKPOINT_PIN_OR_BYTES_DRIFT")
        try:
            payload = torch.load(source, weights_only=True, map_location="cpu")
        except Exception as exc:
            raise RuntimeError("RSS_VNEXT_GENERATOR_CHECKPOINT_DECODE_INVALID") from exc
        modules = self._modules()
        if (
            type(payload) is not dict
            or set(payload) != {
                "schema", "stage", "vae_scale_factor",
                "scheduler_config_sha256", "module_states",
            }
            or payload["schema"] != SCHEMA or payload["stage"] != self.stage
            or payload["vae_scale_factor"] != self.vae_scale_factor
            or payload["scheduler_config_sha256"] != _scheduler_digest(self.scheduler)
            or type(payload["module_states"]) is not dict
            or set(payload["module_states"]) != set(modules)
        ):
            raise RuntimeError("RSS_VNEXT_GENERATOR_STAGE_CHECKPOINT_ABI_DRIFT")
        for name, module in modules.items():
            weights = payload["module_states"][name]
            if (
                type(weights) is not dict
                or set(weights) != set(module.state_dict())
                or any(not isinstance(x, Tensor) or
                       (torch.is_floating_point(x) and not bool(torch.isfinite(x).all()))
                       for x in weights.values())
            ):
                raise RuntimeError("RSS_VNEXT_GENERATOR_MODULE_STATE_DRIFT:" + name)
        for name, module in modules.items():
            try:
                module.load_state_dict(payload["module_states"][name], strict=True)
            except RuntimeError as exc:
                raise RuntimeError("RSS_VNEXT_GENERATOR_MODULE_SHAPE_DRIFT:" + name) from exc
            module.eval()
            module.requires_grad_(False)
        self.checkpoint_sha256 = expected_sha256

    @torch.no_grad()
    def generate(
        self, *, source_tokens: Tensor, target_cameras: Tensor,
        target_modalities: Tensor, height: int, width: int,
        steps: int, seed: int,
    ) -> RSSvNextStageGeneratedImagesV1:
        if self.checkpoint_sha256 is None:
            raise RuntimeError("RSS_VNEXT_GENERATOR_UNPINNED_TASK_WEIGHTS")
        if (
            type(steps) is not int or not 1 <= steps <= 100
            or type(seed) is not int or not 0 <= seed < 2**63
            or type(height) is not int or type(width) is not int
            or min(height, width) < 16 or height * width > _MAX_IMAGE_PIXELS
            or height % self.vae_scale_factor or width % self.vae_scale_factor
            or source_tokens.ndim != 4 or source_tokens.shape[1] != 2
            or target_cameras.ndim != 3 or target_cameras.shape[0] != source_tokens.shape[0]
            or target_cameras.shape[2] != 13
        ):
            raise ValueError("RSS_VNEXT_GENERATOR_INPUT_OR_RESOURCE_DOMAIN")
        b, v = target_cameras.shape[:2]
        if (
            v != (2 if self.stage == STAGE_CANONICAL else 12)
            or target_modalities.shape != (b, v)
            or target_modalities.dtype != torch.long
            or not bool(torch.isfinite(target_cameras).all())
            or not bool(torch.isfinite(source_tokens).all())
            or (self.stage == STAGE_CANONICAL and bool((target_modalities != 0).any()))
        ):
            raise ValueError("RSS_VNEXT_GENERATOR_TARGET_CAMERA_OR_MODALITY_DOMAIN")
        if self.stage == STAGE_MULTIVIEW and (
            bool((target_modalities[:, :6] != 0).any())
            or bool((target_modalities[:, 6:] != 1).any())
            or not torch.allclose(target_cameras[:, :6], target_cameras[:, 6:], atol=1e-5, rtol=0)
        ):
            raise ValueError("RSS_VNEXT_GENERATOR_RGB_NORMAL_VIEW_PAIRING_REQUIRED")
        context = self.conditioner(source_tokens, target_cameras, target_modalities)
        device = source_tokens.device
        if context.device != device:
            raise ValueError("RSS_VNEXT_GENERATOR_DEVICE_DRIFT")
        try:
            self.scheduler.set_timesteps(steps, device=device)
        except TypeError:
            self.scheduler.set_timesteps(steps)
        rng = torch.Generator(device=device).manual_seed(seed)
        shape = (b * v, 4, height // self.vae_scale_factor, width // self.vae_scale_factor)
        dtype = next(self.denoiser.parameters()).dtype
        latents = torch.randn(shape, generator=rng, device=device, dtype=dtype)
        latents = latents * float(self.scheduler.init_noise_sigma)
        for timestep in self.scheduler.timesteps:
            values = latents
            if self.rowwise_attention is not None:
                values = self.rowwise_attention(
                    values.reshape(b, v, *values.shape[1:]),
                    target_cameras, target_modalities,
                ).reshape_as(latents)
            output = self.denoiser(
                values, timestep,
                encoder_hidden_states=context.to(values.dtype),
            )
            noise = output.sample
            if noise.shape != latents.shape or not bool(torch.isfinite(noise).all()):
                raise RuntimeError("RSS_VNEXT_GENERATOR_DENOISER_NONFINITE_OR_SHAPE")
            latents = self.scheduler.step(noise, timestep, latents).prev_sample
        decoded = self.vae.decode(
            latents / float(self.vae.config.scaling_factor),
        ).sample
        if decoded.shape != (b * v, 3, height, width):
            raise RuntimeError("RSS_VNEXT_GENERATOR_VAE_OUTPUT_DOMAIN")
        rgb = ((decoded.float() + 1.) / 2.).clamp(0, 1)
        if not bool(torch.isfinite(rgb).all()):
            raise RuntimeError("RSS_VNEXT_GENERATOR_NONFINITE_RGB")
        if self.stage == STAGE_MULTIVIEW:
            # Training ABI is RGB modalities [0:6], normal modalities [6:12].
            # Do not transpose them at inference: that silently corrupts both
            # appearance and predicted geometry even for a valid checkpoint.
            colors = rgb[: b * 6]
            normal = F.normalize(
                rgb[b * 6:].reshape(b, 6, 3, height, width) * 2. - 1.,
                dim=2, eps=1e-6,
            )
            camera = target_cameras[:, :6]
        else:
            normal = None
            colors = rgb
            camera = target_cameras
        logits = self.alpha_head(colors)
        if logits.shape != (colors.shape[0], 1, height, width):
            raise RuntimeError("RSS_VNEXT_GENERATOR_ALPHA_READOUT_SHAPE")
        alpha = torch.sigmoid(logits).reshape(b, -1, 1, height, width)
        if not bool(torch.isfinite(alpha).all()):
            raise RuntimeError("RSS_VNEXT_GENERATOR_ALPHA_NONFINITE")
        rgba = torch.cat((
            colors.reshape(b, -1, 3, height, width), alpha,
        ), dim=2)
        return RSSvNextStageGeneratedImagesV1(
            rgba=rgba, cameras=camera,
            normals=normal, stage=self.stage,
            checkpoint_sha256=self.checkpoint_sha256,
            generation_seed=seed,
        )


def admitted_alpha_head_loss_v1(
    alpha_head: nn.Module, *, renderer_rgb: Tensor, renderer_alpha: Tensor,
    trusted_valid_mask: Tensor,
) -> Tensor:
    """Separate trainable alpha supervision from independently rendered GT."""
    if (
        renderer_rgb.ndim != 4 or renderer_rgb.shape[1] != 3
        or renderer_alpha.shape != renderer_rgb[:, :1].shape
        or trusted_valid_mask.shape != renderer_alpha.shape
        or not all(bool(torch.isfinite(x).all()) for x in
                   (renderer_rgb, renderer_alpha, trusted_valid_mask))
        or bool((renderer_alpha < 0).any() or (renderer_alpha > 1).any())
        or bool((trusted_valid_mask < 0).any() or (trusted_valid_mask > 1).any())
        or not bool((trusted_valid_mask > 0).any())
    ):
        raise ValueError("RSS_VNEXT_GENERATOR_ALPHA_SUPERVISION_INVALID")
    prediction = alpha_head(renderer_rgb)
    if prediction.shape != renderer_alpha.shape:
        raise ValueError("RSS_VNEXT_GENERATOR_ALPHA_READOUT_SHAPE")
    error = F.binary_cross_entropy_with_logits(
        prediction, renderer_alpha, reduction="none",
    )
    return (error * trusted_valid_mask).sum() / trusted_valid_mask.sum()


__all__ = [
    "SCHEMA", "RSSvNextPinnedDiffusionStageV1",
    "RSSvNextStageGeneratedImagesV1", "admitted_alpha_head_loss_v1",
]
