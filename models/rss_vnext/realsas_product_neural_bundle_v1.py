from __future__ import annotations

"""Single immutable neural release: canonicalizer + multiview + shared H(x).

All trainable state and frozen VAE are inside the same checkpoint artifact.
The original two source observations remain separately sealed. This is a
release *packaging* court, not scientific-quality or ProductRevision approval.
"""

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import nn

from .realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
    PairedSourceDiffusionConditionerV1,
    RowWiseRGBNormalViewAttentionV1,
)
from .realsas_two_stage_sampler_v1 import RSSvNextPinnedDiffusionStageV1
from .inference_checkpoint_v1 import (
    checkpoint_payload_v2, load_inference_checkpoint_v1,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel

BUNDLE_SCHEMA = "RealSaS.RSSvNext.UnifiedThreeStageNeuralRelease.v1"
MAX_BUNDLE_BYTES = 15 * 1024 * 1024 * 1024


def _digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_plain_config(config: Any) -> dict:
    try:
        raw = json.loads(json.dumps(dict(config), sort_keys=True, default=str))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("RSS_VNEXT_BUNDLE_MODULE_CONFIG_NOT_JSON") from exc
    if type(raw) is not dict or not raw:
        raise RuntimeError("RSS_VNEXT_BUNDLE_MODULE_CONFIG_EMPTY")
    return raw


def _stage_spec(stage: RSSvNextPinnedDiffusionStageV1) -> dict[str, Any]:
    if (
        type(stage) is not RSSvNextPinnedDiffusionStageV1
        or type(stage.denoiser).__name__ != "UNet2DConditionModel"
        or type(stage.vae).__name__ != "AutoencoderKL"
        or type(stage.scheduler).__name__ != "DDIMScheduler"
        or type(stage.alpha_head) is not nn.Conv2d
        or stage.alpha_head.in_channels != 3
        or stage.alpha_head.out_channels != 1
        or stage.alpha_head.kernel_size != (1, 1)
        or stage.alpha_head.bias is None
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_UNSUPPORTED_STAGE_ARCHITECTURE")
    cond = stage.conditioner
    spec = {
        "stage": stage.stage,
        "denoiser": _as_plain_config(stage.denoiser.config),
        "vae": _as_plain_config(stage.vae.config),
        "scheduler": _as_plain_config(stage.scheduler.config),
        "conditioner": {
            "source_width": cond.source_projection.in_features,
            "conditioning_width": cond.source_projection.out_features,
            "heads": cond.cross.num_heads,
        },
        "vae_scale_factor": stage.vae_scale_factor,
        "alpha": "CONV_3_TO_1_KERNEL_ONE",
    }
    row = stage.rowwise_attention
    if stage.stage == STAGE_MULTIVIEW:
        if type(row) is not RowWiseRGBNormalViewAttentionV1:
            raise RuntimeError("RSS_VNEXT_BUNDLE_MISSING_MULTIVIEW_ATTENTION")
        spec["rowwise"] = {
            "latent_channels": row.latent_channels,
            "hidden_width": row.feature.out_features,
            "heads": row.attention.num_heads,
            "max_tokens_per_row": row.max_tokens_per_row,
            "row_chunk": row.row_chunk,
        }
    else:
        if row is not None:
            raise RuntimeError("RSS_VNEXT_BUNDLE_CANONICAL_ROW_ATTENTION_FORBIDDEN")
        spec["rowwise"] = None
    return spec


def _build_stage_from_spec(
    spec: dict[str, Any], *, expected_stage: str,
) -> RSSvNextPinnedDiffusionStageV1:
    if (
        type(spec) is not dict
        or set(spec) != {
            "stage", "denoiser", "vae", "scheduler", "conditioner",
            "vae_scale_factor", "alpha", "rowwise",
        }
        or spec["stage"] != expected_stage
        or spec["alpha"] != "CONV_3_TO_1_KERNEL_ONE"
        or type(spec["conditioner"]) is not dict
        or set(spec["conditioner"]) != {
            "source_width", "conditioning_width", "heads",
        }
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_STAGE_SPEC_DRIFT")
    try:
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
    except ImportError as exc:
        raise RuntimeError("RSS_VNEXT_BUNDLE_DIFFUSERS_RUNTIME_REQUIRED") from exc
    unet = UNet2DConditionModel.from_config(spec["denoiser"])
    vae = AutoencoderKL.from_config(spec["vae"])
    scheduler = DDIMScheduler.from_config(spec["scheduler"])
    values = spec["conditioner"]
    conditioner = PairedSourceDiffusionConditionerV1(
        source_width=values["source_width"],
        conditioning_width=values["conditioning_width"],
        heads=values["heads"],
    )
    row = None
    if expected_stage == STAGE_MULTIVIEW:
        cfg = spec["rowwise"]
        if type(cfg) is not dict or set(cfg) != {
            "latent_channels", "hidden_width", "heads",
            "max_tokens_per_row", "row_chunk",
        }:
            raise RuntimeError("RSS_VNEXT_BUNDLE_ROW_SPEC_DRIFT")
        row = RowWiseRGBNormalViewAttentionV1(**cfg)
    elif spec["rowwise"] is not None:
        raise RuntimeError("RSS_VNEXT_BUNDLE_CANONICAL_ROW_DRIFT")
    return RSSvNextPinnedDiffusionStageV1(
        stage=expected_stage, denoiser=unet, vae=vae,
        conditioner=conditioner, alpha_head=nn.Conv2d(3, 1, 1),
        scheduler=scheduler, vae_scale_factor=spec["vae_scale_factor"],
        rowwise_attention=row,
    )


@dataclass(frozen=True)
class RSSvNextPinnedNeuralBundleV1:
    model: RSSvNextUnifiedCharacterModel
    canonical: RSSvNextPinnedDiffusionStageV1
    multiview: RSSvNextPinnedDiffusionStageV1
    checkpoint_sha256: str
    image_size: int
    denoise_steps: int
    source_authority: str = "TWO_ORIGINAL_OBSERVATIONS_ONLY"


def save_product_neural_bundle_v1(
    path: str | Path, *,
    model: RSSvNextUnifiedCharacterModel,
    canonical: RSSvNextPinnedDiffusionStageV1,
    multiview: RSSvNextPinnedDiffusionStageV1,
    image_size: int = 512,
    denoise_steps: int = 30,
) -> str:
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or canonical.stage != STAGE_CANONICAL or multiview.stage != STAGE_MULTIVIEW
        or model.cfg.constructed_total_views != 0
        or type(image_size) is not int or not 64 <= image_size <= 1024
        or image_size % 8 or type(denoise_steps) is not int
        or not 4 <= denoise_steps <= 80
        or canonical.conditioner.source_projection.in_features != model.cfg.iris.image_token_dim
        or multiview.conditioner.source_projection.in_features != model.cfg.iris.image_token_dim
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_NEURAL_ARCHITECTURE_OR_POLICY_INVALID")
    spec_a, spec_b = _stage_spec(canonical), _stage_spec(multiview)
    # A single CAS/ModelRelease artifact now binds all three trained stages.
    with tempfile.TemporaryDirectory(prefix="rss-vnext-release-build-") as temp:
        root = Path(temp)
        reconstruction = root / "reconstruction.pt"
        stage_a, stage_b = root / "canonical.pt", root / "multiview.pt"
        torch.save(checkpoint_payload_v2(model), reconstruction)
        canonical.save_training_weights(stage_a)
        multiview.save_training_weights(stage_b)
        payload = {
            "schema": BUNDLE_SCHEMA,
            "architecture_id": model.architecture_id,
            "construction": "TWO_ORIGINAL_IMAGES_TO_CANONICAL_TO_SIX_RGB_NORMAL_TO_SHARED_H",
            "source_observation_count": 2,
            "generated_conditioning_count": 6,
            "source_authority_escalation": False,
            "image_size": image_size,
            "denoise_steps": denoise_steps,
            "component_bytes": {
                "reconstruction": reconstruction.read_bytes(),
                "canonical": stage_a.read_bytes(),
                "multiview": stage_b.read_bytes(),
            },
            "component_sha256": {
                "reconstruction": _digest(reconstruction),
                "canonical": _digest(stage_a),
                "multiview": _digest(stage_b),
            },
            "stage_specs": {"canonical": spec_a, "multiview": spec_b},
        }
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, tmp)
        if tmp.stat().st_size > MAX_BUNDLE_BYTES:
            raise RuntimeError("RSS_VNEXT_BUNDLE_RESOURCE_BUDGET")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return _digest(target)


def load_product_neural_bundle_v1(
    path: str | Path, *, expected_sha256: str,
    vision_encoder: nn.Module, device: str | torch.device = "cpu",
    product_mode: bool = True,
) -> RSSvNextPinnedNeuralBundleV1:
    source = Path(path).resolve()
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
        or not source.is_file() or not 0 < source.stat().st_size <= MAX_BUNDLE_BYTES
        or _digest(source) != expected_sha256
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_CHECKPOINT_PIN_OR_BYTES")
    try:
        package = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("RSS_VNEXT_BUNDLE_CHECKPOINT_DECODE_INVALID") from exc
    if (
        type(package) is not dict
        or set(package) != {
            "schema", "architecture_id", "construction",
            "source_observation_count", "generated_conditioning_count",
            "source_authority_escalation", "image_size", "denoise_steps",
            "component_bytes", "component_sha256", "stage_specs",
        }
        or package["schema"] != BUNDLE_SCHEMA
        or package["architecture_id"] != RSSvNextUnifiedCharacterModel.architecture_id
        or package["construction"] != "TWO_ORIGINAL_IMAGES_TO_CANONICAL_TO_SIX_RGB_NORMAL_TO_SHARED_H"
        or package["source_observation_count"] != 2
        or package["generated_conditioning_count"] != 6
        or package["source_authority_escalation"] is not False
        or type(package["image_size"]) is not int
        or not 64 <= package["image_size"] <= 1024
        or package["image_size"] % 8
        or type(package["denoise_steps"]) is not int
        or not 4 <= package["denoise_steps"] <= 80
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_MANIFEST_INVALID")
    blobs, digests, specs = (
        package["component_bytes"], package["component_sha256"],
        package["stage_specs"],
    )
    required = {"reconstruction", "canonical", "multiview"}
    if (
        type(blobs) is not dict or set(blobs) != required
        or type(digests) is not dict or set(digests) != required
        or type(specs) is not dict or set(specs) != {"canonical", "multiview"}
        or any(
            type(blobs[k]) is not bytes
            or len(blobs[k]) > MAX_BUNDLE_BYTES
            or sha256(blobs[k]).hexdigest() != digests[k]
            for k in required
        )
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_COMPONENT_BINDING_DRIFT")
    with tempfile.TemporaryDirectory(prefix="rss-vnext-release-read-") as temp:
        root = Path(temp)
        files = {name: root / (name + ".pt") for name in required}
        for name in required:
            files[name].write_bytes(blobs[name])
        model = load_inference_checkpoint_v1(
            files["reconstruction"], expected_sha256=digests["reconstruction"],
            vision_encoder=vision_encoder, device=device,
            product_mode=product_mode,
        )
        if not getattr(model, "construction_requires_generated_multiview", False):
            raise RuntimeError("RSS_VNEXT_BUNDLE_RECONSTRUCTION_NOT_SIX_VIEW")
        stages = {}
        for name, stage_id in (
            ("canonical", STAGE_CANONICAL), ("multiview", STAGE_MULTIVIEW),
        ):
            runner = _build_stage_from_spec(specs[name], expected_stage=stage_id)
            runner.load_pinned_task_weights(
                files[name], expected_sha256=digests[name],
            )
            for module in runner._modules().values():
                module.to(device=device)
            stages[name] = runner
    if (
        model.cfg.constructed_total_views != 0
        or stages["canonical"].conditioner.source_projection.in_features != model.cfg.iris.image_token_dim
        or stages["multiview"].conditioner.source_projection.in_features != model.cfg.iris.image_token_dim
    ):
        raise RuntimeError("RSS_VNEXT_BUNDLE_GENERATOR_RECONSTRUCTION_DIMENSION_DRIFT")
    return RSSvNextPinnedNeuralBundleV1(
        model=model,
        canonical=stages["canonical"],
        multiview=stages["multiview"],
        checkpoint_sha256=expected_sha256,
        image_size=package["image_size"],
        denoise_steps=package["denoise_steps"],
    )


__all__ = [
    "BUNDLE_SCHEMA", "RSSvNextPinnedNeuralBundleV1",
    "save_product_neural_bundle_v1", "load_product_neural_bundle_v1",
]
