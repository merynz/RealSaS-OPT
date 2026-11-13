from __future__ import annotations

"""One execution: two original images -> canonical -> RGB/normal six -> H(x).

This is the same trainable scene constructor used by the six-view reconstruction
trainer. Stage task weights MUST be byte-pinned, and their output cannot be
substituted for original source observations. This module has no ProductPASS.
"""

from dataclasses import dataclass
from hashlib import sha256
import json

import torch
from torch import Tensor

from .realsas_two_stage_sampler_v1 import RSSvNextPinnedDiffusionStageV1
from .realsas_cleanroom_diffusion_training_v1 import (
    STAGE_CANONICAL, STAGE_MULTIVIEW,
)
from .unified_character_model import RSSvNextUnifiedCharacterModel


@dataclass(frozen=True)
class RSSvNextTwoSourceConstructionV1:
    proposal: dict
    canonical_rgba: Tensor
    generated_rgba: Tensor
    generated_normals: Tensor
    generated_cameras: Tensor
    canonical_checkpoint_sha256: str
    multiview_checkpoint_sha256: str
    construction_checkpoint_binding_sha256: str
    original_observed_view_count: int = 2
    generated_view_count: int = 6
    generated_is_observed: bool = False
    product_revision_minted: bool = False


@torch.no_grad()
def run_two_source_neural_construction_v1(
    *, model: RSSvNextUnifiedCharacterModel,
    observed_rgba: Tensor, original_camera_features: Tensor,
    source_binding_sha256: str,
    canonical_cameras: Tensor, six_view_cameras: Tensor,
    canonical_sampler: RSSvNextPinnedDiffusionStageV1,
    multiview_sampler: RSSvNextPinnedDiffusionStageV1,
    image_size: int, denoise_steps: int,
    canonical_seed: int, multiview_seed: int,
    max_topology_tokens: int | None = None,
    generate_controls: bool = False,
) -> RSSvNextTwoSourceConstructionV1:
    """Runs actual pinned stage samplers, never substitutes static test views.

    Until original arbitrary-view camera/pose intake is admitted, the final
    Compiler binding requires a legitimate original 13D camera pair. This
    function does not invent calibration to bypass that upstream dependency.
    """
    if (
        type(model) is not RSSvNextUnifiedCharacterModel
        or model.training or not isinstance(source_binding_sha256, str)
        or len(source_binding_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_binding_sha256)
        or canonical_sampler.stage != STAGE_CANONICAL
        or multiview_sampler.stage != STAGE_MULTIVIEW
        or canonical_sampler.checkpoint_sha256 is None
        or multiview_sampler.checkpoint_sha256 is None
    ):
        raise RuntimeError("RSS_VNEXT_CONSTRUCTION_MODEL_OR_TASK_RELEASE_UNQUALIFIED")
    tokens, _, _ = model._original_source_tokens(
        observed_rgba, original_camera_features,
    )
    b = observed_rgba.shape[0]
    if (
        canonical_cameras.shape != (b, 2, 13)
        or six_view_cameras.shape != (b, 6, 13)
        or canonical_cameras.device != observed_rgba.device
        or six_view_cameras.device != observed_rgba.device
    ):
        raise ValueError("RSS_VNEXT_CONSTRUCTION_VIRTUAL_CAMERA_DOMAIN")
    normal_modality = torch.zeros(
        (b, 2), device=observed_rgba.device, dtype=torch.long,
    )
    canonical = canonical_sampler.generate(
        source_tokens=tokens,
        target_cameras=canonical_cameras,
        target_modalities=normal_modality,
        height=image_size, width=image_size,
        steps=denoise_steps, seed=canonical_seed,
    )
    if (
        canonical.stage != STAGE_CANONICAL
        or canonical.rgba.shape != (b, 2, 4, image_size, image_size)
        or canonical.normals is not None
        or canonical.observation_authority
    ):
        raise RuntimeError("RSS_VNEXT_CONSTRUCTION_CANONICAL_OUTPUT_INVALID")
    canonical_tokens, _ = model.vision(canonical.rgba)
    doubled_cameras = torch.cat((six_view_cameras, six_view_cameras), dim=1)
    modalities = torch.cat((
        torch.zeros((b, 6), dtype=torch.long, device=observed_rgba.device),
        torch.ones((b, 6), dtype=torch.long, device=observed_rgba.device),
    ), dim=1)
    generated = multiview_sampler.generate(
        source_tokens=canonical_tokens,
        target_cameras=doubled_cameras,
        target_modalities=modalities,
        height=image_size, width=image_size,
        steps=denoise_steps, seed=multiview_seed,
    )
    if (
        generated.stage != STAGE_MULTIVIEW
        or generated.rgba.shape != (b, 6, 4, image_size, image_size)
        or generated.normals is None
        or generated.normals.shape != (b, 6, 3, image_size, image_size)
        or generated.observation_authority
        or not torch.equal(generated.cameras, six_view_cameras)
    ):
        raise RuntimeError("RSS_VNEXT_CONSTRUCTION_MULTIVIEW_OUTPUT_INVALID")
    combined = sha256(json.dumps({
        "canonical_task_checkpoint_sha256": canonical.checkpoint_sha256,
        "multiview_task_checkpoint_sha256": generated.checkpoint_sha256,
        "source_binding_sha256": source_binding_sha256,
        "construction_abi": "RealSaS.RSSvNext.PairedTwoStageConstruction.v1",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    prediction = model.propose_teacher_free(
        observed_rgba, original_camera_features,
        generated_rgba=generated.rgba,
        generated_cameras=generated.cameras,
        generated_normals=generated.normals,
        generation_checkpoint_sha256=combined,
        source_binding_sha256=source_binding_sha256,
        max_topology_tokens=max_topology_tokens,
        generate_controls=generate_controls,
        executable_constraint_operators=(
            None if not generate_controls else
            _allowed_product_constraint_operators_v1()
        ),
    )
    if (
        prediction["state"].scene_conditioning_authority
        != "GENERATED_SIX_VIEW_SHARED_SCENE"
        or prediction["state"].generated_view_count != 6
        or prediction["state"].observed_image_tokens.shape[1] != 2
    ):
        raise RuntimeError("RSS_VNEXT_CONSTRUCTION_SOURCE_OR_SCENE_DRIFT")
    return RSSvNextTwoSourceConstructionV1(
        proposal=prediction,
        canonical_rgba=canonical.rgba,
        generated_rgba=generated.rgba,
        generated_normals=generated.normals,
        generated_cameras=generated.cameras,
        canonical_checkpoint_sha256=canonical.checkpoint_sha256,
        multiview_checkpoint_sha256=generated.checkpoint_sha256,
        construction_checkpoint_binding_sha256=combined,
    )


def _allowed_product_constraint_operators_v1() -> tuple[str, ...]:
    from compiler.realsas_compiler_core.rig_constraint_schema_v1 import (
        RIG_CONSTRAINT_OPERATOR_IDS_V1,
    )
    return tuple(RIG_CONSTRAINT_OPERATOR_IDS_V1)


__all__ = [
    "RSSvNextTwoSourceConstructionV1",
    "run_two_source_neural_construction_v1",
]
