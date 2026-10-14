from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .contracts_v1 import validate_preserve_classes_v1


def _as_probabilities_v1(
    semantic: torch.Tensor,
    *,
    input_is_logits: bool,
) -> torch.Tensor:
    if semantic.ndim < 2:
        raise ValueError("UNIFIED_IRIS_SEMANTIC_TENSOR_RANK_INVALID")
    if not torch.isfinite(semantic).all():
        raise ValueError("UNIFIED_IRIS_SEMANTIC_NONFINITE")
    if input_is_logits:
        return torch.softmax(semantic, dim=-1)
    if torch.any(semantic < 0.0) or torch.any(semantic > 1.0):
        raise ValueError("UNIFIED_IRIS_SEMANTIC_PROBABILITY_OUT_OF_RANGE")
    sums = semantic.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
        raise ValueError("UNIFIED_IRIS_SEMANTIC_PROBABILITY_NOT_NORMALIZED")
    return semantic


def semantic_equivalent_sdf_v1(
    sdf: torch.Tensor,
    semantic: torch.Tensor,
    preserve_classes: Sequence[int],
    *,
    input_is_logits: bool = False,
) -> torch.Tensor:
    """Construct an SDF for a preserved semantic subset.

    Clean-room implementation of the semantic-equivalent SDF formulation
    described in StdGEN/StdGEN++. For a preserved subset P the semantic margin
    is max(non-P) - max(P), and the equivalent field is max(original_sdf,
    semantic_margin). This keeps the original exterior hard constraint while
    allowing differentiable extraction of selected semantic regions.

    RealSaS uses this only for normalized categorical/subset semantics. Subject-
    local owner embeddings remain relational descriptors and are not silently
    converted into global categorical IDs.
    """

    if sdf.shape != semantic.shape[:-1]:
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_SDF_SHAPE_INVALID")
    if not torch.isfinite(sdf).all():
        raise ValueError("UNIFIED_IRIS_SDF_NONFINITE")

    probabilities = _as_probabilities_v1(semantic, input_is_logits=input_is_logits)
    class_count = probabilities.shape[-1]
    preserve = validate_preserve_classes_v1(preserve_classes, class_count)

    if len(preserve) == class_count:
        return sdf

    preserve_index = torch.tensor(preserve, device=semantic.device, dtype=torch.long)
    complement = tuple(i for i in range(class_count) if i not in preserve)
    complement_index = torch.tensor(complement, device=semantic.device, dtype=torch.long)

    preserve_score = torch.index_select(probabilities, -1, preserve_index).amax(dim=-1)
    reject_score = torch.index_select(probabilities, -1, complement_index).amax(dim=-1)
    semantic_margin = reject_score - preserve_score
    return torch.maximum(sdf, semantic_margin)


def semantic_equivalent_nerf_weights_v1(
    density_raw: torch.Tensor,
    deltas: torch.Tensor,
    preserve_probability: torch.Tensor,
    *,
    density_bias: float = 0.0,
) -> torch.Tensor:
    """Volume-rendering weights for one arbitrary preserved region set.

    Clean-room implementation of the semantic-equivalent NeRF formulation in
    StdGEN++. The preserved semantic mass gates opacity. The API intentionally
    accepts an arbitrary `preserve_probability` rather than fixed human class
    IDs, so RealSaS can later supply probabilities from subject-local,
    permutation-invariant semantic regions.

    This is training evidence only. NeRF density is not canonical geometry
    authority; the signed field plus compiler-qualified explicit carrier remain
    the geometry authority path.
    """

    if density_raw.shape != deltas.shape or density_raw.shape != preserve_probability.shape:
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_SHAPE_INVALID")
    if not torch.isfinite(density_raw).all() or not torch.isfinite(deltas).all():
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_NONFINITE")
    if torch.any(deltas < 0):
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_NEGATIVE_DELTA")
    if torch.any(preserve_probability < 0.0) or torch.any(preserve_probability > 1.0):
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_PROBABILITY_OUT_OF_RANGE")

    sigma = F.softplus(density_raw + float(density_bias))
    alpha = 1.0 - torch.exp(-sigma * deltas)
    equivalent_alpha = alpha * preserve_probability
    shifted = torch.cat(
        [torch.ones_like(equivalent_alpha[..., :1]), 1.0 - equivalent_alpha + 1e-10],
        dim=-1,
    )
    transmittance = torch.cumprod(shifted, dim=-1)[..., :-1]
    return equivalent_alpha * transmittance


def semantic_equivalent_nerf_render_v1(
    density_raw: torch.Tensor,
    rgb: torch.Tensor,
    deltas: torch.Tensor,
    preserve_probability: torch.Tensor,
    *,
    white_background: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB and accumulated opacity for a generic semantic region set."""

    if rgb.shape[:-1] != density_raw.shape or rgb.shape[-1] not in (3, 4):
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_RGB_SHAPE_INVALID")
    if not torch.isfinite(rgb).all():
        raise ValueError("UNIFIED_IRIS_EQUIVALENT_NERF_RGB_NONFINITE")
    weights = semantic_equivalent_nerf_weights_v1(
        density_raw,
        deltas,
        preserve_probability,
    )
    color = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
    opacity = weights.sum(dim=-1)
    if white_background:
        color = color + (1.0 - opacity).unsqueeze(-1)
    return color, opacity


def coarse_to_fine_active_mask_v1(
    coarse_sdf: torch.Tensor,
    high_resolution_shape: tuple[int, int, int],
    *,
    dilation_kernel: int = 3,
) -> torch.Tensor:
    """StdGEN++-style sparse proposal mask for high-resolution field queries.

    The coarse negative-SDF occupancy is morphologically dilated and then
    nearest-neighbor upsampled. Expensive fine SDF/deformation/semantic queries
    can be restricted to the returned active domain. Additional RealSaS-specific
    owner-boundary/uncertainty proposals should be unioned by callers rather
    than weakening this geometry safety mask.
    """

    if coarse_sdf.ndim not in (3, 4):
        raise ValueError("UNIFIED_IRIS_COARSE_SDF_GRID_RANK_INVALID")
    if not torch.isfinite(coarse_sdf).all():
        raise ValueError("UNIFIED_IRIS_COARSE_SDF_NONFINITE")
    if len(high_resolution_shape) != 3 or min(map(int, high_resolution_shape)) <= 0:
        raise ValueError("UNIFIED_IRIS_HIGH_RESOLUTION_SHAPE_INVALID")
    kernel = int(dilation_kernel)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("UNIFIED_IRIS_DILATION_KERNEL_MUST_BE_ODD_POSITIVE")

    single = coarse_sdf.ndim == 3
    grid = coarse_sdf.unsqueeze(0) if single else coarse_sdf
    occupancy = (grid < 0.0).to(dtype=torch.float32).unsqueeze(1)
    dilated = F.max_pool3d(
        occupancy,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    high = F.interpolate(
        dilated,
        size=tuple(map(int, high_resolution_shape)),
        mode="nearest",
    ).squeeze(1) > 0.5
    return high[0] if single else high


def union_refinement_evidence_v1(
    geometry_active: torch.Tensor,
    *,
    semantic_boundary: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
    semantic_threshold: float = 0.5,
    uncertainty_threshold: float = 0.5,
) -> torch.Tensor:
    """RealSaS extension for generic high-detail refinement allocation.

    StdGEN++ specializes a second branch on the face. RealSaS cannot assume a
    human face, so refinement is allocated generically to geometry-active,
    semantic-boundary, and high-uncertainty regions. All optional tensors must
    already be aligned to the high-resolution extraction grid.
    """

    if geometry_active.dtype != torch.bool:
        raise ValueError("UNIFIED_IRIS_GEOMETRY_ACTIVE_MASK_MUST_BE_BOOL")
    result = geometry_active.clone()
    for name, tensor, threshold in (
        ("SEMANTIC_BOUNDARY", semantic_boundary, semantic_threshold),
        ("UNCERTAINTY", uncertainty, uncertainty_threshold),
    ):
        if tensor is None:
            continue
        if tensor.shape != result.shape:
            raise ValueError(f"UNIFIED_IRIS_{name}_REFINEMENT_SHAPE_INVALID")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"UNIFIED_IRIS_{name}_REFINEMENT_NONFINITE")
        result |= tensor >= float(threshold)
    return result


__all__ = [
    "coarse_to_fine_active_mask_v1",
    "semantic_equivalent_nerf_render_v1",
    "semantic_equivalent_nerf_weights_v1",
    "semantic_equivalent_sdf_v1",
    "union_refinement_evidence_v1",
]
