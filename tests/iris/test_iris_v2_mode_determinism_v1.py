from __future__ import annotations

import torch
from torch import nn

from models.iris.v2.local_refinement_v2 import LocalDepthRefinementV2
from models.iris.v2.ray_modes_v2 import RayModesV2, extract_ray_modes_v2


def test_equal_probability_modes_use_canonical_depth_index_order() -> None:
    logits = torch.full((1, 1, 9), -8.0)
    logits[..., 2] = 4.0
    logits[..., 6] = 4.0
    modes = extract_ray_modes_v2(logits, max_modes=3, min_probability=0.01)
    assert modes.mode_indices[0, 0, :2].tolist() == [2, 6]


def test_flat_local_maximum_plateau_collapses_to_one_mode() -> None:
    logits = torch.full((1, 1, 8), -8.0)
    logits[..., 3] = 5.0
    logits[..., 4] = 5.0
    modes = extract_ray_modes_v2(logits, max_modes=3, min_probability=0.01)
    valid_indices = modes.mode_indices[0, 0][modes.mode_valid[0, 0]].tolist()
    assert valid_indices == [3]


def test_low_confidence_multimodal_ray_is_not_automatically_ambiguous() -> None:
    logits = torch.zeros((1, 1, 32))
    logits[..., 5] = 0.2
    logits[..., 20] = 0.19
    modes = extract_ray_modes_v2(
        logits,
        max_modes=3,
        min_probability=0.02,
        ambiguity_ratio=0.55,
        ambiguity_primary_probability_min=0.05,
    )
    assert modes.mode_valid[0, 0, 0]
    assert not bool(modes.ambiguous[0, 0])


class _FixedOffset(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)


def test_boundary_refinement_has_same_local_authority_as_interior_bin_spacing() -> None:
    refine = LocalDepthRefinementV2(hidden_dim=4)
    refine.offset = _FixedOffset()
    hidden = torch.zeros((1, 1, 4, 4))
    depth = torch.tensor([[[0.0, 1.0, 2.0, 3.0]]])
    modes = RayModesV2(
        mode_indices=torch.tensor([[[0, 3]]]),
        mode_scores=torch.zeros((1, 1, 2)),
        mode_probabilities=torch.ones((1, 1, 2)),
        mode_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        ambiguous=torch.zeros((1, 1), dtype=torch.bool),
    )
    out = refine(hidden, depth, modes)
    lower_delta = abs(float(out[0, 0, 0] - depth[0, 0, 0]))
    upper_delta = abs(float(out[0, 0, 1] - depth[0, 0, 3]))
    assert abs(lower_delta - upper_delta) < 1e-7
