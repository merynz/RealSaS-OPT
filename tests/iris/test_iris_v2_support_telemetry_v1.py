from __future__ import annotations

import torch

from models.iris.v2.depth_output_head_v2 import DepthSupportUncertaintyHeadV2
from models.iris.v2.ray_modes_v2 import RayModesV2


def test_support_head_preserves_conditional_and_mode_probability_separately() -> None:
    head = DepthSupportUncertaintyHeadV2(hidden_dim=4)
    hidden = torch.zeros((1, 1, 3, 4))
    modes = RayModesV2(
        mode_indices=torch.tensor([[[0, 1]]]),
        mode_scores=torch.tensor([[[2.0, 1.0]]]),
        mode_probabilities=torch.tensor([[[0.75, 0.25]]]),
        mode_valid=torch.ones((1, 1, 2), dtype=torch.bool),
        ambiguous=torch.zeros((1, 1), dtype=torch.bool),
    )
    out = head(hidden, modes)
    assert out.conditional_support_probability is not None
    assert out.mode_probability is not None
    torch.testing.assert_close(out.mode_probability, modes.mode_probabilities)
    torch.testing.assert_close(
        out.support_probability,
        out.conditional_support_probability * out.mode_probability,
    )


def test_invalid_mode_zeroes_all_three_support_quantities() -> None:
    head = DepthSupportUncertaintyHeadV2(hidden_dim=4)
    hidden = torch.zeros((1, 1, 2, 4))
    modes = RayModesV2(
        mode_indices=torch.tensor([[[0, -1]]]),
        mode_scores=torch.tensor([[[1.0, -1e4]]]),
        mode_probabilities=torch.tensor([[[1.0, 0.0]]]),
        mode_valid=torch.tensor([[[True, False]]]),
        ambiguous=torch.zeros((1, 1), dtype=torch.bool),
    )
    out = head(hidden, modes)
    assert float(out.support_probability[0, 0, 1]) == 0.0
    assert float(out.conditional_support_probability[0, 0, 1]) == 0.0
    assert float(out.mode_probability[0, 0, 1]) == 0.0
