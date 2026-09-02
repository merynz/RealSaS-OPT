from __future__ import annotations

import torch

from experiments.geppetto_arachne_r6_20260901.arachne_tail_objective_v1 import (
    top_fraction_row_l1_tail_v1,
)


def test_tail_objective_selects_largest_valid_rows_only() -> None:
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    pred = torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.0, 1.0]]])
    mask = torch.tensor([[True, True, True, False]])
    # Valid row-L1 errors are 0.0, 0.2, 1.0. With fraction 1/3, top-1 == 1.0.
    got = top_fraction_row_l1_tail_v1(pred, teacher, mask, fraction=1.0 / 3.0)
    assert torch.allclose(got, torch.tensor(1.0))


def test_tail_objective_is_permutation_invariant_over_surface_rows() -> None:
    g = torch.Generator().manual_seed(7)
    teacher = torch.softmax(torch.randn(1, 17, 5, generator=g), dim=-1)
    pred = torch.softmax(torch.randn(1, 17, 5, generator=g), dim=-1)
    mask = torch.ones((1, 17), dtype=torch.bool)
    perm = torch.randperm(17, generator=g)
    a = top_fraction_row_l1_tail_v1(pred, teacher, mask, fraction=0.10)
    b = top_fraction_row_l1_tail_v1(pred[:, perm], teacher[:, perm], mask[:, perm], fraction=0.10)
    assert torch.allclose(a, b)
