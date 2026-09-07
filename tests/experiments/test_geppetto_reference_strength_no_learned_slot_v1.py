from __future__ import annotations

import torch

from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_no_learned_slot_v1 import (
    ARCHITECTURE_ID,
    GeppettoReferenceStrengthNoLearnedSlotV1,
    assert_no_learned_view_slot_identity_v1,
    deterministic_canonical_view_code_v1,
)


def test_view_code_is_fixed_finite_and_direction_distinguishing() -> None:
    a = deterministic_canonical_view_code_v1(64)
    b = deterministic_canonical_view_code_v1(64)
    assert a.shape == (8, 64)
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    assert torch.unique(a, dim=0).shape[0] == 8


def test_candidate_has_no_learned_absolute_view_slot_parameter() -> None:
    model = GeppettoReferenceStrengthNoLearnedSlotV1()
    assert model.config.architecture_id == ARCHITECTURE_ID
    assert_no_learned_view_slot_identity_v1(model)
    assert "encoder.view_index" in dict(model.named_buffers())
    assert "encoder.view_index" not in dict(model.named_parameters())
