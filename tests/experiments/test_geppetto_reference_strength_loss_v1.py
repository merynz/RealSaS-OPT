from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_candidate_v1 import (
    GeppettoReferenceStrengthRawOutputV1,
)
from experiments.geppetto_reference_strength_fullstack_v1.geppetto_reference_strength_loss_v1 import (
    PreparedReferenceStrengthTargetV1,
    reference_strength_loss_v1,
)


def _output(*, teacher_feedback_used: bool = False, teacher_target_used: bool = True):
    j, n, d = 3, 5, 8
    z = torch.zeros
    return GeppettoReferenceStrengthRawOutputV1(
        coarse_positions_normalized=z((1, j, 3), requires_grad=True),
        positions_normalized=z((1, j, 3)),
        control_states=z((1, j, d)),
        stop_logits=z((1, j), requires_grad=True),
        existence_logits=z((1, j), requires_grad=True),
        root_logits=z((1, j), requires_grad=True),
        salience_logits=z((1, j), requires_grad=True),
        support_presence_logits=z((1, j), requires_grad=True),
        support_logits=z((1, j, n), requires_grad=True),
        internal_parent_logits=z((1, j, j), requires_grad=True),
        all_pair_parent_logits=z((1, j, j), requires_grad=True),
        position_log_sigma=z((1, j, 3), requires_grad=True),
        surface_attention=z((1, j, n)),
        diffusion_loss=torch.tensor(0.25, requires_grad=True),
        teacher_target_used=teacher_target_used,
        teacher_feedback_used=teacher_feedback_used,
    )


def _target():
    return PreparedReferenceStrengthTargetV1(
        positions_normalized=np.asarray(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.2], [-0.1, 0.0, 0.3]],
            dtype=np.float32,
        ),
        parent_indices=np.asarray([-1, 0, 0], dtype=np.int64),
        root_mask=np.asarray([True, False, False]),
        nearest_surface_indices=np.asarray(
            [[0, 1], [2, 3], [3, 4]], dtype=np.int64
        ),
    )


def test_loss_is_finite_and_backpropagates_without_teacher_feedback() -> None:
    out = _output()
    pieces = reference_strength_loss_v1(out, _target())
    assert set(pieces) >= {
        "position_smooth_l1", "diffusion", "stop", "root",
        "internal_parent", "final_parent", "support_index", "total",
    }
    assert all(torch.isfinite(v).all() for v in pieces.values())
    pieces["total"].backward()
    assert out.coarse_positions_normalized.grad is not None
    assert out.stop_logits.grad is not None
    assert out.all_pair_parent_logits.grad is not None


def test_teacher_feedback_causes_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="teacher feedback"):
        reference_strength_loss_v1(_output(teacher_feedback_used=True), _target())


def test_missing_teacher_objective_flag_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires teacher objective"):
        reference_strength_loss_v1(_output(teacher_target_used=False), _target())


def test_parent_before_child_target_is_required() -> None:
    target = PreparedReferenceStrengthTargetV1(
        positions_normalized=_target().positions_normalized,
        parent_indices=np.asarray([-1, 2, 0], dtype=np.int64),
        root_mask=np.asarray([True, False, False]),
        nearest_surface_indices=_target().nearest_surface_indices,
    )
    with pytest.raises(ValueError, match="parent-before-child"):
        reference_strength_loss_v1(_output(), target)
