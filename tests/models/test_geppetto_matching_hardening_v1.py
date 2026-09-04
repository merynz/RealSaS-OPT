from __future__ import annotations

import inspect
import numpy as np
import torch

from models.geppetto.v2.geppetto_loss_v2 import GeppettoLossV2, canonical_geometry_assignment_v2
from models.geppetto.v2.geppetto_candidate_v2 import GeppettoRawOutputV2
from models.geppetto.v2.training_targets_v1 import GeppettoTeacherTargetV1


def _output() -> GeppettoRawOutputV2:
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
    modes = torch.stack([pos, pos + torch.tensor([0.0, 0.1, 0.0])], dim=2)
    parent = torch.tensor([[[ -9.0, 2.0, 0.0], [3.0, -9.0, 1.0], [0.5, 2.5, -9.0]]], dtype=torch.float32)
    return GeppettoRawOutputV2(
        positions_normalized=pos,
        position_log_sigma=torch.zeros((1, 3, 3)),
        existence_logits=torch.tensor([[2.0, 2.0, 2.0]]),
        stop_logits=torch.tensor([[-2.0, -1.0, 2.0]]),
        root_logits=torch.tensor([[2.5, -1.5, -2.0]]),
        support_presence_logits=torch.tensor([[2.0, 2.0, 2.0]]),
        parent_logits=parent,
        support_logits=torch.tensor([[[3.0, 2.0, 1.0], [2.5, 2.0, 1.5], [1.0, 2.0, 3.0]]]),
        control_states=torch.zeros((1, 3, 4)),
        abstain_logits=torch.tensor([-2.0]),
        position_modes_normalized=modes,
        position_mode_log_sigma=torch.zeros((1, 3, 2, 3)),
        position_mode_logits=torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]]),
    )


def _target(perm: np.ndarray | None = None) -> GeppettoTeacherTargetV1:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    parent = np.asarray([-1, 0, 1], np.int64)
    root = np.asarray([True, False, False])
    if perm is None:
        return GeppettoTeacherTargetV1(positions, parent, root, True)
    perm = np.asarray(perm, np.int64)
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm), dtype=np.int64)
    new_parent = []
    for old_id in perm.tolist():
        old_parent = int(parent[old_id])
        new_parent.append(-1 if old_parent < 0 else int(inverse[old_parent]))
    return GeppettoTeacherTargetV1(positions[perm], np.asarray(new_parent, np.int64), root[perm], True)


def test_coincident_teacher_rows_are_topology_loss_permutation_invariant() -> None:
    output = _output()
    loss = GeppettoLossV2(support_topk=2)
    surface = torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    a = loss(output, [_target()], surface, mask)
    b = loss(output, [_target(np.asarray([1, 0, 2], np.int64))], surface, mask)
    torch.testing.assert_close(a["total"], b["total"], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(a["root"], b["root"], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(a["parent"], b["parent"], atol=1e-7, rtol=1e-7)


def test_geometry_assignment_uses_no_fractional_float_tie_authority() -> None:
    source = inspect.getsource(canonical_geometry_assignment_v2)
    assert "astype(np.float64) +" not in source
    assert "composite integer" in source
    primary = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], np.float64)
    teacher = np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64)
    q, t = canonical_geometry_assignment_v2(primary, teacher)
    assert sorted((int(qi), tuple(teacher[int(ti)])) for qi, ti in zip(q, t)) == [
        (0, (0.0, 0.0, 0.0)),
        (1, (1.0, 0.0, 0.0)),
        (2, (2.0, 0.0, 0.0)),
    ]
