from __future__ import annotations

"""Chain supervision is source-native arithmetic, not synthetic wing labels."""
import pytest
import torch

from models.rss_vnext.qualified_chain_skin_v1 import (
    qualified_source_chain_skin_loss_v1,
)


def _fixture():
    # Source-native joint order: body0, body1, wing0, wing1, unknown.
    source = torch.tensor([[
        [.7, .3, 0., 0., 0.],
        [0., 0., .6, .4, 0.],
        [.1, 0., .45, .45, 0.],
        [0., 0., 0., .4, .6],
    ]], dtype=torch.float32)
    owners = torch.tensor([10, 10, 42, 42, -1], dtype=torch.long)
    mask = torch.ones((1, 4), dtype=torch.bool)
    return source, owners, mask


def test_exact_chain_partition_replays_source_and_preserves_gradient_path():
    source, owners, mask = _fixture()
    student = source.clone().detach().requires_grad_(True)
    r = qualified_source_chain_skin_loss_v1(
        predicted_skin=student, source_skin=source,
        joint_chain_ids=owners, qualified_vertex_mask=mask,
    )
    assert r.qualified_chain_count == 2
    assert r.eligible_vertex_count == 3
    assert r.eligible_conditional_count > 0
    assert r.total.item() == pytest.approx(0.)
    changed = source.clone()
    changed[0, 2] = torch.tensor([.45, .15, .2, .2, 0.])
    changed.requires_grad_()
    r = qualified_source_chain_skin_loss_v1(
        predicted_skin=changed, source_skin=source,
        joint_chain_ids=owners, qualified_vertex_mask=mask,
    )
    assert r.total.item() > 0.
    r.total.backward()
    assert changed.grad is not None
    assert changed.grad[0, 2, :4].abs().sum().item() > 0.
    assert changed.grad[0, 3].abs().sum().item() == 0.


def test_names_and_numbering_have_no_effect_and_unknown_is_not_negative():
    source, owners, mask = _fixture()
    student = source.clone()
    renamed = torch.tensor([900, 900, 3, 3, -1], dtype=torch.long)
    a = qualified_source_chain_skin_loss_v1(
        predicted_skin=student, source_skin=source,
        joint_chain_ids=owners, qualified_vertex_mask=mask,
    )
    b = qualified_source_chain_skin_loss_v1(
        predicted_skin=student, source_skin=source,
        joint_chain_ids=renamed, qualified_vertex_mask=mask,
    )
    assert a.total.item() == b.total.item() == 0.
    assert a.eligible_vertex_count == b.eligible_vertex_count == 3


def test_missing_or_false_chain_truth_fails_closed():
    source, owners, mask = _fixture()
    def run(*, w=source, c=owners, v=mask):
        return qualified_source_chain_skin_loss_v1(
            predicted_skin=source, source_skin=w,
            joint_chain_ids=c, qualified_vertex_mask=v,
        )
    with pytest.raises(ValueError, match="NO_SOURCE_CHAINS"):
        run(c=torch.full_like(owners, -1))
    with pytest.raises(ValueError, match="ALL_VERTICES_AMBIGUOUS"):
        run(c=torch.tensor([-1, -1, -1, -1, 3]))
    with pytest.raises(ValueError, match="NO_QUALIFIED_VERTICES"):
        run(v=torch.zeros_like(mask))
    with pytest.raises(ValueError, match="SOURCE_NOT_NORMALIZED"):
        run(w=source * .5)
    with pytest.raises(ValueError, match="SOURCE_DOMAIN_INVALID"):
        run(c=owners.float())
    with pytest.raises(ValueError, match="SOURCE_DOMAIN_INVALID"):
        bad = source.clone()
        bad[0, 0, 0] = float("nan")
        run(w=bad)
