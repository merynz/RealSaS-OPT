import copy

import torch

from models.geppetto.challengers.ar01_skeleton_causal_v1 import (
    GeppettoAR01ConfigV1,
    GeppettoAR01SkeletonCausalV1,
)
from models.geppetto.v2.geppetto_candidate_v2 import GeppettoCandidateConfigV2


def _fixture():
    torch.manual_seed(17)
    features = torch.randn(2, 20, 24)
    surface_pos = torch.randn(2, 20, 3)
    mask = torch.ones(2, 20, dtype=torch.bool)
    raster_xy = torch.tanh(torch.randn(2, 20, 8, 2))
    raster_valid = torch.ones(2, 20, 8, dtype=torch.bool)
    support = torch.ones(2, 20, 8, dtype=torch.bool)
    teacher = torch.tanh(torch.randn(2, 4, 3))
    parents = torch.tensor([[-1, 0, 1, 1], [-1, 0, 0, 2]], dtype=torch.long)
    return features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents


def _model():
    base = GeppettoCandidateConfigV2(
        surface_feature_dim=24,
        model_dim=12,
        attention_heads=3,
        position_scale=1.25,
        dropout=0.0,
    )
    return GeppettoAR01SkeletonCausalV1(GeppettoAR01ConfigV1(base=base))


def _forward(model, enabled, teacher, parents):
    features, surface_pos, mask, raster_xy, raster_valid, support, _, _ = _fixture()
    return model.forward_ar01(
        features,
        surface_pos,
        mask,
        raster_xy,
        raster_valid,
        support,
        decode_steps=4,
        mechanical_feedback_enabled=enabled,
        teacher_positions=teacher,
        teacher_parent_indices=parents,
    )


def test_ar01_same_class_same_state_initially_bit_identical_across_feedback_gate():
    features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents = _fixture()
    torch.manual_seed(123)
    model = _model().eval()
    with torch.no_grad():
        ar0 = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=False,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
        ar1 = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=True,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
    assert torch.equal(ar0.positions_normalized, ar1.positions_normalized)
    assert torch.equal(ar0.parent_logits, ar1.parent_logits)
    assert model._last_attention_token_count == 20


def test_ar01_gate_off_is_invariant_to_teacher_mechanical_state_after_channel_is_opened():
    features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents = _fixture()
    torch.manual_seed(123)
    model = _model().eval()
    with torch.no_grad():
        model.mechanical_feedback_fuse[-1].weight.fill_(0.01)
        model.mechanical_feedback_fuse[-1].bias.zero_()
        teacher_perturbed = teacher.clone()
        teacher_perturbed[:, 1:] += 0.37
        a = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=False,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
        b = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=False,
            teacher_positions=teacher_perturbed, teacher_parent_indices=parents,
        )
    assert torch.equal(a.positions_normalized, b.positions_normalized)


def test_ar01_gate_on_causally_changes_later_joint_when_teacher_mechanics_change():
    features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents = _fixture()
    torch.manual_seed(123)
    model = _model().eval()
    with torch.no_grad():
        model.mechanical_feedback_fuse[-1].weight.fill_(0.01)
        model.mechanical_feedback_fuse[-1].bias.zero_()
        teacher_perturbed = teacher.clone()
        teacher_perturbed[:, 1] += torch.tensor([0.31, -0.22, 0.17])
        a = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=True,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
        b = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=True,
            teacher_positions=teacher_perturbed, teacher_parent_indices=parents,
        )
    # Step 0/1 are produced before the altered step-1 mechanical state is fed back.
    assert torch.equal(a.positions_normalized[:, :2], b.positions_normalized[:, :2])
    assert float((a.positions_normalized[:, 2:] - b.positions_normalized[:, 2:]).abs().max()) > 0.0


def test_ar01_feedback_channel_has_gradient_only_when_enabled():
    features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents = _fixture()
    for enabled, expect_nonzero in ((False, False), (True, True)):
        torch.manual_seed(123)
        model = _model()
        out = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=enabled,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
        loss = out.positions_normalized[:, 2:].square().mean()
        loss.backward()
        grad = model.mechanical_feedback_fuse[-1].weight.grad
        nonzero = bool(grad is not None and torch.count_nonzero(grad).item() > 0)
        assert nonzero is expect_nonzero


def test_ar01_parent_logits_are_strictly_causal():
    features, surface_pos, mask, raster_xy, raster_valid, support, teacher, parents = _fixture()
    torch.manual_seed(123)
    model = _model().eval()
    with torch.no_grad():
        out = model.forward_ar01(
            features, surface_pos, mask, raster_xy, raster_valid, support,
            decode_steps=4, mechanical_feedback_enabled=True,
            teacher_positions=teacher, teacher_parent_indices=parents,
        )
    for step in range(4):
        assert torch.equal(
            out.parent_logits[:, step, step:],
            torch.full_like(out.parent_logits[:, step, step:], -1e4),
        )
