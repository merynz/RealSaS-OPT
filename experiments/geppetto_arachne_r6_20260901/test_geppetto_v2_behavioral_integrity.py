from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from experiments.geppetto_arachne_r6_20260901.geppetto_candidate_v2 import GeppettoRawOutputV2
from experiments.geppetto_arachne_r6_20260901.geppetto_loss_v2 import GeppettoLossV2
from experiments.geppetto_arachne_r6_20260901.training_targets_v1 import GeppettoTeacherTargetV1


def _target(j: int) -> GeppettoTeacherTargetV1:
    x = np.linspace(-0.45, 0.45, j, dtype=np.float32)
    positions = np.stack([x, 0.11 * np.sin(np.arange(j)), 0.07 * np.cos(np.arange(j))], axis=-1).astype(np.float32)
    parents = np.asarray([-1] + list(range(j - 1)), np.int64)
    roots = np.asarray([True] + [False] * (j - 1), bool)
    return GeppettoTeacherTargetV1(positions, parents, roots, True)


def _raw_output(target: GeppettoTeacherTargetV1, *, surface_count: int | None = None) -> tuple[GeppettoRawOutputV2, torch.Tensor, torch.Tensor]:
    tp = torch.as_tensor(target.positions_normalized, dtype=torch.float32)
    j = len(tp)
    n = max(j, int(surface_count or j))
    surface = torch.zeros((1, n, 3), dtype=torch.float32)
    surface[0, :j] = tp
    if n > j:
        tail = torch.linspace(-0.9, 0.9, n - j)
        surface[0, j:, 0] = tail
        surface[0, j:, 1] = tail.flip(0) * 0.3
    valid = torch.ones((1, n), dtype=torch.bool)

    modes = tp[None, :, None, :].repeat(1, 1, 3, 1)
    mode_sigma = torch.zeros_like(modes)
    mode_logits = torch.full((1, j, 3), -12.0)
    mode_logits[:, :, 0] = 12.0
    positions = modes[:, :, 0].clone()
    position_sigma = mode_sigma[:, :, 0].clone()

    parent = torch.zeros((1, j, j), dtype=torch.float32)
    parent[:, torch.arange(j), torch.arange(j)] = -1e4
    support = torch.zeros((1, j, n), dtype=torch.float32)

    out = GeppettoRawOutputV2(
        positions_normalized=positions,
        position_log_sigma=position_sigma,
        existence_logits=torch.full((1, j), 12.0),
        stop_logits=torch.full((1, j), -12.0),
        root_logits=torch.where(torch.as_tensor(target.root_mask)[None], torch.tensor(12.0), torch.tensor(-12.0)),
        support_presence_logits=torch.full((1, j), 12.0),
        parent_logits=parent,
        support_logits=support,
        control_states=torch.zeros((1, j, 8), dtype=torch.float32),
        abstain_logits=torch.full((1,), -12.0),
        position_modes_normalized=modes,
        position_mode_log_sigma=mode_sigma,
        position_mode_logits=mode_logits,
    )
    return out, surface, valid


def test_secondary_mode_cannot_hide_wrong_shipping_map_locus() -> None:
    target = GeppettoTeacherTargetV1(
        np.asarray([[0.0, 0.0, 0.0]], np.float32),
        np.asarray([-1], np.int64),
        np.asarray([True]),
        True,
    )
    base, surface, valid = _raw_output(target, surface_count=4)
    eps = 1e-6
    wrong = torch.tensor([1.0, 1.0, 1.0])
    exact = torch.zeros(3)
    far = torch.tensor([-0.8, 0.7, -0.6])
    logits = torch.tensor([[[eps, 0.0, -12.0]]], dtype=torch.float32)
    sigma = torch.full((1, 1, 3, 3), -8.0)

    good_modes = torch.stack([exact, wrong, far])[None, None]
    bad_modes = torch.stack([wrong, exact, far])[None, None]

    good = base.__class__(
        **{**base.__dict__,
           "positions_normalized": exact[None, None],
           "position_log_sigma": sigma[:, :, 0],
           "position_modes_normalized": good_modes,
           "position_mode_log_sigma": sigma,
           "position_mode_logits": logits}
    )
    bad = base.__class__(
        **{**base.__dict__,
           "positions_normalized": wrong[None, None],
           "position_log_sigma": sigma[:, :, 0],
           "position_modes_normalized": bad_modes,
           "position_mode_log_sigma": sigma,
           "position_mode_logits": logits}
    )

    loss = GeppettoLossV2(support_topk=2)
    a = loss(good, [target], surface, valid)
    b = loss(bad, [target], surface, valid)

    # A secondary exact Gaussian must not make a wrong shipping MAP locus nearly
    # equivalent to an exact shipping MAP locus. This is a decode-alignment law,
    # not a family-specific tolerance.
    assert float(b["total"] - a["total"]) > 1.0
    assert float(a["total"]) >= 0.0
    assert float(b["total"]) >= 0.0


def test_stop_loss_cannot_dilute_the_terminal_positive_with_sequence_length() -> None:
    j = 17
    target = _target(j)
    out, surface, valid = _raw_output(target)
    p = 1.0 / float(j)
    logit = math.log(p / (1.0 - p))
    out.stop_logits[:] = logit

    losses = GeppettoLossV2(support_topk=2)(out, [target], surface, valid)
    terminal_positive_penalty = F.softplus(torch.tensor(-logit))

    # generate() uses a 0-logit / 0.5-probability first-hit boundary. The one
    # terminal positive therefore cannot receive only 1/J of the stop objective.
    assert losses["stop"] >= 0.5 * terminal_positive_penalty


def test_root_evidence_is_ranked_not_class_frequency_gamed() -> None:
    j = 9
    target = _target(j)
    out, surface, valid = _raw_output(target)
    p = 1.0 / float(j)
    logit = math.log(p / (1.0 - p))
    out.root_logits[:] = logit

    losses = GeppettoLossV2(support_topk=2)(out, [target], surface, valid)

    # With one true root, a constant class-prior score is not useful to the
    # Compiler's root selection. Equal positive/negative ranking starts at ln 2.
    assert losses["root"] >= math.log(2.0) - 1e-5


def test_parent_objective_matches_one_parent_ranking_semantics() -> None:
    j = 5
    target = _target(j)
    out, surface, valid = _raw_output(target)
    out.parent_logits.zero_()
    out.parent_logits[:, torch.arange(j), torch.arange(j)] = -1e4

    losses = GeppettoLossV2(support_topk=2)(out, [target], surface, valid)

    # Every non-root child chooses one parent among J-1 candidates. Uniform
    # evidence therefore has categorical loss ln(J-1), not sparse-pair BCE H(1/J).
    assert losses["parent"] >= math.log(float(j - 1)) - 1e-5


def test_support_objective_matches_topk_ranking_semantics() -> None:
    target = _target(1)
    n = 20
    topk = 2
    out, surface, valid = _raw_output(target, surface_count=n)
    p = topk / float(n)
    logit = math.log(p / (1.0 - p))
    out.support_logits[:] = logit

    losses = GeppettoLossV2(support_topk=topk)(out, [target], surface, valid)

    # propose() consumes support logits by ranking/top-k. A constant sparse class
    # prior must not be a cheap solution; equal positive/negative ranking is ln 2.
    assert losses["support"] >= math.log(2.0) - 1e-5
