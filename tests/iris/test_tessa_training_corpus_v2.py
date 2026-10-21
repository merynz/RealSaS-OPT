from __future__ import annotations

import torch

from models.tessa.v2.config_v2 import TESSAStructuralizerConfigV2
from models.tessa.v2.training_corpus_v2 import (
    TESSATrainingWindowPolicyV2,
    build_global_history_training_windows_v2,
    reconstruct_global_sequence_v2,
)


def _teacher_sequence() -> torch.Tensor:
    cfg = TESSAStructuralizerConfigV2()
    # One component with enough adjacent faces to exceed the 4096-token local
    # window. Exact geometry is immaterial; tokenizer grammar is authoritative.
    tokens = [cfg.BOS, cfg.COMPONENT_BEGIN, cfg.CHART_BEGIN]
    for i in range(1500):
        if i == 0 or i % 7 == 0:
            tokens.append(cfg.FACE_BREAK)
            tokens.extend([(i + j) % cfg.coordinate_bins for j in range(9)])
        else:
            tokens.extend([(i + j) % cfg.coordinate_bins for j in range(3)])
    tokens.extend([cfg.CHART_END, cfg.COMPONENT_END, cfg.EOS])
    return torch.tensor(tokens, dtype=torch.long)


def _v4_like_windows(sequence: torch.Tensor):
    cfg = TESSAStructuralizerConfigV2()
    width = 2560
    scored = 512
    rows = []
    labels = []
    offsets = []
    transitions = len(sequence) - 1
    for score_start in range(0, transitions, scored):
        score_end = min(transitions, score_start + scored)
        end = score_end
        start = max(0, end - width)
        x = sequence[start:end]
        y = torch.full((len(x),), cfg.PAD, dtype=torch.long)
        local_start = score_start - start
        y[local_start : score_end - start] = sequence[score_start + 1 : score_end + 1]
        xp = torch.full((width,), cfg.PAD, dtype=torch.long)
        yp = torch.full((width,), cfg.PAD, dtype=torch.long)
        xp[: len(x)] = x
        yp[: len(y)] = y
        rows.append(xp); labels.append(yp); offsets.append(start)
    # V4 fixed bank may contain inactive PAD rows.
    for _ in range(3):
        rows.append(torch.full((width,), cfg.PAD, dtype=torch.long))
        labels.append(torch.full((width,), cfg.PAD, dtype=torch.long))
        offsets.append(0)
    return torch.stack(rows), torch.stack(labels), torch.tensor(offsets), len(rows) - 3


def test_reconstructs_one_global_sequence_from_overlapping_v4_windows():
    sequence = _teacher_sequence()
    x, y, offsets, count = _v4_like_windows(sequence)
    rebuilt = reconstruct_global_sequence_v2(x, y, offsets, count)
    assert torch.equal(rebuilt, sequence)


def test_global_training_windows_score_every_transition_once_and_exercise_history():
    cfg = TESSAStructuralizerConfigV2()
    policy = TESSATrainingWindowPolicyV2()
    sequence = _teacher_sequence()
    x, y, offsets = build_global_history_training_windows_v2(sequence, policy=policy, config=cfg)
    assert x.shape == y.shape
    assert x.shape[1] == 4608
    assert int((y != cfg.PAD).sum()) == len(sequence) - 1
    active_lengths = (x != cfg.PAD).sum(dim=1)
    assert int(active_lengths.max()) == 4608
    assert bool((active_lengths > cfg.local_attention_window).any())
    assert int(offsets.max()) > 0


def test_reconstruction_fails_closed_on_overlap_drift():
    sequence = _teacher_sequence()
    x, y, offsets, count = _v4_like_windows(sequence)
    # Pick an overlap covered by a later window and mutate only one copy.
    row = min(2, count - 1)
    pos = 10
    original = int(x[row, pos])
    x[row, pos] = (original + 1) % TESSAStructuralizerConfigV2().coordinate_bins
    try:
        reconstruct_global_sequence_v2(x, y, offsets, count)
    except ValueError as exc:
        assert "OVERLAP_DRIFT" in str(exc)
    else:
        raise AssertionError("expected overlap drift to fail closed")
