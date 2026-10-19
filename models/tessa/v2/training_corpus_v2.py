from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .config_v2 import TESSAStructuralizerConfigV2
from .generation_v2 import TESSAGrammarStateV2


@dataclass(frozen=True)
class TESSATrainingWindowPolicyV2:
    context_tokens: int = 4608
    score_tokens: int = 512
    stride_tokens: int = 512

    def validate(self, config: TESSAStructuralizerConfigV2) -> None:
        if min(self.context_tokens, self.score_tokens, self.stride_tokens) <= 0:
            raise ValueError("TESSA_V2_TRAINING_WINDOW_NONPOSITIVE")
        if self.context_tokens <= config.local_attention_window:
            raise ValueError("TESSA_V2_TRAINING_WINDOW_MUST_EXERCISE_GLOBAL_HISTORY")
        if self.score_tokens > self.context_tokens:
            raise ValueError("TESSA_V2_TRAINING_SCORE_EXCEEDS_CONTEXT")
        if self.stride_tokens != self.score_tokens:
            raise ValueError("TESSA_V2_TRAINING_TARGETS_MUST_BE_SCORED_EXACTLY_ONCE")


def _contiguous_nonpad_prefix(row: torch.Tensor, pad: int) -> int:
    if row.ndim != 1:
        raise ValueError("TESSA_V2_RECONSTRUCT_ROW_RANK_INVALID")
    mask = row.ne(int(pad))
    count = int(mask.sum().item())
    if count and not bool(mask[:count].all().item()):
        raise ValueError("TESSA_V2_RECONSTRUCT_NONPAD_NOT_PREFIX")
    if count < row.numel() and bool(mask[count:].any().item()):
        raise ValueError("TESSA_V2_RECONSTRUCT_PAD_HOLE")
    return count


def reconstruct_global_sequence_v2(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    offsets: torch.Tensor,
    window_count: int | torch.Tensor,
    *,
    config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
) -> torch.Tensor:
    """Reconstruct the exact global topology sequence from sealed V4 windows.

    V4 stores progressively expanding/sliding teacher-forced contexts and scores
    only the newest 512 next-token targets.  This function uses overlap equality
    as a fail-closed integrity check and returns the one global token sequence.
    """

    config.validate()
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("TESSA_V2_RECONSTRUCT_WINDOW_SHAPE_INVALID")
    if offsets.ndim != 1 or offsets.shape[0] != input_ids.shape[0]:
        raise ValueError("TESSA_V2_RECONSTRUCT_OFFSET_SHAPE_INVALID")
    count = int(window_count.item()) if isinstance(window_count, torch.Tensor) else int(window_count)
    if count <= 0 or count > input_ids.shape[0]:
        raise ValueError("TESSA_V2_RECONSTRUCT_WINDOW_COUNT_INVALID")

    assignments: dict[int, int] = {}

    def assign(position: int, token: int) -> None:
        if position < 0 or token < 0 or token >= config.vocab_size or token == config.PAD:
            raise ValueError("TESSA_V2_RECONSTRUCT_ASSIGNMENT_INVALID")
        previous = assignments.get(position)
        if previous is not None and previous != token:
            raise ValueError(
                f"TESSA_V2_RECONSTRUCT_OVERLAP_DRIFT:{position}:{previous}:{token}"
            )
        assignments[position] = token

    for row_index in range(count):
        offset = int(offsets[row_index].item())
        if offset < 0:
            raise ValueError("TESSA_V2_RECONSTRUCT_NEGATIVE_OFFSET")
        x = input_ids[row_index].long()
        y = labels[row_index].long()
        x_count = _contiguous_nonpad_prefix(x, config.PAD)
        if x_count <= 0:
            raise ValueError("TESSA_V2_RECONSTRUCT_EMPTY_ACTIVE_WINDOW")
        for local in range(x_count):
            assign(offset + local, int(x[local].item()))
        label_positions = torch.nonzero(y.ne(config.PAD), as_tuple=False).flatten().tolist()
        for local in label_positions:
            assign(offset + int(local) + 1, int(y[local].item()))

    if not assignments:
        raise ValueError("TESSA_V2_RECONSTRUCT_NO_ASSIGNMENTS")
    last = max(assignments)
    missing = [position for position in range(last + 1) if position not in assignments]
    if missing:
        raise ValueError(f"TESSA_V2_RECONSTRUCT_GLOBAL_HOLE:{missing[:8]}")
    sequence = torch.tensor([assignments[i] for i in range(last + 1)], dtype=torch.long)
    if int(sequence[0].item()) != config.BOS or int(sequence[-1].item()) != config.EOS:
        raise ValueError("TESSA_V2_RECONSTRUCT_BOS_EOS_INVALID")
    if bool(sequence.eq(config.PAD).any().item()):
        raise ValueError("TESSA_V2_RECONSTRUCT_PAD_IN_GLOBAL_SEQUENCE")

    # The same streaming grammar used at inference must accept the reconstructed
    # teacher sequence. This guards against silently training a different codec.
    state = TESSAGrammarStateV2.from_prefix(sequence.tolist(), config=config)
    if not state.complete:
        raise ValueError("TESSA_V2_RECONSTRUCT_GRAMMAR_NOT_COMPLETE")
    return sequence


def build_global_history_training_windows_v2(
    sequence: torch.Tensor | Iterable[int],
    *,
    policy: TESSATrainingWindowPolicyV2 = TESSATrainingWindowPolicyV2(),
    config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-width windows that score every transition exactly once.

    Later windows carry >4096 tokens of context, so V2's causal global-history
    summaries are exercised during the serious fit rather than remaining a
    source-only mechanism.
    """

    config.validate()
    policy.validate(config)
    seq = torch.as_tensor(tuple(sequence) if not isinstance(sequence, torch.Tensor) else sequence, dtype=torch.long).flatten()
    if seq.numel() < 2 or int(seq[0]) != config.BOS or int(seq[-1]) != config.EOS:
        raise ValueError("TESSA_V2_TRAINING_GLOBAL_SEQUENCE_INVALID")
    if int(seq.min()) < 0 or int(seq.max()) >= config.vocab_size or bool(seq.eq(config.PAD).any()):
        raise ValueError("TESSA_V2_TRAINING_GLOBAL_TOKEN_INVALID")

    transition_count = int(seq.numel() - 1)
    windows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    offsets: list[int] = []
    scored = torch.zeros(transition_count, dtype=torch.int16)

    for score_start in range(0, transition_count, policy.stride_tokens):
        score_end = min(transition_count, score_start + policy.score_tokens)
        context_end = score_end
        context_start = max(0, context_end - policy.context_tokens)
        x = seq[context_start:context_end]
        y = torch.full((x.numel(),), config.PAD, dtype=torch.long)
        local_score_start = score_start - context_start
        local_score_end = score_end - context_start
        if local_score_start < 0 or local_score_end > x.numel():
            raise AssertionError("TESSA_V2_TRAINING_WINDOW_SCORE_RANGE_INTERNAL")
        y[local_score_start:local_score_end] = seq[
            score_start + 1 : score_end + 1
        ]
        scored[score_start:score_end] += 1

        x_pad = torch.full((policy.context_tokens,), config.PAD, dtype=torch.long)
        y_pad = torch.full((policy.context_tokens,), config.PAD, dtype=torch.long)
        x_pad[: x.numel()] = x
        y_pad[: y.numel()] = y
        windows.append(x_pad)
        labels.append(y_pad)
        offsets.append(context_start)

    if not bool(scored.eq(1).all().item()):
        bad = torch.nonzero(scored.ne(1), as_tuple=False).flatten()[:8].tolist()
        raise AssertionError(f"TESSA_V2_TRAINING_TARGET_SCORE_CARDINALITY:{bad}")
    return (
        torch.stack(windows, dim=0),
        torch.stack(labels, dim=0),
        torch.tensor(offsets, dtype=torch.long),
    )
