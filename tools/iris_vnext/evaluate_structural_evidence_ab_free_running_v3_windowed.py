#!/usr/bin/env python3
from __future__ import annotations

"""Exact-windowed fail-closed evaluator wrapper for frozen Structural-Evidence H1.

The frozen TESSA V3 checkpoint inherits the shipped V2 decoder whose self-attention
is WindowedCausalSelfAttentionV2. The previous evaluation wrapper accidentally
called the separate HierarchicalCausalSelfAttentionV2 incremental backend and
therefore failed before the first generated token.

This wrapper changes no scientific contract. It installs an incremental backend
that is algebraically matched to WindowedCausalSelfAttentionV2 and refuses to run
FIT8 until a deterministic tiny-model parity test against model.decode_tokens()
passes across both query-chunk and local-window boundaries.
"""

import json

import torch

from models.tessa.v2.court_v2 import evaluate_decoded_topology_v2
from models.tessa.v2.generation_v2 import (
    TESSAGeneratedCandidateV2,
    TESSAGenerationPolicyV2,
    TESSAGrammarStateV2,
    _choose_token,
)
from models.tessa.v2.windowed_incremental_generation_v2 import (
    TESSAWindowedIncrementalStateV2,
    assert_windowed_incremental_reference_parity_v2,
    incremental_decode_step_windowed_v2,
)
import tools.iris_vnext.evaluate_structural_evidence_ab_free_running_v2 as base


@torch.no_grad()
def generate_failclosed_windowed(
    model,
    *,
    memory: torch.Tensor,
    policy: TESSAGenerationPolicyV2,
    seed: int | None,
) -> TESSAGeneratedCandidateV2:
    """Strict greedy free-running on the checkpoint's actual windowed decoder."""

    policy.validate()
    if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[-1] != model.config.d_model:
        raise ValueError("TESSA_V2_WINDOWED_GENERATION_REQUIRES_SINGLE_ASSET_MEMORY")

    previous = model.training
    model.eval()
    grammar = TESSAGrammarStateV2()
    cache = TESSAWindowedIncrementalStateV2.for_model(model)
    tokens = [model.config.BOS]
    logp_sum = 0.0
    generator = None
    if seed is not None:
        generator = torch.Generator(device=memory.device)
        generator.manual_seed(int(seed))
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"

    try:
        logits = incremental_decode_step_windowed_v2(
            model,
            memory=memory,
            token_ids=torch.tensor(
                [model.config.BOS], device=memory.device, dtype=torch.long
            ),
            state=cache,
        )[0]
        for _ in range(policy.maximum_new_tokens):
            allowed = grammar.allowed_token_mask(config=model.config, device=memory.device)
            if not bool(allowed.any()):
                stop_reason = base.NO_ALLOWED_STOP
                break
            token, logp = _choose_token(
                logits,
                allowed,
                policy=policy,
                generator=generator,
            )
            tokens.append(token)
            logp_sum += logp
            grammar.advance(token, config=model.config)
            if grammar.complete:
                stop_reason = "EOS"
                break
            logits = incremental_decode_step_windowed_v2(
                model,
                memory=memory,
                token_ids=torch.tensor([token], device=memory.device, dtype=torch.long),
                state=cache,
            )[0]
    finally:
        model.train(previous)

    court = evaluate_decoded_topology_v2(tokens, config=model.config)
    return TESSAGeneratedCandidateV2(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(grammar.complete),
        stop_reason=stop_reason,
        token_log_probability_sum=float(logp_sum),
    )


def main() -> int:
    print("FREE_RUN PHASE stage=windowed_incremental_reference_parity_begin", flush=True)
    parity = assert_windowed_incremental_reference_parity_v2(device="cpu")
    print(
        "FREE_RUN PHASE stage=windowed_incremental_reference_parity_pass "
        + json.dumps(parity, sort_keys=True),
        flush=True,
    )
    base.generate_failclosed = generate_failclosed_windowed
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
