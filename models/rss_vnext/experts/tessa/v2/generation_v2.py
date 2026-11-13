from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch

from .config_v2 import TESSAStructuralizerConfigV2
from .court_v2 import TESSADecodedTopologyCourtV2, evaluate_decoded_topology_v2
from .model_v2 import TESSAStructuralizerV2


class TESSAGrammarPhaseV2(str, Enum):
    EXPECT_COMPONENT_BEGIN = "EXPECT_COMPONENT_BEGIN"
    EXPECT_CHART_BEGIN = "EXPECT_CHART_BEGIN"
    EXPECT_FACE_OR_CHART_END = "EXPECT_FACE_OR_CHART_END"
    READ_FACE_BREAK_VERTEX = "READ_FACE_BREAK_VERTEX"
    READ_ADJACENT_VERTEX = "READ_ADJACENT_VERTEX"
    EXPECT_COMPONENT_END = "EXPECT_COMPONENT_END"
    EXPECT_COMPONENT_OR_EOS = "EXPECT_COMPONENT_OR_EOS"
    COMPLETE = "COMPLETE"


@dataclass
class TESSAGrammarStateV2:
    """Strict streaming state for the V1-compatible TESSA mesh grammar."""

    phase: TESSAGrammarPhaseV2 = TESSAGrammarPhaseV2.EXPECT_COMPONENT_BEGIN
    coordinate_tokens_remaining: int = 0
    face_count: int = 0
    component_count: int = 0
    current_component_face_count: int = 0
    restart_count: int = 0
    token_count: int = 1  # BOS is already present.
    complete: bool = False

    @classmethod
    def from_prefix(
        cls,
        token_ids: Iterable[int],
        *,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> "TESSAGrammarStateV2":
        config.validate()
        tokens = tuple(map(int, token_ids))
        if not tokens or tokens[0] != config.BOS:
            raise ValueError("TESSA_V2_GENERATION_BOS_REQUIRED")
        state = cls()
        for token in tokens[1:]:
            state.advance(token, config=config)
        return state

    def allowed_token_mask(
        self,
        *,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        config.validate()
        mask = torch.zeros(config.vocab_size, dtype=torch.bool, device=device)
        if self.complete or self.phase == TESSAGrammarPhaseV2.COMPLETE:
            return mask

        coordinate_slice = slice(0, config.coordinate_bins)
        if self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_BEGIN:
            mask[config.COMPONENT_BEGIN] = True
        elif self.phase == TESSAGrammarPhaseV2.EXPECT_CHART_BEGIN:
            mask[config.CHART_BEGIN] = True
        elif self.phase == TESSAGrammarPhaseV2.READ_FACE_BREAK_VERTEX:
            mask[coordinate_slice] = True
        elif self.phase == TESSAGrammarPhaseV2.READ_ADJACENT_VERTEX:
            mask[coordinate_slice] = True
        elif self.phase == TESSAGrammarPhaseV2.EXPECT_FACE_OR_CHART_END:
            if self.face_count < config.max_faces:
                # A coordinate starts one adjacent face vertex (3 xyz tokens).
                mask[coordinate_slice] = True
                mask[config.FACE_BREAK] = True
            if self.current_component_face_count > 0:
                mask[config.CHART_END] = True
        elif self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_END:
            mask[config.COMPONENT_END] = True
        elif self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_OR_EOS:
            mask[config.COMPONENT_BEGIN] = True
            mask[config.EOS] = True
        else:
            raise RuntimeError("TESSA_V2_GENERATION_UNKNOWN_PHASE")
        return mask

    def advance(
        self,
        token: int,
        *,
        config: TESSAStructuralizerConfigV2 = TESSAStructuralizerConfigV2(),
    ) -> None:
        config.validate()
        token = int(token)
        if token < 0 or token >= config.vocab_size:
            raise ValueError("TESSA_V2_GENERATION_TOKEN_OUT_OF_RANGE")
        allowed = self.allowed_token_mask(config=config)
        if not bool(allowed[token]):
            raise ValueError(
                f"TESSA_V2_GENERATION_GRAMMAR_VIOLATION:{self.phase.value}:{token}"
            )
        self.token_count += 1

        is_coordinate = 0 <= token < config.coordinate_bins
        if self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_BEGIN:
            self.component_count += 1
            self.current_component_face_count = 0
            self.phase = TESSAGrammarPhaseV2.EXPECT_CHART_BEGIN
            return
        if self.phase == TESSAGrammarPhaseV2.EXPECT_CHART_BEGIN:
            self.phase = TESSAGrammarPhaseV2.EXPECT_FACE_OR_CHART_END
            return
        if self.phase == TESSAGrammarPhaseV2.EXPECT_FACE_OR_CHART_END:
            if token == config.CHART_END:
                self.phase = TESSAGrammarPhaseV2.EXPECT_COMPONENT_END
                return
            if token == config.FACE_BREAK:
                self.restart_count += 1
                self.coordinate_tokens_remaining = 9
                self.phase = TESSAGrammarPhaseV2.READ_FACE_BREAK_VERTEX
                return
            if is_coordinate:
                self.coordinate_tokens_remaining = 2
                self.phase = TESSAGrammarPhaseV2.READ_ADJACENT_VERTEX
                return
        if self.phase in (
            TESSAGrammarPhaseV2.READ_FACE_BREAK_VERTEX,
            TESSAGrammarPhaseV2.READ_ADJACENT_VERTEX,
        ):
            if not is_coordinate:
                raise AssertionError("TESSA_V2_GENERATION_INTERNAL_COORDINATE_DRIFT")
            self.coordinate_tokens_remaining -= 1
            if self.coordinate_tokens_remaining < 0:
                raise AssertionError("TESSA_V2_GENERATION_COORDINATE_COUNTER_UNDERFLOW")
            if self.coordinate_tokens_remaining == 0:
                self.face_count += 1
                self.current_component_face_count += 1
                if self.face_count > config.max_faces:
                    raise ValueError("TESSA_V2_GENERATION_FACE_POLICY_EXCEEDED")
                self.phase = TESSAGrammarPhaseV2.EXPECT_FACE_OR_CHART_END
            return
        if self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_END:
            self.phase = TESSAGrammarPhaseV2.EXPECT_COMPONENT_OR_EOS
            return
        if self.phase == TESSAGrammarPhaseV2.EXPECT_COMPONENT_OR_EOS:
            if token == config.EOS:
                self.phase = TESSAGrammarPhaseV2.COMPLETE
                self.complete = True
            else:
                self.component_count += 1
                self.current_component_face_count = 0
                self.phase = TESSAGrammarPhaseV2.EXPECT_CHART_BEGIN
            return
        raise AssertionError("TESSA_V2_GENERATION_STATE_TRANSITION_UNREACHABLE")


@dataclass(frozen=True)
class TESSAGenerationPolicyV2:
    maximum_new_tokens: int = 131072
    sampling: bool = False
    temperature: float = 1.0
    top_k: int = 0

    def validate(self) -> None:
        if self.maximum_new_tokens <= 0:
            raise ValueError("TESSA_V2_GENERATION_MAX_TOKENS_NONPOSITIVE")
        if not torch.isfinite(torch.tensor(float(self.temperature))) or self.temperature <= 0.0:
            raise ValueError("TESSA_V2_GENERATION_TEMPERATURE_INVALID")
        if self.top_k < 0:
            raise ValueError("TESSA_V2_GENERATION_TOP_K_NEGATIVE")


@dataclass
class TESSAGeneratedCandidateV2:
    token_ids: tuple[int, ...]
    topology_court: TESSADecodedTopologyCourtV2
    completed: bool
    stop_reason: str
    token_log_probability_sum: float


def _choose_token(
    logits: torch.Tensor,
    allowed: torch.Tensor,
    *,
    policy: TESSAGenerationPolicyV2,
    generator: torch.Generator | None,
) -> tuple[int, float]:
    if logits.ndim != 1 or allowed.shape != logits.shape:
        raise ValueError("TESSA_V2_GENERATION_LOGIT_MASK_SHAPE_INVALID")
    if not bool(allowed.any()):
        raise RuntimeError("TESSA_V2_GENERATION_NO_ALLOWED_TOKEN")
    masked = logits.float() / float(policy.temperature)
    masked = masked.masked_fill(~allowed, float("-inf"))
    if policy.top_k > 0:
        k = min(int(policy.top_k), int(allowed.sum().item()))
        threshold = torch.topk(masked, k=k).values[-1]
        masked = masked.masked_fill(masked < threshold, float("-inf"))
    log_probability = torch.log_softmax(masked, dim=-1)
    if policy.sampling:
        probability = torch.softmax(masked, dim=-1)
        token = int(torch.multinomial(probability, 1, generator=generator).item())
    else:
        token = int(torch.argmax(masked).item())
    return token, float(log_probability[token].item())


@torch.no_grad()
def generate_tessa_candidate_v2(
    model: TESSAStructuralizerV2,
    *,
    memory: torch.Tensor,
    policy: TESSAGenerationPolicyV2 = TESSAGenerationPolicyV2(),
    seed: int | None = None,
) -> TESSAGeneratedCandidateV2:
    """Strict free-running single-asset generation followed by topology court.

    This is a correctness/reference path, not the final high-throughput decoder:
    it re-evaluates the prefix and intentionally makes no 32k-scale throughput
    claim. A cached incremental implementation must prove token parity against
    this reference before replacing it.
    """

    policy.validate()
    cfg = model.config
    cfg.validate()
    if memory.ndim != 3 or memory.shape[0] != 1 or memory.shape[-1] != cfg.d_model:
        raise ValueError("TESSA_V2_GENERATION_REQUIRES_SINGLE_ASSET_MEMORY")

    device = memory.device
    tokens = [cfg.BOS]
    state = TESSAGrammarStateV2()
    logp_sum = 0.0
    rng = None
    if seed is not None:
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))

    model_was_training = model.training
    model.eval()
    stop_reason = "MAXIMUM_NEW_TOKENS_REACHED"
    try:
        for _ in range(policy.maximum_new_tokens):
            input_ids = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
            output = model.decode_tokens(memory=memory, input_ids=input_ids)
            logits = output.logits[0, -1]
            allowed = state.allowed_token_mask(config=cfg, device=device)
            token, logp = _choose_token(logits, allowed, policy=policy, generator=rng)
            tokens.append(token)
            logp_sum += logp
            state.advance(token, config=cfg)
            if state.complete:
                stop_reason = "EOS"
                break
    finally:
        model.train(model_was_training)

    court = evaluate_decoded_topology_v2(tokens, config=cfg)
    return TESSAGeneratedCandidateV2(
        token_ids=tuple(tokens),
        topology_court=court,
        completed=bool(state.complete),
        stop_reason=stop_reason,
        token_log_probability_sum=float(logp_sum),
    )
