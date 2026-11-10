from __future__ import annotations

"""Train the *discrete* topology policy with verified motion consequences.

NOT post-hoc "fix the mesh" code. With a fixed face set G3's continuous
surrogates backpropagate to rest positions and skin. They do NOT backpropagate
to the categorical choice of triangles. This module sends the dynamic
consequence back to the autoregressive topology decoder by preference loss
on candidate token log-probabilities. Source geometry/pose/court evidence is
used ONLY to produce train-time supervision, never as student conditions.

This is a preregistration candidate, not a final architecture/loss freeze.
"""
from dataclasses import dataclass
import hashlib

import torch
from torch import Tensor
import torch.nn.functional as F

from models.tessa.v3.config_v3 import TESSAStructuralizerConfigV3


@dataclass(frozen=True)
class RSSvNextMotionCourt:
    """Qualified, source-matched consequence labels for candidate ranking.

    Static and dynamic costs are nonnegative, lower is better. All candidate
    costs must derive from exactly the same source, motion test suite and
    canonical coordinate binding. Provenance is checked by the evidence
    builder and constrained here to explicit, nonempty identity strings.

    This is NOT the final Truth Foundry evidence/target schema.
    """

    source_sha256: str
    motion_suite_hash: str
    canonical_binding_hash: str
    # All topology candidates must be judged under the same qualified teacher
    # kinematics / skin-transfer protocol, not their independently predicted skins.
    source_skin_protocol_hash: str
    evaluation_stage: str  # must measure the original structural mesh before any polish
    candidate_hashes: tuple[str, ...]
    reference_authority: str  # exact source-relative comparison, NOT predicted-view authority
    static_cost: Tensor       # [K] source-match metric
    dynamic_cost: Tensor      # [K] source-relative G3/multi-pose metric
    valid_mask: Tensor        # [K] independent qualified hard-court/teacher coverage
    candidate_source_sha256: tuple[str, ...]
    candidate_motion_suite_hash: tuple[str, ...]
    candidate_canonical_binding_hash: tuple[str, ...]
    candidate_source_skin_protocol_hash: tuple[str, ...]

    def validate(self, k: int) -> None:
        if self.reference_authority != "QUALIFIED_SOURCE_RELATIVE_MOTION_COURT":
            raise ValueError("RSS_VNEXT_MOTION_SOURCE_AUTHORITY_INVALID")
        if self.evaluation_stage != "PRE_DYNAMIC_REFINEMENT_STRUCTURAL_MESH":
            raise ValueError("RSS_VNEXT_MOTION_POSTHOC_REPAIR_REWARD_FORBIDDEN")
        if min(map(len, (self.source_sha256, self.motion_suite_hash,
                         self.canonical_binding_hash,
                         self.source_skin_protocol_hash))) == 0:
            raise ValueError("RSS_VNEXT_MOTION_COURT_PROVENANCE_MISSING")
        for name, hashes, canonical in (
            ("SOURCE", self.candidate_source_sha256, self.source_sha256),
            ("MOTION", self.candidate_motion_suite_hash, self.motion_suite_hash),
            ("BINDING", self.candidate_canonical_binding_hash, self.canonical_binding_hash),
            ("SKIN_PROTOCOL", self.candidate_source_skin_protocol_hash, self.source_skin_protocol_hash),
        ):
            if len(hashes) != k or any(x != canonical for x in hashes):
                raise ValueError(f"RSS_VNEXT_MOTION_{name}_CANDIDATE_PROVENANCE_DRIFT")
        if len(self.candidate_hashes) != k or len(set(self.candidate_hashes)) != k:
            raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_HASHES_NONUNIQUE")
        if self.static_cost.shape != (k,) or self.dynamic_cost.shape != (k,) or self.valid_mask.shape != (k,):
            raise ValueError("RSS_VNEXT_MOTION_COURT_SHAPE_INVALID")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("RSS_VNEXT_MOTION_VALID_MASK_BOOL_REQUIRED")
        if not (torch.isfinite(self.static_cost).all() and torch.isfinite(self.dynamic_cost).all()):
            raise ValueError("RSS_VNEXT_MOTION_NONFINITE_COST")
        if bool((self.static_cost < 0).any() or (self.dynamic_cost < 0).any()):
            raise ValueError("RSS_VNEXT_MOTION_NEGATIVE_COST")


@dataclass(frozen=True)
class RSSvNextTopologyPreferenceConfig:
    static_equivalence_tolerance: float = 0.05
    min_dynamic_cost_gap: float = 0.05
    temperature: float = 1.0
    max_pairs: int = 32
    max_candidates: int = 8

    def validate(self):
        if (self.static_equivalence_tolerance < 0
                or self.min_dynamic_cost_gap <= 0 or self.temperature <= 0
                or self.max_pairs < 1 or not 2 <= self.max_candidates <= 32):
            raise ValueError("RSS_VNEXT_MOTION_PREFERENCE_CONFIG_INVALID")


@dataclass
class RSSvNextTopologyPreferenceResult:
    loss: Tensor
    candidate_log_prob: Tensor
    pair_count: int
    eligible_candidates: int
    preferred_pairs: tuple[tuple[int, int], ...]
    authority: str = "TRAINING_SOURCE_RELATIVE_PREFERENCE_ONLY"


def topology_candidate_log_probs(
    topology_decoder,
    structural_memory: Tensor,
    candidate_tokens: Tensor,
    *,
    config: TESSAStructuralizerConfigV3,
) -> Tensor:
    """Teacher-forced *sequence scoring*, not teacher topology conditioning.

    [1,K,L] includes BOS, at least one token, EOS, then PAD. Input prefix at
    position t predicts label at t+1, so the target never conditions itself.
    Returns length-normalized log probability per candidate, [K].
    """
    config.validate()
    if structural_memory.ndim != 3 or structural_memory.shape[0] != 1:
        raise ValueError("RSS_VNEXT_MOTION_DECODER_SINGLE_SOURCE_MEMORY")
    if (candidate_tokens.ndim != 3 or candidate_tokens.shape[0] != 1
            or candidate_tokens.shape[2] < 3 or candidate_tokens.dtype != torch.long):
        raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_TOKENS_INVALID")
    k = candidate_tokens.shape[1]
    if k < 2:
        raise ValueError("RSS_VNEXT_MOTION_REQUIRES_MULTIPLE_TOPOLOGY_CANDIDATES")
    if int(candidate_tokens.min()) < 0 or int(candidate_tokens.max()) >= config.vocab_size:
        raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_VOCAB")
    if not bool((candidate_tokens[:, :, 0] == config.BOS).all()):
        raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_BOS")
    # One EOS per candidate, only PAD after it. Training always knows where
    # valid sequence ends, independently of the autoregressive logits.
    for idx in range(k):
        seq = candidate_tokens[0, idx].tolist()
        if seq.count(config.EOS) != 1:
            raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_EOS")
        cut = seq.index(config.EOS)
        if cut < 2 or any(tok == config.PAD for tok in seq[:cut + 1]):
            raise ValueError("RSS_VNEXT_MOTION_EARLY_PAD")
        if any(tok != config.PAD for tok in seq[cut + 1:]):
            raise ValueError("RSS_VNEXT_MOTION_NONPAD_AFTER_EOS")
    # Repeated memory shares gradients with the one common two-image encoder.
    memory = structural_memory.expand(k, -1, -1)
    inp = candidate_tokens[0, :, :-1]
    label = candidate_tokens[0, :, 1:]
    logits = topology_decoder.decode_tokens(memory=memory, input_ids=inp).logits
    if logits.shape != (k, inp.shape[1], config.vocab_size):
        raise ValueError("RSS_VNEXT_MOTION_TOPOLOGY_LOGITS_SHAPE")
    log_probability = F.log_softmax(logits.float(), dim=-1)
    picked = log_probability.gather(-1, label.unsqueeze(-1)).squeeze(-1)
    valid = label != config.PAD
    count = valid.sum(-1)
    return (picked * valid).sum(-1) / count.clamp_min(1)


def source_relative_topology_preference_loss(
    candidate_log_prob: Tensor,
    candidate_tokens: Tensor,
    court: RSSvNextMotionCourt,
    *,
    config: RSSvNextTopologyPreferenceConfig = RSSvNextTopologyPreferenceConfig(),
) -> RSSvNextTopologyPreferenceResult:
    """Positive loss drives the *structural logits* toward better moving meshes.

    Pairwise comparison is permitted ONLY for candidates from the same source,
    same motion suite, same coordinate binding, comparable static quality.
    No preference is invented from absent pose/source evidence.
    """
    config.validate()
    if candidate_log_prob.ndim != 1:
        raise ValueError("RSS_VNEXT_MOTION_LOGPROB_SHAPE")
    k = candidate_log_prob.numel()
    if k < 2 or k > config.max_candidates or candidate_tokens.shape[:2] != (1, k):
        raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_COUNT")
    if not bool(torch.isfinite(candidate_log_prob).all()):
        raise ValueError("RSS_VNEXT_MOTION_LOGPROB_NONFINITE")
    court.validate(k)
    # Candidate identity must bind to the *exact* grammar token content; a
    # metadata-only hash is not adequate to protect this consequence join.
    actual_hashes = []
    for i in range(k):
        seq = candidate_tokens[0, i].detach().to(
            device="cpu", dtype=torch.int64,
        ).contiguous()
        # Hash sequence with padding as stored by the materialization court.
        actual_hashes.append(hashlib.sha256(seq.numpy().tobytes()).hexdigest())
    if tuple(actual_hashes) != court.candidate_hashes:
        raise ValueError("RSS_VNEXT_MOTION_CANDIDATE_TOKEN_HASH_DRIFT")
    if not (court.static_cost.device == candidate_log_prob.device
            and court.dynamic_cost.device == candidate_log_prob.device
            and court.valid_mask.device == candidate_log_prob.device):
        raise ValueError("RSS_VNEXT_MOTION_COST_DEVICE_DRIFT")
    pairs: list[tuple[int, int]] = []
    terms: list[Tensor] = []
    # Require distinct *actual* sequences, not merely distinct metadata hashes.
    for i in range(k):
        if not bool(court.valid_mask[i]):
            continue
        for j in range(i + 1, k):
            if not bool(court.valid_mask[j]):
                continue
            if torch.equal(candidate_tokens[0, i], candidate_tokens[0, j]):
                continue
            static_gap = float((court.static_cost[i] - court.static_cost[j]).abs().detach())
            if static_gap > config.static_equivalence_tolerance:
                continue
            dynamic_gap = float((court.dynamic_cost[i] - court.dynamic_cost[j]).abs().detach())
            if dynamic_gap < config.min_dynamic_cost_gap:
                continue
            winner, loser = (i, j) if court.dynamic_cost[i] < court.dynamic_cost[j] else (j, i)
            terms.append(F.softplus(
                -(candidate_log_prob[winner] - candidate_log_prob[loser]) / config.temperature,
            ))
            pairs.append((winner, loser))
            if len(terms) == config.max_pairs:
                break
        if len(terms) == config.max_pairs:
            break
    if not terms:
        raise ValueError("RSS_VNEXT_MOTION_NO_QUALIFIED_DISTINCT_PREFERENCE_PAIR")
    return RSSvNextTopologyPreferenceResult(
        loss=torch.stack(terms).mean(),
        candidate_log_prob=candidate_log_prob,
        pair_count=len(pairs),
        eligible_candidates=int(court.valid_mask.sum()),
        preferred_pairs=tuple(pairs),
    )


__all__ = [
    "RSSvNextMotionCourt", "RSSvNextTopologyPreferenceConfig",
    "RSSvNextTopologyPreferenceResult", "topology_candidate_log_probs",
    "source_relative_topology_preference_loss",
]
