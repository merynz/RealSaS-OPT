from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from coords import grid_to_cell_index
from matcher import MatchResult, MatcherConfig, match_query


@dataclass
class CandidateQualification:
    candidate_index: int
    reciprocal_supported: bool
    reciprocal_reverse_rank: int | None
    cycle_support_views: List[int]


@dataclass
class QualifiedMatchEvidence:
    forward: MatchResult
    qualifications: List[CandidateQualification]
    fine_field_hw: tuple[int, int]


def exact_cell_rank(candidate_coords, truth_coord, hw: tuple[int, int], topk: int | None = None) -> int | None:
    """1-based rank of first candidate in the exact containing field cell; no pixel tolerance."""
    c = np.asarray(candidate_coords, np.float32)
    if topk is not None:
        c = c[:topk]
    if len(c) == 0:
        return None
    h, w = hw
    tx = int(grid_to_cell_index(float(truth_coord[0]), w))
    ty = int(grid_to_cell_index(float(truth_coord[1]), h))
    for i, q in enumerate(c, 1):
        qx = int(grid_to_cell_index(float(q[0]), w))
        qy = int(grid_to_cell_index(float(q[1]), h))
        if qx == tx and qy == ty:
            return i
    return None


def reciprocal_support(
    outputs,
    images,
    source_view: int,
    target_view: int,
    source_query,
    target_candidate,
    cfg: MatcherConfig,
    reverse_topk: int = 8,
):
    """Check whether target candidate maps back into source query's exact fine-field cell."""
    reverse = match_query(outputs, images, target_view, source_view, target_candidate, cfg)
    rank = exact_cell_rank(reverse.top_coords, source_query, reverse.fine_field_hw, reverse_topk)
    return rank is not None, rank


def cycle_support_views(
    outputs,
    images,
    source_view: int,
    target_view: int,
    source_query,
    target_candidate,
    cfg: MatcherConfig,
    cycle_branch_k: int = 2,
    return_topk: int = 8,
):
    """Third-view cycle evidence; returns supporting view IDs without hard filtering.

    Path: target candidate -> third-view top-k branch -> source. A third view supports the
    candidate if any bounded branch returns to the exact fine-field cell containing source_query.
    """
    support = []
    n_views = int(outputs["Z_coarse"].shape[1])
    for third in range(n_views):
        if third in (source_view, target_view):
            continue
        mid = match_query(outputs, images, target_view, third, target_candidate, cfg)
        ok = False
        for q3 in mid.top_coords[:cycle_branch_k]:
            back = match_query(outputs, images, third, source_view, q3, cfg)
            if exact_cell_rank(back.top_coords, source_query, back.fine_field_hw, return_topk) is not None:
                ok = True
                break
        if ok:
            support.append(third)
    return support


def qualify_match(
    outputs,
    images,
    source_view: int,
    target_view: int,
    source_query,
    cfg: MatcherConfig = MatcherConfig(),
    reciprocal_topk: int = 8,
    cycle_branch_k: int = 2,
    cycle_return_topk: int = 8,
    max_candidates: int | None = None,
) -> QualifiedMatchEvidence:
    """Attach deterministic support evidence to a set-valued forward match.

    This function does not delete candidates, invent singletons, or turn support into a hard
    confidence threshold. SurfaceBuilder/calibration may consume the evidence later.
    """
    forward = match_query(outputs, images, source_view, target_view, source_query, cfg)
    candidates = forward.top_coords
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    q = []
    for i, target_candidate in enumerate(candidates):
        recip, rank = reciprocal_support(
            outputs,
            images,
            source_view,
            target_view,
            source_query,
            target_candidate,
            cfg,
            reciprocal_topk,
        )
        cycles = cycle_support_views(
            outputs,
            images,
            source_view,
            target_view,
            source_query,
            target_candidate,
            cfg,
            cycle_branch_k,
            cycle_return_topk,
        )
        q.append(
            CandidateQualification(
                candidate_index=i,
                reciprocal_supported=recip,
                reciprocal_reverse_rank=rank,
                cycle_support_views=cycles,
            )
        )
    return QualifiedMatchEvidence(forward, q, forward.fine_field_hw)
