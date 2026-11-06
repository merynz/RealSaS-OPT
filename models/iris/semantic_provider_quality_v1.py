from __future__ import annotations

"""Fail-closed quality gates for frozen semantic mask providers.

These checks run before semantic evidence is allowed to reach canonical-M fusion.
They detect the failure observed in the first Knight oracle, where many unrelated
labels collapsed to an almost identical whole-character mask.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SemanticProviderQualityV1:
    accepted_proposal_count: int
    distinct_label_count: int
    cross_label_pair_count: int
    near_duplicate_cross_label_pair_count: int
    maximum_cross_label_iou: float
    whole_subject_proposal_count: int
    whole_subject_fraction: float
    status: str
    failure_code: str | None


def mask_iou(a, b) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def audit_semantic_proposals_v1(
    proposals,
    *,
    subject_masks,
    near_duplicate_iou: float = 0.97,
    maximum_duplicate_pair_fraction: float = 0.10,
    whole_subject_coverage: float = 0.92,
    maximum_whole_subject_fraction: float = 0.25,
    minimum_distinct_labels: int = 3,
    minimum_proposals: int = 6,
) -> SemanticProviderQualityV1:
    """Reject collapsed or non-localized provider outputs.

    `subject_masks` is one foreground bool mask per source view. A proposal that
    covers almost the full visible subject is not useful part semantics and is
    counted as whole-subject evidence regardless of its declared label.
    """

    rows = list(proposals)
    subject = [np.asarray(x, dtype=bool) for x in subject_masks]
    labels = {str(x["label"]) for x in rows}
    pair_count = duplicate_count = 0
    maximum_iou = 0.0
    whole = 0

    for row in rows:
        view = int(row["view_index"])
        mask = np.asarray(row["mask"], dtype=bool)
        denom = int(np.count_nonzero(subject[view]))
        coverage = float(np.count_nonzero(mask & subject[view]) / max(denom, 1))
        if coverage >= whole_subject_coverage:
            whole += 1

    by_view = {}
    for i, row in enumerate(rows):
        by_view.setdefault(int(row["view_index"]), []).append(i)
    for ids in by_view.values():
        for ai in range(len(ids)):
            a = rows[ids[ai]]
            for bi in range(ai + 1, len(ids)):
                b = rows[ids[bi]]
                if str(a["label"]) == str(b["label"]):
                    continue
                pair_count += 1
                iou = mask_iou(a["mask"], b["mask"])
                maximum_iou = max(maximum_iou, iou)
                if iou >= near_duplicate_iou:
                    duplicate_count += 1

    duplicate_fraction = duplicate_count / max(pair_count, 1)
    whole_fraction = whole / max(len(rows), 1)
    failure = None
    if len(rows) < minimum_proposals:
        failure = "INSUFFICIENT_SEMANTIC_PROPOSALS"
    elif len(labels) < minimum_distinct_labels:
        failure = "INSUFFICIENT_LABEL_DIVERSITY"
    elif duplicate_fraction > maximum_duplicate_pair_fraction:
        failure = "PROVIDER_MASK_COLLAPSE"
    elif whole_fraction > maximum_whole_subject_fraction:
        failure = "WHOLE_SUBJECT_MASK_DOMINANCE"

    return SemanticProviderQualityV1(
        accepted_proposal_count=len(rows),
        distinct_label_count=len(labels),
        cross_label_pair_count=pair_count,
        near_duplicate_cross_label_pair_count=duplicate_count,
        maximum_cross_label_iou=maximum_iou,
        whole_subject_proposal_count=whole,
        whole_subject_fraction=whole_fraction,
        status="PASS" if failure is None else "FAIL",
        failure_code=failure,
    )


def require_semantic_provider_quality_v1(*args, **kwargs) -> SemanticProviderQualityV1:
    report = audit_semantic_proposals_v1(*args, **kwargs)
    if report.failure_code is not None:
        raise RuntimeError(report.failure_code)
    return report
