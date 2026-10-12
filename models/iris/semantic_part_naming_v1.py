from __future__ import annotations

"""Provider-neutral region naming helpers for frozen part-semantic fields.

The functions in this module do not load external models and do not touch IRIS
geometry.  They consume dense per-label part probabilities plus a local mask,
then either emit one label or abstain.  This keeps the expensive PartCLIPSeg
runtime out of CI while making the qualification math directly testable.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartNamingDecisionV1:
    label_index: int
    top_probability: float
    second_probability: float
    margin: float
    localization_margin: float
    background_probability: float
    accepted: bool
    failure_code: str | None


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p) - np.log1p(-p)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    z -= float(np.max(z))
    e = np.exp(z)
    return e / max(float(np.sum(e)), 1e-12)


def classify_part_region_v1(
    *,
    part_probabilities,
    background_probability,
    mask,
    foreground,
    temperature: float = 0.55,
    minimum_top_probability: float = 0.34,
    minimum_class_margin: float = 0.07,
    minimum_localization_margin: float = 0.10,
    minimum_part_vs_background_margin: float = 0.02,
) -> PartNamingDecisionV1:
    """Classify one local candidate from dense frozen part probabilities.

    `part_probabilities` is [K,H,W], already aligned to the source image.
    Scores combine absolute in-mask confidence with contrast against the same
    label's visible-subject baseline.  A candidate may abstain for ambiguity,
    weak localization, or background dominance.
    """

    fields = np.asarray(part_probabilities, dtype=np.float64)
    bg = np.asarray(background_probability, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    fg = np.asarray(foreground, dtype=bool)
    if fields.ndim != 3 or fields.shape[0] < 2:
        raise ValueError("part_probabilities must be [K,H,W] with K>=2")
    if bg.shape != fields.shape[1:] or mask.shape != bg.shape or fg.shape != bg.shape:
        raise ValueError("part naming image shape mismatch")
    local = mask & fg
    if int(local.sum()) == 0:
        raise ValueError("empty local part candidate")
    if int(fg.sum()) == 0:
        raise ValueError("empty subject foreground")

    logits = _logit(fields)
    in_logits = logits[:, local].mean(axis=1)
    subject_baseline = np.median(logits[:, fg], axis=1)
    localization = in_logits - subject_baseline

    # Absolute confidence remains important; pure contrast can promote a tiny
    # relative bump for a label that the part model considers globally unlikely.
    absolute = fields[:, local].mean(axis=1)
    combined = in_logits + 0.65 * localization
    class_prob = _softmax(combined, temperature)
    order = np.argsort(-class_prob)
    top = int(order[0])
    second = int(order[1])
    top_prob = float(class_prob[top])
    second_prob = float(class_prob[second])
    margin = top_prob - second_prob
    loc_margin = float(localization[top])
    abs_top = float(absolute[top])
    bg_prob = float(np.mean(bg[local]))

    failure = None
    if top_prob < float(minimum_top_probability):
        failure = "PART_NAMING_LOW_TOP_PROBABILITY"
    elif margin < float(minimum_class_margin):
        failure = "PART_NAMING_AMBIGUOUS"
    elif loc_margin < float(minimum_localization_margin):
        failure = "PART_NAMING_NOT_LOCALIZED"
    elif abs_top - bg_prob < float(minimum_part_vs_background_margin):
        failure = "PART_NAMING_BACKGROUND_DOMINANT"

    return PartNamingDecisionV1(
        label_index=top if failure is None else -1,
        top_probability=top_prob,
        second_probability=second_prob,
        margin=float(margin),
        localization_margin=loc_margin,
        background_probability=bg_prob,
        accepted=failure is None,
        failure_code=failure,
    )
