from __future__ import annotations

"""Contact-preserving post projection for Stage42 visual motion.

The geometry safety operator may repair disconnected source charts independently.
This module consumes a qualified contact set and only adds per-domain translations
needed to preserve the pre-repair contact deltas. Translation cannot change
triangle area/condition, so it composes safely after the existing SE(2) repair.
"""

import numpy as np

from .types import QualificationError


OPERATOR_ID = "QUALIFIED_CONTACT_DOMAIN_TRANSLATION_PROJECTION_V1"
POLICY = {
    "schema": "RealSaS.VisualContactMotionPolicy.v1",
    "operator_id": OPERATOR_ID,
    "constraint_target": "PRE_SAFETY_CONNECTED_PRESENTATION_DELTA",
    "unknowns": "ONE_XY_TRANSLATION_PER_QUALIFIED_CONTACT_DOMAIN",
    "regularization": 1e-8,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _solve_domain_translations(
    *,
    baseline_xy: np.ndarray,
    repaired_xy: np.ndarray,
    domain_id: np.ndarray,
    contact_pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)
    domains = np.asarray(domain_id, dtype=np.int32)
    if not len(pairs):
        return repaired_xy.copy(), np.zeros((0, 2), dtype=np.float64)

    active_domains = np.unique(domains[pairs])
    index = {int(domain): i for i, domain in enumerate(active_domains.tolist())}
    rows = []
    rhs = []
    for a, b in pairs.tolist():
        da, db = int(domains[a]), int(domains[b])
        if da == db:
            continue
        row = np.zeros(len(active_domains), dtype=np.float64)
        row[index[db]] = 1.0
        row[index[da]] = -1.0
        rows.append(row)
        rhs.append(
            (baseline_xy[b] - baseline_xy[a]) - (repaired_xy[b] - repaired_xy[a])
        )
    if not rows:
        return repaired_xy.copy(), np.zeros((len(active_domains), 2), dtype=np.float64)

    matrix = np.asarray(rows, dtype=np.float64)
    target = np.asarray(rhs, dtype=np.float64)
    ridge = float(POLICY["regularization"])
    augmented = np.vstack((matrix, np.sqrt(ridge) * np.eye(len(active_domains))))
    augmented_target = np.vstack((target, np.zeros((len(active_domains), 2))))
    translation, *_ = np.linalg.lstsq(augmented, augmented_target, rcond=None)

    # Contact equations constrain only relative translation. Keep the safety
    # result's absolute placement by removing the common null-space motion.
    translation -= translation.mean(axis=0, keepdims=True)
    result = repaired_xy.copy()
    for domain, row in index.items():
        result[domains == int(domain)] += translation[row]
    return result, translation


def apply_contact_projection(
    *,
    baseline_field,
    repaired_field,
    domain_id,
    contact_pairs,
) -> tuple[np.ndarray, dict]:
    baseline = np.asarray(baseline_field, dtype=np.float64)
    repaired = np.asarray(repaired_field, dtype=np.float64)
    domains = np.asarray(domain_id, dtype=np.int32)
    pairs = np.asarray(contact_pairs, dtype=np.int64).reshape(-1, 2)
    if (
        baseline.shape != repaired.shape
        or baseline.ndim != 2
        or baseline.shape[1] != 3
        or domains.shape != (len(baseline),)
        or np.any(domains < 0)
        or np.any(pairs < 0)
        or np.any(pairs >= len(baseline))
        or not np.isfinite(baseline).all()
        or not np.isfinite(repaired).all()
    ):
        raise QualificationError("VISUAL_CONTACT_MOTION_INPUT_INVALID")

    result = repaired.copy()
    result[:, :2], translations = _solve_domain_translations(
        baseline_xy=baseline[:, :2],
        repaired_xy=repaired[:, :2],
        domain_id=domains,
        contact_pairs=pairs,
    )
    before = (
        np.linalg.norm(
            repaired[pairs[:, 0], :2] - repaired[pairs[:, 1], :2]
            - (baseline[pairs[:, 0], :2] - baseline[pairs[:, 1], :2]),
            axis=1,
        )
        if len(pairs)
        else np.zeros(0)
    )
    after = (
        np.linalg.norm(
            result[pairs[:, 0], :2] - result[pairs[:, 1], :2]
            - (baseline[pairs[:, 0], :2] - baseline[pairs[:, 1], :2]),
            axis=1,
        )
        if len(pairs)
        else np.zeros(0)
    )
    return result, {
        "contact_pair_count": int(len(pairs)),
        "maximum_contact_delta_residual_before_px": float(np.max(before, initial=0.0)),
        "maximum_contact_delta_residual_after_px": float(np.max(after, initial=0.0)),
        "maximum_domain_translation_px": float(
            np.max(np.linalg.norm(translations, axis=1), initial=0.0)
        ),
    }


def compile_contact_projection(
    *,
    domain_id,
    contact_pairs,
    baseline_clip_fields,
    repaired_clip_fields,
) -> dict:
    if set(baseline_clip_fields) != set(repaired_clip_fields) or not baseline_clip_fields:
        raise QualificationError("VISUAL_CONTACT_MOTION_CLIP_SET_INVALID")
    outputs = {}
    rows = {}
    max_after = max_shift = 0.0
    for key in baseline_clip_fields:
        baseline = np.asarray(baseline_clip_fields[key], dtype=np.float64)
        repaired = np.asarray(repaired_clip_fields[key], dtype=np.float64)
        if baseline.shape != repaired.shape or baseline.ndim != 3 or baseline.shape[2] != 3:
            raise QualificationError("VISUAL_CONTACT_MOTION_CLIP_SHAPE_INVALID")
        frames = []
        diagnostics = []
        for fi in range(len(baseline)):
            value, diag = apply_contact_projection(
                baseline_field=baseline[fi],
                repaired_field=repaired[fi],
                domain_id=domain_id,
                contact_pairs=contact_pairs,
            )
            frames.append(value)
            diagnostics.append(diag)
            max_after = max(max_after, diag["maximum_contact_delta_residual_after_px"])
            max_shift = max(max_shift, diag["maximum_domain_translation_px"])
        outputs[key] = np.asarray(frames)
        rows[key] = diagnostics
    return {
        "projected_fields": outputs,
        "frame_diagnostics": rows,
        "maximum_contact_delta_residual_after_px": float(max_after),
        "maximum_domain_translation_px": float(max_shift),
    }
