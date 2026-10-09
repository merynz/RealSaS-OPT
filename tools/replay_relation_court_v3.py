"""Fast V7 contact court with exact source cuts and rigid domain closure.

This is a research-only causal court over the already sealed V6 evidence.  It
narrows contact truth to exact source-chart cuts that also pass the canonical /
skin qualification, keeps one pivot relation per domain pair, and asks whether a
proper SE(2) post-projection can close those contacts without reopening M/G/W.

The court deliberately treats qualified contact as the higher presentation
constraint: pre-safety chart separation is retained as a diagnostic, not as the
target that a qualified seam must preserve.
"""

from __future__ import annotations

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_contact_v1 import (
    ARTICULATED_CONTINUITY,
    ATTACHMENT_CONTACT,
    MATERIAL_CONTINUITY,
    QualifiedVisualContactSet,
)
from tools import replay_relation_court_v2 as strict


court = strict.court

POLICY = {
    "schema": "RealSaS.ExactSourceCutRigidContactCourt.v1",
    "source_contact": "EXACT_SOURCE_CUT_AFTER_CANONICAL_SKIN_QUALIFICATION",
    "relation_reduction": "ONE_PIVOT_PER_DOMAIN_PAIR__MATERIAL_PREFERRED",
    "projection": "INCREMENTAL_PROPER_SE2_PER_CONTACT_DOMAIN",
    "constraint_target": "COINCIDENT_QUALIFIED_SOURCE_CUT_PIVOT",
    "maximum_iterations": 6,
    "convergence_px": 1.0e-7,
    "regularization": 1.0e-10,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}


def _pivot_qualify(**kwargs) -> QualifiedVisualContactSet:
    value = strict._strict_qualify(**kwargs)
    if not len(value.pairs):
        return value

    pairs = np.asarray(value.pairs, dtype=np.int64)
    codes = np.asarray(value.relation_codes, dtype=np.int8)
    domains = np.asarray(value.domain_pairs, dtype=np.int32)
    distances = np.asarray(value.rest_distances_px, dtype=np.float64)
    scores = np.asarray(value.evidence_scores, dtype=np.float64)

    keep = []
    for domain_pair in sorted({tuple(map(int, row)) for row in domains.tolist()}):
        idx = np.flatnonzero(np.all(domains == np.asarray(domain_pair), axis=1))
        # A material seam is stronger evidence than an articulated pivot, then
        # prefer explicit attachment contact, evidence score, and stable indices.
        relation_priority = {
            MATERIAL_CONTINUITY: 0,
            ATTACHMENT_CONTACT: 1,
            ARTICULATED_CONTINUITY: 2,
        }
        best = min(
            idx.tolist(),
            key=lambda i: (
                relation_priority[int(codes[i])],
                float(distances[i]),
                -float(scores[i]),
                int(pairs[i, 0]),
                int(pairs[i, 1]),
            ),
        )
        keep.append(best)

    keep = np.asarray(keep, dtype=np.int64)
    out_pairs = pairs[keep].copy()
    out_codes = codes[keep].copy()
    out_domains = domains[keep].copy()
    out_distances = distances[keep].copy()
    out_scores = scores[keep].copy()
    digest = content_sha256(
        {
            "schema": POLICY["schema"],
            "parent_contact_hash": value.contact_hash,
            "policy_hash": content_sha256(POLICY),
            "pairs": out_pairs.astype(int).tolist(),
            "relation_codes": out_codes.astype(int).tolist(),
            "domain_pairs": out_domains.astype(int).tolist(),
        }
    )
    return QualifiedVisualContactSet(
        view_index=value.view_index,
        pairs=out_pairs,
        relation_codes=out_codes,
        domain_pairs=out_domains,
        rest_distances_px=out_distances,
        evidence_scores=out_scores,
        contact_hash=digest,
    )


def _rigid_contact_frame(*, baseline, repaired, domain_id, contact_pairs):
    baseline = np.asarray(baseline, dtype=np.float64)
    repaired = np.asarray(repaired, dtype=np.float64)
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
        raise QualificationError("EXACT_CONTACT_RIGID_COURT_INPUT_INVALID")
    if not len(pairs):
        return repaired.copy(), {
            "contact_pair_count": 0,
            "maximum_contact_delta_residual_before_px": 0.0,
            "maximum_contact_delta_residual_after_px": 0.0,
            "maximum_contact_gap_after_px": 0.0,
            "maximum_domain_translation_px": 0.0,
            "maximum_domain_rotation_rad": 0.0,
            "iteration_count": 0,
        }

    active = np.unique(domains[pairs])
    domain_index = {int(d): i for i, d in enumerate(active.tolist())}
    masks = {int(d): np.flatnonzero(domains == int(d)) for d in active.tolist()}
    result = repaired.copy()
    original_centroids = {
        d: repaired[idx, :2].mean(axis=0) for d, idx in masks.items()
    }
    total_rotation = {d: 0.0 for d in masks}
    iteration_count = 0

    # Linearized pose-graph solve, followed by exact rigid application.  Contact
    # rows are small (one qualified pivot per domain relation), so the system is
    # substantially cheaper than re-running any upstream stage.
    for iteration in range(int(POLICY["maximum_iterations"])):
        iteration_count = iteration + 1
        current_delta = result[pairs[:, 0], :2] - result[pairs[:, 1], :2]
        max_gap = float(np.max(np.linalg.norm(current_delta, axis=1), initial=0.0))
        if max_gap <= float(POLICY["convergence_px"]):
            break

        centers = {d: result[idx, :2].mean(axis=0) for d, idx in masks.items()}
        matrix = np.zeros((2 * len(pairs), 3 * len(active)), dtype=np.float64)
        target = np.zeros(2 * len(pairs), dtype=np.float64)
        for ri, (a, b) in enumerate(pairs.tolist()):
            da, db = int(domains[a]), int(domains[b])
            ia, ib = domain_index[da], domain_index[db]
            pa, pb = result[a, :2], result[b, :2]
            va = pa - centers[da]
            vb = pb - centers[db]
            # d(R(theta)v)/dtheta at theta=0 = [-vy, vx].
            ja = np.asarray([-va[1], va[0]], dtype=np.float64)
            jb = np.asarray([-vb[1], vb[0]], dtype=np.float64)
            row = 2 * ri
            matrix[row, 3 * ia] = ja[0]
            matrix[row + 1, 3 * ia] = ja[1]
            matrix[row, 3 * ia + 1] = 1.0
            matrix[row + 1, 3 * ia + 2] = 1.0
            matrix[row, 3 * ib] = -jb[0]
            matrix[row + 1, 3 * ib] = -jb[1]
            matrix[row, 3 * ib + 1] = -1.0
            matrix[row + 1, 3 * ib + 2] = -1.0
            target[row : row + 2] = -(pa - pb)

        ridge = float(POLICY["regularization"])
        augmented = np.vstack(
            (matrix, np.sqrt(ridge) * np.eye(matrix.shape[1], dtype=np.float64))
        )
        augmented_target = np.concatenate(
            (target, np.zeros(matrix.shape[1], dtype=np.float64))
        )
        step, *_ = np.linalg.lstsq(augmented, augmented_target, rcond=None)
        if not np.isfinite(step).all():
            raise QualificationError("EXACT_CONTACT_RIGID_COURT_SOLVE_NONFINITE")

        for domain, di in domain_index.items():
            theta = float(step[3 * di])
            translation = step[3 * di + 1 : 3 * di + 3]
            center = centers[domain]
            c, s = np.cos(theta), np.sin(theta)
            rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
            idx = masks[domain]
            result[idx, :2] = (
                (result[idx, :2] - center) @ rotation.T + center + translation
            )
            total_rotation[domain] += theta

    before_vector = (
        repaired[pairs[:, 0], :2]
        - repaired[pairs[:, 1], :2]
        - (baseline[pairs[:, 0], :2] - baseline[pairs[:, 1], :2])
    )
    after_vector = (
        result[pairs[:, 0], :2]
        - result[pairs[:, 1], :2]
        - (baseline[pairs[:, 0], :2] - baseline[pairs[:, 1], :2])
    )
    final_gap = np.linalg.norm(
        result[pairs[:, 0], :2] - result[pairs[:, 1], :2], axis=1
    )
    centroid_shift = [
        np.linalg.norm(result[idx, :2].mean(axis=0) - original_centroids[d])
        for d, idx in masks.items()
    ]
    return result, {
        "contact_pair_count": int(len(pairs)),
        "maximum_contact_delta_residual_before_px": float(
            np.max(np.linalg.norm(before_vector, axis=1), initial=0.0)
        ),
        "maximum_contact_delta_residual_after_px": float(
            np.max(np.linalg.norm(after_vector, axis=1), initial=0.0)
        ),
        "maximum_contact_gap_after_px": float(np.max(final_gap, initial=0.0)),
        "maximum_domain_translation_px": float(max(centroid_shift, default=0.0)),
        "maximum_domain_rotation_rad": float(
            max((abs(v) for v in total_rotation.values()), default=0.0)
        ),
        "iteration_count": int(iteration_count),
    }


def _rigid_compile_contact_projection(
    *, domain_id, contact_pairs, baseline_clip_fields, repaired_clip_fields
):
    if set(baseline_clip_fields) != set(repaired_clip_fields) or not baseline_clip_fields:
        raise QualificationError("EXACT_CONTACT_RIGID_COURT_CLIP_SET_INVALID")
    outputs = {}
    frame_diagnostics = {}
    max_delta = max_gap = max_shift = max_rotation = 0.0
    for key in baseline_clip_fields:
        baseline = np.asarray(baseline_clip_fields[key], dtype=np.float64)
        repaired = np.asarray(repaired_clip_fields[key], dtype=np.float64)
        if baseline.shape != repaired.shape or baseline.ndim != 3 or baseline.shape[2] != 3:
            raise QualificationError("EXACT_CONTACT_RIGID_COURT_CLIP_SHAPE_INVALID")
        rows = []
        frames = []
        for fi in range(len(baseline)):
            frame, diag = _rigid_contact_frame(
                baseline=baseline[fi],
                repaired=repaired[fi],
                domain_id=domain_id,
                contact_pairs=contact_pairs,
            )
            frames.append(frame)
            rows.append(diag)
            max_delta = max(max_delta, diag["maximum_contact_delta_residual_after_px"])
            max_gap = max(max_gap, diag["maximum_contact_gap_after_px"])
            max_shift = max(max_shift, diag["maximum_domain_translation_px"])
            max_rotation = max(max_rotation, diag["maximum_domain_rotation_rad"])
        outputs[key] = np.asarray(frames)
        frame_diagnostics[key] = rows
    return {
        "projected_fields": outputs,
        "frame_diagnostics": frame_diagnostics,
        "maximum_contact_delta_residual_after_px": float(max_delta),
        "maximum_contact_gap_after_px": float(max_gap),
        "maximum_domain_translation_px": float(max_shift),
        "maximum_domain_rotation_rad": float(max_rotation),
    }


def _contact_truth_metrics(
    *,
    rest_positions,
    baseline_positions,
    posed_positions,
    pairs,
    relation_codes,
    material_tolerance_px: float = 1.0e-6,
    articulated_tolerance_px: float = 1.5,
) -> dict:
    rest = np.asarray(rest_positions, dtype=np.float64)
    baseline = np.asarray(baseline_positions, dtype=np.float64)
    posed = np.asarray(posed_positions, dtype=np.float64)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(relation_codes, dtype=np.int8)
    if (
        rest.shape != baseline.shape
        or rest.shape != posed.shape
        or rest.ndim != 2
        or rest.shape[1] != 2
        or codes.shape != (len(pairs),)
        or np.any(pairs < 0)
        or np.any(pairs >= len(rest))
        or not all(np.isfinite(x).all() for x in (rest, baseline, posed))
    ):
        raise QualificationError("EXACT_CONTACT_RIGID_COURT_METRIC_INPUT_INVALID")
    if not len(pairs):
        return {
            "qualified_contact_relations_passed": True,
            "contact_pair_count": 0,
            "contact_failure_count": 0,
            "maximum_contact_repair_delta_residual_px": 0.0,
            "maximum_contact_gap_growth_px": 0.0,
        }

    rest_distance = np.linalg.norm(rest[pairs[:, 0]] - rest[pairs[:, 1]], axis=1)
    baseline_distance = np.linalg.norm(
        baseline[pairs[:, 0]] - baseline[pairs[:, 1]], axis=1
    )
    posed_distance = np.linalg.norm(posed[pairs[:, 0]] - posed[pairs[:, 1]], axis=1)
    repair_delta = np.abs(posed_distance - baseline_distance)
    gap_growth = np.maximum(posed_distance - rest_distance, 0.0)
    tolerance = np.where(
        codes == MATERIAL_CONTINUITY,
        float(material_tolerance_px),
        float(articulated_tolerance_px),
    )
    # Qualified seam/contact truth is the acceptance condition.  The amount by
    # which an already-separated pre-safety baseline had to move remains a
    # diagnostic and must not veto restoring a qualified contact.
    failure = gap_growth > tolerance
    return {
        "qualified_contact_relations_passed": bool(not np.any(failure)),
        "contact_pair_count": int(len(pairs)),
        "contact_failure_count": int(np.count_nonzero(failure)),
        "maximum_contact_repair_delta_residual_px": float(
            np.max(repair_delta, initial=0.0)
        ),
        "maximum_contact_gap_growth_px": float(np.max(gap_growth, initial=0.0)),
    }


court.qualify_visual_contacts = _pivot_qualify
court.compile_contact_projection = _rigid_compile_contact_projection
court.contact_relation_metrics = _contact_truth_metrics


if __name__ == "__main__":
    court.main()
