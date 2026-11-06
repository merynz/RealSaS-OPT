from __future__ import annotations

"""Conservative multiview semantic fusion on canonical M.

V1 normalized label votes by the total surviving semantic weight on each face.
That makes a lone surviving label look maximally confident even when it has only
one weak source-view witness.  V2 keeps the same canonical M and downstream
relation contracts, but calibrates confidence with absolute evidence and requires
independent multiview support for the winning label.

No geometry, rig, skin, motion, runtime or product authority is modified.
"""

import numpy as np

from models.iris.semantic_evidence_v1 import FrozenSemanticMaskEvidenceV1
from .semantic_relation_evidence_v1 import (
    MeshSemanticEvidenceV1,
    _connected_semantic_owners,
    _face_visibility_samples,
)

SCHEMA = "RealSaS.SemanticMFusion.v2"


def calibrate_label_view_scores_v2(
    view_scores: np.ndarray,
    *,
    minimum_face_confidence: float = 0.45,
    minimum_label_support_views: int = 2,
):
    """Calibrate per-face label evidence without relative-confidence inflation.

    ``view_scores`` has shape ``[face, label, view]`` and stores at most one
    score per label/view (the maximum score if multiple same-label proposals hit
    the face in that view).

    For each label:
      dominance = label_score_sum / all_label_score_sum
      absolute = mean score over views that actually support that label
      calibrated = dominance * absolute

    A face qualifies only when the winning label itself has support from at
    least ``minimum_label_support_views`` independent views.
    """
    s = np.asarray(view_scores, dtype=np.float64)
    if s.ndim != 3 or s.shape[2] != 8:
        raise ValueError("view_scores must have shape [face,label,8]")
    if not np.isfinite(s).all() or np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("view_scores outside [0,1]")
    if not 0.0 <= float(minimum_face_confidence) <= 1.0:
        raise ValueError("minimum_face_confidence outside [0,1]")
    if int(minimum_label_support_views) < 1 or int(minimum_label_support_views) > 8:
        raise ValueError("minimum_label_support_views outside [1,8]")

    fcount, k, _ = s.shape
    if k == 0:
        return (
            np.zeros((fcount, 0), dtype=np.float64),
            np.zeros(fcount, dtype=np.float64),
            np.full(fcount, -1, dtype=np.int16),
            np.zeros((fcount, 8), dtype=bool),
            np.zeros((fcount, 0), dtype=np.int16),
        )

    support = s > 0.0
    support_count = support.sum(axis=2).astype(np.int16)
    label_sum = s.sum(axis=2)
    absolute = np.divide(
        label_sum,
        support_count,
        out=np.zeros_like(label_sum),
        where=support_count > 0,
    )
    total = label_sum.sum(axis=1)
    dominance = np.divide(
        label_sum,
        total[:, None],
        out=np.zeros_like(label_sum),
        where=total[:, None] > 0,
    )
    calibrated = dominance * absolute

    winner = calibrated.argmax(axis=1)
    rows = np.arange(fcount)
    confidence = calibrated[rows, winner]
    winner_support_count = support_count[rows, winner]
    qualified = (
        (confidence >= float(minimum_face_confidence))
        & (winner_support_count >= int(minimum_label_support_views))
    )
    label_index = np.where(qualified, winner, -1).astype(np.int16)

    winning_support = np.zeros((fcount, 8), dtype=bool)
    if np.any(qualified):
        qrows = rows[qualified]
        winning_support[qrows] = support[qrows, winner[qualified], :]

    return calibrated, confidence, label_index, winning_support, support_count


def lift_mask_evidence_to_m_v2(
    *,
    rest,
    faces,
    cameras,
    evidence: FrozenSemanticMaskEvidenceV1,
    minimum_face_confidence: float = 0.45,
    minimum_label_support_views: int = 2,
) -> MeshSemanticEvidenceV1:
    """Fuse modal 2D semantics onto M with label-specific multiview support."""
    rest = np.asarray(rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if rest.ndim != 2 or rest.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("invalid M domain")

    fcount = len(faces)
    k = len(evidence.labels)
    view_scores = np.zeros((fcount, k, 8), dtype=np.float64)
    samples = _face_visibility_samples(
        rest, faces, cameras, image_shape=(evidence.height, evidence.width)
    )
    by_view = [np.where(evidence.proposal_view_index == v)[0] for v in range(8)]

    for view in range(8):
        sx, sy, visible = samples[view]
        if not np.any(visible):
            continue
        for pi in by_view[view]:
            hit = visible & evidence.proposal_masks[pi, sy, sx]
            if not np.any(hit):
                continue
            score = float(evidence.proposal_score[pi])
            label_index = int(evidence.proposal_label_index[pi])
            # Same label / same view is corroboration, not additive probability.
            current = view_scores[:, label_index, view]
            current[hit] = np.maximum(current[hit], score)

    calibrated, confidence, label_index, winning_support, _ = calibrate_label_view_scores_v2(
        view_scores,
        minimum_face_confidence=minimum_face_confidence,
        minimum_label_support_views=minimum_label_support_views,
    )
    owner_index, owner_ids, owner_labels, owner_faces = _connected_semantic_owners(
        faces, label_index, evidence.labels
    )
    return MeshSemanticEvidenceV1(
        label_vocab=evidence.labels,
        face_probabilities=calibrated.astype(np.float32),
        face_confidence=confidence.astype(np.float32),
        face_support_views=winning_support,
        face_label_index=label_index,
        face_owner_index=owner_index,
        owner_ids=owner_ids,
        owner_label_index=owner_labels,
        owner_face_indices=owner_faces,
        source_evidence_hash=evidence.evidence_hash(),
        schema_version=SCHEMA,
    )
