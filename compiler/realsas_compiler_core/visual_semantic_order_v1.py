from __future__ import annotations

"""Semantic presentation order compiled from frozen skin/attachment ownership.

Inter-slot occlusion is decided by one stable rank per semantic motion owner;
canonical camera depth remains authoritative only inside a slot. This is the
Spine-class drawing-order idea expressed over RealSaS' already-qualified
mechanical evidence, without changing M/G/W.
"""

import numpy as np

from .hashing import content_sha256
from .types import QualificationError


OPERATOR_ID = "SEMANTIC_SLOT_ORDER_WITH_INTRA_SLOT_CANONICAL_DEPTH_V1"
POLICY = {
    "schema": "RealSaS.SemanticPresentationOrderPolicy.v1",
    "operator_id": OPERATOR_ID,
    "body_slot_source": "DOMINANT_FROZEN_SKIN_COEFFICIENT",
    "target_attachment_slot_source": "EXPLICIT_TARGET_SLOT_OWNER",
    "inter_slot_order": "FRAME_MEDIAN_CANONICAL_DEPTH_NEAR_TO_FAR",
    "intra_slot_order": "CANONICAL_CAMERA_DEPTH",
    "temporal_tie_break": "PREVIOUS_FRAME_ORDER_THEN_STABLE_SLOT_ID",
    "relative_tie_band": 1e-4,
    "semantic_depth_band": 1.0,
    "intra_slot_depth_fraction": 0.25,
    "mechanical_state_mutation_authorized": False,
    "runtime_inference_authorized": False,
}
CONTRACT = "SEMANTIC_SLOT_ORDER_THEN_CANONICAL_DEPTH_WITHIN_SLOT_V1"


def semantic_vertex_slots(
    *,
    motion_blend_coefficients,
    vertex_attachment_owner,
    attachment_target_joint_by_owner=None,
) -> np.ndarray:
    blend = np.asarray(motion_blend_coefficients, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    if (
        blend.ndim != 2
        or not len(blend)
        or not blend.shape[1]
        or owners.shape != (len(blend),)
        or np.any(owners < 0)
        or not np.isfinite(blend).all()
        or np.any(blend < -1e-12)
        or not np.allclose(blend.sum(axis=1), 1.0, atol=1e-8, rtol=0)
    ):
        raise QualificationError("SEMANTIC_ORDER_SLOT_INPUT_INVALID")
    slots = np.argmax(blend, axis=1).astype(np.int32)
    target = {int(k): int(v) for k, v in (attachment_target_joint_by_owner or {}).items()}
    joint_count = int(blend.shape[1])
    for owner in np.unique(owners):
        if int(owner) == 0:
            continue
        if int(owner) not in target:
            raise QualificationError("SEMANTIC_ORDER_ATTACHMENT_TARGET_MISSING")
        # Attachment identity stays distinct from the body joint it follows.
        slots[owners == owner] = joint_count + int(owner) - 1
    return slots


def semantic_face_slots(
    *,
    visual_faces,
    motion_blend_coefficients,
    vertex_attachment_owner,
    attachment_target_joint_by_owner=None,
) -> np.ndarray:
    faces = np.asarray(visual_faces, dtype=np.int64)
    blend = np.asarray(motion_blend_coefficients, dtype=np.float64)
    owners = np.asarray(vertex_attachment_owner, dtype=np.int32)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or np.any(faces < 0)
        or np.any(faces >= len(blend))
    ):
        raise QualificationError("SEMANTIC_ORDER_FACE_INPUT_INVALID")
    result = np.empty(len(faces), dtype=np.int32)
    joint_count = int(blend.shape[1])
    for fi, face in enumerate(faces):
        face_owners = owners[face]
        attachment = face_owners[face_owners > 0]
        if len(attachment):
            if np.any(attachment != attachment[0]):
                raise QualificationError("SEMANTIC_ORDER_FACE_ATTACHMENT_MIXED")
            result[fi] = joint_count + int(attachment[0]) - 1
        else:
            result[fi] = int(np.argmax(blend[face].mean(axis=0)))
    return result


def _stable_slot_order(depth: np.ndarray, slots: np.ndarray, previous: list[int] | None):
    unique = sorted(map(int, np.unique(slots)))
    medians = {slot: float(np.median(depth[slots == slot])) for slot in unique}
    values = np.asarray(list(medians.values()), dtype=np.float64)
    span = float(np.ptp(values)) if len(values) else 0.0
    tie = max(1e-9, span * float(POLICY["relative_tie_band"]))
    previous_rank = {slot: i for i, slot in enumerate(previous or unique)}
    ordered = sorted(
        unique,
        key=lambda slot: (
            int(np.floor(medians[slot] / tie + 0.5)),
            previous_rank.get(slot, len(unique) + slot),
            slot,
        ),
    )
    return ordered, medians


def compile_semantic_order(
    *,
    canonical_depths,
    semantic_vertex_slot,
) -> dict:
    depths = np.asarray(canonical_depths, dtype=np.float64)
    slots = np.asarray(semantic_vertex_slot, dtype=np.int32)
    if (
        depths.ndim != 2
        or slots.shape != (depths.shape[1],)
        or np.any(slots < 0)
        or not np.isfinite(depths).all()
        or np.any(depths <= 0)
    ):
        raise QualificationError("SEMANTIC_ORDER_DEPTH_INPUT_INVALID")
    unique = sorted(map(int, np.unique(slots)))
    slot_to_column = {slot: i for i, slot in enumerate(unique)}
    rank_rows = np.full((len(depths), len(unique)), -1, dtype=np.int32)
    effective = np.empty_like(depths)
    previous = None
    transitions = 0
    medians_rows = []

    for fi, depth in enumerate(depths):
        order, medians = _stable_slot_order(depth, slots, previous)
        if previous is not None and order != previous:
            transitions += 1
        previous = order
        rank = {slot: i for i, slot in enumerate(order)}
        for slot, value in rank.items():
            rank_rows[fi, slot_to_column[slot]] = int(value)

        # Each semantic slot gets a disjoint numeric band. Canonical depth only
        # resolves self-occlusion inside that band.
        row = np.empty_like(depth)
        for slot in unique:
            selected = slots == slot
            local = depth[selected]
            lo, hi = float(np.min(local)), float(np.max(local))
            if hi - lo <= 1e-12:
                normalized = np.zeros_like(local)
            else:
                normalized = (local - lo) / (hi - lo)
            row[selected] = (
                1.0
                + float(rank[slot]) * float(POLICY["semantic_depth_band"])
                + normalized * float(POLICY["intra_slot_depth_fraction"])
            )
        effective[fi] = row
        medians_rows.append([medians[slot] for slot in unique])

    payload = {
        "schema": "RealSaS.SemanticPresentationOrderCompiled.v1",
        "operator_id": OPERATOR_ID,
        "contract": CONTRACT,
        "policy_hash": content_sha256(POLICY),
        "slot_ids": unique,
        "rank_rows": rank_rows.astype(int).tolist(),
    }
    return {
        "slot_ids": np.asarray(unique, dtype=np.int32),
        "rank_rows": rank_rows,
        "canonical_slot_medians": np.asarray(medians_rows, dtype=np.float64),
        "effective_depths": effective,
        "transition_count": int(transitions),
        "order_hash": content_sha256(payload),
        "contract": CONTRACT,
    }


def semantic_order_metrics(
    *,
    effective_depths,
    semantic_vertex_slot,
    slot_ids,
    rank_rows,
) -> dict:
    depths = np.asarray(effective_depths, dtype=np.float64)
    slots = np.asarray(semantic_vertex_slot, dtype=np.int32)
    slot_ids = np.asarray(slot_ids, dtype=np.int32)
    ranks = np.asarray(rank_rows, dtype=np.int32)
    if (
        depths.ndim != 2
        or slots.shape != (depths.shape[1],)
        or ranks.shape != (len(depths), len(slot_ids))
        or not np.isfinite(depths).all()
    ):
        raise QualificationError("SEMANTIC_ORDER_METRIC_INPUT_INVALID")
    failures = 0
    minimum_gap = np.inf
    for fi in range(len(depths)):
        rank = {int(slot): int(ranks[fi, i]) for i, slot in enumerate(slot_ids)}
        ordered = sorted(rank, key=rank.get)
        for near, far in zip(ordered, ordered[1:]):
            near_max = float(np.max(depths[fi, slots == near]))
            far_min = float(np.min(depths[fi, slots == far]))
            gap = far_min - near_max
            minimum_gap = min(minimum_gap, gap)
            if gap <= 0:
                failures += 1
    if not np.isfinite(minimum_gap):
        minimum_gap = 0.0
    return {
        "semantic_occlusion_passed": failures == 0,
        "semantic_order_failure_count": int(failures),
        "minimum_inter_slot_depth_gap": float(minimum_gap),
        "semantic_slot_count": int(len(slot_ids)),
    }
