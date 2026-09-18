from __future__ import annotations

"""Controlled 8-view observation contract for RealSaS mainline.

This module deliberately separates positive evidence, negative evidence and absence
of observation. A projection outside a source frame is UNKNOWN and may never be
used as negative geometry evidence.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError


OBSERVATION_CONTRACT_SCHEMA = "RealSaS.ControlledObservationContract.v1"
REQUIRED_DIRECTION_COUNT = 8


class ObservationEvidence(IntEnum):
    NEGATIVE = -1
    UNKNOWN = 0
    POSITIVE = 1


@dataclass(frozen=True)
class ObservationViewAuditV1:
    view_index: int
    foreground_pixel_count: int
    bbox_xyxy: tuple[int, int, int, int]
    minimum_margin_px: int
    border_contacts: tuple[str, ...]


@dataclass(frozen=True)
class ControlledObservationSetAuditV1:
    views: tuple[ObservationViewAuditV1, ...]
    minimum_required_margin_px: int
    minimum_observed_margin_px: int
    exact_direction_count: int
    full_subject_frame_pass: bool
    audit_hash: str
    schema_version: str = OBSERVATION_CONTRACT_SCHEMA


def _alpha_mask(value) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2:
        raise QualificationError("OBSERVATION_ALPHA_MUST_BE_HXW")
    if a.dtype == np.bool_:
        mask = a.copy()
    else:
        if not np.isfinite(a).all():
            raise QualificationError("OBSERVATION_ALPHA_NONFINITE")
        mask = a > 0
    if not np.any(mask):
        raise QualificationError("OBSERVATION_ALPHA_EMPTY")
    return mask


def audit_controlled_observation_set_v1(
    alpha_masks: Sequence[np.ndarray],
    *,
    minimum_margin_px: int = 8,
) -> ControlledObservationSetAuditV1:
    if len(alpha_masks) != REQUIRED_DIRECTION_COUNT:
        raise QualificationError(
            f"OBSERVATION_EXACTLY_{REQUIRED_DIRECTION_COUNT}_DIRECTIONS_REQUIRED"
        )
    if int(minimum_margin_px) < 1:
        raise QualificationError("OBSERVATION_MINIMUM_MARGIN_MUST_BE_POSITIVE")

    shape = None
    rows: list[ObservationViewAuditV1] = []
    for view_index, raw in enumerate(alpha_masks):
        mask = _alpha_mask(raw)
        if shape is None:
            shape = mask.shape
        elif mask.shape != shape:
            raise QualificationError("OBSERVATION_RESOLUTION_DRIFT")
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        contacts: list[str] = []
        if x0 == 0:
            contacts.append("LEFT")
        if x1 == w - 1:
            contacts.append("RIGHT")
        if y0 == 0:
            contacts.append("TOP")
        if y1 == h - 1:
            contacts.append("BOTTOM")
        margin = int(min(x0, y0, (w - 1) - x1, (h - 1) - y1))
        rows.append(
            ObservationViewAuditV1(
                view_index=view_index,
                foreground_pixel_count=int(np.count_nonzero(mask)),
                bbox_xyxy=(x0, y0, x1, y1),
                minimum_margin_px=margin,
                border_contacts=tuple(contacts),
            )
        )

    observed_min = min(row.minimum_margin_px for row in rows)
    passed = all(
        not row.border_contacts and row.minimum_margin_px >= int(minimum_margin_px)
        for row in rows
    )
    payload = {
        "schema": OBSERVATION_CONTRACT_SCHEMA,
        "minimum_required_margin_px": int(minimum_margin_px),
        "views": [
            {
                "view_index": row.view_index,
                "foreground_pixel_count": row.foreground_pixel_count,
                "bbox_xyxy": list(row.bbox_xyxy),
                "minimum_margin_px": row.minimum_margin_px,
                "border_contacts": list(row.border_contacts),
            }
            for row in rows
        ],
    }
    return ControlledObservationSetAuditV1(
        views=tuple(rows),
        minimum_required_margin_px=int(minimum_margin_px),
        minimum_observed_margin_px=int(observed_min),
        exact_direction_count=len(rows),
        full_subject_frame_pass=bool(passed),
        audit_hash=content_sha256(payload),
    )


def require_controlled_observation_set_v1(
    alpha_masks: Sequence[np.ndarray],
    *,
    minimum_margin_px: int = 8,
) -> ControlledObservationSetAuditV1:
    audit = audit_controlled_observation_set_v1(
        alpha_masks,
        minimum_margin_px=minimum_margin_px,
    )
    if not audit.full_subject_frame_pass:
        failures = {
            row.view_index: {
                "margin": row.minimum_margin_px,
                "contacts": row.border_contacts,
            }
            for row in audit.views
            if row.border_contacts
            or row.minimum_margin_px < audit.minimum_required_margin_px
        }
        raise QualificationError(f"OBSERVATION_FULL_SUBJECT_FRAME_FAIL:{failures}")
    return audit


def classify_projected_observation_evidence_v1(
    xy,
    alpha_mask,
) -> np.ndarray:
    """Classify projected samples without conflating OOF with background.

    Pixel coordinates use the ordinary image-domain convention [0,width) x
    [0,height). Samples outside that domain are UNKNOWN. Samples inside are
    POSITIVE when the source alpha is foreground, otherwise NEGATIVE.
    """

    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise QualificationError("OBSERVATION_PROJECTED_XY_INVALID")
    mask = _alpha_mask(alpha_mask)
    h, w = mask.shape

    state = np.full(len(points), int(ObservationEvidence.UNKNOWN), dtype=np.int8)
    inside = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(w))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(h))
    )
    if not np.any(inside):
        return state
    indices = np.flatnonzero(inside)
    px = np.floor(points[indices, 0]).astype(np.int64)
    py = np.floor(points[indices, 1]).astype(np.int64)
    state[indices] = np.where(
        mask[py, px],
        int(ObservationEvidence.POSITIVE),
        int(ObservationEvidence.NEGATIVE),
    ).astype(np.int8)
    return state


def aggregate_multiview_evidence_v1(states) -> np.ndarray:
    """Union eight view states with POSITIVE > NEGATIVE > UNKNOWN semantics.

    A sample is POSITIVE when any view positively observes it. It is NEGATIVE only
    when at least one view observes background and no view observes foreground.
    All-UNKNOWN remains UNKNOWN.
    """

    s = np.asarray(states, dtype=np.int8)
    if s.ndim != 2 or s.shape[0] != REQUIRED_DIRECTION_COUNT:
        raise QualificationError("OBSERVATION_EVIDENCE_REQUIRES_8XN")
    valid = np.isin(
        s,
        [
            int(ObservationEvidence.NEGATIVE),
            int(ObservationEvidence.UNKNOWN),
            int(ObservationEvidence.POSITIVE),
        ],
    )
    if not bool(np.all(valid)):
        raise QualificationError("OBSERVATION_EVIDENCE_ENUM_INVALID")
    out = np.full(s.shape[1], int(ObservationEvidence.UNKNOWN), dtype=np.int8)
    any_positive = np.any(s == int(ObservationEvidence.POSITIVE), axis=0)
    any_negative = np.any(s == int(ObservationEvidence.NEGATIVE), axis=0)
    out[any_negative] = int(ObservationEvidence.NEGATIVE)
    out[any_positive] = int(ObservationEvidence.POSITIVE)
    return out
