from __future__ import annotations

"""Fail-closed geometry checks for prompted semantic masks.

This module is deliberately model-agnostic.  It does not decide what a region is;
it checks whether a proposed mask is at least geometrically compatible with the
semantic localization prompt that produced it.  Multimodal adjudication remains
separate and non-authoritative.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


CANONICAL_SEMANTIC_LABELS = (
    "body",
    "head",
    "arm",
    "forearm",
    "hand",
    "cape",
    "cloth",
    "armor",
    "helmet",
    "horn",
    "sword",
    "shield",
)


@dataclass(frozen=True)
class MaskShapeCourtV1:
    status: str
    failure_code: str | None
    mask_pixels: int
    subject_coverage: float
    prompt_box_coverage: float
    mask_box_iou: float
    center_retained: bool
    connected_component_count: int
    largest_component_fraction: float


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return inter / union if union > 0 else 0.0


def _components(mask: np.ndarray) -> list[int]:
    """Return 4-connected component sizes without SciPy dependency."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for y0, x0 in zip(*np.where(mask & ~seen)):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def normalized_box_to_pixels(box_1000, *, width: int, height: int) -> tuple[int, int, int, int]:
    if len(box_1000) != 4:
        raise ValueError("bbox_1000 must contain four coordinates")
    vals = [float(v) for v in box_1000]
    if not all(0.0 <= v <= 1000.0 for v in vals):
        raise ValueError("bbox_1000 coordinates must be in [0,1000]")
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox_1000 must have positive area")
    px = (
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
        int(round(x2 * width / 1000.0)),
        int(round(y2 * height / 1000.0)),
    )
    ax1, ay1, ax2, ay2 = px
    ax1 = min(max(ax1, 0), width - 1)
    ay1 = min(max(ay1, 0), height - 1)
    ax2 = min(max(ax2, ax1 + 1), width)
    ay2 = min(max(ay2, ay1 + 1), height)
    return ax1, ay1, ax2, ay2


def normalized_point_to_pixels(point_1000, *, width: int, height: int) -> tuple[int, int]:
    if len(point_1000) != 2:
        raise ValueError("center_1000 must contain two coordinates")
    x, y = [float(v) for v in point_1000]
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        raise ValueError("center_1000 coordinates must be in [0,1000]")
    return (
        min(max(int(round(x * width / 1000.0)), 0), width - 1),
        min(max(int(round(y * height / 1000.0)), 0), height - 1),
    )


def audit_prompted_mask_v1(
    mask,
    *,
    subject_mask,
    prompt_box,
    prompt_center=None,
    minimum_pixels: int = 48,
    maximum_subject_coverage: float = 0.90,
    minimum_mask_box_iou: float = 0.08,
    minimum_largest_component_fraction: float = 0.80,
    maximum_components: int = 4,
) -> MaskShapeCourtV1:
    mask = np.asarray(mask, dtype=bool)
    subject = np.asarray(subject_mask, dtype=bool)
    if mask.shape != subject.shape:
        raise ValueError("mask and subject_mask shape mismatch")
    pixels = int(mask.sum())
    subject_pixels = int(subject.sum())
    subject_coverage = pixels / max(subject_pixels, 1)

    x1, y1, x2, y2 = [int(v) for v in prompt_box]
    box_area = max((x2 - x1) * (y2 - y1), 1)
    in_box = int(mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)].sum())
    prompt_box_coverage = in_box / box_area

    mask_box = _bbox_from_mask(mask)
    mask_box_iou = _bbox_iou(mask_box, prompt_box) if mask_box else 0.0

    center_retained = True
    if prompt_center is not None:
        cx, cy = [int(v) for v in prompt_center]
        center_retained = 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1] and bool(mask[cy, cx])

    comps = _components(mask)
    largest_fraction = (comps[0] / pixels) if pixels > 0 and comps else 0.0

    failure = None
    if pixels < int(minimum_pixels):
        failure = "MASK_TOO_SMALL"
    elif subject_coverage >= float(maximum_subject_coverage):
        failure = "MASK_WHOLE_SUBJECT_DOMINANT"
    elif mask_box_iou < float(minimum_mask_box_iou):
        failure = "MASK_PROMPT_BOX_MISMATCH"
    elif not center_retained:
        failure = "MASK_DROPPED_PROMPT_CENTER"
    elif len(comps) > int(maximum_components) and largest_fraction < float(minimum_largest_component_fraction):
        failure = "MASK_FRAGMENTED"
    elif largest_fraction < float(minimum_largest_component_fraction):
        failure = "MASK_MULTI_REGION"

    return MaskShapeCourtV1(
        status="PASS" if failure is None else "FAIL",
        failure_code=failure,
        mask_pixels=pixels,
        subject_coverage=float(subject_coverage),
        prompt_box_coverage=float(prompt_box_coverage),
        mask_box_iou=float(mask_box_iou),
        center_retained=bool(center_retained),
        connected_component_count=len(comps),
        largest_component_fraction=float(largest_fraction),
    )


def validate_multiview_oracle_payload_v1(payload: dict, *, allowed_labels: Iterable[str] = CANONICAL_SEMANTIC_LABELS) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("oracle payload must be an object")
    if payload.get("schema") != "RealSaS.MultiviewSemanticOracle.v1":
        raise ValueError("oracle schema mismatch")
    allowed = set(str(x) for x in allowed_labels)
    instances = payload.get("instances")
    if not isinstance(instances, list) or not instances:
        raise ValueError("oracle must contain non-empty instances")
    seen = set()
    for row in instances:
        iid = str(row.get("instance_id", "")).strip()
        label = str(row.get("label", "")).strip()
        if not iid or iid in seen:
            raise ValueError("instance_id missing or duplicated")
        seen.add(iid)
        if label not in allowed:
            raise ValueError(f"unsupported semantic label:{label}")
        conf = float(row.get("confidence", -1.0))
        if not 0.0 <= conf <= 1.0:
            raise ValueError("instance confidence outside [0,1]")
        views = row.get("views")
        if not isinstance(views, dict):
            raise ValueError("instance views must be an object")
        for view in range(8):
            item = views.get(str(view))
            if item is None:
                raise ValueError(f"instance {iid} missing view {view}")
            present = bool(item.get("present", False))
            if present:
                box = item.get("bbox_1000")
                center = item.get("center_1000")
                if box is None or center is None:
                    raise ValueError("present view needs bbox_1000 and center_1000")
                normalized_box_to_pixels(box, width=1000, height=1000)
                normalized_point_to_pixels(center, width=1000, height=1000)
    return payload
