#!/usr/bin/env python3
from __future__ import annotations

"""R8 V4 oracle entrypoint with deterministic cross-instance collision court.

R7 demonstrated that 8-view semantic discovery + single-view Qwen grounding can
produce useful boxes, but per-instance mask adjudication cannot detect when the
same visible pixels are simultaneously accepted as different semantic instances
(e.g. arm/hand/sword). R8 keeps R7 discovery/grounding and R4 per-instance mask
court, then adds a deterministic same-view cross-instance collision court before
proposals are serialized.

Policy is conservative:
- same-label masks from distinct physical instances with near-identical IoU are
  both rejected for that view;
- hierarchical coarse/specific pairs prefer the specific visible-surface label
  (helmet > head, armor > body, hand > arm) when masks are near-identical;
- other cross-label near-identical masks reject both for that view.

No geometry, training, runtime behavior, or M authority changes are introduced.
"""

import json
from pathlib import Path

import numpy as np

from tools.research import run_multiview_semantic_oracle_v4 as v4
from tools.research import run_multiview_semantic_oracle_v4_r7 as r7  # noqa: F401 - activates R7 localization


_COLLISION_IOU = 0.90
_SPECIFIC_WINS = {
    frozenset(("head", "helmet")): "helmet",
    frozenset(("body", "armor")): "armor",
    frozenset(("arm", "hand")): "hand",
}

_BASE_MASK_ADJUDICATION = v4._run_mask_adjudication


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(aa, bb).sum())
    union = int(np.logical_or(aa, bb).sum())
    return float(inter / union) if union else 0.0


def collision_action(label_a: str, label_b: str, *, iou: float, threshold: float = _COLLISION_IOU):
    """Return labels to reject for a near-identical same-view collision."""
    if float(iou) < float(threshold):
        return ()
    a, b = str(label_a), str(label_b)
    if a == b:
        return ("A", "B")
    winner = _SPECIFIC_WINS.get(frozenset((a, b)))
    if winner is not None:
        if a == winner:
            return ("B",)
        return ("A",)
    return ("A", "B")


def apply_cross_instance_collision_court(payload: dict, rows: list[dict], *, threshold: float = _COLLISION_IOU):
    decisions = {str(x["instance_id"]): dict(x) for x in payload.get("instances", [])}
    for item in decisions.values():
        item["bad_views"] = sorted({int(v) for v in item.get("bad_views", [])})
        item.setdefault("collision_views", [])

    collisions = []
    by_view: dict[int, list[dict]] = {}
    for row in rows:
        iid = str(row["instance_id"])
        decision = decisions.get(iid)
        if decision is None or str(decision.get("verdict", "UNKNOWN")) != "PASS":
            continue
        by_view.setdefault(int(row["view_index"]), []).append(row)

    for view, view_rows in sorted(by_view.items()):
        for i in range(len(view_rows)):
            a = view_rows[i]
            for j in range(i + 1, len(view_rows)):
                b = view_rows[j]
                if str(a["instance_id"]) == str(b["instance_id"]):
                    continue
                iou = _mask_iou(a["mask"], b["mask"])
                actions = collision_action(str(a["label"]), str(b["label"]), iou=iou, threshold=threshold)
                if not actions:
                    continue

                rejected = []
                for token, row in (("A", a), ("B", b)):
                    if token not in actions:
                        continue
                    iid = str(row["instance_id"])
                    decision = decisions[iid]
                    decision["bad_views"] = sorted(set(decision.get("bad_views", [])) | {int(view)})
                    decision["collision_views"] = sorted(set(decision.get("collision_views", [])) | {int(view)})
                    decision["reason_code"] = "CROSS_INSTANCE_COLLISION"
                    rejected.append(iid)

                collisions.append({
                    "view_index": int(view),
                    "a_instance_id": str(a["instance_id"]),
                    "a_label": str(a["label"]),
                    "b_instance_id": str(b["instance_id"]),
                    "b_label": str(b["label"]),
                    "iou": float(iou),
                    "rejected_instance_ids": rejected,
                    "policy": (
                        "SAME_LABEL_BOTH_FAIL"
                        if str(a["label"]) == str(b["label"])
                        else "SPECIFIC_WINS"
                        if frozenset((str(a["label"]), str(b["label"]))) in _SPECIFIC_WINS
                        else "INCOMPATIBLE_BOTH_FAIL"
                    ),
                })

    out = dict(payload)
    order = [str(x["instance_id"]) for x in payload.get("instances", [])]
    out["instances"] = [decisions[iid] for iid in order]
    out["cross_instance_collision_court"] = {
        "schema": "RealSaS.CrossInstanceMaskCollisionCourt.v1",
        "near_identical_iou_threshold": float(threshold),
        "collision_count": len(collisions),
        "collisions": collisions,
        "hierarchical_specific_wins": [
            {"pair": sorted(pair), "winner": winner}
            for pair, winner in sorted(_SPECIFIC_WINS.items(), key=lambda x: tuple(sorted(x[0])))
        ],
    }
    return out


def _run_mask_adjudication_r8(model, processor, images, rows, work: Path, *, max_new_tokens: int):
    payload, raw, overlay_paths = _BASE_MASK_ADJUDICATION(
        model, processor, images, rows, work, max_new_tokens=max_new_tokens
    )
    payload = apply_cross_instance_collision_court(payload, rows)
    court = payload["cross_instance_collision_court"]
    audit_path = work.parent / "cross_instance_collision_audit.json"
    audit_path.write_text(json.dumps(court, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw = raw + "\n\n===== DETERMINISTIC CROSS-INSTANCE COLLISION COURT =====\n" + json.dumps(court, sort_keys=True)
    return payload, raw, overlay_paths


v4._run_mask_adjudication = _run_mask_adjudication_r8


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
