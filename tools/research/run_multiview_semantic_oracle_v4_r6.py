#!/usr/bin/env python3
from __future__ import annotations

"""R6 V4 oracle entrypoint with single-view Qwen grounding.

R5 failure evidence showed that semantic discovery was strong, but the
multi-view localization prompt was overfit to its numeric schema example and
repeated the same V0 coordinates across every instance. R6 keeps the successful
8-view semantic discovery, then grounds one discovered physical instance in one
view at a time using Qwen3-VL's native 2D grounding vocabulary (bbox_2d /
point_2d). No numeric coordinate example is included in the localization prompt.

Only instances with at least two independently grounded visible views survive to
prompted SAM2. Repeated localization signatures across different physical
instances are rejected as likely template/copy failures. Existing R4 per-instance
mask adjudication and canonical-M fusion remain unchanged.
"""

import json
from pathlib import Path

from tools.research import run_multiview_semantic_oracle_v4 as v4
from tools.research import run_multiview_semantic_oracle_v4_r3 as r3  # local-path normalization + strict JSON
from tools.research import run_multiview_semantic_oracle_v4_r4 as r4  # noqa: F401 - activates per-instance mask court
from tools.research import run_multiview_semantic_oracle_v4_r5 as r5


_GROUND_SCHEMA = "RealSaS.SingleViewInstanceGrounding.v1"
_MIN_VISIBLE_VIEWS = 2


def _discovery_prompt_r6(*, retry: bool = False) -> str:
    labels = ",".join(v4.CANONICAL_SEMANTIC_LABELS)
    prefix = "Previous answer was invalid. Regenerate from the images. " if retry else ""
    return (
        prefix
        + "All 8 images V0..V7 show the SAME stylized character at equal scale. "
        "Reason across all views and discover stable physical instances useful for animation. "
        "Do NOT localize pixels yet. Do NOT output view directions as parts. "
        f"Allowed coarse labels only: {labels}. "
        "Omit unsupported concepts. A pointed helmet contour is not automatically a horn. "
        "A long rigid held object is not an arm. Distinguish duplicate same-label physical instances with a short "
        "view-invariant descriptor based on appearance or associated objects (for example sword-side versus shield-side); "
        "do not use screen-left/screen-right because the camera rotates around the subject. "
        "Return at most 12 instances. Output ONE minified JSON object only using keys schema,summary,instances; "
        "each instance must contain id,label,descriptor,confidence. "
        f"The schema value must be {r5._DISCOVERY_SCHEMA}. "
        "No relations, boxes, markdown, comments, ellipses or trailing commas."
    )


def _single_view_prompt(instance: dict, view: int, *, retry: bool = False) -> str:
    prefix = "Previous answer was invalid. Re-evaluate the image from scratch. " if retry else ""
    return (
        prefix
        + f"This is orthographic view V{view} of the same stylized character. "
        + f"Locate ONLY this already-discovered physical instance if it is visibly supported: "
        + f"id={instance['id']}, label={instance['label']}, descriptor={instance['descriptor']!r}. "
        + "Do not discover or name any other part. Use the descriptor to distinguish same-label instances. "
        + "If this exact physical instance is hidden, absent, or too uncertain in this view, return present=false. "
        + "If visible, return a tight bbox_2d around ONLY that part and one interior point_2d. "
        + "bbox_2d is [xmin,ymin,xmax,ymax], point_2d is [x,y], and every coordinate must be an integer from 0 to 1000. "
        + "Infer coordinates from the image; there are intentionally no example coordinates to copy. "
        + f"Return ONE minified JSON object only. Required schema value: {_GROUND_SCHEMA}. "
        + "Required keys: schema,id,label,view,present,bbox_2d,point_2d. "
        + "When present=false, bbox_2d and point_2d must be null. No markdown, prose, comments or trailing comma."
    )


def _validate_single_view(payload: dict, *, instance: dict, view: int) -> dict:
    if not isinstance(payload, dict) or payload.get("schema") != _GROUND_SCHEMA:
        raise ValueError("single-view grounding schema mismatch")
    if str(payload.get("id", "")).strip() != instance["id"]:
        raise ValueError("single-view grounding id mismatch")
    if str(payload.get("label", "")).strip() != instance["label"]:
        raise ValueError("single-view grounding label mismatch")
    if int(payload.get("view", -1)) != int(view):
        raise ValueError("single-view grounding view mismatch")
    present = payload.get("present")
    if not isinstance(present, bool):
        raise ValueError("single-view grounding present must be bool")
    if not present:
        if payload.get("bbox_2d") is not None or payload.get("point_2d") is not None:
            raise ValueError("absent grounding must use null bbox_2d/point_2d")
        return {
            "schema": _GROUND_SCHEMA,
            "id": instance["id"],
            "label": instance["label"],
            "view": int(view),
            "present": False,
            "bbox_2d": None,
            "point_2d": None,
        }

    box = payload.get("bbox_2d")
    point = payload.get("point_2d")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("visible grounding bbox_2d must contain four values")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("visible grounding point_2d must contain two values")
    vals = [float(x) for x in [*box, *point]]
    if not all(0.0 <= x <= 1000.0 for x in vals):
        raise ValueError("single-view grounding coordinate outside [0,1000]")
    x1, y1, x2, y2 = [float(x) for x in box]
    px, py = [float(x) for x in point]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("single-view grounding box has nonpositive area")
    if not (x1 <= px <= x2 and y1 <= py <= y2):
        raise ValueError("single-view grounding point must lie inside bbox_2d")
    return {
        "schema": _GROUND_SCHEMA,
        "id": instance["id"],
        "label": instance["label"],
        "view": int(view),
        "present": True,
        "bbox_2d": [x1, y1, x2, y2],
        "point_2d": [px, py],
    }


def _call_single_view(model, processor, image_path: Path, instance: dict, view: int, work: Path):
    stem = f"ground_{instance['id']}_{instance['label']}_V{view}"
    last_error = None
    for attempt in range(2):
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path.resolve()),
                    "resized_height": 768,
                    "resized_width": 768,
                },
                {
                    "type": "text",
                    "text": _single_view_prompt(instance, view, retry=attempt == 1),
                },
            ],
        }]
        raw = v4._qwen_generate(model, processor, messages, max_new_tokens=384)
        (work / f"{stem}_attempt{attempt + 1}.txt").write_text(raw, encoding="utf-8")
        try:
            payload = r3._extract_json_strict(raw)
            clean = _validate_single_view(payload, instance=instance, view=view)
            (work / f"{stem}_accepted.json").write_text(
                json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return clean
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
    (work / f"{stem}_FAIL_CLOSED.txt").write_text(str(last_error) + "\n", encoding="utf-8")
    return None


def _signature(row: dict):
    if not row.get("present"):
        return None
    return tuple(round(float(x), 4) for x in [*row["bbox_2d"], *row["point_2d"]])


def _run_localization_oracle_r6(model, processor, paths: list[Path], *, max_new_tokens: int):
    if len(paths) != 8:
        raise RuntimeError("R6_REQUIRES_EIGHT_SOURCE_VIEWS")
    work = paths[0].parent.parent
    work.mkdir(parents=True, exist_ok=True)

    discovery, discovery_raw = r5._call_typed_json(
        model,
        processor,
        paths,
        _discovery_prompt_r6,
        r5._validate_discovery,
        work,
        "semantic_discovery",
        max_new_tokens=min(max(int(max_new_tokens), 1024), 2048),
    )
    if discovery is None:
        raise RuntimeError("SEMANTIC_DISCOVERY_TWO_ATTEMPTS_INVALID")

    grounded = []
    for instance in discovery["instances"]:
        for view, path in enumerate(paths):
            row = _call_single_view(model, processor, path, instance, view, work)
            if row is not None:
                grounded.append(row)

    # Detect template/copy failures: one exact localization signature reused for
    # different physical instances is not credible evidence. Drop those rows.
    by_signature = {}
    for row in grounded:
        sig = _signature(row)
        if sig is not None:
            by_signature.setdefault(sig, []).append(row)
    bad_signatures = set()
    for sig, rows in by_signature.items():
        instance_ids = {row["id"] for row in rows}
        labels = {row["label"] for row in rows}
        if len(instance_ids) >= 2 and len(labels) >= 2:
            bad_signatures.add(sig)

    canonical_instances = []
    localization_failures = []
    duplicate_rejections = []
    for instance in discovery["instances"]:
        rows = []
        for row in grounded:
            if row["id"] != instance["id"] or not row.get("present"):
                continue
            if _signature(row) in bad_signatures:
                duplicate_rejections.append({"id": row["id"], "label": row["label"], "view": row["view"]})
                continue
            rows.append(row)

        if len(rows) < _MIN_VISIBLE_VIEWS:
            localization_failures.append({
                "id": instance["id"],
                "label": instance["label"],
                "visible_views": [row["view"] for row in rows],
            })
            continue

        views = {
            str(view): {
                "present": False,
                "bbox_1000": None,
                "center_1000": None,
                "confidence": 0.0,
            }
            for view in range(8)
        }
        for row in rows:
            views[str(row["view"])] = {
                "present": True,
                "bbox_1000": list(row["bbox_2d"]),
                "center_1000": list(row["point_2d"]),
                # Qwen grounding confidence is not calibrated here. Use a neutral
                # factor; downstream confidence still includes discovery, SAM IoU
                # and multimodal mask adjudication.
                "confidence": 1.0,
            }
        canonical_instances.append({
            "instance_id": instance["id"],
            "label": instance["label"],
            "confidence": instance["confidence"],
            "views": views,
            "relations": [],
        })

    diagnostics = {
        "schema": "RealSaS.SingleViewGroundingDiagnostics.v1",
        "discovered_instance_count": len(discovery["instances"]),
        "grounding_record_count": len(grounded),
        "surviving_instance_count": len(canonical_instances),
        "failed_instances": localization_failures,
        "duplicate_signature_rejections": duplicate_rejections,
        "view_confidence_semantics": "UNSCORED_NEUTRAL_1.0",
    }
    (work / "single_view_grounding_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if not canonical_instances:
        raise RuntimeError("SEMANTIC_DISCOVERY_NO_INSTANCE_SURVIVED_SINGLE_VIEW_GROUNDING")

    canonical = {
        "schema": "RealSaS.MultiviewSemanticOracle.v1",
        "subject_summary": discovery.get("summary", ""),
        "instances": canonical_instances,
    }
    v4.validate_multiview_oracle_payload_v1(canonical)
    return canonical, discovery_raw


# R3 import keeps local-path normalization active; R4 import keeps per-instance
# mask adjudication active. Override only discovery/localization with R6.
v4._run_localization_oracle = _run_localization_oracle_r6


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
