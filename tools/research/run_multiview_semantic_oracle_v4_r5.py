#!/usr/bin/env python3
from __future__ import annotations

"""R5 two-stage multiview semantic oracle.

Observed R4 failure evidence showed that one large call could discover sensible
semantic instances but struggled when it also had to emit all per-view boxes and
relations.  R5 separates those responsibilities:

1. all 8 views -> short semantic instance discovery only;
2. one discovered physical instance at a time -> all 8 views -> visible-view
   box/point localization only;
3. deterministic expansion to the existing RealSaS.MultiviewSemanticOracle.v1;
4. unchanged prompted SAM2, deterministic mask court, R4 per-instance mask
   adjudication, and canonical-M fusion.

Relations are intentionally omitted from this segmentation court. They remain a
separate compiler qualification problem and cannot invalidate otherwise useful
semantic/localization evidence.
"""

import json
from pathlib import Path

from tools.research import run_multiview_semantic_oracle_v4 as v4
from tools.research import run_multiview_semantic_oracle_v4_r3 as r3
from tools.research import run_multiview_semantic_oracle_v4_r4 as r4  # noqa: F401 - activates R4 mask court


_DISCOVERY_SCHEMA = "RealSaS.MultiviewSemanticDiscovery.v1"
_LOCALIZATION_SCHEMA = "RealSaS.MultiviewInstanceLocalization.v1"
_MAX_INSTANCES = 12
_MIN_VISIBLE_VIEWS = 2


def _views_content(paths: list[Path], prompt: str):
    content = []
    for view, path in enumerate(paths):
        content.append({"type": "text", "text": f"V{view}"})
        content.append({
            "type": "image",
            "image": str(path.resolve()),
            "resized_height": 512,
            "resized_width": 512,
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _discovery_prompt(*, retry: bool = False) -> str:
    labels = ",".join(v4.CANONICAL_SEMANTIC_LABELS)
    prefix = "Previous answer was invalid. Regenerate from the images. " if retry else ""
    return (
        prefix
        + "All 8 images V0..V7 show the SAME stylized character at equal scale. "
        "Reason across all views and discover stable physical instances useful for animation. "
        "Do NOT localize pixels yet. Do NOT output view directions as parts. "
        f"Allowed coarse labels only: {labels}. "
        "Omit unsupported concepts. A pointed helmet contour is not automatically a horn. "
        "A long rigid held object is not an arm. Distinguish duplicate physical instances with a short descriptor, "
        "for example sword-side hand vs shield-side hand. "
        f"Return at most {_MAX_INSTANCES} instances. Output ONE minified JSON object only using: "
        f'{{"schema":"{_DISCOVERY_SCHEMA}","summary":"short","instances":['
        '{"id":"I00","label":"sword","descriptor":"held long blade","confidence":0.9}]}. '
        "No relations, boxes, markdown, comments, ellipses or trailing commas."
    )


def _validate_discovery(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("schema") != _DISCOVERY_SCHEMA:
        raise ValueError("semantic discovery schema mismatch")
    rows = payload.get("instances")
    if not isinstance(rows, list) or not rows or len(rows) > _MAX_INSTANCES:
        raise ValueError("semantic discovery instances missing or over limit")
    allowed = set(v4.CANONICAL_SEMANTIC_LABELS)
    seen = set()
    clean = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("semantic discovery instance must be object")
        iid = str(row.get("id", "")).strip()
        label = str(row.get("label", "")).strip()
        descriptor = " ".join(str(row.get("descriptor", "")).strip().split())
        confidence = float(row.get("confidence", -1.0))
        if not iid or iid in seen:
            raise ValueError("semantic discovery id missing or duplicated")
        seen.add(iid)
        if label not in allowed:
            raise ValueError(f"semantic discovery unsupported label:{label}")
        if not descriptor or len(descriptor) > 96:
            raise ValueError("semantic discovery descriptor missing or too long")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("semantic discovery confidence outside [0,1]")
        clean.append({
            "id": iid,
            "label": label,
            "descriptor": descriptor,
            "confidence": confidence,
        })
    return {
        "schema": _DISCOVERY_SCHEMA,
        "summary": str(payload.get("summary", "")).strip(),
        "instances": clean,
    }


def _localization_prompt(instance: dict, *, retry: bool = False) -> str:
    prefix = "Previous answer was invalid. Regenerate from the images. " if retry else ""
    iid, label, descriptor = instance["id"], instance["label"], instance["descriptor"]
    return (
        prefix
        + f"Localize ONE physical instance across the same 8 views: id={iid}, label={label}, descriptor={descriptor!r}. "
        "Do not discover or name any other part. For each view where THIS SAME physical instance is visibly supported, "
        "give a tight box around only that part plus one interior point. Omit hidden, absent or uncertain views. "
        "Coordinates are normalized integers 0..1000. Each visible view value is exactly "
        "[x1,y1,x2,y2,cx,cy,confidence]. "
        f"Output ONE minified JSON object only: {{\"schema\":\"{_LOCALIZATION_SCHEMA}\","
        f"\"id\":\"{iid}\",\"label\":\"{label}\",\"v\":{{\"0\":[100,200,300,700,210,430,0.9]}}}}. "
        "No markdown, comments, prose, ellipses or trailing commas."
    )


def _validate_localization(payload: dict, *, instance: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("schema") != _LOCALIZATION_SCHEMA:
        raise ValueError("instance localization schema mismatch")
    if str(payload.get("id", "")).strip() != instance["id"]:
        raise ValueError("instance localization id mismatch")
    if str(payload.get("label", "")).strip() != instance["label"]:
        raise ValueError("instance localization label mismatch")
    visible = payload.get("v")
    if not isinstance(visible, dict):
        raise ValueError("instance localization v must be object")
    clean = {}
    for key, value in visible.items():
        key = str(key)
        if key not in {str(v) for v in range(8)}:
            raise ValueError(f"instance localization invalid view:{key}")
        if not isinstance(value, list) or len(value) != 7:
            raise ValueError("instance localization view must contain seven values")
        x1, y1, x2, y2, cx, cy, confidence = value
        coords = [float(x1), float(y1), float(x2), float(y2), float(cx), float(cy)]
        if not all(0.0 <= x <= 1000.0 for x in coords):
            raise ValueError("instance localization coordinate outside [0,1000]")
        if float(x2) <= float(x1) or float(y2) <= float(y1):
            raise ValueError("instance localization box has nonpositive area")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("instance localization confidence outside [0,1]")
        clean[key] = [float(x1), float(y1), float(x2), float(y2), float(cx), float(cy), confidence]
    if len(clean) < _MIN_VISIBLE_VIEWS:
        raise ValueError(f"instance localization insufficient multiview support:{len(clean)}")
    return {"schema": _LOCALIZATION_SCHEMA, "id": instance["id"], "label": instance["label"], "v": clean}


def _call_typed_json(model, processor, paths, prompt_factory, validator, work: Path, stem: str, *, max_new_tokens: int):
    last_error = None
    for attempt in range(2):
        prompt = prompt_factory(retry=attempt == 1)
        raw = v4._qwen_generate(
            model,
            processor,
            _views_content(paths, prompt),
            max_new_tokens=max_new_tokens,
        )
        (work / f"{stem}_attempt{attempt + 1}.txt").write_text(raw, encoding="utf-8")
        try:
            payload = r3._extract_json_strict(raw)
            clean = validator(payload)
            (work / f"{stem}_accepted.json").write_text(
                json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return clean, raw
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
    (work / f"{stem}_FAIL_CLOSED.txt").write_text(str(last_error) + "\n", encoding="utf-8")
    return None, ""


def _run_localization_oracle_r5(model, processor, paths: list[Path], *, max_new_tokens: int):
    if len(paths) != 8:
        raise RuntimeError("R5_REQUIRES_EIGHT_SOURCE_VIEWS")
    work = paths[0].parent.parent
    work.mkdir(parents=True, exist_ok=True)

    discovery, discovery_raw = _call_typed_json(
        model,
        processor,
        paths,
        _discovery_prompt,
        _validate_discovery,
        work,
        "semantic_discovery",
        max_new_tokens=min(max(int(max_new_tokens), 1024), 2048),
    )
    if discovery is None:
        raise RuntimeError("SEMANTIC_DISCOVERY_TWO_ATTEMPTS_INVALID")

    canonical_instances = []
    aggregate_raw = ["===== DISCOVERY =====\n" + discovery_raw]
    localization_failures = []

    for index, instance in enumerate(discovery["instances"]):
        def prompt_factory(*, retry: bool, _instance=instance):
            return _localization_prompt(_instance, retry=retry)

        def validator(payload, _instance=instance):
            return _validate_localization(payload, instance=_instance)

        localized, raw = _call_typed_json(
            model,
            processor,
            paths,
            prompt_factory,
            validator,
            work,
            f"localize_{index:02d}_{instance['id']}_{instance['label']}",
            max_new_tokens=1536,
        )
        if localized is None:
            localization_failures.append(instance["id"])
            continue
        aggregate_raw.append(f"===== LOCALIZE {instance['id']}:{instance['label']} =====\n{raw}")
        views = {
            str(view): {
                "present": False,
                "bbox_1000": None,
                "center_1000": None,
                "confidence": 0.0,
            }
            for view in range(8)
        }
        for key, values in localized["v"].items():
            x1, y1, x2, y2, cx, cy, confidence = values
            views[key] = {
                "present": True,
                "bbox_1000": [x1, y1, x2, y2],
                "center_1000": [cx, cy],
                "confidence": confidence,
            }
        canonical_instances.append({
            "instance_id": instance["id"],
            "label": instance["label"],
            "confidence": instance["confidence"],
            "views": views,
            "relations": [],
        })

    if not canonical_instances:
        raise RuntimeError("SEMANTIC_DISCOVERY_NO_INSTANCE_SURVIVED_LOCALIZATION")

    canonical = {
        "schema": "RealSaS.MultiviewSemanticOracle.v1",
        "subject_summary": discovery.get("summary", ""),
        "instances": canonical_instances,
    }
    v4.validate_multiview_oracle_payload_v1(canonical)
    (work / "semantic_localization_failures.json").write_text(
        json.dumps({"failed_instance_ids": localization_failures}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return canonical, "\n\n".join(aggregate_raw)


# R3 keeps local-path normalization; R4 keeps per-instance mask adjudication.
# Replace only the overloaded discovery+localization call with R5 two-stage court.
v4._run_localization_oracle = _run_localization_oracle_r5


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
