#!/usr/bin/env python3
from __future__ import annotations

"""R4 V4 oracle entrypoint with per-instance mask adjudication.

Imports R3 so compact localization, raw-attempt persistence, typed retry and
local-image-path normalization remain active.  R4 changes only the multimodal
mask court: each physical instance is judged in its own short call.  Two invalid
JSON replies for one instance degrade that instance to UNKNOWN (candidate
rejected) rather than corrupting or authorizing evidence.
"""

import json
from pathlib import Path

from tools.research import run_multiview_semantic_oracle_v4 as v4
from tools.research import run_multiview_semantic_oracle_v4_r3 as r3


_ITEM_SCHEMA = "RealSaS.MultiviewMaskAdjudicationItem.v1"
_ALLOWED_VERDICTS = {"PASS", "FAIL", "UNKNOWN"}
_ALLOWED_REASONS = {"GOOD", "FRAGMENT", "MULTI_PART", "WRONG_PART", "BOUNDARY_POOR", "INSUFFICIENT"}


def _item_prompt(instance_id: str, label: str, *, retry: bool = False) -> str:
    prefix = "Previous answer was invalid JSON. Regenerate from the image. " if retry else ""
    return (
        prefix
        + f"Judge the RED SAM2 mask for instance {instance_id}:{label} across the 8-view sheet. "
        "Yellow is the semantic localization box. PASS only if the red mask follows ONE coherent physical part "
        "matching the label in the visible views. FAIL for fragment, merged/multiple parts, wrong part, or poor boundary. "
        "UNKNOWN if evidence is insufficient. Return ONE minified JSON object only: "
        f'{{"schema":"{_ITEM_SCHEMA}","instance_id":"{instance_id}","label":"{label}",'
        '"verdict":"PASS|FAIL|UNKNOWN","semantic_match":true,"confidence":0.0,'
        '"bad_views":[0],"reason_code":"GOOD|FRAGMENT|MULTI_PART|WRONG_PART|BOUNDARY_POOR|INSUFFICIENT"}}. '
        "No markdown, comments, prose or trailing comma."
    )


def _validate_item(payload: dict, *, instance_id: str, label: str) -> dict:
    if not isinstance(payload, dict) or payload.get("schema") != _ITEM_SCHEMA:
        raise ValueError("mask item schema mismatch")
    if str(payload.get("instance_id", "")) != instance_id:
        raise ValueError("mask item instance mismatch")
    if str(payload.get("label", "")) != label:
        raise ValueError("mask item label mismatch")
    verdict = str(payload.get("verdict", "UNKNOWN")).upper()
    if verdict not in _ALLOWED_VERDICTS:
        raise ValueError("mask item verdict invalid")
    semantic_match = bool(payload.get("semantic_match", False))
    confidence = float(payload.get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("mask item confidence outside [0,1]")
    reason = str(payload.get("reason_code", "INSUFFICIENT")).upper()
    if reason not in _ALLOWED_REASONS:
        raise ValueError("mask item reason invalid")
    raw_bad = payload.get("bad_views", [])
    if not isinstance(raw_bad, list):
        raise ValueError("mask item bad_views must be list")
    bad_views = []
    for value in raw_bad:
        view = int(value)
        if not 0 <= view <= 7:
            raise ValueError("mask item bad view outside [0,7]")
        if view not in bad_views:
            bad_views.append(view)
    return {
        "instance_id": instance_id,
        "label": label,
        "verdict": verdict,
        "semantic_match": semantic_match,
        "confidence": confidence,
        "bad_views": bad_views,
        "reason_code": reason,
    }


def _unknown(instance_id: str, label: str) -> dict:
    return {
        "instance_id": instance_id,
        "label": label,
        "verdict": "UNKNOWN",
        "semantic_match": False,
        "confidence": 0.0,
        "bad_views": [],
        "reason_code": "INSUFFICIENT",
    }


def _run_mask_adjudication_r4(model, processor, images, rows, work: Path, *, max_new_tokens: int):
    instances = []
    seen = set()
    for row in rows:
        key = (str(row["instance_id"]), str(row["label"]))
        if key not in seen:
            seen.add(key)
            instances.append(key)
    if not instances:
        raise RuntimeError("NO_PROMPTED_INSTANCES_FOR_ADJUDICATION")

    work.mkdir(parents=True, exist_ok=True)
    decisions = []
    aggregate_raw = []
    overlay_paths = []

    for index, (iid, label) in enumerate(instances):
        sheet = v4._overlay_sheet(images, rows, instance_id=iid, label=label)
        path = work / f"{index:02d}_{iid}_{label}.png"
        sheet.save(path)
        overlay_paths.append(path)

        decision = None
        last_error = None
        for attempt in range(2):
            prompt = _item_prompt(iid, label, retry=attempt == 1)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path.resolve()), "resized_height": 512, "resized_width": 1024},
                    {"type": "text", "text": prompt},
                ],
            }]
            raw = v4._qwen_generate(model, processor, messages, max_new_tokens=max(512, min(int(max_new_tokens), 1024)))
            raw_path = work / f"{index:02d}_{iid}_{label}_attempt{attempt + 1}.txt"
            raw_path.write_text(raw, encoding="utf-8")
            aggregate_raw.append(f"===== {iid}:{label} attempt {attempt + 1} =====\n{raw}\n")
            try:
                payload = r3._extract_json_strict(raw)
                decision = _validate_item(payload, instance_id=iid, label=label)
                break
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                last_error = exc

        if decision is None:
            decision = _unknown(iid, label)
            (work / f"{index:02d}_{iid}_{label}_FAIL_CLOSED.txt").write_text(
                str(last_error) + "\n", encoding="utf-8"
            )
        decisions.append(decision)

    payload = {
        "schema": "RealSaS.MultiviewMaskAdjudication.v1",
        "instances": decisions,
    }
    return payload, "\n".join(aggregate_raw), overlay_paths


# R3 side effects are active on import; replace only the mask court.
v4._run_mask_adjudication = _run_mask_adjudication_r4


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
