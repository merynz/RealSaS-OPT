#!/usr/bin/env python3
from __future__ import annotations

"""Robust runtime entrypoint for the V4 multiview semantic oracle.

R3 keeps the V4 authority boundary unchanged while fixing two operational
failures observed in Colab:

1. local ``file://`` image URIs are normalized to absolute paths before
   Transformers consumes them (R2 behavior);
2. the localization oracle uses a compact generation-only schema, persists raw
   attempts before parsing, and performs one explicit retry when JSON is
   syntactically invalid or fails the compact contract.

The compact payload is deterministically expanded into the existing
``RealSaS.MultiviewSemanticOracle.v1`` contract before the unchanged validator,
SAM2 prompted segmentation, mask courts and M fusion consume it.

No malformed output is silently repaired.  Two invalid attempts fail closed.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from tools.research import run_multiview_semantic_oracle_v4 as v4


COMPACT_SCHEMA = "RealSaS.MultiviewSemanticOracleCompact.v1"
MAX_INSTANCES = 12
_ALLOWED_RELATIONS = {"HELD_BY", "IN_FRONT_OF", "BEHIND", "ATTACHED_TO"}


def normalize_local_image_sources_v1(messages):
    """Return a copy with ``file://`` image URIs converted to absolute paths."""

    out = deepcopy(messages)
    for message in out:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            value = item.get("image")
            if not isinstance(value, str) or not value.startswith("file://"):
                continue
            parsed = urlparse(value)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                raise RuntimeError(f"UNSUPPORTED_LOCAL_IMAGE_URI:{value}")
            item["image"] = str(Path(unquote(parsed.path)).resolve())
    return out


def _extract_json_strict(text: str) -> dict:
    """Extract one JSON object without syntax repair."""

    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("MULTIMODAL_JSON_NOT_FOUND")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MULTIMODAL_JSON_INVALID:{exc}") from exc


def _compact_prompt() -> str:
    labels = ",".join(v4.CANONICAL_SEMANTIC_LABELS)
    return f"""
You are a non-authoritative multiview semantic oracle for an animation compiler.
All 8 images V0..V7 show the SAME stylized character at equal scale.
Reason across ALL views before answering. Do not infer a part from silhouette resemblance alone.
Allowed labels: {labels}.

Return at most {MAX_INSTANCES} animation-relevant physical instances. Prefer omission over guessing.
A pointed helmet contour is not automatically a horn. A long rigid held object is not an arm.
Use one stable id per physical instance across views.

For each visible view, encode exactly seven numbers:
[x1,y1,x2,y2,cx,cy,confidence]
where coordinates are integers 0..1000, box is tight around ONLY that part, center is an interior point, confidence is 0..1.
Do NOT include invisible/uncertain views in `v`.
Relations are optional triples [TYPE,target_id,confidence].

Return MINIFIED JSON ONLY using exactly this compact schema:
{{"schema":"{COMPACT_SCHEMA}","summary":"short","instances":[{{"id":"I00","label":"sword","confidence":0.9,"v":{{"0":[100,200,300,700,210,430,0.9]}},"r":[["HELD_BY","I01",0.9]]}}]}}
No markdown. No comments. No ellipses. No trailing commas.
""".strip()


def _retry_prompt() -> str:
    return f"""
Your previous response was not valid compact JSON. Re-read the same 8 images and regenerate from scratch.
Return at most {MAX_INSTANCES} instances. Output ONE minified JSON object only.
Required schema: {COMPACT_SCHEMA}.
Each instance: id,label,confidence,v,r.
`v` contains only visible view keys 0..7; each value is [x1,y1,x2,y2,cx,cy,confidence].
`r` is a list of [HELD_BY|IN_FRONT_OF|BEHIND|ATTACHED_TO,target_id,confidence].
No prose, markdown, comments, ellipses or trailing commas.
""".strip()


def expand_compact_oracle_v1(payload: dict) -> dict:
    """Validate compact generation data and expand to canonical V1 oracle IR."""

    if not isinstance(payload, dict) or payload.get("schema") != COMPACT_SCHEMA:
        raise ValueError("compact oracle schema mismatch")
    rows = payload.get("instances")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_INSTANCES:
        raise ValueError("compact oracle instances missing or over limit")

    allowed_labels = set(v4.CANONICAL_SEMANTIC_LABELS)
    seen: set[str] = set()
    expanded = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("compact instance must be object")
        iid = str(row.get("id", "")).strip()
        label = str(row.get("label", "")).strip()
        if not iid or iid in seen:
            raise ValueError("compact instance id missing or duplicated")
        seen.add(iid)
        if label not in allowed_labels:
            raise ValueError(f"unsupported compact label:{label}")
        confidence = float(row.get("confidence", -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("compact instance confidence outside [0,1]")

        visible = row.get("v", {})
        if not isinstance(visible, dict):
            raise ValueError("compact visible views must be object")
        views = {
            str(view): {
                "present": False,
                "bbox_1000": None,
                "center_1000": None,
                "confidence": 0.0,
            }
            for view in range(8)
        }
        for key, value in visible.items():
            if str(key) not in views:
                raise ValueError(f"compact invalid view:{key}")
            if not isinstance(value, list) or len(value) != 7:
                raise ValueError("compact visible view must have seven values")
            x1, y1, x2, y2, cx, cy, vconf = value
            coords = [float(x1), float(y1), float(x2), float(y2), float(cx), float(cy)]
            if not all(0.0 <= x <= 1000.0 for x in coords):
                raise ValueError("compact coordinate outside [0,1000]")
            if float(x2) <= float(x1) or float(y2) <= float(y1):
                raise ValueError("compact box must have positive area")
            vconf = float(vconf)
            if not 0.0 <= vconf <= 1.0:
                raise ValueError("compact view confidence outside [0,1]")
            views[str(key)] = {
                "present": True,
                "bbox_1000": [float(x1), float(y1), float(x2), float(y2)],
                "center_1000": [float(cx), float(cy)],
                "confidence": vconf,
            }

        relations = []
        raw_relations = row.get("r", [])
        if not isinstance(raw_relations, list):
            raise ValueError("compact relations must be list")
        for rel in raw_relations:
            if not isinstance(rel, list) or len(rel) != 3:
                raise ValueError("compact relation must have three values")
            rel_type, target, rconf = str(rel[0]), str(rel[1]), float(rel[2])
            if rel_type not in _ALLOWED_RELATIONS:
                raise ValueError(f"unsupported compact relation:{rel_type}")
            if not 0.0 <= rconf <= 1.0:
                raise ValueError("compact relation confidence outside [0,1]")
            relations.append({
                "type": rel_type,
                "target_instance_id": target,
                "confidence": rconf,
            })

        expanded.append({
            "instance_id": iid,
            "label": label,
            "confidence": confidence,
            "views": views,
            "relations": relations,
        })

    canonical = {
        "schema": "RealSaS.MultiviewSemanticOracle.v1",
        "subject_summary": str(payload.get("summary", "")).strip(),
        "instances": expanded,
    }
    v4.validate_multiview_oracle_payload_v1(canonical)
    return canonical


_original_qwen_generate = v4._qwen_generate


def _qwen_generate_r3(model, processor, messages, *, max_new_tokens: int) -> str:
    return _original_qwen_generate(
        model,
        processor,
        normalize_local_image_sources_v1(messages),
        max_new_tokens=max_new_tokens,
    )


def _content_for_views(paths: list[Path], prompt: str):
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


def _run_localization_oracle_r3(model, processor, paths: list[Path], *, max_new_tokens: int):
    """Persist raw attempts, parse strictly, retry once, expand to canonical IR."""

    if not paths:
        raise RuntimeError("ORACLE_SOURCE_VIEWS_EMPTY")
    work = paths[0].parent.parent
    work.mkdir(parents=True, exist_ok=True)

    attempts = [(_compact_prompt(), "oracle_raw_attempt1.txt")]
    last_error: Exception | None = None
    for attempt in range(2):
        prompt, filename = attempts[attempt] if attempt < len(attempts) else (_retry_prompt(), "oracle_raw_attempt2.txt")
        messages = _content_for_views(paths, prompt)
        raw = _qwen_generate_r3(
            model,
            processor,
            messages,
            max_new_tokens=max(int(max_new_tokens), 6144),
        )
        (work / filename).write_text(raw, encoding="utf-8")
        try:
            compact = _extract_json_strict(raw)
            canonical = expand_compact_oracle_v1(compact)
            (work / f"oracle_compact_attempt{attempt + 1}.json").write_text(
                json.dumps(compact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return canonical, raw
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            last_error = exc
            if attempt == 0:
                attempts.append((_retry_prompt(), "oracle_raw_attempt2.txt"))
                continue
            break
    raise RuntimeError(f"MULTIMODAL_ORACLE_TWO_ATTEMPTS_INVALID:{last_error}") from last_error


# Apply both runtime compatibility and compact typed retry without changing core V4.
v4._qwen_generate = _qwen_generate_r3
v4._run_localization_oracle = _run_localization_oracle_r3


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
