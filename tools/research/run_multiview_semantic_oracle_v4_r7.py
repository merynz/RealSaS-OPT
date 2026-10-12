#!/usr/bin/env python3
from __future__ import annotations

"""R7 compatibility layer for Qwen single-view grounding tokens.

R6 produced useful bbox/point grounding, but the strict validator rejected
semantically equivalent view identifiers such as ``V5`` and
``orthographic view V6`` because it called ``int(...)`` directly.

R7 changes only view-token normalization before the unchanged R6 validator.
Malformed JSON, wrong labels, invalid boxes/points, ambiguous view strings and
out-of-range views remain fail-closed.
"""

from copy import deepcopy
import re

from tools.research import run_multiview_semantic_oracle_v4 as v4
from tools.research import run_multiview_semantic_oracle_v4_r6 as r6


_VIEW_PATTERNS = (
    re.compile(r"^\s*([0-7])\s*$", re.IGNORECASE),
    re.compile(r"^\s*V\s*([0-7])\s*$", re.IGNORECASE),
    re.compile(r"^\s*view\s+V?\s*([0-7])\s*$", re.IGNORECASE),
    re.compile(r"^\s*orthographic\s+view\s+V?\s*([0-7])\s*$", re.IGNORECASE),
)


def normalize_view_token_v1(value) -> int:
    if isinstance(value, bool):
        raise ValueError("single-view grounding view bool is invalid")
    if isinstance(value, int):
        if 0 <= value <= 7:
            return int(value)
        raise ValueError("single-view grounding view outside [0,7]")
    if isinstance(value, float) and value.is_integer():
        ivalue = int(value)
        if 0 <= ivalue <= 7:
            return ivalue
        raise ValueError("single-view grounding view outside [0,7]")
    if not isinstance(value, str):
        raise ValueError("single-view grounding view token type invalid")
    for pattern in _VIEW_PATTERNS:
        match = pattern.fullmatch(value)
        if match:
            return int(match.group(1))
    raise ValueError(f"single-view grounding view token invalid:{value}")


_original_validate = r6._validate_single_view


def _validate_single_view_r7(payload: dict, *, instance: dict, view: int) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("single-view grounding payload must be object")
    normalized = deepcopy(payload)
    normalized["view"] = normalize_view_token_v1(normalized.get("view", -1))
    return _original_validate(normalized, instance=instance, view=view)


# R6 _call_single_view resolves its validator through the r6 module globals.
r6._validate_single_view = _validate_single_view_r7


def main() -> int:
    return v4.main()


if __name__ == "__main__":
    raise SystemExit(main())
