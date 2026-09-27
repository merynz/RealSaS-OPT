from __future__ import annotations

"""Domain-neutral typed IR JSON serialization helpers.

Keep this module free of geometry, motion, appearance, runtime and product-state
imports so adapter implementation closures do not couple unrelated domains.
"""

from dataclasses import asdict
import json
from pathlib import Path


def write_ir_json(path: str | Path, value) -> Path:
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    payload=value.to_dict() if hasattr(value,"to_dict") else asdict(value)
    p.write_text(
        json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False)+"\n",
        encoding="utf-8",
    )
    return p
