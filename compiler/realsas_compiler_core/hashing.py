from __future__ import annotations
from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json, math
from typing import Any

def canonicalize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value=value.to_dict()
    elif is_dataclass(value):
        value=asdict(value)
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k,v in sorted(value.items(), key=lambda kv:str(kv[0]))}
    if isinstance(value, (tuple,list)):
        return [canonicalize(v) for v in value]
    if isinstance(value, set):
        return [canonicalize(v) for v in sorted(value,key=repr)]
    if isinstance(value,float):
        if not math.isfinite(value): raise ValueError("non-finite canonical value")
        return value
    if value is None or isinstance(value,(str,int,bool)): return value
    return repr(value)

def canonical_bytes(value:Any)->bytes:
    return json.dumps(canonicalize(value),ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def content_sha256(value:Any)->str: return sha256(canonical_bytes(value)).hexdigest()
