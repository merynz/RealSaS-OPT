from __future__ import annotations

"""Frozen RealSaS M4 v0.5 CDT numerical authority.

The historical source is stored as a compressed ASCII payload so the byte stream can
be hash-verified before execution.  The decoded source SHA-256 must remain exactly
``dd21ae3fc570b2eb5d3c439a5c31fd25fed2ad0743d34a71119d5b391806854d``.

This package is numerical machinery only.  Current Compiler IR/provenance/authority
semantics remain owned by the current adapter and qualifiers.
"""

import base64
import hashlib
from pathlib import Path
import sys
import types
import zlib

HISTORICAL_CDT_SOURCE_SHA256 = "dd21ae3fc570b2eb5d3c439a5c31fd25fed2ad0743d34a71119d5b391806854d"
HISTORICAL_CDT_SOURCE_BYTES = 20221
_PAYLOAD_NAME = "cdt_production.py.zlib.b85"
_MODULE_NAME = "realsas_compiler_core._historical_v05_cdt_production"


def historical_cdt_source_bytes() -> bytes:
    payload = (Path(__file__).with_name(_PAYLOAD_NAME)).read_text(encoding="ascii").strip()
    try:
        raw = zlib.decompress(base64.b85decode(payload.encode("ascii")))
    except Exception as exc:
        raise RuntimeError("HISTORICAL_CDT_PAYLOAD_DECODE_FAILED") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != HISTORICAL_CDT_SOURCE_BYTES:
        raise RuntimeError(f"HISTORICAL_CDT_SOURCE_SIZE_DRIFT:{len(raw)}")
    if digest != HISTORICAL_CDT_SOURCE_SHA256:
        raise RuntimeError(f"HISTORICAL_CDT_SOURCE_HASH_DRIFT:{digest}")
    return raw


def load_historical_cdt_module_v05():
    existing = sys.modules.get(_MODULE_NAME)
    if existing is not None:
        return existing
    raw = historical_cdt_source_bytes()
    module = types.ModuleType(_MODULE_NAME)
    module.__file__ = f"<historical-cdt-v05:{HISTORICAL_CDT_SOURCE_SHA256}>"
    module.__package__ = __package__
    sys.modules[_MODULE_NAME] = module
    try:
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


def triangulate_production_cdt(*args, **kwargs):
    return load_historical_cdt_module_v05().triangulate_production_cdt(*args, **kwargs)


__all__ = [
    "HISTORICAL_CDT_SOURCE_SHA256",
    "HISTORICAL_CDT_SOURCE_BYTES",
    "historical_cdt_source_bytes",
    "load_historical_cdt_module_v05",
    "triangulate_production_cdt",
]
