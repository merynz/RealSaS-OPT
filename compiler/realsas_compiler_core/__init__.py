"""Canonical RealSaS compiler entry surface.

Current typed compiler code is ordinary source. The small historical execution
closure is stored as lossless base64 transport chunks, reconstructed and SHA
verified, then extracted to a filesystem cache before import. Heavy historical
proof/repair/numerical authorities remain SHA-bound external provenance and are
not silently promoted into the current execution closure.
"""
from __future__ import annotations

from pathlib import Path as _Path
import base64 as _base64
import hashlib as _hashlib
import io as _io
import json as _json
import shutil as _shutil
import sys as _sys
import tempfile as _tempfile
import zipfile as _zipfile

_bundle_dir = _Path(__file__).resolve().parents[1] / "vendor" / "realsas_v05_current_execution_closure.b64"
_manifest_path = _bundle_dir / "manifest.json"
if not _manifest_path.is_file():
    raise ImportError(f"RealSaS compiler vendor manifest missing: {_manifest_path}")
_m = _json.loads(_manifest_path.read_text(encoding="utf-8"))
if _m.get("scope") != "CURRENT_IRIS_TO_COMPILER_EXECUTION_CLOSURE":
    raise ImportError("RealSaS compiler vendor scope mismatch")

_encoded_parts = []
for _rec in _m.get("part_records", []):
    _p = _bundle_dir / _rec["name"]
    if not _p.is_file():
        raise ImportError(f"RealSaS compiler vendor part missing: {_p}")
    _b = _p.read_bytes()
    if len(_b) != int(_rec["chars"]):
        raise ImportError(f"RealSaS compiler vendor part size mismatch: {_p.name}")
    _got = _hashlib.sha256(_b).hexdigest()
    if _got != _rec["sha256"]:
        raise ImportError(f"RealSaS compiler vendor part SHA mismatch: {_p.name}: {_got}")
    _encoded_parts.append(_b.decode("ascii"))
_raw = _base64.b64decode("".join(_encoded_parts), validate=True)
if len(_raw) != int(_m["raw_size_bytes"]):
    raise ImportError("RealSaS compiler vendor raw size mismatch")
_raw_sha = _hashlib.sha256(_raw).hexdigest()
if _raw_sha != _m["raw_sha256"]:
    raise ImportError(f"RealSaS compiler vendor raw SHA mismatch: {_raw_sha}")

_cache_root = _Path(_tempfile.gettempdir()) / "realsas_compiler_vendor" / _raw_sha[:20]
_marker = _cache_root / ".verified_sha256"
if not (_marker.is_file() and _marker.read_text(encoding="ascii").strip() == _raw_sha):
    if _cache_root.exists():
        _shutil.rmtree(_cache_root)
    _cache_root.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(_io.BytesIO(_raw), "r") as _zf:
        for _info in _zf.infolist():
            _name = _Path(_info.filename)
            if _name.is_absolute() or ".." in _name.parts:
                raise ImportError(f"unsafe path in RealSaS compiler vendor archive: {_info.filename}")
        _zf.extractall(_cache_root)
    for _rec in _m.get("records", []):
        _p = _cache_root / _rec["path"]
        if not _p.is_file():
            raise ImportError(f"RealSaS compiler restored entry missing: {_rec['path']}")
        _b = _p.read_bytes()
        if len(_b) != int(_rec["bytes"]):
            raise ImportError(f"RealSaS compiler restored entry size mismatch: {_rec['path']}")
        _got = _hashlib.sha256(_b).hexdigest()
        if _got != _rec["sha256"]:
            raise ImportError(f"RealSaS compiler restored entry SHA mismatch: {_rec['path']}: {_got}")
    _marker.write_text(_raw_sha + "\n", encoding="ascii")

if str(_cache_root) not in _sys.path:
    _sys.path.insert(0, str(_cache_root))

from .types import *
from .v4_types import *
from .surface import *
from .rig import *
from .skin import *
from .mesh_binding import *
from .directional_binding import *
from .product import *
from .v4 import *
from .compile_transaction import *
from .api import CompilerFacade
from .bundle_routes import *
