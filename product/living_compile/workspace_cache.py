from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

from .common import LivingCompileError, product_bundle_root, within


_COMPILE_ID = re.compile(r"^cmp_[0-9a-f]{32}$")
_REQUIRED = {
    "bundle_manifest.json",
    "puppet/canonical_puppet_graph_v3.json",
    "proof/product_proof_bundle_ir.json",
}
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_MAX_ENTRY_BYTES = 256 * 1024 * 1024


def _validate_compile_id(value: str) -> str:
    value = str(value or "").strip()
    if not _COMPILE_ID.fullmatch(value):
        raise LivingCompileError("invalid public compile id")
    return value


def _validate_cached_bundle(root: Path) -> None:
    for relative in _REQUIRED:
        path = within(root, relative)
        if not path.is_file():
            raise LivingCompileError(f"workspace cache missing required file: {relative}")


def materialize_workspace_bundle(compile_id: str, payload: bytes) -> str:
    compile_id = _validate_compile_id(compile_id)
    if not payload or len(payload) > _MAX_BUNDLE_BYTES:
        raise LivingCompileError("workspace bundle size invalid")
    bundle_root = product_bundle_root()
    if bundle_root is None:
        raise LivingCompileError("REALSAS_PRODUCT_BUNDLE_ROOT is required for production workspace materialization")
    bundle_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(payload).hexdigest()
    relative = f"public/{compile_id}/{digest}"
    target = within(bundle_root, relative)
    if target.is_dir():
        _validate_cached_bundle(target)
        return relative

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=parent))
    try:
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload), "r")
        except zipfile.BadZipFile as exc:
            raise LivingCompileError("workspace bundle is not a valid zip") from exc
        with archive:
            names: set[str] = set()
            total = 0
            for info in archive.infolist():
                name = info.filename.replace("\\", "/")
                if not name or name.startswith("/") or "../" in name or name.endswith("/.."):
                    raise LivingCompileError("workspace bundle contains unsafe path")
                if info.file_size > _MAX_ENTRY_BYTES:
                    raise LivingCompileError("workspace bundle entry exceeds size limit")
                total += info.file_size
                if total > _MAX_BUNDLE_BYTES:
                    raise LivingCompileError("workspace bundle expanded size exceeds limit")
                names.add(name)
                if info.is_dir():
                    within(temp, name).mkdir(parents=True, exist_ok=True)
                    continue
                destination = within(temp, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
            missing = sorted(_REQUIRED - names)
            if missing:
                raise LivingCompileError("workspace bundle missing required files: " + ", ".join(missing))
        _validate_cached_bundle(temp)
        try:
            os.rename(temp, target)
        except FileExistsError:
            _validate_cached_bundle(target)
        return relative
    finally:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
