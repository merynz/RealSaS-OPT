"""Fail-closed, read-only installation inspection for RSS.vNext model slots.

This module does not load pickle, execute checkpoints, confer model-release
authority, or enable product compilation. A qualified execution graph remains
a separate gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

SCHEMA = "RealSaS.BustedCheckpointInstallation.v1"
FAMILIES = ("IRIS", "TESSA", "AXIS", "MIRA")
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_ARCH = re.compile(r"RealSaS\.[A-Za-z0-9_.-]{3,160}\Z")


def inspect_installation(manifest_path: str | Path | None, root: str | Path | None) -> dict[str, Any]:
    """Inspect a pinned four-family manifest without exposing host paths."""
    status: dict[str, Any] = {
        "schema_version": "RealSaS.BustedCheckpointInspection.v1",
        "manifest_valid": False,
        "checkpoints_present": False,
        "compile_available": False,
        "product_authority": False,
        "slots": {family: {"status": "NOT_CONFIGURED"} for family in FAMILIES},
        "blockers": [],
    }
    if not manifest_path or not root:
        status["blockers"].append("CHECKPOINT_INSTALLATION_NOT_CONFIGURED")
        return status
    base = Path(root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not base.is_dir() or not manifest.is_file() or not manifest.is_relative_to(base):
        status["blockers"].append("CHECKPOINT_INSTALLATION_PATH_INVALID")
        return status
    try:
        if manifest.stat().st_size > 1024 * 1024:
            raise ValueError("MANIFEST_TOO_LARGE")
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {"schema", "models"} or document["schema"] != SCHEMA:
            raise ValueError("MANIFEST_SCHEMA_INVALID")
        models = document["models"]
        if not isinstance(models, dict) or set(models) != set(FAMILIES):
            raise ValueError("MANIFEST_SLOTS_INVALID")
    except (OSError, UnicodeError, ValueError, TypeError):
        status["blockers"].append("CHECKPOINT_INSTALLATION_MANIFEST_INVALID")
        return status
    status["manifest_valid"] = True
    seen: set[Path] = set()
    for family in FAMILIES:
        spec = models[family]
        slot = status["slots"][family]
        if not isinstance(spec, dict) or set(spec) != {"path", "sha256", "architecture_id"}:
            slot["status"] = "CONTRACT_INVALID"
            continue
        relative, digest, architecture = spec["path"], spec["sha256"], spec["architecture_id"]
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts or not isinstance(digest, str)
            or not _HEX.fullmatch(digest) or not isinstance(architecture, str)
            or not _ARCH.fullmatch(architecture) or not architecture.startswith("RealSaS." + family + ".")):
            slot["status"] = "CONTRACT_INVALID"
            continue
        target = (base / relative).resolve()
        if not target.is_relative_to(base) or target in seen:
            slot["status"] = "PATH_INVALID"
            continue
        seen.add(target)
        if not target.is_file():
            slot["status"] = "MISSING"
            continue
        hasher = hashlib.sha256()
        try:
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
        except OSError:
            slot["status"] = "UNREADABLE"
            continue
        slot["status"] = "DIGEST_VERIFIED" if hasher.hexdigest() == digest else "DIGEST_MISMATCH"
        slot["architecture_id"] = architecture
        slot["content_sha256"] = digest
    status["checkpoints_present"] = all(
        slot["status"] == "DIGEST_VERIFIED" for slot in status["slots"].values()
    )
    if not status["checkpoints_present"]:
        status["blockers"].append("CHECKPOINT_INSTALLATION_INCOMPLETE")
    status["blockers"].append("PRODUCT_MODELRELEASE_COMPILER_PROOF_GATE_NOT_CLOSED")
    return status


def inspect_environment() -> dict[str, Any]:
    return inspect_installation(
        os.environ.get("BUSTED_CHECKPOINT_MANIFEST"),
        os.environ.get("BUSTED_CHECKPOINT_ROOT"),
    )
