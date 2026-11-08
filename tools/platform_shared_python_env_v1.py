"""Shared immutable developer-service venv, keyed by pinned Python and requirements.

Only developer Platform processes may use this cache. Research/training envs
remain independently frozen. One physical environment is shared by releases
that have identical, explicitly enumerated requirements. Release/venv is a
symlink, not an in-place mutable environment. Never garbage-collect this cache
as a side effect of release retention.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

if __package__:
    from .platform_ci_bootstrap import ensure_venv, fingerprint
else:
    from platform_ci_bootstrap import ensure_venv, fingerprint


def requirement_closure(source: Path) -> list[Path]:
    directory = (source / "requirements").resolve(strict=True)
    visited: set[Path] = set()
    ordered: list[Path] = []

    def visit(raw: Path) -> None:
        resolved = raw.resolve(strict=True)
        if resolved.parent != directory or not resolved.is_file():
            raise RuntimeError("DEVELOPER_REQUIREMENT_OUTSIDE_PINNED_DIRECTORY")
        if resolved in visited:
            return
        visited.add(resolved)
        ordered.append(resolved)
        for line in resolved.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            included = re.fullmatch(r"-r\s+([a-zA-Z0-9._-]+\.txt)", stripped)
            if included:
                visit(directory / included.group(1))
            elif stripped.startswith("-r") or stripped.startswith("--requirement"):
                raise RuntimeError("DEVELOPER_REQUIREMENT_UNSUPPORTED_INCLUDE:" + stripped)

    visit(directory / "platform-deployment.txt")
    return ordered


def provision_shared_venv(root: Path, source: Path, release: Path) -> tuple[Path, str]:
    if release.is_symlink() or not release.is_dir() or source != release / "source":
        raise RuntimeError("SHARED_VENV_RELEASE_SCOPE_DRIFT")
    cache_root = root / "shared_python_envs"
    if cache_root.is_symlink():
        raise RuntimeError("SHARED_VENV_ROOT_SYMLINK_REFUSED")
    requirements = requirement_closure(source)
    digest = fingerprint([], requirements)  # Python patch version + every recursive requirements file SHA
    target = cache_root / "venvs" / digest
    stamp = target / ".realsas-env.json"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_dir() or not stamp.is_file():
            raise RuntimeError("SHARED_VENV_EXISTING_STATE_INVALID_NO_REBUILD")
        try:
            state = json.loads(stamp.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError("SHARED_VENV_STAMP_INVALID_NO_REBUILD") from exc
        if state.get("fingerprint") != digest or not (target / "bin/python").is_file():
            raise RuntimeError("SHARED_VENV_IDENTITY_DRIFT_NO_REBUILD")
    # Existing CI builder: global persistent lock + per-fingerprint lock +
    # atomic directory swap. Reuses complete environment without pip reinstall.
    receipt = ensure_venv(cache_root, [], requirement_files=requirements)
    if Path(receipt["venv"]) != target or receipt["fingerprint"] != digest:
        raise RuntimeError("SHARED_VENV_BOOTSTRAP_IDENTITY_DRIFT")
    link = release / "venv"
    if link.exists() or link.is_symlink():
        raise RuntimeError("RELEASE_VENV_ALREADY_EXISTS")
    link.symlink_to(target)
    if link.resolve(strict=True) != target:
        raise RuntimeError("SHARED_VENV_SYMLINK_IDENTITY_DRIFT")
    return link, digest
