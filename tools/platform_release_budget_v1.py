"""Fail-closed developer Platform capacity guard before creating a new release.

WSL ext4 df alone is misleading because it shows the maximum virtual disk size,
not the underlying Windows drive's free space. Therefore guard /mnt/c too.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

RELEASE_DIR = re.compile(r"[0-9a-f]{40}-[0-9a-f]{8}\Z")
MAX_EXISTING_RELEASES = 12
MIN_HOST_C_FREE = 10 * (1 << 30)
MIN_LINUX_FREE = 8 * (1 << 30)


def require_release_budget(root: Path) -> None:
    releases = root / "releases"
    if releases.is_dir():
        count = sum(1 for p in releases.iterdir()
                    if p.is_dir() and not p.is_symlink()
                    and RELEASE_DIR.fullmatch(p.name))
        if count >= MAX_EXISTING_RELEASES:
            raise RuntimeError(
                f"REALSAS_RELEASE_CAP_REACHED:{count}>={MAX_EXISTING_RELEASES}. "
                "Clean unused developer releases using the explicit safety court, "
                "not arbitrary recursive deletion.")
    c = Path("/mnt/c")
    if c.is_mount() and shutil.disk_usage(c).free < MIN_HOST_C_FREE:
        raise RuntimeError("WINDOWS_C_FREE_BELOW_10_GIB: WSL ext4 may report free "
                           "space even while its backing Windows disk is full")
    existing = root if root.is_dir() else root.parent
    if existing.is_dir() and shutil.disk_usage(existing).free < MIN_LINUX_FREE:
        raise RuntimeError("LINUX_FREE_BELOW_8_GIB: blocking new immutable release")
