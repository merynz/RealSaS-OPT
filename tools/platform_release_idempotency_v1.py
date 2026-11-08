"""Prevent duplicate full Platform installs for an unchanged canonical commit.

This is intentionally conservative: only the current, recorded, complete release
can be reused. Older matching-SHA directories require an independent equivalence
court before reuse. No migration, symlink switch or service restart occurs here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

RELEASE_DIR = re.compile(r"[0-9a-f]{40}-[0-9a-f]{8}\Z")


def matching_current_release(root: Path, expected_sha: str, authority: Path) -> Path | None:
    marker = root / "deployment.json"
    if not marker.is_file() or marker.is_symlink():
        return None
    data = json.loads(marker.read_text())
    if data.get("schema") != "RealSaS.DeveloperDeployment.v1" or data.get("root") != str(root):
        raise RuntimeError("DEPLOY_IDEMPOTENCY_MARKER_DRIFT")
    if data.get("code_sha") != expected_sha:
        return None
    if data.get("state") != "SERVICES_ACTIVE":
        raise RuntimeError("MATCHING_SHA_PREVIOUS_DEPLOY_NOT_COMPLETE: do not duplicate without explicit operator recovery")
    if data.get("authority_root") != str(authority):
        raise RuntimeError("MATCHING_SHA_AUTHORITY_ROOT_DRIFT")
    link = root / "current"
    if not link.is_symlink():
        raise RuntimeError("MATCHING_SHA_ACTIVE_RELEASE_LINK_MISSING")
    release = link.resolve(strict=True)
    releases_dir = (root / "releases").resolve(strict=True)
    if release.parent != releases_dir or not RELEASE_DIR.fullmatch(release.name):
        raise RuntimeError("MATCHING_SHA_RELEASE_OUTSIDE_MANAGED_ROOT")
    if str(release) != data.get("current_release") or not release.name.startswith(expected_sha + "-"):
        raise RuntimeError("MATCHING_SHA_ACTIVE_RELEASE_ID_DRIFT")
    for p in (release / "source", release / "venv/bin/python",
              release / "bin/realsas-api", release / "bin/realsas-control-worker",
              release / "bin/temporal", release / "bin/realsas-python"):
        if not p.exists():
            raise RuntimeError("MATCHING_SHA_RELEASE_INCOMPLETE:" + str(p))
    return release
