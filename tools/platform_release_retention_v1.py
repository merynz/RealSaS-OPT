#!/usr/bin/env python3
"""Fail-closed RealSaS developer release garbage collection.

Default is read-only. --apply requires an exact confirmation string, an inactive
RealSaS Platform systemd target, no running process referencing a candidate,
and preservation of both deployment marker references and current symlink.
Never accesses artifacts, database, research authority or other directories.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

RELEASE_NAME = re.compile(r"[0-9a-f]{40}-[0-9a-f]{8}\Z")
CONFIRM = "PRUNE-REALSAS-OLD-RELEASES"


def checked_root(root: Path) -> tuple[Path, Path, Path, dict]:
    root = root.expanduser().absolute()
    if os.geteuid() == 0:
        raise RuntimeError("RUN_AS_NORMAL_USER_NOT_ROOT")
    if root.is_symlink() or not root.is_dir() or len(root.parts) < 4:
        raise RuntimeError("UNSAFE_ROOT_OR_MISSING")
    releases = root / "releases"
    current_link = root / "current"
    marker = root / "deployment.json"
    if releases.is_symlink() or not releases.is_dir():
        raise RuntimeError("UNSAFE_RELEASE_DIRECTORY")
    if not current_link.is_symlink() or not marker.is_file() or marker.is_symlink():
        raise RuntimeError("CURRENT_LINK_OR_DEPLOYMENT_MARKER_MISSING")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("schema") != "RealSaS.DeveloperDeployment.v1" or data.get("root") != str(root):
        raise RuntimeError("DEPLOYMENT_MARKER_IDENTITY_DRIFT")
    current = current_link.resolve(strict=True)
    if current.parent != releases.resolve(strict=True) or not RELEASE_NAME.fullmatch(current.name):
        raise RuntimeError("ACTIVE_RELEASE_OUTSIDE_MANAGED_ROOT")
    recorded = data.get("current_release")
    if recorded and Path(recorded).resolve(strict=True) != current:
        raise RuntimeError("ACTIVE_RELEASE_MARKER_MISMATCH")
    return root, releases, current, data


def plan(root: Path, retain: int = 6) -> dict:
    if not 5 <= retain <= 30:
        raise RuntimeError("RETENTION_COUNT_OUT_OF_SAFE_RANGE")
    root, releases, current, data = checked_root(root)
    actual = []
    skipped = []
    for entry in releases.iterdir():
        if entry.is_symlink() or not entry.is_dir() or not RELEASE_NAME.fullmatch(entry.name):
            skipped.append(entry.name)
            continue
        if entry.resolve() != entry:
            raise RuntimeError("UNSAFE_NESTED_RELEASE_SYMLINK")
        actual.append(entry)
    actual.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    keep = set(actual[:retain]) | {current}
    for key in ("current_release", "previous_release"):
        ref = data.get(key)
        if ref:
            p = Path(ref)
            if not p.is_absolute() or p.parent != releases:
                raise RuntimeError("UNSAFE_DEPLOYMENT_RELEASE_REFERENCE:" + key)
            if p.is_dir() and not p.is_symlink():
                keep.add(p)
    for p in root.iterdir():
        if p.is_symlink():
            try:
                ref = p.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            if ref.parent == releases:
                keep.add(ref)
    candidates = [p for p in actual if p not in keep]
    return {"root": str(root), "active": str(current),
            "release_count": len(actual), "keep_count": len(set(actual) & keep),
            "candidate_count": len(candidates), "keep": sorted(str(x) for x in keep),
            "candidates": [str(x) for x in candidates],
            "skipped_unmanaged_entries": skipped}


def target_inactive() -> None:
    for name in ("target", "postgres", "temporal", "migrate", "api", "control", "engine"):
        unit = "realsas-platform.target" if name == "target" else f"realsas-platform-{name}.service"
        r = subprocess.run(["systemctl", "--user", "is-active", unit],
                           capture_output=True, text=True, check=False, timeout=10)
        if r.returncode != 3 or r.stdout.strip() not in ("inactive", "failed"):
            raise RuntimeError("PLATFORM_UNITS_MUST_BE_STOPPED_BEFORE_PRUNE: " +
                               unit + ":" + r.stdout.strip())


def check_no_references(candidates: list[str]) -> None:
    if not candidates:
        return
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        parts = []
        for attr in ("exe", "cwd", "root"):
            try:
                parts.append(os.readlink(proc / attr))
            except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
                pass
        for attr in ("cmdline", "maps"):
            try:
                data = (proc / attr).read_bytes()
                parts.append(data.decode("utf-8", "replace"))
            except (PermissionError, FileNotFoundError, ProcessLookupError, OSError):
                pass
        haystack = "\n".join(parts)
        for candidate in candidates:
            if candidate in haystack:
                raise RuntimeError("LIVE_PROCESS_REFERENCES_OLD_RELEASE: " + proc.name)


def check_no_mounted_subtrees(candidates: list[str]) -> None:
    try:
        info = Path("/proc/self/mountinfo").read_text()
    except OSError as exc:
        raise RuntimeError("MOUNTINFO_UNAVAILABLE") from exc
    mountpoints = [line.split(" - ", 1)[0].split()[4].replace("\\040", " ")
                   for line in info.splitlines() if " - " in line and len(line.split()) >= 5]
    for candidate in candidates:
        for mount in mountpoints:
            if mount == candidate or mount.startswith(candidate + "/"):
                raise RuntimeError("MOUNTPOINT_INSIDE_DELETE_CANDIDATE:" + candidate)


def apply(root: Path, retain: int, confirmation: str) -> dict:
    if confirmation != CONFIRM:
        raise RuntimeError("EXPLICIT_CONFIRMATION_REQUIRED")
    preview = plan(root, retain)
    target_inactive()
    check_no_references(preview["candidates"])
    check_no_mounted_subtrees(preview["candidates"])
    removed = 0
    for candidate in preview["candidates"]:
        now = plan(root, retain)
        if candidate not in now["candidates"]:
            raise RuntimeError("RELEASE_PLAN_CHANGED__STOPPED")
        p = Path(candidate)
        if not p.is_dir() or p.is_symlink() or p.parent != Path(now["root"]) / "releases":
            raise RuntimeError("RELEASE_PATH_DRIFT__STOPPED")
        shutil.rmtree(p)
        removed += 1
        print("REMOVED", p.name, flush=True)
    return {"removed": removed, "remaining": plan(root, retain)["release_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() / "realsas_platform")
    p.add_argument("--retain", type=int, default=6)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()
    report = plan(args.root, args.retain)
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))
    print("CANDIDATE_COUNT:", report["candidate_count"])
    print("PREVIEW_CANDIDATES_FIRST_10:", [Path(x).name for x in report["candidates"][:10]])
    if args.apply:
        print("APPLY_RESULT:", json.dumps(apply(args.root, args.retain, args.confirm)))
    else:
        print("DRY_RUN_ONLY: no files deleted")


if __name__ == "__main__":
    main()
