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



def historical_release_references(root: Path, reference_index: Path) -> tuple[set[Path], int]:
    """Protect every release path mentioned by audited research text records.

    The index is a newline-separated list of *existing original record paths*
    produced by a read-only grep. Source documents must live in exact allowlisted
    roots; missing or unreadable documents fail closed. No scientific file writes.
    This is an additional safety net, not a proof of comprehensive dependency
    closure across all binary or remote records.
    """
    index = reference_index.expanduser().absolute()
    if index.is_symlink() or not index.is_file():
        raise RuntimeError("HISTORICAL_REFERENCE_INDEX_MISSING_OR_UNTRUSTED")
    authority = root.parent / "realsas_authority"
    allowed = (authority.resolve(strict=True),
               (root / "receipts").resolve(strict=True),
               (root / "research-results").resolve(strict=True))
    matcher = re.compile(
        re.escape(str(root / "releases")) +
        r"/([0-9a-f]{40}-[0-9a-f]{8})(?![0-9a-f-])"
    )
    protected: set[Path] = set()
    count = 0
    with index.open(encoding="utf-8") as handles:
        for raw in handles:
            name = raw.strip()
            if not name:
                continue
            record = Path(name)
            if not record.is_absolute() or record.is_symlink() or not record.is_file():
                raise RuntimeError("HISTORICAL_REFERENCE_RECORD_MISSING_OR_UNTRUSTED:" + name)
            resolved = record.resolve(strict=True)
            if not any(resolved.is_relative_to(p) for p in allowed):
                raise RuntimeError("HISTORICAL_REFERENCE_RECORD_OUTSIDE_ALLOWED_ROOT:" + name)
            count += 1
            try:
                with record.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        for match in matcher.finditer(line):
                            protected.add(root / "releases" / match.group(1))
            except OSError as exc:
                raise RuntimeError("HISTORICAL_REFERENCE_RECORD_READ_FAILED:" + name) from exc
    if count == 0:
        raise RuntimeError("EMPTY_HISTORICAL_REFERENCE_INDEX_REQUIRES_REVIEW")
    return protected, count

def plan(root: Path, retain: int = 6, reference_index: Path | None = None) -> dict:
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
    historical = set()
    inspected_records = 0
    if reference_index is not None:
        historical, inspected_records = historical_release_references(root, reference_index)
        keep |= historical & set(actual)
    candidates = [p for p in actual if p not in keep]
    return {"root": str(root), "active": str(current),
            "release_count": len(actual), "keep_count": len(set(actual) & keep),
            "candidate_count": len(candidates), "keep": sorted(str(x) for x in keep),
            "candidates": [str(x) for x in candidates],
            "protected_historical_release_count": len(historical & set(actual)),
            "inspected_historical_reference_records": inspected_records,
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


def apply(root: Path, retain: int, confirmation: str, reference_index: Path | None = None) -> dict:
    if confirmation != CONFIRM:
        raise RuntimeError("EXPLICIT_CONFIRMATION_REQUIRED")
    if reference_index is None:
        raise RuntimeError("HISTORICAL_REFERENCE_REVIEW_REQUIRED_FOR_DELETION")
    preview = plan(root, retain, reference_index)
    target_inactive()
    check_no_references(preview["candidates"])
    check_no_mounted_subtrees(preview["candidates"])
    removed = 0
    for candidate in preview["candidates"]:
        now = plan(root, retain, reference_index)
        if candidate not in now["candidates"]:
            raise RuntimeError("RELEASE_PLAN_CHANGED__STOPPED")
        p = Path(candidate)
        if not p.is_dir() or p.is_symlink() or p.parent != Path(now["root"]) / "releases":
            raise RuntimeError("RELEASE_PATH_DRIFT__STOPPED")
        shutil.rmtree(p)
        removed += 1
        print("REMOVED", p.name, flush=True)
    return {"removed": removed, "remaining": plan(root, retain, reference_index)["release_count"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.home() / "realsas_platform")
    p.add_argument("--retain", type=int, default=6)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--reference-file-list", type=Path, help="Audited list of historical text records containing release paths")
    p.add_argument("--confirm", default="")
    args = p.parse_args()
    report = plan(args.root, args.retain, args.reference_file_list)
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))
    print("CANDIDATE_COUNT:", report["candidate_count"])
    print("PREVIEW_CANDIDATES_FIRST_10:", [Path(x).name for x in report["candidates"][:10]])
    if args.apply:
        print("APPLY_RESULT:", json.dumps(apply(args.root, args.retain, args.confirm, args.reference_file_list)))
    else:
        print("DRY_RUN_ONLY: no files deleted")


if __name__ == "__main__":
    main()
