#!/usr/bin/env python3
"""Bounded GC for persistent per-runner RealSaS tool caches.

Correctness rule: never prune arbitrary files out of a reusable environment or
Go module tree. Venvs are evicted as whole fingerprint directories; generic
build/download caches are evicted as whole namespaces only under disk pressure.
The external artifact CAS has a different lifetime contract and is intentionally
not collected here until executor leases are wired.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .platform_cache_lock import cache_lease
else:
    from platform_cache_lock import cache_lease


@dataclass(frozen=True)
class Entry:
    path: Path
    size: int
    mtime: float
    kind: str
    lock_path: Path | None = None


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if not child.is_file() or child.is_symlink():
            continue
        try:
            total += child.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _entry_mtime(path: Path) -> float:
    stamp = path / ".realsas-env.json"
    try:
        return (stamp if stamp.is_file() else path).stat().st_mtime
    except FileNotFoundError:
        return 0.0


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return

    def make_writable(_func, target, _exc):
        try:
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except OSError:
            pass
        try:
            if Path(target).is_dir():
                shutil.rmtree(target, onerror=make_writable)
            else:
                Path(target).unlink(missing_ok=True)
        except OSError:
            pass

    shutil.rmtree(path, onerror=make_writable)


def scan_entries(root: Path) -> tuple[list[Entry], dict[str, int]]:
    entries: list[Entry] = []
    namespace_bytes: dict[str, int] = {}

    temp_root = root / ".tmp"
    if temp_root.is_dir():
        for child in temp_root.iterdir():
            entries.append(Entry(child, _tree_size(child), _entry_mtime(child), "temporary"))

    venv_root = root / "venvs"
    if venv_root.is_dir():
        for child in venv_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                entries.append(Entry(
                    child,
                    _tree_size(child),
                    _entry_mtime(child),
                    "venv",
                    venv_root / f"{child.name}.lock",
                ))

    # These caches are internally structured. Treating them as a single atomic
    # namespace avoids a half-deleted wheel/module/build entry being reused.
    for namespace in ("go-build", "pip", "go-mod"):
        path = root / namespace
        size = _tree_size(path)
        namespace_bytes[namespace] = size
        if path.exists() and size:
            entries.append(Entry(path, size, _entry_mtime(path), f"namespace:{namespace}"))

    return entries, namespace_bytes


def collect(
    root: Path,
    *,
    soft_max_bytes: int,
    target_max_bytes: int,
    max_age_seconds: int,
    min_free_bytes: int,
    dry_run: bool,
    now: float | None = None,
) -> dict:
    try:
        with cache_lease(root, exclusive=True, nonblocking=True):
            return _collect_idle(
                root, soft_max_bytes=soft_max_bytes, target_max_bytes=target_max_bytes,
                max_age_seconds=max_age_seconds, min_free_bytes=min_free_bytes,
                dry_run=dry_run, now=now,
            )
    except BlockingIOError:
        return {
            "schema": "RealSaS.PlatformCacheGC.v2", "root": str(root),
            "dry_run": dry_run, "deleted": [], "deleted_count": 0,
            "safe_to_execute": False, "status": "LIVE_CACHE_LEASE_HELD",
        }


def _collect_idle(
    root: Path, *, soft_max_bytes: int, target_max_bytes: int,
    max_age_seconds: int, min_free_bytes: int, dry_run: bool,
    now: float | None,
) -> dict:
    if target_max_bytes > soft_max_bytes:
        raise ValueError("GC_TARGET_EXCEEDS_SOFT_MAX")
    if min(soft_max_bytes, target_max_bytes, min_free_bytes) < 0:
        raise ValueError("GC_LIMIT_NEGATIVE")

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    entries, namespace_bytes = scan_entries(root)
    before_bytes = _tree_size(root)
    free_before = shutil.disk_usage(root).free
    deleted: list[dict] = []
    retained = {entry.path: entry for entry in entries}

    venv_ages = [max(0.0, now - entry.mtime) for entry in entries if entry.kind == "venv"]

    def evict(entry: Entry, reason: str) -> bool:
        if entry.path not in retained:
            return False
        handle = None
        try:
            if entry.lock_path is not None:
                handle = entry.lock_path.open("a+b")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not dry_run:
                _remove_tree(entry.path)
                if entry.path.exists():
                    raise OSError(f"CACHE_ENTRY_REMOVAL_INCOMPLETE:{entry.path}")
            deleted.append({
                "path": str(entry.path), "kind": entry.kind,
                "size_bytes": entry.size, "reason": reason,
            })
            retained.pop(entry.path, None)
            return True
        except BlockingIOError:
            return False
        finally:
            if handle is not None:
                handle.close()

    # Incomplete bootstrap/download state is never reusable.
    for entry in sorted(entries, key=lambda item: item.mtime):
        if entry.kind == "temporary":
            evict(entry, "TEMPORARY_OR_INCOMPLETE")

    # Venv stamps are touched on every successful cache hit, therefore age is a
    # real recency signal. Download/build namespace mtimes are not reliable LRU.
    for entry in sorted(retained.values(), key=lambda item: item.mtime):
        if entry.kind == "venv" and max_age_seconds >= 0 and now - entry.mtime > max_age_seconds:
            evict(entry, "VENV_MAX_AGE_EXPIRED")

    reclaimed = sum(row["size_bytes"] for row in deleted)
    current_bytes = max(0, before_bytes - reclaimed)
    free_estimate = free_before + reclaimed

    if current_bytes > soft_max_bytes or free_estimate < min_free_bytes:
        # First discard least-recently-used complete venvs.
        pressure_candidates = sorted(
            (entry for entry in retained.values() if entry.kind == "venv"),
            key=lambda item: item.mtime,
        )
        # Then drop cheap/reconstructable cache namespaces atomically. Go module
        # sources are last because redownloading them is more expensive.
        for namespace in ("go-build", "pip", "go-mod"):
            pressure_candidates.extend(
                entry for entry in retained.values()
                if entry.kind == f"namespace:{namespace}"
            )

        for entry in pressure_candidates:
            if current_bytes <= target_max_bytes and free_estimate >= min_free_bytes:
                break
            if evict(entry, "LRU_OR_DISK_PRESSURE"):
                current_bytes = max(0, current_bytes - entry.size)
                free_estimate += entry.size

    after_bytes_estimate = max(0, before_bytes - sum(row["size_bytes"] for row in deleted))
    free_after_estimate = free_before + sum(row["size_bytes"] for row in deleted)
    safe = after_bytes_estimate <= soft_max_bytes and free_after_estimate >= min_free_bytes
    actual_after = _tree_size(root) if not dry_run else None
    if not dry_run:
        safe = actual_after <= soft_max_bytes and shutil.disk_usage(root).free >= min_free_bytes

    return {
        "schema": "RealSaS.PlatformCacheGC.v2",
        "root": str(root),
        "dry_run": dry_run,
        "entry_count_before": len(entries),
        "venv_count_before": sum(entry.kind == "venv" for entry in entries),
        "venv_age_seconds_oldest": max(venv_ages) if venv_ages else None,
        "venv_age_seconds_newest": min(venv_ages) if venv_ages else None,
        "namespace_bytes_before": namespace_bytes,
        "before_bytes": before_bytes,
        "after_bytes_estimate": after_bytes_estimate,
        "after_bytes_actual": actual_after,
        "reclaimed_bytes": before_bytes - after_bytes_estimate,
        "free_bytes_before": free_before,
        "free_bytes_after_estimate": free_after_estimate,
        "soft_max_bytes": soft_max_bytes,
        "target_max_bytes": target_max_bytes,
        "min_free_bytes": min_free_bytes,
        "max_age_seconds": max_age_seconds,
        "deleted": deleted,
        "deleted_count": len(deleted),
        "safe_to_execute": safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--soft-max-gib", type=float, required=True)
    parser.add_argument("--target-max-gib", type=float, required=True)
    parser.add_argument("--max-age-days", type=float, default=30)
    parser.add_argument("--min-free-gib", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    gib = 1024 ** 3
    report = collect(
        args.root,
        soft_max_bytes=int(args.soft_max_gib * gib),
        target_max_bytes=int(args.target_max_gib * gib),
        max_age_seconds=int(args.max_age_days * 86400),
        min_free_bytes=int(args.min_free_gib * gib),
        dry_run=args.dry_run,
    )
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["safe_to_execute"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
