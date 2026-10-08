#!/usr/bin/env python3
"""Bounded GC for persistent RealSaS self-hosted caches and artifact CAS roots."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

TEMP_SUFFIXES = (".part", ".tmp", ".incomplete")


@dataclass(frozen=True)
class Entry:
    path: Path
    size: int
    mtime: float
    temporary: bool


def _is_locked(path: Path) -> bool:
    lock = path.with_name(path.name + ".lock")
    if not lock.exists():
        return False
    try:
        with lock.open("a+b") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        return True
    return False


def scan(root: Path) -> list[Entry]:
    if not root.exists():
        return []
    rows: list[Entry] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or "/.locks/" in str(path):
            continue
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        rows.append(Entry(path=path, size=stat.st_size, mtime=stat.st_mtime,
                          temporary=path.name.endswith(TEMP_SUFFIXES)))
    return rows


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
    if target_max_bytes > soft_max_bytes:
        raise ValueError("GC_TARGET_EXCEEDS_SOFT_MAX")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    entries = scan(root)
    before_bytes = sum(entry.size for entry in entries)
    free_before = shutil.disk_usage(root).free
    deleted: list[dict] = []
    retained = {entry.path: entry for entry in entries}

    def evict(entry: Entry, reason: str) -> None:
        if entry.path not in retained or _is_locked(entry.path):
            return
        deleted.append({"path": str(entry.path), "size_bytes": entry.size, "reason": reason})
        if not dry_run:
            entry.path.unlink(missing_ok=True)
        retained.pop(entry.path, None)

    for entry in sorted(entries, key=lambda item: item.mtime):
        if entry.temporary:
            evict(entry, "TEMPORARY_OR_INCOMPLETE")

    for entry in sorted(retained.values(), key=lambda item: item.mtime):
        if max_age_seconds >= 0 and now - entry.mtime > max_age_seconds:
            evict(entry, "MAX_AGE_EXPIRED")

    current_bytes = sum(entry.size for entry in retained.values())
    free_estimate = free_before + sum(row["size_bytes"] for row in deleted)
    if current_bytes > soft_max_bytes or free_estimate < min_free_bytes:
        for entry in sorted(retained.values(), key=lambda item: item.mtime):
            if current_bytes <= target_max_bytes and free_estimate >= min_free_bytes:
                break
            before = len(deleted)
            evict(entry, "LRU_WATERMARK")
            if len(deleted) != before:
                current_bytes -= entry.size
                free_estimate += entry.size

    after_bytes = sum(entry.size for entry in retained.values())
    free_after_estimate = free_before + sum(row["size_bytes"] for row in deleted)
    safe = after_bytes <= soft_max_bytes and free_after_estimate >= min_free_bytes
    return {
        "schema": "RealSaS.PlatformCacheGC.v1",
        "root": str(root),
        "dry_run": dry_run,
        "before_bytes": before_bytes,
        "after_bytes_estimate": after_bytes,
        "reclaimed_bytes": before_bytes - after_bytes,
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
