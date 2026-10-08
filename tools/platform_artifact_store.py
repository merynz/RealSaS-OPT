#!/usr/bin/env python3
"""Hash-verified external evidence hydration into a bounded local content-addressed store.

Hydration only makes external bytes locally available. It never mints scientific
qualification, product authority, or stage cache hits.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

INVENTORY_SCHEMA = "RealSaS.ExternalInputInventory.v1"
REPORT_SCHEMA = "RealSaS.PlatformArtifactHydration.v1"


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inventory(inventory: dict) -> list[dict]:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError("INPUT_INVENTORY_SCHEMA_INVALID")
    required = inventory.get("required_files")
    if not isinstance(required, list) or not required:
        raise ValueError("INPUT_INVENTORY_REQUIRED_FILES_INVALID")
    seen_names: set[str] = set()
    seen_hashes: set[str] = set()
    for row in required:
        name = row.get("name")
        digest = row.get("sha256")
        size = row.get("size_bytes")
        if not isinstance(name, str) or Path(name).name != name or name in seen_names:
            raise ValueError("INPUT_INVENTORY_NAMES_INVALID")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"INPUT_INVENTORY_SHA256_INVALID:{name}")
        if digest in seen_hashes:
            raise ValueError(f"INPUT_INVENTORY_DUPLICATE_CONTENT_ADDRESS:{name}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"INPUT_INVENTORY_SIZE_INVALID:{name}")
        source = row.get("source")
        if source is not None:
            if not isinstance(source, dict) or source.get("provider") != "google_drive":
                raise ValueError(f"INPUT_INVENTORY_SOURCE_INVALID:{name}")
            file_id = source.get("file_id")
            if not isinstance(file_id, str) or not file_id.strip():
                raise ValueError(f"INPUT_INVENTORY_SOURCE_FILE_ID_INVALID:{name}")
        seen_names.add(name)
        seen_hashes.add(digest)
    return required


def cas_path(cas_root: Path, digest: str) -> Path:
    return cas_root / "sha256" / digest[:2] / digest[2:]


def verify_exact(path: Path, row: dict) -> tuple[bool, str]:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return False, "MISSING"
    if size != row["size_bytes"]:
        return False, f"SIZE_MISMATCH:{size}"
    digest = sha256_file(path)
    if digest != row["sha256"]:
        return False, f"SHA256_MISMATCH:{digest}"
    return True, "EXACT"


@contextlib.contextmanager
def digest_lock(cas_root: Path, digest: str):
    lock_dir = cas_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{digest}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def fetch_google_drive_with_rclone(*, file_id: str, destination: Path, remote: str) -> None:
    remote_name = remote.rstrip(":")
    if not remote_name or ":" in remote_name:
        raise ValueError("RCLONE_REMOTE_NAME_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rclone", "backend", "copyid", f"{remote_name}:", file_id, str(destination)],
        check=True,
    )


def hydrate_row(
    row: dict,
    *,
    cas_root: Path,
    fetcher: Callable[..., None],
    rclone_remote: str,
    allow_remote: bool = True,
) -> dict:
    destination = cas_path(cas_root, row["sha256"])
    started = time.monotonic()
    with digest_lock(cas_root, row["sha256"]):
        exact, reason = verify_exact(destination, row)
        if exact:
            os.utime(destination, None)
            return {
                "name": row["name"], "sha256": row["sha256"], "size_bytes": row["size_bytes"],
                "status": "CAS_HIT", "path": str(destination),
                "wall_seconds": round(time.monotonic() - started, 6),
            }
        if destination.exists():
            destination.unlink()
        if not allow_remote:
            return {
                "name": row["name"], "sha256": row["sha256"], "size_bytes": row["size_bytes"],
                "status": "MISS_REMOTE_DISABLED", "path": str(destination), "reason": reason,
                "wall_seconds": round(time.monotonic() - started, 6),
            }
        source = row.get("source")
        if not source:
            return {
                "name": row["name"], "sha256": row["sha256"], "size_bytes": row["size_bytes"],
                "status": "MISS_NO_SOURCE_LOCATOR", "path": str(destination),
                "wall_seconds": round(time.monotonic() - started, 6),
            }
        if source.get("provider") != "google_drive":
            raise ValueError(f"UNSUPPORTED_ARTIFACT_PROVIDER:{source.get('provider')}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cas_root / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{row['sha256']}.", suffix=".part", dir=tmp_dir)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            fetcher(file_id=source["file_id"], destination=tmp, remote=rclone_remote)
            exact, reason = verify_exact(tmp, row)
            if not exact:
                raise RuntimeError(f"HYDRATED_BYTES_INVALID:{row['name']}:{reason}")
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
        return {
            "name": row["name"], "sha256": row["sha256"], "size_bytes": row["size_bytes"],
            "status": "HYDRATED", "path": str(destination), "provider": source["provider"],
            "provider_file_id": source["file_id"],
            "wall_seconds": round(time.monotonic() - started, 6),
        }


def hydrate_inventory(
    inventory: dict,
    *,
    cas_root: Path,
    fetcher: Callable[..., None] = fetch_google_drive_with_rclone,
    rclone_remote: str = "realsas-drive",
    allow_remote: bool = True,
) -> dict:
    required = validate_inventory(inventory)
    cas_root = cas_root.expanduser().resolve()
    cas_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = [
        hydrate_row(row, cas_root=cas_root, fetcher=fetcher,
                    rclone_remote=rclone_remote, allow_remote=allow_remote)
        for row in required
    ]
    exact = all(row["status"] in {"CAS_HIT", "HYDRATED"} for row in rows)
    return {
        "schema": REPORT_SCHEMA,
        "subject": inventory.get("subject"),
        "authority_class": inventory.get("authority_class"),
        "cas_root": str(cas_root),
        "files": rows,
        "required_file_count": len(rows),
        "verified_file_count": sum(row["status"] in {"CAS_HIT", "HYDRATED"} for row in rows),
        "all_input_bytes_verified": exact,
        "scientific_pass_claimed": False,
        "qualification_minted": False,
        "stage_cache_hit_minted": False,
        "wall_seconds": round(time.monotonic() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, default=Path("~/.cache/realsas/artifacts"))
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_DRIVE_REMOTE", "realsas-drive"))
    parser.add_argument("--offline", action="store_true", help="Verify CAS only; never fetch remote bytes")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    report = hydrate_inventory(inventory, cas_root=args.cas_root,
                               rclone_remote=args.rclone_remote, allow_remote=not args.offline)
    payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if not report["all_input_bytes_verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
