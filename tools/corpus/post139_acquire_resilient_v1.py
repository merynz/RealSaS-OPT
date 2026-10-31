"""Resilient acquisition wrapper for Post139 source recovery.

This wrapper intentionally reuses the canonical acquisition implementation from
``post139_source_inventory_v1`` while replacing only transient transport policy:

* Google Drive files resume into the same workspace;
* zero-byte partials are removed before retry;
* retry/backoff is bounded and explicit;
* already-complete workspaces are verified and reused;
* acquisition success is accepted only with an ``ACQUIRED_VERIFIED`` receipt.

It does not mint corpus/training authority.  Its output is consumed by the same
inventory -> Blender audit -> compile -> seal chain as a fresh acquisition.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tools.corpus import post139_source_inventory_v1 as source_inventory


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verified(out: Path) -> bool:
    receipt = out / "ACQUISITION_RECEIPT.json"
    raw = out / "raw"
    if not receipt.is_file() or not raw.is_dir():
        return False
    try:
        payload = _read_json(receipt)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ACQUIRED_VERIFIED" and any(p.is_file() for p in raw.rglob("*"))


def _resilient_gdrive_item(gdown: Any, item: Any, raw: Path, attempts: int = 10) -> Path:
    target = raw / item.path
    target.parent.mkdir(parents=True, exist_ok=True)
    delays = [0, 3, 8, 15, 30, 60, 90, 120, 180, 240]
    last_error: Exception | None = None
    for attempt in range(attempts):
        if target.is_file() and target.stat().st_size > 0:
            return target
        if target.exists() and target.stat().st_size == 0:
            target.unlink(missing_ok=True)
        if delays[min(attempt, len(delays) - 1)]:
            time.sleep(delays[min(attempt, len(delays) - 1)])
        try:
            # id= is less brittle than rebuilding a uc URL and still supports resume.
            result = gdown.download(
                id=item.id,
                output=str(target),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
            if result and target.is_file() and target.stat().st_size > 0:
                return target
        except Exception as exc:  # transport retry boundary
            last_error = exc
    raise RuntimeError(f"POST139_RESILIENT_GDRIVE_FILE_FAILED:{item.id}:{item.path}:{last_error}")


def acquire_once(registry: Path, source_id: str, out: Path, *, clean: bool) -> None:
    # Patch only the transport primitive; semantic/acquisition source resolution stays canonical.
    source_inventory.download_gdrive_item = _resilient_gdrive_item
    args = argparse.Namespace(
        registry=str(registry),
        source_id=source_id,
        out=str(out),
        clean=clean,
    )
    source_inventory.acquire(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--base-delay-seconds", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="Discard an already verified workspace and reacquire")
    args = parser.parse_args()

    registry = Path(args.registry).resolve()
    out = Path(args.out).resolve()
    if not args.force and _verified(out):
        print("POST139_RESILIENT_ACQUIRE_REUSE", args.source_id)
        return 0

    last_error: Exception | None = None
    for attempt in range(max(1, args.attempts)):
        if attempt:
            delay = args.base_delay_seconds * (2 ** (attempt - 1))
            print("POST139_RESILIENT_ACQUIRE_RETRY", args.source_id, attempt + 1, delay)
            time.sleep(delay)
        try:
            # First attempt starts clean to avoid stale evidence. Subsequent attempts preserve
            # successfully downloaded source bytes so Drive recovery is resumable.
            acquire_once(registry, args.source_id, out, clean=(attempt == 0))
            if not _verified(out):
                raise RuntimeError("POST139_RESILIENT_RECEIPT_NOT_VERIFIED")
            receipt = _read_json(out / "ACQUISITION_RECEIPT.json")
            print(
                "POST139_RESILIENT_ACQUIRE_PASS",
                args.source_id,
                receipt.get("acquisition_kind"),
                receipt.get("downloaded_file_count", "n/a"),
            )
            return 0
        except Exception as exc:
            last_error = exc
            print("POST139_RESILIENT_ACQUIRE_ATTEMPT_FAILED", args.source_id, attempt + 1, repr(exc))

    print("POST139_RESILIENT_ACQUIRE_FAILED", args.source_id, repr(last_error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
