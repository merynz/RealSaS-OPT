"""Enumerate a Quaternius official Drive publisher tree without downloading bytes.

This tool exists to keep publisher inventory complete even when public per-file
transport is rate-limited. It emits only immutable listing evidence; it does not
acquire source bytes and cannot mint corpus/training authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path, PurePosixPath

from tools.corpus import post139_source_inventory_v1 as source_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    registry = source_inventory.load_registry(Path(args.registry))
    row = source_inventory.source_row(registry, args.source_id)
    if row.get("acquisition", {}).get("kind") != "QUATERNIUS_OFFICIAL_PAGE_DRIVE":
        raise SystemExit(f"POST139_PUBLISHER_CENSUS_SOURCE_KIND:{args.source_id}")

    folder_id, page_url = source_inventory.quaternius_drive_folder(row)
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("POST139_GDOWN_REQUIRED") from exc

    scratch = Path(args.out).resolve().parent / (Path(args.out).stem + "-scratch")
    listing = gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{folder_id}",
        output=str(scratch),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
    )
    if not listing:
        raise SystemExit(f"POST139_PUBLISHER_CENSUS_LIST_EMPTY:{args.source_id}")

    rows = []
    counts: Counter[str] = Counter()
    for item in listing:
        path = PurePosixPath(item.path).as_posix()
        extension = Path(path).suffix.lower() or "<none>"
        counts[extension] += 1
        rows.append({
            "file_id": item.id,
            "path": path,
            "extension": extension,
            "byte_state": "PUBLISHER_LISTED_NOT_ACQUIRED",
        })
    rows.sort(key=lambda value: value["path"])
    projection = [
        {"file_id": value["file_id"], "path": value["path"], "extension": value["extension"]}
        for value in rows
    ]
    payload = {
        "schema": source_inventory.SCHEMA_PUBLISHER_TREE,
        "source_id": args.source_id,
        "transport": "QUATERNIUS_OFFICIAL_PUBLIC_DRIVE_LISTING_ONLY",
        "publisher_page_raw_url": page_url,
        "resolved_drive_folder_id": folder_id,
        "file_count": len(rows),
        "extension_counts": dict(sorted(counts.items())),
        "acquired_file_count": 0,
        "files": rows,
        "tree_digest": source_inventory.digest(projection),
    }
    source_inventory.write_json(Path(args.out).resolve(), payload)
    print(
        "POST139_PUBLISHER_TREE_CENSUS_PASS",
        args.source_id,
        len(rows),
        payload["tree_digest"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
