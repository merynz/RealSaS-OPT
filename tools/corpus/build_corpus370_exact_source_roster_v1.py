#!/usr/bin/env python3
from __future__ import annotations

"""Seal the exact 370 raw-source roster from a verified source bundle manifest.

The roster is identity/provenance only. It does not promote source presence to a
truth-family PASS and never grants training authority.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path

SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def build(bundle_manifest_path: Path, archive_sha256: str, expected_subject_count: int = 370) -> dict:
    bundle_manifest_path = bundle_manifest_path.resolve()
    manifest = json.loads(bundle_manifest_path.read_text())
    subjects = list(manifest.get("subjects") or ())
    require(len(subjects) == expected_subject_count, "CORPUS370_EXACT_ROSTER_COUNT")
    require(manifest.get("training_authorized") is False, "CORPUS370_SOURCE_BUNDLE_PREAUTHORIZED")
    require(len(archive_sha256) == 64, "CORPUS370_ARCHIVE_SHA256")
    seen_ids = set()
    seen_rows = []
    for raw in subjects:
        subject_id = str(raw.get("subject_id") or "")
        source_sha = str(raw.get("source_sha256") or "")
        require(subject_id and subject_id not in seen_ids, "CORPUS370_EXACT_ROSTER_SUBJECT_ID")
        require(len(source_sha) == 64, "CORPUS370_EXACT_ROSTER_SOURCE_SHA")
        seen_ids.add(subject_id)
        seen_rows.append(
            {
                "subject_id": subject_id,
                "origin": raw.get("origin"),
                "source_id": raw.get("source_id"),
                "source_format": raw.get("source_format"),
                "source_relative_path": raw.get("source_relative_path"),
                "source_sha256": source_sha,
                "training_authorized": False,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "EXACT_RAW_SOURCE_ROSTER__NO_FAMILY_OR_TRAINING_AUTHORITY",
        "subject_count": len(seen_rows),
        "origin_counts": dict(sorted(collections.Counter(row["origin"] for row in seen_rows).items())),
        "format_counts": dict(sorted(collections.Counter(row["source_format"] for row in seen_rows).items())),
        "source_bundle_manifest_sha256": sha256(bundle_manifest_path),
        "source_bundle_archive_sha256": archive_sha256,
        "subjects": sorted(seen_rows, key=lambda row: row["subject_id"]),
        "training_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-manifest", type=Path, required=True)
    parser.add_argument("--source-bundle-archive-sha256", required=True)
    parser.add_argument("--expected-subject-count", type=int, default=370)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source_bundle_manifest, args.source_bundle_archive_sha256, args.expected_subject_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("schema", "subject_count", "origin_counts", "format_counts", "source_bundle_manifest_sha256", "source_bundle_archive_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
