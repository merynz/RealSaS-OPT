#!/usr/bin/env python3
from __future__ import annotations

"""Discover and launch the canonical Full139 family sweep on a mounted Drive.

The launcher never trusts a filename alone. It recursively discovers historical
readiness/source-cache manifests, validates their schema/status/subject sets, and
requires exactly one compatible pair. Historical split/training flags remain
inventory-only because the invoked V3 sweep ignores them by contract.

Typical Colab use after Drive mount:
  python tools/corpus/launch_full139_truth_family_sweep_drive_v1.py \
    --drive-root /content/drive/MyDrive --execute
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

READINESS_SCHEMA = "RealSaS.Truth139V32PretrainTruthLedger.v2"
READINESS_STATUS = "PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY"
CACHE_SCHEMA = "RealSaS.Truth139FullSourceCache.v1"
CACHE_STATUS = "PASS_TRUTH139_FULL139_SOURCE_BYTES_CACHED__TRUTH_NOT_PERFORMED"
SUBJECT_COUNT = 139


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def readiness_subjects(value: dict) -> set[str] | None:
    if (
        value.get("schema") != READINESS_SCHEMA
        or value.get("status") != READINESS_STATUS
        or value.get("subject_count") != SUBJECT_COUNT
    ):
        return None
    rows = value.get("subjects")
    if not isinstance(rows, dict) or len(rows) != SUBJECT_COUNT:
        return None
    required = {"geometry", "geometry_court", "rgb", "rgb_court", "stage48", "tessa_teacher", "tessa_court"}
    if any(not required.issubset(dict(row or {})) for row in rows.values()):
        return None
    return set(map(str, rows))


def cache_subjects(value: dict) -> set[str] | None:
    if (
        value.get("schema") != CACHE_SCHEMA
        or value.get("status") != CACHE_STATUS
        or value.get("subject_count") != SUBJECT_COUNT
        or value.get("source_failures") not in ({}, None)
    ):
        return None
    rows = value.get("subjects")
    if not isinstance(rows, list) or len(rows) != SUBJECT_COUNT:
        return None
    result = set()
    for row in rows:
        row = dict(row or {})
        subject = str(row.get("subject_id") or "")
        if not subject or row.get("valid") is not True:
            return None
        if Path(str(row.get("corpus_relative_path") or "")).suffix.lower() != ".gltf":
            return None
        result.add(subject)
    return result if len(result) == SUBJECT_COUNT else None


def discover(root: Path) -> tuple[list[tuple[Path, set[str]]], list[tuple[Path, set[str]]]]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(str(root))
    readiness = []
    caches = []
    # Restrict filename families to keep discovery cheap; schema/status still decide authority.
    for path in root.rglob("V32_TRAINING_READINESS_RECEIPT.json"):
        value = load_json(path)
        subjects = readiness_subjects(value or {})
        if subjects is not None:
            readiness.append((path.resolve(), subjects))
    for path in root.rglob("FULL_SOURCE_CACHE_MANIFEST.json"):
        value = load_json(path)
        subjects = cache_subjects(value or {})
        if subjects is not None:
            caches.append((path.resolve(), subjects))
    return readiness, caches


def compatible_pairs(readiness, caches):
    result = []
    for rp, rs in readiness:
        for cp, cs in caches:
            if rs == cs:
                result.append((rp, cp))
    return result


def run(cmd: list[str], cwd: Path) -> None:
    print(json.dumps({"event": "EXEC", "argv": cmd}, sort_keys=True), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--subject-id")
    parser.add_argument("--max-subjects", type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    readiness, caches = discover(args.drive_root)
    pairs = compatible_pairs(readiness, caches)
    discovery = {
        "schema": "RealSaS.Full139DriveSweepDiscovery.v1",
        "drive_root": str(args.drive_root.resolve()),
        "readiness_candidate_count": len(readiness),
        "source_cache_candidate_count": len(caches),
        "compatible_pair_count": len(pairs),
        "training_authorized": False,
    }
    print(json.dumps(discovery, sort_keys=True), flush=True)
    if len(pairs) != 1:
        detail = {
            "readiness": [str(path) for path, _ in readiness],
            "source_caches": [str(path) for path, _ in caches],
            "compatible_pairs": [[str(a), str(b)] for a, b in pairs],
        }
        raise RuntimeError("FULL139_DRIVE_DISCOVERY_NOT_UNIQUE:" + json.dumps(detail, sort_keys=True))
    readiness_path, cache_path = pairs[0]

    output_root = (
        args.output_root.resolve()
        if args.output_root
        else readiness_path.parent / "CORPUS400_FULL139_TRUTH_FAMILY_SWEEP_V1"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    selection = {
        **discovery,
        "selected_readiness": str(readiness_path),
        "selected_source_cache": str(cache_path),
        "output_root": str(output_root),
        "historical_split_inherited": False,
        "historical_training_authority_inherited": False,
    }
    (output_root / "FULL139_DRIVE_SWEEP_SELECTION_V1.json").write_text(
        json.dumps(selection, sort_keys=True, indent=2) + "\n"
    )

    base = [
        sys.executable,
        str((repo_root / "tools/corpus/run_full139_truth_family_sweep_v3.py").resolve()),
        "--historical-readiness", str(readiness_path),
        "--source-cache-manifest", str(cache_path),
        "--output-root", str(output_root),
    ]
    run(base + ["--plan-only"], repo_root)
    if not args.execute:
        print(json.dumps({"status": "PASS_FULL139_DRIVE_PLAN__EXECUTION_NOT_REQUESTED", "output_root": str(output_root), "training_authorized": False}, sort_keys=True))
        return

    cmd = list(base)
    if args.subject_id:
        cmd += ["--subject-id", args.subject_id]
    if args.max_subjects is not None:
        cmd += ["--max-subjects", str(args.max_subjects)]
    if args.continue_on_error:
        cmd += ["--continue-on-error"]
    run(cmd, repo_root)
    receipt = output_root / "FULL139_TRUTH_FAMILY_SWEEP_RECEIPT_V1.json"
    if not receipt.is_file():
        raise RuntimeError("FULL139_DRIVE_SWEEP_RECEIPT_MISSING")
    value = json.loads(receipt.read_text())
    print(json.dumps({"status": value.get("status"), "subject_count": value.get("subject_count"), "training_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
