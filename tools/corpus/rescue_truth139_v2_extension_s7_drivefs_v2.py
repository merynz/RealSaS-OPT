#!/usr/bin/env python3
"""DriveFS-safe launcher for the scoped Truth139 V2-extension S7 rescue.

The original rescue runner used ``shutil.copy2`` to preserve the failed 129/139
receipt before mutation. Google DriveFS in Colab can reject copy-stat metadata
operations even when a plain byte copy is valid. This launcher performs the
same provenance guard with an atomic byte-exact copy, then delegates to the
canonical rescue implementation unchanged.

This file is intentionally executable both as ``python -m tools.corpus...`` and
as ``python tools/corpus/...py``. The latter is how Colab rescue cells invoke it,
so the repository root is inserted into ``sys.path`` before package imports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.corpus.build_truth139_dense_graphics_gbuffer_v1 import sha256
from tools.corpus.rescue_truth139_v2_extension_s7_v1 import rescue


def _atomic_byte_copy(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    staged = destination.with_suffix(destination.suffix + ".partial")
    staged.write_bytes(data)
    staged.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--parent-subjects", required=True, type=Path)
    parser.add_argument("--rescue-subjects", required=True, type=Path)
    parser.add_argument("--expected-parent-receipt-sha256", required=True)
    parser.add_argument("--expected-parent-receipt-identity-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    batch_root = args.batch_root.resolve()
    receipt = batch_root / "FULL139_V2_EXTENSION_RECEIPT.json"
    backup = batch_root / "FULL139_V2_EXTENSION_RECEIPT_PRE_RESCUE_129OF139.json"

    print("[DRIVEFS-PREFLIGHT] batch_root =", batch_root, flush=True)
    if not receipt.is_file():
        raise FileNotFoundError("DRIVEFS_RESCUE_PARENT_RECEIPT_MISSING:" + str(receipt))

    current_sha = sha256(receipt)
    print("[DRIVEFS-PREFLIGHT] parent receipt sha256 =", current_sha, flush=True)
    if current_sha != args.expected_parent_receipt_sha256:
        raise ValueError(
            "DRIVEFS_RESCUE_PARENT_RECEIPT_SHA_MISMATCH:"
            + current_sha
            + "!="
            + args.expected_parent_receipt_sha256
        )

    parent = json.loads(receipt.read_text())
    current_identity = str(parent.get("receipt_identity_sha256") or "")
    print("[DRIVEFS-PREFLIGHT] parent receipt identity =", current_identity, flush=True)
    if current_identity != args.expected_parent_receipt_identity_sha256:
        raise ValueError(
            "DRIVEFS_RESCUE_PARENT_RECEIPT_IDENTITY_MISMATCH:"
            + current_identity
            + "!="
            + args.expected_parent_receipt_identity_sha256
        )

    if not backup.is_file():
        print("[DRIVEFS-PREFLIGHT] creating byte-exact backup without copy2 metadata", flush=True)
        _atomic_byte_copy(receipt, backup)
    backup_sha = sha256(backup)
    print("[DRIVEFS-PREFLIGHT] backup sha256 =", backup_sha, flush=True)
    if backup_sha != args.expected_parent_receipt_sha256:
        raise ValueError("DRIVEFS_RESCUE_BACKUP_SHA_MISMATCH")
    print("[DRIVEFS-PREFLIGHT PASS] receipt backup is byte-exact", flush=True)

    result = rescue(
        batch_root=batch_root,
        parent_subjects=args.parent_subjects,
        rescue_subjects=args.rescue_subjects,
        expected_parent_receipt_sha256=args.expected_parent_receipt_sha256,
        expected_parent_receipt_identity_sha256=args.expected_parent_receipt_identity_sha256,
        device=args.device,
    )
    print("\n[S7-RESCUE-FINAL]")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
