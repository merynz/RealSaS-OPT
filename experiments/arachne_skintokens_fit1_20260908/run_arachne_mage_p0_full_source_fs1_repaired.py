#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import build_arachne_mage_p0_full_source_fs1 as impl

EXPECTED_IMPLEMENTATION_GIT_BLOB = "4e5e54e893f145a1e57f76ee8a5751b38fceca0f"
EXPECTED_REPAIRED_BRIDGE_SHA256 = "3a2f4d586a14fda2f714d24e7f3e83b5c1a830f8ee41ac3f18921b32cf9384f8"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def main() -> None:
    implementation_path = Path(impl.__file__).resolve()
    got = git_blob_sha(implementation_path)
    if got != EXPECTED_IMPLEMENTATION_GIT_BLOB:
        raise RuntimeError(
            f"FS1_IMPLEMENTATION_GIT_BLOB_DRIFT:{got}:{EXPECTED_IMPLEMENTATION_GIT_BLOB}"
        )
    impl.EXPECTED_BRIDGE_SHA = EXPECTED_REPAIRED_BRIDGE_SHA256
    impl.main()


if __name__ == "__main__":
    main()
