from __future__ import annotations

"""Read-only exact Git-blob audit of append-only native runtime source seals.

It diagnoses historical/working tree drift without granting source authority,
writing a seal, or mutating immutable prior extensions.
"""
import json
from pathlib import Path
import subprocess

from tools import verify_native_runtime_source_seal_v2 as seal


def audit() -> dict:
    base = seal._load_json(seal.BASE)
    expected: dict[str, dict] = {
        str(row["path"]): {
            "git_blob_sha1": str(row["git_blob_sha1"]),
            "size_bytes": int(row["size_bytes"]),
            "last_seal": "base",
        }
        for row in base["files"]
    }
    for version, path in enumerate(seal.EXTENSION_PATHS, start=1):
        ext = seal._load_json(path)
        for label in ("replacements", "additions"):
            for row in ext.get(label, []):
                expected[str(row["path"])] = {
                    "git_blob_sha1": str(row["git_blob_sha1"]),
                    "size_bytes": int(row["size_bytes"]),
                    "last_seal": f"v{version}",
                }
    tracked = sorted(row for row in subprocess.check_output(
        ["git", "ls-files", "runtime/realsas_cpp"], text=True,
    ).splitlines() if row)
    actual = {}
    for name in tracked:
        path = Path(name)
        actual[name] = {
            "git_blob_sha1": subprocess.check_output(
                ["git", "hash-object", "--", name], text=True,
            ).strip(),
            "size_bytes": path.stat().st_size,
        }
    unsealed = [{"path": path, **actual[path]} for path in tracked if path not in expected]
    obsolete = sorted(set(expected) - set(tracked))
    changed = [
        {"path": path, "previous": expected[path], "current": actual[path]}
        for path in tracked if path in expected and (
            expected[path]["git_blob_sha1"] != actual[path]["git_blob_sha1"]
            or expected[path]["size_bytes"] != actual[path]["size_bytes"]
        )
    ]
    return {
        "schema": "RealSaS.NativeRuntimeSealReadOnlyGapAudit.v1",
        "tracked_count": len(tracked),
        "expected_sealed_count": len(expected),
        "unsealed_additions": unsealed,
        "changed_sealed_paths": changed,
        "previously_sealed_missing_paths": obsolete,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), sort_keys=True, indent=2))
