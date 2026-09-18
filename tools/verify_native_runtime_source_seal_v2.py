from __future__ import annotations

"""Verify the sealed native runtime source lineage without mutating authority.

This is the reusable form of the historical Native runtime source gate. It verifies
the immutable base seal and both append-only Runtime-v4 source extensions against the
exact repository bytes.
"""

import hashlib
import json
from pathlib import Path
import subprocess


BASE = Path("canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json")
EXT1 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V1_20260917.json")
EXT2 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V2_20260917.json")


def _blob_sha1(payload: bytes) -> str:
    return subprocess.check_output(
        ["git", "hash-object", "--stdin"],
        input=payload,
        text=False,
    ).decode("ascii").strip()


def verify() -> dict:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    ext1 = json.loads(EXT1.read_text(encoding="utf-8"))
    ext2 = json.loads(EXT2.read_text(encoding="utf-8"))

    if ext1["schema"] != "realsas.compiler_runtime_source_extension_seal.v1":
        raise RuntimeError("NATIVE_SOURCE_EXT1_SCHEMA_DRIFT")
    if ext1["base_seal"]["path"] != str(BASE):
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_PATH_DRIFT")
    if _blob_sha1(BASE.read_bytes()) != ext1["base_seal"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_BLOB_DRIFT")
    if ext1["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_MUTATION_CLAIM")
    if ext1["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT1_ROLE_DRIFT")

    if ext2["schema"] != "realsas.compiler_runtime_source_extension_seal.v2":
        raise RuntimeError("NATIVE_SOURCE_EXT2_SCHEMA_DRIFT")
    if ext2["prior_extension"]["path"] != str(EXT1):
        raise RuntimeError("NATIVE_SOURCE_EXT2_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT1.read_bytes()) != ext2["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT2_PRIOR_BLOB_DRIFT")
    if ext2["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT2_BASE_MUTATION_CLAIM")
    if ext2["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT2_PRIOR_MUTATION_CLAIM")
    if ext2["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT2_ROLE_DRIFT")

    expected = {
        row["path"]: {
            "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"],
            "git_blob_sha1": row["git_blob_sha1"],
            "source": "historical_base",
        }
        for row in base["files"]
    }

    def apply_extension(ext: dict, label: str) -> None:
        for row in ext["replacements"]:
            path = row["path"]
            if path not in expected:
                raise RuntimeError(f"NATIVE_SOURCE_UNKNOWN_REPLACEMENT:{label}:{path}")
            expected[path] = {
                "size_bytes": int(row["size_bytes"]),
                "git_blob_sha1": row["git_blob_sha1"],
                "source": f"{label}_replacement",
            }
        for row in ext["additions"]:
            path = row["path"]
            if path in expected:
                raise RuntimeError(f"NATIVE_SOURCE_DUPLICATE_ADDITION:{label}:{path}")
            expected[path] = {
                "size_bytes": int(row["size_bytes"]),
                "git_blob_sha1": row["git_blob_sha1"],
                "source": f"{label}_addition",
            }

    apply_extension(ext1, "extension_v1")
    apply_extension(ext2, "extension_v2")

    verified = []
    for path_text, identity in sorted(expected.items()):
        path = Path(path_text)
        payload = path.read_bytes()
        if len(payload) != identity["size_bytes"]:
            raise RuntimeError(
                f"NATIVE_SOURCE_SIZE_DRIFT:{path}:{len(payload)}:{identity['size_bytes']}"
            )
        actual_blob = _blob_sha1(payload)
        if actual_blob != identity["git_blob_sha1"]:
            raise RuntimeError(
                f"NATIVE_SOURCE_BLOB_DRIFT:{path}:{actual_blob}:{identity['git_blob_sha1']}"
            )
        if identity["source"] == "historical_base":
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != identity["sha256"]:
                raise RuntimeError(
                    f"NATIVE_SOURCE_SHA256_DRIFT:{path}:{actual_sha}:{identity['sha256']}"
                )
        verified.append(path_text)

    return {
        "status": "PASS__NATIVE_RUNTIME_SOURCE_SEAL_CHAIN",
        "file_count": len(verified),
        "base_file_count": len(base["files"]),
        "extension_v1_change_count": len(ext1["replacements"]) + len(ext1["additions"]),
        "extension_v2_change_count": len(ext2["replacements"]) + len(ext2["additions"]),
        "verified_paths": verified,
    }


if __name__ == "__main__":
    result = verify()
    print(
        "NATIVE_SOURCE_SEAL_PASS "
        f"files={result['file_count']} "
        f"base={result['base_file_count']} "
        f"ext1={result['extension_v1_change_count']} "
        f"ext2={result['extension_v2_change_count']}"
    )
