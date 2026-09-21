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
EXT3 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V3_20260918.json")
EXT4 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V4_20260918.json")
EXT5 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V5_20260921.json")
EXT6 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V6_20260921.json")
EXT7 = Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V7_20260921.json")


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
    ext3 = json.loads(EXT3.read_text(encoding="utf-8"))
    ext4 = json.loads(EXT4.read_text(encoding="utf-8"))
    ext5 = json.loads(EXT5.read_text(encoding="utf-8"))
    ext6 = json.loads(EXT6.read_text(encoding="utf-8"))
    ext7 = json.loads(EXT7.read_text(encoding="utf-8"))

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

    if ext3["schema"] != "realsas.compiler_runtime_source_extension_seal.v3":
        raise RuntimeError("NATIVE_SOURCE_EXT3_SCHEMA_DRIFT")
    if ext3["prior_extension"]["path"] != str(EXT2):
        raise RuntimeError("NATIVE_SOURCE_EXT3_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT2.read_bytes()) != ext3["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT3_PRIOR_BLOB_DRIFT")
    if ext3["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT3_BASE_MUTATION_CLAIM")
    if ext3["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT3_PRIOR_MUTATION_CLAIM")
    if ext3["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT3_ROLE_DRIFT")
    if ext3["authority"].get("subtree_closure") != "ALL_REPOSITORY_BLOBS_UNDER_RUNTIME_REALSAS_CPP_AT_SEAL_TIME":
        raise RuntimeError("NATIVE_SOURCE_EXT3_SUBTREE_CLOSURE_DRIFT")

    if ext4["schema"] != "realsas.compiler_runtime_source_extension_seal.v4":
        raise RuntimeError("NATIVE_SOURCE_EXT4_SCHEMA_DRIFT")
    if ext4["prior_extension"]["path"] != str(EXT3):
        raise RuntimeError("NATIVE_SOURCE_EXT4_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT3.read_bytes()) != ext4["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT4_PRIOR_BLOB_DRIFT")
    if ext4["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT4_BASE_MUTATION_CLAIM")
    if ext4["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT4_PRIOR_MUTATION_CLAIM")
    if ext4["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT4_ROLE_DRIFT")

    if ext5["schema"] != "realsas.compiler_runtime_source_extension_seal.v5":
        raise RuntimeError("NATIVE_SOURCE_EXT5_SCHEMA_DRIFT")
    if ext5["prior_extension"]["path"] != str(EXT4):
        raise RuntimeError("NATIVE_SOURCE_EXT5_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT4.read_bytes()) != ext5["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT5_PRIOR_BLOB_DRIFT")
    if ext5["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT5_BASE_MUTATION_CLAIM")
    if ext5["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT5_PRIOR_MUTATION_CLAIM")
    if ext5["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT5_ROLE_DRIFT")
    if ext5["authority"].get("subtree_closure") != "ALL_REPOSITORY_BLOBS_UNDER_RUNTIME_REALSAS_CPP_AT_SEAL_TIME":
        raise RuntimeError("NATIVE_SOURCE_EXT5_SUBTREE_CLOSURE_DRIFT")

    if ext6["schema"] != "realsas.compiler_runtime_source_extension_seal.v6":
        raise RuntimeError("NATIVE_SOURCE_EXT6_SCHEMA_DRIFT")
    if ext6["prior_extension"]["path"] != str(EXT5):
        raise RuntimeError("NATIVE_SOURCE_EXT6_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT5.read_bytes()) != ext6["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT6_PRIOR_BLOB_DRIFT")
    if ext6["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT6_BASE_MUTATION_CLAIM")
    if ext6["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT6_PRIOR_MUTATION_CLAIM")
    if ext6["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT6_ROLE_DRIFT")
    if ext6["authority"].get("subtree_closure") != "ALL_REPOSITORY_BLOBS_UNDER_RUNTIME_REALSAS_CPP_AT_SEAL_TIME":
        raise RuntimeError("NATIVE_SOURCE_EXT6_SUBTREE_CLOSURE_DRIFT")

    if ext7["schema"] != "realsas.compiler_runtime_source_extension_seal.v7":
        raise RuntimeError("NATIVE_SOURCE_EXT7_SCHEMA_DRIFT")
    if ext7["prior_extension"]["path"] != str(EXT6):
        raise RuntimeError("NATIVE_SOURCE_EXT7_PRIOR_PATH_DRIFT")
    if _blob_sha1(EXT6.read_bytes()) != ext7["prior_extension"]["git_blob_sha1"]:
        raise RuntimeError("NATIVE_SOURCE_EXT7_PRIOR_BLOB_DRIFT")
    if ext7["authority"]["historical_base_seal_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT7_BASE_MUTATION_CLAIM")
    if ext7["authority"]["prior_extension_mutated"] is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT7_PRIOR_MUTATION_CLAIM")
    if ext7["authority"]["runtime_role"] != "subordinate_deployment_consumer":
        raise RuntimeError("NATIVE_SOURCE_EXT7_ROLE_DRIFT")
    if ext7["authority"].get("subtree_closure") != "ALL_REPOSITORY_BLOBS_UNDER_RUNTIME_REALSAS_CPP_AT_SEAL_TIME":
        raise RuntimeError("NATIVE_SOURCE_EXT7_SUBTREE_CLOSURE_DRIFT")

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
    apply_extension(ext3, "extension_v3")
    apply_extension(ext4, "extension_v4")
    apply_extension(ext5, "extension_v5")
    apply_extension(ext6, "extension_v6")
    apply_extension(ext7, "extension_v7")

    tracked = {
        row.strip()
        for row in subprocess.check_output(
            ["git", "ls-files", "runtime/realsas_cpp"],
            text=True,
        ).splitlines()
        if row.strip()
    }
    if tracked != set(expected):
        missing_from_seal = sorted(tracked - set(expected))
        stale_in_seal = sorted(set(expected) - tracked)
        raise RuntimeError(
            "NATIVE_SOURCE_SUBTREE_SET_DRIFT:"
            + json.dumps(
                {
                    "unsealed_tracked_paths": missing_from_seal,
                    "sealed_paths_missing_from_repo": stale_in_seal,
                },
                sort_keys=True,
            )
        )
    closure = dict(ext7.get("closure") or {})
    if int(closure.get("runtime_realsas_cpp_blob_count", -1)) != len(tracked):
        raise RuntimeError("NATIVE_SOURCE_EXT7_CLOSURE_COUNT_DRIFT")
    if int(closure.get("resulting_sealed_blob_count", -1)) != len(expected):
        raise RuntimeError("NATIVE_SOURCE_EXT7_SEALED_COUNT_DRIFT")
    if int(closure.get("unsealed_repository_blob_count_under_subtree", -1)) != 0:
        raise RuntimeError("NATIVE_SOURCE_EXT7_UNSEALED_COUNT_NONZERO")

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
        "extension_v3_change_count": len(ext3["replacements"]) + len(ext3["additions"]),
        "extension_v4_change_count": len(ext4["replacements"]) + len(ext4["additions"]),
        "extension_v5_change_count": len(ext5["replacements"]) + len(ext5["additions"]),
        "extension_v6_change_count": len(ext6["replacements"]) + len(ext6["additions"]),
        "extension_v7_change_count": len(ext7["replacements"]) + len(ext7["additions"]),
        "subtree_blob_count": len(tracked),
        "verified_paths": verified,
    }


if __name__ == "__main__":
    result = verify()
    print(
        "NATIVE_SOURCE_SEAL_PASS "
        f"files={result['file_count']} "
        f"base={result['base_file_count']} "
        f"ext1={result['extension_v1_change_count']} "
        f"ext2={result['extension_v2_change_count']} "
        f"ext3={result['extension_v3_change_count']} "
        f"ext4={result['extension_v4_change_count']} "
        f"ext5={result['extension_v5_change_count']} "
        f"ext6={result['extension_v6_change_count']} "
        f"ext7={result['extension_v7_change_count']} "
        f"subtree={result['subtree_blob_count']}"
    )
