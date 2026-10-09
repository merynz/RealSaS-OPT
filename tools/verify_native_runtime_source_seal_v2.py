from __future__ import annotations

"""Verify the append-only native runtime source lineage without mutating authority.

Historical seals are immutable provenance.  This verifier walks the complete
ordered extension chain, reconstructs the exact sealed runtime/realsas_cpp
subtree and compares it with Git.  New runtime consumers therefore have to be
sealed explicitly rather than becoming an untracked second authority.
"""

import hashlib
import json
from pathlib import Path
import subprocess


BASE = Path("canonical/COMPILER_RUNTIME_PROMOTION_SOURCE_SEAL_V1_20260903.json")
EXTENSION_PATHS = (
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V1_20260917.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V2_20260917.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V3_20260918.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V4_20260918.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V5_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V6_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V7_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V8_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V9_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V10_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V11_20260921.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V12_20260927.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V13_20260927.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V14_20260927.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V15_20260927.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V16_20261001.json"),
    Path("canonical/COMPILER_RUNTIME_SOURCE_EXTENSION_SEAL_V17_20261009.json"),
)
SUBTREE_CLOSURE = "ALL_REPOSITORY_BLOBS_UNDER_RUNTIME_REALSAS_CPP_AT_SEAL_TIME"
RUNTIME_ROLE = "subordinate_deployment_consumer"

# Compatibility aliases retained for scripts/tests that imported the old
# explicit constants.  The verifier itself is data-driven now.
for _index, _path in enumerate(EXTENSION_PATHS, start=1):
    globals()[f"EXT{_index}"] = _path


def _blob_sha1(payload: bytes) -> str:
    return subprocess.check_output(
        ["git", "hash-object", "--stdin"], input=payload, text=False
    ).decode("ascii").strip()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"NATIVE_SOURCE_SEAL_NOT_OBJECT:{path}")
    return value


def _verify_chain(base: dict, extensions: list[dict]) -> None:
    if len(extensions) != len(EXTENSION_PATHS) or not extensions:
        raise RuntimeError("NATIVE_SOURCE_EXTENSION_CHAIN_LENGTH_DRIFT")

    first = extensions[0]
    if first.get("schema") != "realsas.compiler_runtime_source_extension_seal.v1":
        raise RuntimeError("NATIVE_SOURCE_EXT1_SCHEMA_DRIFT")
    if first.get("base_seal", {}).get("path") != str(BASE):
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_PATH_DRIFT")
    if _blob_sha1(BASE.read_bytes()) != first.get("base_seal", {}).get("git_blob_sha1"):
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_BLOB_DRIFT")
    authority = dict(first.get("authority") or {})
    if authority.get("historical_base_seal_mutated") is not False:
        raise RuntimeError("NATIVE_SOURCE_EXT1_BASE_MUTATION_CLAIM")
    if authority.get("runtime_role") != RUNTIME_ROLE:
        raise RuntimeError("NATIVE_SOURCE_EXT1_ROLE_DRIFT")

    for version, (path, ext) in enumerate(
        zip(EXTENSION_PATHS[1:], extensions[1:]), start=2
    ):
        expected_schema = f"realsas.compiler_runtime_source_extension_seal.v{version}"
        if ext.get("schema") != expected_schema:
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_SCHEMA_DRIFT")
        previous_path = EXTENSION_PATHS[version - 2]
        prior = dict(ext.get("prior_extension") or {})
        if prior.get("path") != str(previous_path):
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_PRIOR_PATH_DRIFT")
        if _blob_sha1(previous_path.read_bytes()) != prior.get("git_blob_sha1"):
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_PRIOR_BLOB_DRIFT")
        authority = dict(ext.get("authority") or {})
        if authority.get("historical_base_seal_mutated") is not False:
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_BASE_MUTATION_CLAIM")
        if authority.get("prior_extension_mutated") is not False:
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_PRIOR_MUTATION_CLAIM")
        if authority.get("runtime_role") != RUNTIME_ROLE:
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_ROLE_DRIFT")
        # Immutable V4 predates consistent retention of this metadata field.
        # Its bytes remain pinned by V5 and the final exact subtree check still applies.
        historical_v4_omission = version == 4 and "subtree_closure" not in authority
        if version >= 3 and not historical_v4_omission and authority.get("subtree_closure") != SUBTREE_CLOSURE:
            raise RuntimeError(f"NATIVE_SOURCE_EXT{version}_SUBTREE_CLOSURE_DRIFT")


def _apply_extension(expected: dict[str, dict], ext: dict, label: str) -> None:
    for row in ext.get("replacements", []):
        path = str(row["path"])
        if path not in expected:
            raise RuntimeError(f"NATIVE_SOURCE_UNKNOWN_REPLACEMENT:{label}:{path}")
        expected[path] = {
            "size_bytes": int(row["size_bytes"]),
            "git_blob_sha1": str(row["git_blob_sha1"]),
            "source": f"{label}_replacement",
        }
    for row in ext.get("additions", []):
        path = str(row["path"])
        if path in expected:
            raise RuntimeError(f"NATIVE_SOURCE_DUPLICATE_ADDITION:{label}:{path}")
        expected[path] = {
            "size_bytes": int(row["size_bytes"]),
            "git_blob_sha1": str(row["git_blob_sha1"]),
            "source": f"{label}_addition",
        }


def verify() -> dict:
    base = _load_json(BASE)
    extensions = [_load_json(path) for path in EXTENSION_PATHS]
    _verify_chain(base, extensions)

    expected = {
        str(row["path"]): {
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
            "git_blob_sha1": str(row["git_blob_sha1"]),
            "source": "historical_base",
        }
        for row in base["files"]
    }
    for version, ext in enumerate(extensions, start=1):
        _apply_extension(expected, ext, f"extension_v{version}")

    tracked = {
        row.strip()
        for row in subprocess.check_output(
            ["git", "ls-files", "runtime/realsas_cpp"], text=True
        ).splitlines()
        if row.strip()
    }
    if tracked != set(expected):
        raise RuntimeError(
            "NATIVE_SOURCE_SUBTREE_SET_DRIFT:"
            + json.dumps(
                {
                    "unsealed_tracked_paths": sorted(tracked - set(expected)),
                    "sealed_paths_missing_from_repo": sorted(set(expected) - tracked),
                },
                sort_keys=True,
            )
        )

    # The latest extension is the only closure declaration that can describe the
    # current reconstructed subtree.  Using an older closure here would silently
    # make an append-only addition unverifiable.
    closure = dict(extensions[-1].get("closure") or {})
    if int(closure.get("runtime_realsas_cpp_blob_count", -1)) != len(tracked):
        raise RuntimeError("NATIVE_SOURCE_LATEST_CLOSURE_COUNT_DRIFT")
    if int(closure.get("resulting_sealed_blob_count", -1)) != len(expected):
        raise RuntimeError("NATIVE_SOURCE_LATEST_SEALED_COUNT_DRIFT")
    if int(closure.get("unsealed_repository_blob_count_under_subtree", -1)) != 0:
        raise RuntimeError("NATIVE_SOURCE_LATEST_UNSEALED_COUNT_NONZERO")

    verified = []
    for path_text, identity in sorted(expected.items()):
        payload = Path(path_text).read_bytes()
        if len(payload) != identity["size_bytes"]:
            raise RuntimeError(
                f"NATIVE_SOURCE_SIZE_DRIFT:{path_text}:{len(payload)}:{identity['size_bytes']}"
            )
        actual_blob = _blob_sha1(payload)
        if actual_blob != identity["git_blob_sha1"]:
            raise RuntimeError(
                f"NATIVE_SOURCE_BLOB_DRIFT:{path_text}:{actual_blob}:{identity['git_blob_sha1']}"
            )
        if identity["source"] == "historical_base":
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != identity["sha256"]:
                raise RuntimeError(
                    f"NATIVE_SOURCE_SHA256_DRIFT:{path_text}:{actual_sha}:{identity['sha256']}"
                )
        verified.append(path_text)

    result = {
        "status": "PASS__NATIVE_RUNTIME_SOURCE_SEAL_CHAIN",
        "file_count": len(verified),
        "base_file_count": len(base["files"]),
        "subtree_blob_count": len(tracked),
        "verified_paths": verified,
    }
    for version, ext in enumerate(extensions, start=1):
        result[f"extension_v{version}_change_count"] = len(ext.get("replacements", [])) + len(
            ext.get("additions", [])
        )
    return result


if __name__ == "__main__":
    result = verify()
    extension_text = " ".join(
        f"ext{version}={result[f'extension_v{version}_change_count']}"
        for version in range(1, len(EXTENSION_PATHS) + 1)
    )
    print(
        "NATIVE_SOURCE_SEAL_PASS "
        f"files={result['file_count']} "
        f"base={result['base_file_count']} "
        f"{extension_text} "
        f"subtree={result['subtree_blob_count']}"
    )
