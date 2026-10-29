#!/usr/bin/env python3
"""Hydrate only the exact October-4 Quaternius byte closure required for Truth139 raw audit.

This tool is deliberately narrower than full corpus reconstruction.  It copies the 395
ledger-bound Quaternius GLTF/GLB primaries from the sealed acquisition Drive folders, discovers
and copies only their local external buffer/image dependencies, copies the exact source-page
snapshots required by those packs from an existing exact seal working copy, and preserves the
original seal metadata bytes.

Every copied byte is checked against FILE_HASHES.sha256 before becoming part of the audit closure.
The output MUST NOT be described as a fully rehydrated corpus: files unrelated to the Truth139 raw
structural audit are intentionally absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote, urlparse

from tools.corpus import run_truth139_sealed_raw_audit_v1 as seal

SCHEMA = "RealSaS.Truth139AuditClosureHydrationReceipt.v1"
REQUIRED_PRIMARY_COUNT = 395
SEAL_FILES = (
    "manifest/MANIFEST.json",
    "manifest/FILE_HASHES.sha256",
    "manifest/CORPUS_SEAL.json",
    "manifest/CORPUS_SEAL_MATERIAL.txt",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_copy(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected:
        return
    staged = destination.with_suffix(destination.suffix + ".partial")
    staged.unlink(missing_ok=True)
    shutil.copyfile(source, staged)
    actual = sha256(staged)
    if actual != expected:
        staged.unlink(missing_ok=True)
        raise ValueError(f"TRUTH139_AUDIT_CLOSURE_LOCAL_COPY_HASH_DRIFT:{destination}:{expected}:{actual}")
    os.replace(staged, destination)


def _rclone_copy_exact(remote: str, folder_id: str, relative: str, destination: Path, expected: str) -> str:
    if not folder_id or "/" in folder_id or " " in folder_id:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_DRIVE_FOLDER_ID_INVALID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected:
        return "CACHE_HIT"
    staged = destination.with_suffix(destination.suffix + ".partial")
    staged.unlink(missing_ok=True)
    source = f"{remote},root_folder_id={folder_id}:{relative}"
    subprocess.run(["rclone", "copyto", source, str(staged)], check=True, timeout=900)
    if not staged.is_file():
        raise FileNotFoundError("TRUTH139_AUDIT_CLOSURE_RCLONE_OUTPUT_MISSING:" + relative)
    actual = sha256(staged)
    if actual != expected:
        staged.unlink(missing_ok=True)
        raise ValueError(f"TRUTH139_AUDIT_CLOSURE_REMOTE_HASH_DRIFT:{relative}:{expected}:{actual}")
    os.replace(staged, destination)
    return "HYDRATED"


def _normalize_dependency(source_relative: str, uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_REMOTE_URI_FORBIDDEN:" + source_relative)
    decoded = unquote(parsed.path)
    if not decoded:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_DEPENDENCY_URI_EMPTY:" + source_relative)
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(source_relative), decoded))
    parts = candidate.split("/")
    if len(parts) < 4 or parts[0:2] != ["raw", "quaternius"] or ".." in parts:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_DEPENDENCY_ESCAPE:" + candidate)
    if parts[2] != source_relative.split("/")[2]:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_CROSS_PACK_DEPENDENCY_FORBIDDEN:" + candidate)
    return candidate


def _document(source: Path) -> dict:
    if source.suffix.lower() == ".gltf":
        return json.loads(source.read_text(encoding="utf-8"))
    if source.suffix.lower() == ".glb":
        from tools.observation.source_scene_contract_v2 import _json_and_chunks
        doc, _chunks = _json_and_chunks(source)
        return doc
    raise ValueError("TRUTH139_AUDIT_CLOSURE_PRIMARY_FORMAT")


def _dependency_relatives(source_path: Path, source_relative: str) -> list[str]:
    doc = _document(source_path)
    rows = list(doc.get("buffers") or ()) + list(doc.get("images") or ())
    found = set()
    for row in rows:
        uri = str(row.get("uri") or "").strip()
        if not uri or uri.startswith("data:"):
            continue
        found.add(_normalize_dependency(source_relative, uri))
    return sorted(found)


def _find_exact_seal_roots(search_roots: tuple[Path, ...]) -> list[Path]:
    found = []
    seen = set()
    for base in search_roots:
        if not base.exists():
            continue
        for material in base.glob("**/manifest/CORPUS_SEAL_MATERIAL.txt"):
            root = material.parent.parent.resolve()
            if root in seen:
                continue
            seen.add(root)
            try:
                seal.verify_seal(root)
            except Exception:
                continue
            found.append(root)
    return sorted(found, key=lambda path: str(path))


def hydrate(*, metadata_root: Path, destination: Path, rclone_remote: str) -> dict:
    metadata_root = metadata_root.resolve()
    destination = destination.resolve()
    manifest, ledger, _ = seal.verify_seal(metadata_root)
    if destination == metadata_root or metadata_root in destination.parents:
        raise ValueError("TRUTH139_AUDIT_CLOSURE_DESTINATION_MUST_BE_SEPARATE")

    bindings = {
        str(row.get("id")): row
        for row in manifest.get("results") or ()
        if str(row.get("provider") or "").lower() == "quaternius"
    }
    primaries = sorted(
        relative
        for relative in ledger
        if relative.startswith("raw/quaternius/") and relative.lower().endswith((".gltf", ".glb"))
    )
    if len(primaries) != REQUIRED_PRIMARY_COUNT:
        raise ValueError(f"TRUTH139_AUDIT_CLOSURE_PRIMARY_CARDINALITY:{len(primaries)}")

    required_packs = sorted({relative.split("/")[2] for relative in primaries})
    if any(not str((bindings.get(pack) or {}).get("drive_folder_id") or "") for pack in required_packs):
        missing = [pack for pack in required_packs if not str((bindings.get(pack) or {}).get("drive_folder_id") or "")]
        raise ValueError("TRUTH139_AUDIT_CLOSURE_PACK_DRIVE_BINDING_MISSING:" + ",".join(missing))

    destination.mkdir(parents=True, exist_ok=True)
    # Seal metadata are exact historical bytes, copied before any raw hydration.
    for relative in SEAL_FILES:
        source = metadata_root / relative
        if not source.is_file():
            raise FileNotFoundError("TRUTH139_AUDIT_CLOSURE_SEAL_FILE_MISSING:" + relative)
        _atomic_copy(source, destination / relative, sha256(source))

    primary_rows = []
    dependency_relatives = set()
    for index, relative in enumerate(primaries, 1):
        expected = ledger[relative]
        pack = relative.split("/")[2]
        remote_relative = "/".join(relative.split("/")[3:])
        folder_id = str(bindings[pack]["drive_folder_id"])
        target = destination / relative
        transport = _rclone_copy_exact(rclone_remote, folder_id, remote_relative, target, expected)
        deps = _dependency_relatives(target, relative)
        dependency_relatives.update(deps)
        primary_rows.append({
            "relative_path": relative,
            "sha256": expected,
            "pack_id": pack,
            "drive_folder_id": folder_id,
            "dependency_count": len(deps),
            "transport": transport,
        })
        if index <= 3 or index % 25 == 0 or index == len(primaries):
            print(f"[TRUTH139_AUDIT_HYDRATE_PRIMARY] {index}/{len(primaries)} {relative} deps={len(deps)}", flush=True)

    dependency_rows = []
    for index, relative in enumerate(sorted(dependency_relatives), 1):
        expected = ledger.get(relative)
        if expected is None:
            raise ValueError("TRUTH139_AUDIT_CLOSURE_DEPENDENCY_NOT_IN_LEDGER:" + relative)
        pack = relative.split("/")[2]
        folder_id = str(bindings[pack]["drive_folder_id"])
        remote_relative = "/".join(relative.split("/")[3:])
        transport = _rclone_copy_exact(rclone_remote, folder_id, remote_relative, destination / relative, expected)
        dependency_rows.append({"relative_path": relative, "sha256": expected, "pack_id": pack, "transport": transport})
        if index <= 3 or index % 50 == 0 or index == len(dependency_relatives):
            print(f"[TRUTH139_AUDIT_HYDRATE_DEP] {index}/{len(dependency_relatives)} {relative}", flush=True)

    # Source-page snapshots are corpus provenance, not pack-folder payloads.  Search exact seal
    # working copies and copy the first hash-correct byte instance for each required pack.
    exact_roots = _find_exact_seal_roots((Path("/home/monster/realsas_authority"), Path("/home/monster")))
    snapshot_rows = []
    for pack in required_packs:
        binding = bindings[pack]
        relative = str(binding.get("source_snapshot") or "")
        expected = str(binding.get("source_snapshot_sha256") or "")
        if not relative or len(expected) != 64 or ledger.get(relative) != expected:
            raise ValueError("TRUTH139_AUDIT_CLOSURE_SOURCE_SNAPSHOT_BINDING:" + pack)
        witnesses = [root / relative for root in exact_roots if (root / relative).is_file() and sha256(root / relative) == expected]
        if not witnesses:
            raise FileNotFoundError("TRUTH139_AUDIT_CLOSURE_SOURCE_SNAPSHOT_UNAVAILABLE:" + pack)
        _atomic_copy(witnesses[0], destination / relative, expected)
        snapshot_rows.append({"pack_id": pack, "relative_path": relative, "sha256": expected, "witness_root": str(witnesses[0].parent.parent)})

    # Final exact-byte verification of every materialized authority byte.
    for row in primary_rows + dependency_rows:
        relative = row["relative_path"]
        if sha256(destination / relative) != row["sha256"]:
            raise RuntimeError("TRUTH139_AUDIT_CLOSURE_FINAL_HASH_DRIFT:" + relative)
    for row in snapshot_rows:
        if sha256(destination / row["relative_path"]) != row["sha256"]:
            raise RuntimeError("TRUTH139_AUDIT_CLOSURE_FINAL_SNAPSHOT_HASH_DRIFT:" + row["relative_path"])

    receipt = {
        "schema": SCHEMA,
        "date": "2026-10-29",
        "status": "PASS_EXACT_TRUTH139_AUDIT_CLOSURE_HYDRATED",
        "destination": str(destination),
        "corpus_seal_identity": {
            "root_sha256": seal.EXPECTED_ROOT_SHA256,
            "manifest_sha256": seal.EXPECTED_MANIFEST_SHA256,
            "ledger_sha256": seal.EXPECTED_LEDGER_SHA256,
            "seal_json_sha256": seal.EXPECTED_SEAL_JSON_SHA256,
        },
        "full_corpus_rehydrated": False,
        "materialization_scope": "395_QUATERNIUS_GLTF_GLB_PLUS_REFERENCED_LOCAL_DEPENDENCIES_PLUS_REQUIRED_SOURCE_PAGE_SNAPSHOTS",
        "required_pack_ids": required_packs,
        "primary_count": len(primary_rows),
        "dependency_count": len(dependency_rows),
        "source_page_snapshot_count": len(snapshot_rows),
        "primaries": primary_rows,
        "dependencies": dependency_rows,
        "source_page_snapshots": snapshot_rows,
        "render_performed": False,
        "training_performed": False,
        "split_created": False,
        "qualification_minted": False,
        "next_gate": "RUN_TRUTH139_SEALED_RAW_AUDIT_V2",
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_DRIVE_REMOTE", "gdrive"))
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = hydrate(
        metadata_root=Path(args.metadata_root),
        destination=Path(args.destination),
        rclone_remote=str(args.rclone_remote),
    )
    output = Path(args.receipt).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix(output.suffix + ".partial")
    staged.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    staged.replace(output)
    print(json.dumps({"status": result["status"], "primary_count": result["primary_count"], "dependency_count": result["dependency_count"], "source_page_snapshot_count": result["source_page_snapshot_count"], "receipt_sha256": result["receipt_sha256"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
