#!/usr/bin/env python3
from __future__ import annotations

"""Hydrate and exhaustively verify the split Corpus370 raw-source bundle.

Transport is not scientific authority. This tool reassembles the exact Drive parts,
verifies every part, the complete archive, the embedded source-bundle manifest and
every manifest-declared member before exposing an extracted source root. No source
byte is repaired or rewritten and no truth family, split, or training authority is
minted here.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tempfile
import zipfile

SCHEMA = "RealSaS.SourceBundleTransportParts.v1"
MANIFEST_SCHEMA = "RealSaS.ExpandedCorpusSourceInputBundle.v1"
RECEIPT_SCHEMA = "RealSaS.Corpus370SourceBundleHydration.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_sha256(stream) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        block = stream.read(8 << 20)
        if not block:
            break
        size += len(block)
        digest.update(block)
    return digest.hexdigest(), size


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def exact_file(path: Path, expected_sha: str, expected_size: int) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_size == expected_size
        and sha256(path) == expected_sha
    )


def fetch_drive_part(*, file_id: str, destination: Path, rclone_remote: str) -> None:
    remote = rclone_remote.rstrip(":")
    require(bool(remote) and ":" not in remote, "CORPUS370_SOURCE_RCLONE_REMOTE")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rclone", "backend", "copyid", f"{remote}:", file_id, str(destination)],
        check=True,
        timeout=900,
    )


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and ".." not in path.parts, "CORPUS370_SOURCE_ZIP_PATH_ESCAPE")
    require(str(path) == name, "CORPUS370_SOURCE_ZIP_PATH_NORMALIZATION")
    return path


def _validate_transport(transport: dict) -> None:
    require(transport.get("schema") == SCHEMA, "CORPUS370_SOURCE_TRANSPORT_SCHEMA")
    require(transport.get("training_authorized") is False, "CORPUS370_SOURCE_TRANSPORT_PREAUTHORIZED")
    require(transport.get("source_bytes_modified") is False, "CORPUS370_SOURCE_TRANSPORT_MUTATED")
    require(int(transport.get("subject_count", -1)) == 370, "CORPUS370_SOURCE_TRANSPORT_SUBJECT_COUNT")
    parts = list(transport.get("parts") or ())
    require(parts and len(parts) == 5, "CORPUS370_SOURCE_TRANSPORT_PART_COUNT")
    require(sum(int(row["size_bytes"]) for row in parts) == int(transport["archive_size_bytes"]), "CORPUS370_SOURCE_TRANSPORT_PART_SIZE_SUM")
    require(len({row["filename"] for row in parts}) == len(parts), "CORPUS370_SOURCE_TRANSPORT_PART_NAME_DUPLICATE")
    require(len({row["drive_file_id"] for row in parts}) == len(parts), "CORPUS370_SOURCE_TRANSPORT_PART_DRIVE_ID_DUPLICATE")
    for row in parts:
        require(len(str(row.get("sha256") or "")) == 64, "CORPUS370_SOURCE_TRANSPORT_PART_SHA")
        require(bool(row.get("drive_file_id")), "CORPUS370_SOURCE_TRANSPORT_PART_DRIVE_ID")
    require(len(str(transport.get("archive_sha256") or "")) == 64, "CORPUS370_SOURCE_TRANSPORT_ARCHIVE_SHA")
    require(len(str(transport.get("source_bundle_manifest_sha256") or "")) == 64, "CORPUS370_SOURCE_TRANSPORT_MANIFEST_SHA")
    require(sum(int(v) for v in (transport.get("origin_counts") or {}).values()) == 370, "CORPUS370_SOURCE_TRANSPORT_ORIGIN_COUNTS")
    require(sum(int(v) for v in (transport.get("format_counts") or {}).values()) == 370, "CORPUS370_SOURCE_TRANSPORT_FORMAT_COUNTS")
    require(sum(int(v) for v in (transport.get("source_id_counts") or {}).values()) == 370, "CORPUS370_SOURCE_TRANSPORT_SOURCE_ID_COUNTS")


def _validate_manifest(manifest: dict, transport: dict) -> None:
    subjects = list(manifest.get("subjects") or ())
    require(manifest.get("schema") == MANIFEST_SCHEMA, "CORPUS370_SOURCE_MANIFEST_SCHEMA")
    require(manifest.get("training_authorized") is False, "CORPUS370_SOURCE_MANIFEST_PREAUTHORIZED")
    require(int(manifest.get("subject_count", -1)) == 370 and len(subjects) == 370, "CORPUS370_SOURCE_MANIFEST_SUBJECT_COUNT")
    require(len({row["subject_id"] for row in subjects}) == 370, "CORPUS370_SOURCE_MANIFEST_SUBJECT_IDS")
    require(len({row["source_sha256"] for row in subjects}) == 370, "CORPUS370_SOURCE_MANIFEST_SOURCE_SHA_UNIQUENESS")
    require(dict(sorted(Counter(row.get("origin") for row in subjects).items())) == dict(sorted((transport.get("origin_counts") or {}).items())), "CORPUS370_SOURCE_MANIFEST_ORIGIN_CENSUS")
    require(dict(sorted(Counter(row.get("source_format") for row in subjects).items())) == dict(sorted((transport.get("format_counts") or {}).items())), "CORPUS370_SOURCE_MANIFEST_FORMAT_CENSUS")
    require(dict(sorted(Counter(row.get("source_id") for row in subjects).items())) == dict(sorted((transport.get("source_id_counts") or {}).items())), "CORPUS370_SOURCE_MANIFEST_SOURCE_ID_CENSUS")


def hydrate(
    transport_path: Path,
    output_root: Path,
    *,
    rclone_remote: str,
    fetcher=fetch_drive_part,
) -> dict:
    transport_path = transport_path.resolve()
    output_root = output_root.resolve()
    transport = json.loads(transport_path.read_text())
    _validate_transport(transport)
    output_root.mkdir(parents=True, exist_ok=True)
    parts_root = output_root / "parts"
    parts_root.mkdir(parents=True, exist_ok=True)

    part_receipts = []
    for index, row in enumerate(transport["parts"]):
        target = parts_root / str(row["filename"])
        expected_sha = str(row["sha256"])
        expected_size = int(row["size_bytes"])
        state = "CAS_HIT"
        if not exact_file(target, expected_sha, expected_size):
            if target.exists():
                target.rename(target.with_name(target.name + ".drift." + str(os.getpid())))
            fd, raw_temp = tempfile.mkstemp(dir=parts_root, prefix=target.name + ".", suffix=".partial")
            os.close(fd)
            temporary = Path(raw_temp)
            try:
                fetcher(file_id=str(row["drive_file_id"]), destination=temporary, rclone_remote=rclone_remote)
                require(exact_file(temporary, expected_sha, expected_size), "CORPUS370_SOURCE_PART_BYTES_DRIFT:" + str(index))
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            state = "HYDRATED"
        require(exact_file(target, expected_sha, expected_size), "CORPUS370_SOURCE_PART_FINAL_DRIFT:" + str(index))
        part_receipts.append({"index": index, "path": str(target), "sha256": expected_sha, "size_bytes": expected_size, "state": state, "drive_file_id": row["drive_file_id"]})
        print(json.dumps({"event": "CORPUS370_SOURCE_PART", "index": index, "state": state, "size_bytes": expected_size}), flush=True)

    archive = output_root / str(transport["archive_filename"])
    expected_archive_sha = str(transport["archive_sha256"])
    expected_archive_size = int(transport["archive_size_bytes"])
    if not exact_file(archive, expected_archive_sha, expected_archive_size):
        fd, raw_temp = tempfile.mkstemp(dir=output_root, prefix=archive.name + ".", suffix=".partial")
        os.close(fd)
        temporary = Path(raw_temp)
        try:
            with temporary.open("wb") as output:
                for row in part_receipts:
                    with Path(row["path"]).open("rb") as source:
                        while True:
                            block = source.read(8 << 20)
                            if not block:
                                break
                            output.write(block)
            require(exact_file(temporary, expected_archive_sha, expected_archive_size), "CORPUS370_SOURCE_ARCHIVE_REASSEMBLY_DRIFT")
            os.replace(temporary, archive)
        finally:
            temporary.unlink(missing_ok=True)
    require(exact_file(archive, expected_archive_sha, expected_archive_size), "CORPUS370_SOURCE_ARCHIVE_FINAL_DRIFT")

    extracted = output_root / "extracted" / expected_archive_sha
    extracted.mkdir(parents=True, exist_ok=True)
    manifest_member = str(transport["source_bundle_manifest_member"])
    expected_manifest_sha = str(transport["source_bundle_manifest_sha256"])
    with zipfile.ZipFile(archive) as source_zip:
        names = source_zip.namelist()
        require(len(names) == len(set(names)), "CORPUS370_SOURCE_ZIP_DUPLICATE_MEMBER")
        for info in source_zip.infolist():
            _safe_member(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            require(mode != stat.S_IFLNK, "CORPUS370_SOURCE_ZIP_SYMLINK_FORBIDDEN")
        manifest_bytes = source_zip.read(manifest_member)
        require(hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha, "CORPUS370_SOURCE_EMBEDDED_MANIFEST_DRIFT")
        manifest = json.loads(manifest_bytes)
        _validate_manifest(manifest, transport)
        declared = {str(row["archive_path"]): row for row in manifest.get("files") or ()}
        require(len(declared) == len(manifest.get("files") or ()), "CORPUS370_SOURCE_MANIFEST_FILE_PATH_DUPLICATE")
        expected_members = set(declared) | {manifest_member}
        require(set(names) == expected_members, "CORPUS370_SOURCE_ZIP_MEMBER_SET_DRIFT")

        for index, (name, row) in enumerate(sorted(declared.items()), 1):
            info = source_zip.getinfo(name)
            expected_size = int(row["size_bytes"])
            expected_sha = str(row["sha256"])
            require(info.file_size == expected_size, "CORPUS370_SOURCE_ZIP_MEMBER_SIZE:" + name)
            with source_zip.open(info, "r") as stream:
                actual_sha, actual_size = stream_sha256(stream)
            require(actual_size == expected_size and actual_sha == expected_sha, "CORPUS370_SOURCE_ZIP_MEMBER_SHA:" + name)
            destination = extracted / _safe_member(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not exact_file(destination, expected_sha, expected_size):
                with source_zip.open(info, "r") as src, destination.with_suffix(destination.suffix + ".partial").open("wb") as dst:
                    while True:
                        block = src.read(8 << 20)
                        if not block:
                            break
                        dst.write(block)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                require(exact_file(temporary, expected_sha, expected_size), "CORPUS370_SOURCE_EXTRACTED_MEMBER_DRIFT:" + name)
                os.replace(temporary, destination)
            if index % 100 == 0 or index == len(declared):
                print(json.dumps({"event": "CORPUS370_SOURCE_MEMBER_VERIFY", "verified": index, "total": len(declared)}), flush=True)

    manifest_path = extracted / manifest_member
    if not manifest_path.exists():
        manifest_path.write_bytes(manifest_bytes)
    require(sha256(manifest_path) == expected_manifest_sha, "CORPUS370_SOURCE_EXTRACTED_MANIFEST_DRIFT")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_EXACT_SOURCE_BUNDLE_HYDRATED_AND_ALL_DECLARED_MEMBERS_REVERIFIED",
        "transport": {"path": str(transport_path), "sha256": sha256(transport_path)},
        "archive": {"path": str(archive), "sha256": expected_archive_sha, "size_bytes": expected_archive_size},
        "source_bundle_manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha},
        "extracted_source_root": str(extracted),
        "subject_count": 370,
        "declared_file_count": len(manifest.get("files") or ()),
        "parts": part_receipts,
        "source_bytes_modified": False,
        "family_authority_minted": False,
        "split_minted": False,
        "training_authorized": False,
    }
    receipt_path = output_root / "CORPUS370_SOURCE_BUNDLE_HYDRATION_V1.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": receipt["schema"], "status": receipt["status"], "subject_count": 370, "declared_file_count": receipt["declared_file_count"], "archive_sha256": expected_archive_sha}, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", type=Path, default=Path(__file__).resolve().parents[2] / "canonical/CORPUS370_SOURCE_BUNDLE_TRANSPORT_V1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rclone-remote", default=os.environ.get("REALSAS_RCLONE_DRIVE_REMOTE", "realsas-drive"))
    args = parser.parse_args()
    hydrate(args.transport, args.output_root, rclone_remote=args.rclone_remote)


if __name__ == "__main__":
    main()
