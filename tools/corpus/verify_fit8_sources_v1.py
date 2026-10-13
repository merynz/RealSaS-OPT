#!/usr/bin/env python3
"""Verify selected raw bytes, without importing or minting research authority."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "canonical/FIT8_DIVERSITY_SELECTION_V1.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_relative(raw: str) -> str:
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    require(bool(normalized) and not path.is_absolute(), "ABSOLUTE_OR_EMPTY_PATH")
    require(not any(part in ("", ".", "..") or ":" in part for part in normalized.split("/")),
            "UNSAFE_RELATIVE_PATH")
    return normalized


def verify(selection_path: Path, audit_bundle: Path, source_root: Path | None) -> dict:
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    require(selection["schema"] == "RealSaS.FIT8DiversitySelection.v1", "SELECTION_SCHEMA")
    rows = selection["selected"]
    require(len(rows) == 8 and selection["selection_policy"]["count"] == 8, "FIT8_COUNT")
    for field in ("subject_key", "coverage_class"):
        require(len({row[field] for row in rows}) == 8, f"DUPLICATE_{field.upper()}")
    require([row["ordinal"] for row in rows] == list(range(1, 9)), "ORDINALS")
    require(len({row["source"]["sha256"] for row in rows}) == 8, "DUPLICATE_SOURCE")
    require(selection["execution_ready"] is False, "IDENTITY_IS_NOT_ADMISSION")
    for flag in ("fit8_pass_claimed", "unseen_claimed", "product_pass_claimed"):
        require(selection[flag] is False, f"UNSUPPORTED_CLAIM:{flag}")
    expected = selection["raw_corpus"]
    require(digest(audit_bundle.read_bytes()) == expected["audit_package_sha256"],
            "AUDIT_BUNDLE_HASH")
    with zipfile.ZipFile(audit_bundle) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir() or info.filename.replace("\\", "/").endswith("/"):
                safe_relative(info.filename.rstrip("/\\"))
                continue
            path = safe_relative(info.filename)
            require(path not in members, "DUPLICATE_ARCHIVE_PATH")
            members[path] = info.filename

        def read(path: str) -> bytes:
            require(path in members, f"AUDIT_MEMBER_MISSING:{path}")
            return archive.read(members[path])

        manifest_bytes = read("manifest/MANIFEST.json")
        ledger_bytes = read("manifest/FILE_HASHES.sha256")
        seal = json.loads(read("manifest/CORPUS_SEAL.json").decode("utf-8-sig"))
        require(digest(manifest_bytes) == expected["manifest_sha256"] == seal["manifest_sha256"],
                "MANIFEST_HASH")
        require(digest(ledger_bytes) == expected["file_ledger_sha256"] == seal["file_ledger_sha256"],
                "LEDGER_HASH")
        require(digest(read("manifest/CORPUS_SEAL_MATERIAL.txt")) == expected["root_sha256"]
                == seal["corpus_root_sha256"], "CORPUS_ROOT_HASH")
        require(seal["corpus_id"] == expected["corpus_id"], "CORPUS_ID")
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        require(manifest["corpus_id"] == expected["corpus_id"], "MANIFEST_CORPUS_ID")
        packs = {row["id"]: row for row in manifest["results"]}
        ledger = {}
        for line in ledger_bytes.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            sha, path = line.split("  ", 1)
            path = safe_relative(path)
            require(path not in ledger, "DUPLICATE_LEDGER_PATH")
            ledger[path] = sha
        verified = []
        for row in rows:
            src = row["source"]
            path = safe_relative(src["corpus_relative_path"])
            require(path.startswith(f"raw/{row['creator']}/{row['pack_id']}/"), "SOURCE_PACK_PATH")
            require(ledger.get(path) == src["sha256"], f"SOURCE_LEDGER_HASH:{path}")
            pack = packs[row["pack_id"]]
            require(pack["status"] == "SEALED_RAW", "PACK_NOT_SEALED_RAW")
            require(pack["provider"] == row["creator"], "SOURCE_CREATOR")
            require(pack["official_url"] == src["official_page"], "OFFICIAL_PAGE")
            require(pack["source_snapshot_sha256"] == src["source_page_sha256"], "SOURCE_PAGE_HASH")
            require(pack["license"] == src["license"], "SOURCE_LICENSE")
            require(pack["drive_folder_id"] == src["locator"]["pack_folder_id"], "PACK_LOCATOR")
            local = source_root / path if source_root else None
            if local is not None and local.is_file():
                require(local.resolve().is_relative_to(source_root.resolve()), "SOURCE_SYMLINK_ESCAPE")
                data = local.read_bytes()
            else:
                data = read(path)
            require(digest(data) == src["sha256"], f"SOURCE_BYTE_HASH:{path}")
            require(len(data) == src["byte_size"], f"SOURCE_BYTE_SIZE:{path}")
            require(PurePosixPath(path).suffix[1:].upper() == src["format"], "SOURCE_FORMAT")
            verified.append({"subject_key": row["subject_key"], "corpus_relative_path": path,
                             "sha256": digest(data), "byte_size": len(data)})
    return {
        "schema": "RealSaS.FIT8SourceIdentityAudit.v1",
        "status": "SOURCE_IDENTITIES_VERIFIED__IMPORT_RENDER_ADMISSION_PENDING",
        "selection_sha256": digest(selection_bytes),
        "verifier_sha256": digest(Path(__file__).read_bytes()),
        "raw_corpus_root_sha256": expected["root_sha256"],
        "audit_package_sha256": expected["audit_package_sha256"],
        "verified_source_count": len(verified), "sources": verified,
        "import_executed": False, "render_executed": False,
        "research_attempt_created": False, "training_executed": False,
        "fit8_pass_claimed": False, "product_pass_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--audit-bundle", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.selection, args.audit_bundle, args.source_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
