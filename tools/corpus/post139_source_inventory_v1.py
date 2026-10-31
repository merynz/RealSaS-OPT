"""Acquire publisher-controlled corpus sources and emit an exact raw inventory.

This utility is intentionally authority-limited.  It records byte identity and
representation structure; it does not admit training subjects by itself.

Examples
--------
python tools/corpus/post139_source_inventory_v1.py acquire \
  --registry canonical/POST139_SOURCE_REGISTRY_V1.json \
  --source-id quaternius-easy-animated-enemy --out /tmp/easy

python tools/corpus/post139_source_inventory_v1.py inventory \
  --source-root /tmp/easy/raw --out /tmp/easy/inventory \
  --source-id quaternius-easy-animated-enemy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_RECEIPT = "RealSaS.Post139.AcquisitionReceipt.v1"
SCHEMA_LEDGER = "RealSaS.Post139.RawFileLedger.v1"
SCHEMA_INVENTORY = "RealSaS.Post139.SourceInventory.v1"
SCHEMA_MANIFEST = "RealSaS.Post139.CandidateManifest.v1"
MODEL_EXTENSIONS = {".gltf", ".glb", ".fbx", ".blend", ".obj"}
AUDIT_EXTENSIONS = {".gltf", ".glb", ".fbx"}
PREFERRED_AUDIT_FORMATS = (".gltf", ".glb", ".fbx")
FORMAT_DIR_NAMES = {
    "fbx", "gltf", "glb", "obj", "blend", "blender", "models_fbx",
    "models_gltf", "models_glb", "models_obj", "models_blend",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "item"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "RealSaS.Post139.SourceRegistry.v1":
        raise ValueError("POST139_SOURCE_REGISTRY_SCHEMA_MISMATCH")
    return payload


def source_row(registry: dict[str, Any], source_id: str) -> dict[str, Any]:
    rows = [row for row in registry["sources"] if row["id"] == source_id]
    if len(rows) != 1:
        raise ValueError(f"POST139_SOURCE_ID_NOT_UNIQUE:{source_id}:{len(rows)}")
    return rows[0]


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"POST139_ZIP_PATH_ESCAPE:{member.filename}")
        zf.extractall(destination)


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "RealSaS-Post139-Audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def quaternius_drive_folder(row: dict[str, Any]) -> tuple[str, str]:
    acq = row["acquisition"]
    repo = acq["publisher_repository"]
    owner, name = repo.split("/", 1)
    page_path = acq["page_path"]
    page_url = f"https://raw.githubusercontent.com/{owner}/{name}/main/{page_path}"
    request = urllib.request.Request(page_url, headers={"User-Agent": "RealSaS-Post139-Audit/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="replace")
    matches = re.findall(r"https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)", html)
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise ValueError(f"POST139_QUATERNIUS_DRIVE_RESOLUTION:{row['id']}:{matches}")
    return matches[0], page_url


def acquire(args: argparse.Namespace) -> None:
    registry = load_registry(Path(args.registry))
    row = source_row(registry, args.source_id)
    out = Path(args.out).resolve()
    raw = out / "raw"
    if out.exists() and args.clean:
        shutil.rmtree(out)
    raw.mkdir(parents=True, exist_ok=True)
    acq = row["acquisition"]
    kind = acq["kind"]
    receipt: dict[str, Any] = {
        "schema": SCHEMA_RECEIPT,
        "source_id": row["id"],
        "source_family": row["source_family"],
        "pack": row["pack"],
        "role": row["role"],
        "license": row["license"],
        "provenance": row["provenance"],
        "acquisition_kind": kind,
        "status": "UNRESOLVED",
        "raw_root": str(raw),
    }

    if kind == "DIRECT_ZIP":
        archive = out / "download.zip"
        download(acq["url"], archive)
        actual_sha = sha256_file(archive)
        actual_size = archive.stat().st_size
        if acq.get("expected_sha256") and actual_sha != acq["expected_sha256"]:
            raise ValueError(f"POST139_ARCHIVE_SHA_MISMATCH:{actual_sha}")
        if acq.get("expected_size_bytes") and actual_size != int(acq["expected_size_bytes"]):
            raise ValueError(f"POST139_ARCHIVE_SIZE_MISMATCH:{actual_size}")
        safe_extract_zip(archive, raw)
        receipt.update({
            "status": "ACQUIRED_VERIFIED",
            "resolved_url": acq["url"],
            "archive_sha256": actual_sha,
            "archive_size_bytes": actual_size,
        })
    elif kind == "GITHUB_REPOSITORY_ARCHIVE":
        repo = acq["repository"]
        ref = acq.get("ref", "main")
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/{ref}"
        archive = out / "repository.zip"
        download(url, archive)
        actual_sha = sha256_file(archive)
        actual_size = archive.stat().st_size
        safe_extract_zip(archive, raw)
        receipt.update({
            "status": "ACQUIRED_VERIFIED",
            "resolved_url": url,
            "archive_sha256": actual_sha,
            "archive_size_bytes": actual_size,
            "repository": repo,
            "ref": ref,
        })
    elif kind == "QUATERNIUS_OFFICIAL_PAGE_DRIVE":
        folder_id, page_url = quaternius_drive_folder(row)
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise RuntimeError("POST139_GDOWN_REQUIRED") from exc
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        downloaded = gdown.download_folder(
            url=url,
            output=str(raw),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
        if downloaded is None:
            raise RuntimeError(f"POST139_GDOWN_FOLDER_FAILED:{row['id']}")
        receipt.update({
            "status": "ACQUIRED_VERIFIED",
            "publisher_page_raw_url": page_url,
            "resolved_drive_folder_id": folder_id,
            "resolved_url": url,
            "downloaded_file_count": len(downloaded),
        })
    elif kind in {"KENNEY_OFFICIAL_PAGE_PENDING_RESOLVER", "NO_AUTO_ACQUIRE"}:
        receipt["status"] = "UNRESOLVED" if kind != "NO_AUTO_ACQUIRE" else "QUARANTINED"
        receipt["reason"] = kind
        write_json(out / "ACQUISITION_RECEIPT.json", receipt)
        print(f"POST139_ACQUISITION_{receipt['status']} {row['id']}")
        return
    else:
        raise ValueError(f"POST139_ACQUISITION_KIND_UNSUPPORTED:{kind}")

    files = [path for path in raw.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"POST139_ACQUISITION_EMPTY:{row['id']}")
    receipt["raw_file_count"] = len(files)
    receipt["raw_tree_digest"] = digest(sorted(
        (path.relative_to(raw).as_posix(), path.stat().st_size, sha256_file(path)) for path in files
    ))
    write_json(out / "ACQUISITION_RECEIPT.json", receipt)
    print("POST139_ACQUISITION_PASS", row["id"], len(files), receipt["raw_tree_digest"])


def strip_format_components(path: PurePosixPath) -> PurePosixPath:
    parts = list(path.parts)
    filtered = [part for part in parts[:-1] if slug(part).replace("-", "_") not in FORMAT_DIR_NAMES]
    return PurePosixPath(*filtered, path.stem)


def representation_key(relative_path: str) -> str:
    pure = PurePosixPath(relative_path)
    stripped = strip_format_components(pure)
    return "/".join(slug(part) for part in stripped.parts)


def inventory(args: argparse.Namespace) -> None:
    root = Path(args.source_root).resolve()
    out = Path(args.out).resolve()
    if not root.is_dir():
        raise SystemExit(f"POST139_SOURCE_ROOT_MISSING:{root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    ledger_rows = []
    extension_counts: Counter[str] = Counter()
    representations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(root).as_posix()
        extension = path.suffix.lower() or "<none>"
        extension_counts[extension] += 1
        row = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "extension": extension,
        }
        ledger_rows.append(row)
        if extension in MODEL_EXTENSIONS:
            representations[representation_key(relative)].append({
                "path": relative,
                "extension": extension,
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            })

    ledger = {
        "schema": SCHEMA_LEDGER,
        "source_id": args.source_id,
        "source_root": str(root),
        "file_count": len(ledger_rows),
        "total_bytes": sum(row["size_bytes"] for row in ledger_rows),
        "extension_counts": dict(sorted(extension_counts.items())),
        "files": ledger_rows,
    }
    ledger["ledger_sha256"] = digest(ledger_rows)

    representation_rows = []
    candidates = []
    used_candidate_ids: Counter[str] = Counter()
    for key, group in sorted(representations.items()):
        group = sorted(group, key=lambda row: (row["extension"], row["path"]))
        by_ext: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_ext[row["extension"]].append(row)
        chosen = None
        for extension in PREFERRED_AUDIT_FORMATS:
            if by_ext.get(extension):
                chosen = by_ext[extension][0]
                break
        record = {
            "representation_key": key,
            "representations": group,
            "format_counts": {ext: len(rows) for ext, rows in sorted(by_ext.items())},
            "audit_source": chosen,
            "audit_status": "READY" if chosen else "NO_SUPPORTED_STRUCTURAL_REPRESENTATION",
        }
        representation_rows.append(record)
        if chosen:
            base_id = slug(key.replace("/", "-"))
            used_candidate_ids[base_id] += 1
            suffix = "" if used_candidate_ids[base_id] == 1 else f"-{used_candidate_ids[base_id]}"
            candidates.append({
                "subject_id": f"{args.source_id}--{base_id}{suffix}",
                "source_id": args.source_id,
                "representation_key": key,
                "relative_path": chosen["path"],
                "source_sha256": chosen["sha256"],
                "source_size_bytes": chosen["size_bytes"],
                "source_extension": chosen["extension"],
                "format_pair_count": len(group),
                "provisional_role": "STRUCTURAL_AUDIT_CANDIDATE",
            })

    inventory_payload = {
        "schema": SCHEMA_INVENTORY,
        "source_id": args.source_id,
        "source_root": str(root),
        "raw_file_count": len(ledger_rows),
        "raw_total_bytes": ledger["total_bytes"],
        "raw_ledger_sha256": ledger["ledger_sha256"],
        "model_file_count": sum(1 for row in ledger_rows if row["extension"] in MODEL_EXTENSIONS),
        "representation_group_count": len(representation_rows),
        "audit_candidate_count": len(candidates),
        "extension_counts": ledger["extension_counts"],
        "representation_groups": representation_rows,
    }
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "source_id": args.source_id,
        "source_root": str(root),
        "inventory_sha256": digest(inventory_payload),
        "candidates": candidates,
    }
    write_json(out / "RAW_FILE_LEDGER.json", ledger)
    write_json(out / "SOURCE_INVENTORY.json", inventory_payload)
    write_json(out / "CANDIDATE_MANIFEST.json", manifest)
    print("POST139_SOURCE_INVENTORY_PASS", args.source_id, len(ledger_rows), len(representation_rows), len(candidates), ledger["ledger_sha256"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--registry", required=True)
    acquire_parser.add_argument("--source-id", required=True)
    acquire_parser.add_argument("--out", required=True)
    acquire_parser.add_argument("--clean", action="store_true")
    inventory_parser = sub.add_parser("inventory")
    inventory_parser.add_argument("--source-root", required=True)
    inventory_parser.add_argument("--source-id", required=True)
    inventory_parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "acquire":
        acquire(args)
    elif args.command == "inventory":
        inventory(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
