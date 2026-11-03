"""Acquire publisher-controlled corpus sources and emit an exact source inventory.

Two inventories are intentionally distinct:

* publisher tree: every authored file/path exposed by the publisher source;
* acquired byte closure: the canonical structural representation lane and its
  appearance/buffer dependencies, with exact SHA-256 and byte sizes.

This lets RealSaS know the complete pack structure without hammering Google
Drive by downloading redundant Blend/OBJ/FBX/glTF copies of the same actor.
Training authority is never minted here.
"""
from __future__ import annotations

import argparse
import hashlib
import html as html_lib
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_RECEIPT = "RealSaS.Post139.AcquisitionReceipt.v1"
SCHEMA_PUBLISHER_TREE = "RealSaS.Post139.PublisherTree.v1"
SCHEMA_LEDGER = "RealSaS.Post139.RawFileLedger.v1"
SCHEMA_INVENTORY = "RealSaS.Post139.SourceInventory.v1"
SCHEMA_MANIFEST = "RealSaS.Post139.CandidateManifest.v1"
MODEL_EXTENSIONS = {".gltf", ".glb", ".fbx", ".blend", ".obj"}
PREFERRED_AUDIT_FORMATS = (".glb", ".gltf", ".fbx")
DEPENDENCY_EXTENSIONS = {".bin", ".png", ".jpg", ".jpeg", ".webp", ".ktx", ".ktx2", ".dds", ".tga", ".bmp"}
FORMAT_DIR_NAMES = {
    "fbx", "gltf", "glb", "obj", "blend", "blender", "models_fbx",
    "models_gltf", "models_glb", "models_obj", "models_blend",
}
USER_AGENT = "Mozilla/5.0 (compatible; RealSaS-Post139-Audit/1.0; +https://github.com/merynz/RealSaS-OPT)"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(path: Path) -> dict[str, Any]:
    payload = read_json(path)
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


def fetch_text(url: str, timeout: int = 90) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Referer": urllib.parse.urljoin(url, "/")})
    with urllib.request.urlopen(request, timeout=180) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def quaternius_drive_folder(row: dict[str, Any]) -> tuple[str, str]:
    acq = row["acquisition"]
    owner, name = acq["publisher_repository"].split("/", 1)
    page_url = f"https://raw.githubusercontent.com/{owner}/{name}/main/{acq['page_path']}"
    document = fetch_text(page_url)
    matches = list(dict.fromkeys(re.findall(r"https://drive\.google\.com/drive/folders/([A-Za-z0-9_-]+)", document)))
    if len(matches) != 1:
        raise ValueError(f"POST139_QUATERNIUS_DRIVE_RESOLUTION:{row['id']}:{matches}")
    return matches[0], page_url


def kenney_official_zip(row: dict[str, Any]) -> tuple[str, str]:
    page_url = row["acquisition"]["page_url"]
    document = html_lib.unescape(fetch_text(page_url))
    absolute = re.findall(r"https://kenney\.nl/media/pages/assets/[^\"'<> ]+?\.zip", document, flags=re.IGNORECASE)
    relative = re.findall(r"(?:href|data-url|data-download)=[\"'](/media/pages/assets/[^\"']+?\.zip)[\"']", document, flags=re.IGNORECASE)
    matches = list(dict.fromkeys(absolute + [urllib.parse.urljoin(page_url, value) for value in relative]))
    if not matches:
        paths = re.findall(r"/media/pages/assets/[^\"'<> ]+?\.zip", document, flags=re.IGNORECASE)
        matches = list(dict.fromkeys(urllib.parse.urljoin(page_url, value) for value in paths))
    if len(matches) != 1:
        raise ValueError(f"POST139_KENNEY_ZIP_RESOLUTION:{row['id']}:{matches}")
    return matches[0], page_url


def local_publisher_tree(raw: Path, source_id: str, transport: str) -> dict[str, Any]:
    rows = []
    extensions: Counter[str] = Counter()
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        relative = path.relative_to(raw).as_posix()
        extension = path.suffix.lower() or "<none>"
        extensions[extension] += 1
        rows.append({
            "path": relative,
            "extension": extension,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "byte_state": "ACQUIRED_HASHED",
        })
    payload = {
        "schema": SCHEMA_PUBLISHER_TREE,
        "source_id": source_id,
        "transport": transport,
        "file_count": len(rows),
        "extension_counts": dict(sorted(extensions.items())),
        "files": rows,
    }
    payload["tree_digest"] = digest(rows)
    return payload


def acquire_zip(row: dict[str, Any], url: str, out: Path, raw: Path, receipt: dict[str, Any]) -> None:
    archive = out / "download.zip"
    download(url, archive)
    actual_sha = sha256_file(archive)
    actual_size = archive.stat().st_size
    acq = row["acquisition"]
    if acq.get("expected_sha256") and actual_sha != acq["expected_sha256"]:
        raise ValueError(f"POST139_ARCHIVE_SHA_MISMATCH:{actual_sha}")
    if acq.get("expected_size_bytes") and actual_size != int(acq["expected_size_bytes"]):
        raise ValueError(f"POST139_ARCHIVE_SIZE_MISMATCH:{actual_size}")
    safe_extract_zip(archive, raw)
    receipt.update({
        "status": "ACQUIRED_VERIFIED",
        "resolved_url": url,
        "archive_sha256": actual_sha,
        "archive_size_bytes": actual_size,
    })


def choose_quaternius_lane(files: list[Any]) -> tuple[str, list[Any]]:
    by_extension: defaultdict[str, list[Any]] = defaultdict(list)
    for item in files:
        by_extension[Path(item.path).suffix.lower()].append(item)
    lane = next((extension for extension in PREFERRED_AUDIT_FORMATS if by_extension.get(extension)), None)
    if lane is None:
        raise RuntimeError("POST139_QUATERNIUS_NO_STRUCTURAL_LANE")
    selected = []
    for item in files:
        extension = Path(item.path).suffix.lower()
        if extension == lane or extension in DEPENDENCY_EXTENSIONS:
            selected.append(item)
    return lane, selected


def download_gdrive_item(gdown: Any, item: Any, raw: Path, attempts: int = 6) -> Path:
    target = raw / item.path
    target.parent.mkdir(parents=True, exist_ok=True)
    delays = [0, 4, 12, 30, 60, 90]
    last_error: Exception | None = None
    for attempt in range(attempts):
        if target.is_file() and target.stat().st_size > 0:
            return target
        if delays[attempt]:
            time.sleep(delays[attempt])
        try:
            result = gdown.download(
                url=f"https://drive.google.com/uc?id={item.id}",
                output=str(target),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
            if result and target.is_file() and target.stat().st_size > 0:
                return target
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"POST139_GDRIVE_FILE_DOWNLOAD_FAILED:{item.id}:{item.path}:{last_error}")


def acquire_quaternius(row: dict[str, Any], out: Path, raw: Path, receipt: dict[str, Any]) -> None:
    folder_id, page_url = quaternius_drive_folder(row)
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("POST139_GDOWN_REQUIRED") from exc
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    listing = gdown.download_folder(
        url=url,
        output=str(raw),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
    )
    if listing is None or not listing:
        raise RuntimeError(f"POST139_GDOWN_FOLDER_LIST_FAILED:{row['id']}")
    publisher_rows = []
    extensions: Counter[str] = Counter()
    for item in listing:
        extension = Path(item.path).suffix.lower() or "<none>"
        extensions[extension] += 1
        publisher_rows.append({
            "file_id": item.id,
            "path": PurePosixPath(item.path).as_posix(),
            "extension": extension,
            "byte_state": "PUBLISHER_LISTED_NOT_ACQUIRED",
        })
    lane, selected = choose_quaternius_lane(listing)
    acquired = []
    for item in selected:
        acquired.append(download_gdrive_item(gdown, item, raw))
    acquired_by_id = {item.id: raw / item.path for item in selected if (raw / item.path).is_file()}
    for row_entry in publisher_rows:
        if row_entry["file_id"] in acquired_by_id:
            path = acquired_by_id[row_entry["file_id"]]
            row_entry.update({
                "byte_state": "ACQUIRED_HASHED",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    publisher_tree = {
        "schema": SCHEMA_PUBLISHER_TREE,
        "source_id": row["id"],
        "transport": "QUATERNIUS_OFFICIAL_PUBLIC_DRIVE",
        "resolved_drive_folder_id": folder_id,
        "file_count": len(publisher_rows),
        "extension_counts": dict(sorted(extensions.items())),
        "canonical_lane_extension": lane,
        "acquired_file_count": len(acquired),
        "files": publisher_rows,
    }
    publisher_tree["tree_digest"] = digest([
        {"file_id": item["file_id"], "path": item["path"], "extension": item["extension"]}
        for item in publisher_rows
    ])
    write_json(out / "PUBLISHER_TREE.json", publisher_tree)
    receipt.update({
        "status": "ACQUIRED_VERIFIED",
        "publisher_page_raw_url": page_url,
        "resolved_drive_folder_id": folder_id,
        "resolved_url": url,
        "publisher_file_count": len(publisher_rows),
        "publisher_tree_digest": publisher_tree["tree_digest"],
        "canonical_lane_extension": lane,
        "downloaded_file_count": len(acquired),
        "byte_scope": "CANONICAL_STRUCTURAL_LANE_PLUS_APPEARANCE_AND_BUFFER_DEPENDENCIES",
    })


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
        "source_id": row["id"], "source_family": row["source_family"], "pack": row["pack"],
        "role": row["role"], "license": row["license"], "provenance": row["provenance"],
        "acquisition_kind": kind, "status": "UNRESOLVED", "raw_root": str(raw),
    }

    if kind == "DIRECT_ZIP":
        acquire_zip(row, acq["url"], out, raw, receipt)
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
            "status": "ACQUIRED_VERIFIED", "resolved_url": url,
            "archive_sha256": actual_sha, "archive_size_bytes": actual_size,
            "repository": repo, "ref": ref,
        })
    elif kind == "QUATERNIUS_OFFICIAL_PAGE_DRIVE":
        acquire_quaternius(row, out, raw, receipt)
    elif kind == "KENNEY_OFFICIAL_PAGE_PENDING_RESOLVER":
        url, page_url = kenney_official_zip(row)
        acquire_zip(row, url, out, raw, receipt)
        receipt["publisher_page_url"] = page_url
        receipt["acquisition_kind"] = "KENNEY_OFFICIAL_PAGE_RESOLVED_ZIP"
    elif kind == "NO_AUTO_ACQUIRE":
        receipt.update({"status": "QUARANTINED", "reason": kind})
        write_json(out / "ACQUISITION_RECEIPT.json", receipt)
        print("POST139_ACQUISITION_QUARANTINED", row["id"])
        return
    else:
        raise ValueError(f"POST139_ACQUISITION_KIND_UNSUPPORTED:{kind}")

    files = [path for path in raw.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"POST139_ACQUISITION_EMPTY:{row['id']}")
    if not (out / "PUBLISHER_TREE.json").is_file():
        write_json(out / "PUBLISHER_TREE.json", local_publisher_tree(raw, row["id"], kind))
    receipt["raw_file_count"] = len(files)
    receipt["raw_tree_digest"] = digest(sorted((path.relative_to(raw).as_posix(), path.stat().st_size, sha256_file(path)) for path in files))
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


def publisher_representation_groups(publisher_tree: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not publisher_tree:
        return []
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in publisher_tree.get("files", []):
        if row.get("extension") not in MODEL_EXTENSIONS:
            continue
        groups[representation_key(row["path"])].append({
            key: row[key]
            for key in ("file_id", "path", "extension", "byte_state", "size_bytes", "sha256")
            if key in row
        })
    result = []
    for key, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda item: (item["extension"], item["path"]))
        result.append({
            "representation_key": key,
            "representations": rows,
            "format_counts": dict(sorted(Counter(item["extension"] for item in rows).items())),
        })
    return result


def inventory(args: argparse.Namespace) -> None:
    root = Path(args.source_root).resolve()
    out = Path(args.out).resolve()
    if not root.is_dir():
        raise SystemExit(f"POST139_SOURCE_ROOT_MISSING:{root}")
    source_meta = source_row(load_registry(Path(args.registry)), args.source_id) if args.registry else None
    publisher_tree_path = root.parent / "PUBLISHER_TREE.json"
    publisher_tree = read_json(publisher_tree_path) if publisher_tree_path.is_file() else None
    files = sorted(path for path in root.rglob("*") if path.is_file())
    ledger_rows = []
    extension_counts: Counter[str] = Counter()
    representations: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in files:
        relative = path.relative_to(root).as_posix()
        extension = path.suffix.lower() or "<none>"
        extension_counts[extension] += 1
        row = {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "extension": extension}
        ledger_rows.append(row)
        if extension in MODEL_EXTENSIONS:
            representations[representation_key(relative)].append({
                "path": relative, "extension": extension,
                "size_bytes": row["size_bytes"], "sha256": row["sha256"],
            })

    ledger = {
        "schema": SCHEMA_LEDGER, "source_id": args.source_id, "source_root": str(root),
        "file_count": len(ledger_rows), "total_bytes": sum(row["size_bytes"] for row in ledger_rows),
        "extension_counts": dict(sorted(extension_counts.items())), "files": ledger_rows,
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
        chosen = next((by_ext[extension][0] for extension in PREFERRED_AUDIT_FORMATS if by_ext.get(extension)), None)
        record = {
            "representation_key": key, "representations": group,
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
                "subject_id": f"{args.source_id}--{base_id}{suffix}", "source_id": args.source_id,
                "representation_key": key, "path": chosen["path"], "relative_path": chosen["path"],
                "source_sha256": chosen["sha256"], "source_size_bytes": chosen["size_bytes"],
                "source_extension": chosen["extension"], "format_pair_count": len(group),
                "provisional_role": "STRUCTURAL_AUDIT_CANDIDATE",
            })

    publisher_groups = publisher_representation_groups(publisher_tree)
    inventory_payload = {
        "schema": SCHEMA_INVENTORY, "source_id": args.source_id, "source_root": str(root),
        "raw_file_count": len(ledger_rows), "raw_total_bytes": ledger["total_bytes"],
        "raw_ledger_sha256": ledger["ledger_sha256"],
        "model_file_count": sum(1 for row in ledger_rows if row["extension"] in MODEL_EXTENSIONS),
        "representation_group_count": len(representation_rows), "audit_candidate_count": len(candidates),
        "extension_counts": ledger["extension_counts"], "representation_groups": representation_rows,
        "publisher_file_count": publisher_tree.get("file_count") if publisher_tree else len(ledger_rows),
        "publisher_extension_counts": publisher_tree.get("extension_counts") if publisher_tree else ledger["extension_counts"],
        "publisher_tree_digest": publisher_tree.get("tree_digest") if publisher_tree else None,
        "publisher_representation_group_count": len(publisher_groups),
        "publisher_representation_groups": publisher_groups,
    }
    manifest = {
        "schema": SCHEMA_MANIFEST, "source_id": args.source_id, "source_root": str(root),
        "inventory_sha256": digest(inventory_payload), "candidates": candidates,
    }
    if source_meta:
        manifest.update({
            "source_family": source_meta["source_family"], "pack": source_meta["pack"],
            "provenance": source_meta["provenance"], "license": source_meta.get("license"),
            "role": source_meta["role"], "candidate_policy": source_meta.get("candidate_policy"),
            "materialization_eligible": bool(source_meta.get("materialization_eligible")),
        })
        inventory_payload.update({
            "source_family": source_meta["source_family"], "pack": source_meta["pack"],
            "role": source_meta["role"], "license": source_meta.get("license"),
            "provenance": source_meta["provenance"],
        })
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
    inventory_parser.add_argument("--registry")
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
