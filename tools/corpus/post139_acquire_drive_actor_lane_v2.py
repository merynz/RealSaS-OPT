"""Acquire mixed Quaternius packs through the official public Drive actor lane.

The publisher page remains the provenance authority.  Google Drive is used in two
separate ways:

1. gdown enumerates the complete public folder without downloading bytes;
2. selected mechanically-relevant model files and exact glTF dependencies are
   streamed by file id from Google's usercontent endpoint.

This avoids gdown's per-file public-link resolver, which is vulnerable to access
frequency failures on large mixed packs.  The complete publisher tree is still
sealed; raw/ contains only the selected truth superset.  Corpus identity,
admission and training denominators are decided later by audit/compiler/sealer.
"""
from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
import shutil
import urllib.parse
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from tools.corpus import post139_acquire_resilient_v1 as resilient
from tools.corpus import post139_source_inventory_v1 as source_inventory


USERCONTENT_URLS = (
    "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
    "https://drive.google.com/uc?export=download&confirm=t&id={file_id}",
)


def _write_stream(response: requests.Response, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.unlink(missing_ok=True)
    with partial.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError("POST139_DRIVE_DIRECT_EMPTY_RESPONSE")
    partial.replace(target)


def _looks_like_file(response: requests.Response) -> bool:
    disposition = response.headers.get("content-disposition", "")
    if "attachment" in disposition.lower() or "filename=" in disposition.lower():
        return True
    content_type = response.headers.get("content-type", "").lower()
    return "text/html" not in content_type and "application/xhtml" not in content_type


def _confirm_url(document: str, base_url: str) -> str | None:
    patterns = (
        r'href="([^"]*(?:confirm|uuid)=[^"]+)"',
        r"href='([^']*(?:confirm|uuid)=[^']+)'",
        r'action="([^"]+/download\?[^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, document, flags=re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(base_url, html.unescape(match.group(1)))
    return None


def _error_hint(document: str) -> str:
    text = re.sub(r"<[^>]+>", " ", document)
    text = " ".join(html.unescape(text).split())
    return text[:600]


def download_drive_file(file_id: str, target: Path, *, attempts: int = 2) -> Path:
    """Stream one public Drive file without relying on gdown's link resolver."""
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.unlink(missing_ok=True)
    errors: list[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": source_inventory.USER_AGENT})
    for template in USERCONTENT_URLS:
        url = template.format(file_id=urllib.parse.quote(file_id, safe=""))
        for attempt in range(max(1, attempts)):
            try:
                response = session.get(url, stream=True, timeout=(20, 180), allow_redirects=True)
                response.raise_for_status()
                if _looks_like_file(response):
                    _write_stream(response, target)
                    return target

                # HTML interstitials are small; read them and follow an explicit
                # Google confirmation link if one exists.
                document = response.text
                confirm = _confirm_url(document, response.url)
                if confirm:
                    confirmed = session.get(confirm, stream=True, timeout=(20, 180), allow_redirects=True)
                    confirmed.raise_for_status()
                    if _looks_like_file(confirmed):
                        _write_stream(confirmed, target)
                        return target
                    errors.append(f"confirm-html:{_error_hint(confirmed.text)}")
                else:
                    errors.append(f"html:{_error_hint(document)}")
            except Exception as exc:  # transport boundary, bounded above
                errors.append(f"{type(exc).__name__}:{exc}")
    raise RuntimeError(f"POST139_DRIVE_DIRECT_FILE_FAILED:{file_id}:{target.name}:{errors[-4:]}")


def _path_matches(path: str, policy: dict[str, Any]) -> bool:
    return resilient._path_matches_actor_policy(path, policy)


def _gltf_dependency_paths(model_path: str, local_path: Path) -> list[str]:
    document = json.loads(local_path.read_text(encoding="utf-8"))
    parent = PurePosixPath(model_path).parent
    result: set[str] = set()
    for collection in (document.get("buffers", []), document.get("images", [])):
        for row in collection:
            uri = row.get("uri") if isinstance(row, dict) else None
            if not isinstance(uri, str) or not uri or uri.startswith("data:"):
                continue
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme or uri.startswith("/"):
                raise RuntimeError(f"POST139_DRIVE_ACTOR_GLTF_EXTERNAL_URI_UNSAFE:{model_path}:{uri}")
            decoded = urllib.parse.unquote(uri).replace("\\", "/")
            normalized = posixpath.normpath((parent / decoded).as_posix())
            if normalized == ".." or normalized.startswith("../"):
                raise RuntimeError(f"POST139_DRIVE_ACTOR_GLTF_DEPENDENCY_ESCAPE:{model_path}:{uri}")
            result.add(normalized)
    return sorted(result)


def acquire(registry_path: Path, pins_path: Path, source_id: str, out: Path) -> None:
    registry = source_inventory.load_registry(registry_path)
    row = source_inventory.source_row(registry, source_id)
    if row.get("acquisition", {}).get("kind") != "QUATERNIUS_OFFICIAL_PAGE_DRIVE":
        raise RuntimeError(f"POST139_DRIVE_ACTOR_SOURCE_KIND:{source_id}")
    pins = resilient._load_pins(pins_path)
    policy = pins.get("quaternius_actor_lane_by_source_id", {}).get(source_id)
    if not isinstance(policy, dict):
        raise RuntimeError(f"POST139_DRIVE_ACTOR_POLICY_MISSING:{source_id}")

    if out.exists():
        shutil.rmtree(out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    folder_id, page_url = source_inventory.quaternius_drive_folder(row)
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError("POST139_GDOWN_REQUIRED") from exc
    drive_url = f"https://drive.google.com/drive/folders/{folder_id}"
    listing = gdown.download_folder(
        url=drive_url,
        output=str(raw),
        quiet=True,
        use_cookies=False,
        remaining_ok=True,
        skip_download=True,
    )
    if not listing:
        raise RuntimeError(f"POST139_DRIVE_ACTOR_LIST_EMPTY:{source_id}")

    publisher_rows: list[dict[str, Any]] = []
    extension_counts: Counter[str] = Counter()
    by_path: dict[str, Any] = {}
    by_casefold: dict[str, Any] = {}
    for item in listing:
        relative = PurePosixPath(item.path).as_posix()
        extension = Path(relative).suffix.lower() or "<none>"
        extension_counts[extension] += 1
        by_path[relative] = item
        by_casefold[relative.casefold()] = item
        publisher_rows.append({
            "file_id": item.id,
            "path": relative,
            "extension": extension,
            "byte_state": "PUBLISHER_LISTED_NOT_ACQUIRED",
        })

    lane: str | None = None
    selected_models: list[Any] = []
    for extension in tuple(policy.get("preferred_formats", source_inventory.PREFERRED_AUDIT_FORMATS)):
        matches = [
            item for item in listing
            if Path(item.path).suffix.lower() == extension
            and _path_matches(PurePosixPath(item.path).as_posix(), policy)
        ]
        if matches:
            lane = extension
            selected_models = sorted(matches, key=lambda item: PurePosixPath(item.path).as_posix())
            break
    if lane is None or not selected_models:
        raise RuntimeError(f"POST139_DRIVE_ACTOR_NO_MODEL_LANE:{source_id}")

    selected_by_id: dict[str, Any] = {item.id: item for item in selected_models}
    for item in selected_models:
        relative = PurePosixPath(item.path).as_posix()
        local_model = download_drive_file(item.id, raw / relative)
        if lane == ".gltf":
            for dependency_path in _gltf_dependency_paths(relative, local_model):
                dependency = by_path.get(dependency_path) or by_casefold.get(dependency_path.casefold())
                if dependency is None:
                    raise RuntimeError(
                        f"POST139_DRIVE_ACTOR_DEPENDENCY_NOT_LISTED:{source_id}:{relative}:{dependency_path}"
                    )
                selected_by_id[dependency.id] = dependency

    # Fetch only dependencies not already fetched with the model lane.
    for item in sorted(selected_by_id.values(), key=lambda value: PurePosixPath(value.path).as_posix()):
        download_drive_file(item.id, raw / PurePosixPath(item.path).as_posix())

    acquired: dict[str, Path] = {}
    for item in selected_by_id.values():
        path = raw / PurePosixPath(item.path).as_posix()
        if path.is_file() and path.stat().st_size > 0:
            acquired[item.id] = path
    for entry in publisher_rows:
        path = acquired.get(entry["file_id"])
        if path is not None:
            entry.update({
                "byte_state": "ACQUIRED_HASHED",
                "size_bytes": path.stat().st_size,
                "sha256": source_inventory.sha256_file(path),
            })

    stable_tree_rows = [
        {"file_id": row_["file_id"], "path": row_["path"], "extension": row_["extension"]}
        for row_ in publisher_rows
    ]
    tree = {
        "schema": source_inventory.SCHEMA_PUBLISHER_TREE,
        "source_id": source_id,
        "transport": "QUATERNIUS_OFFICIAL_PUBLIC_DRIVE_DIRECT_ACTOR_LANE",
        "resolved_drive_folder_id": folder_id,
        "file_count": len(publisher_rows),
        "extension_counts": dict(sorted(extension_counts.items())),
        "canonical_lane_extension": lane,
        "actor_model_count": len(selected_models),
        "actor_model_paths": [PurePosixPath(item.path).as_posix() for item in selected_models],
        "acquired_file_count": len(acquired),
        "files": publisher_rows,
    }
    tree["tree_digest"] = source_inventory.digest(stable_tree_rows)
    source_inventory.write_json(out / "PUBLISHER_TREE.json", tree)

    raw_files = sorted(path for path in raw.rglob("*") if path.is_file())
    receipt = {
        "schema": source_inventory.SCHEMA_RECEIPT,
        "source_id": source_id,
        "source_family": row["source_family"],
        "pack": row["pack"],
        "role": row["role"],
        "license": row["license"],
        "provenance": row["provenance"],
        "acquisition_kind": row["acquisition"]["kind"],
        "status": "ACQUIRED_VERIFIED",
        "raw_root": str(raw),
        "publisher_page_raw_url": page_url,
        "resolved_drive_folder_id": folder_id,
        "resolved_url": drive_url,
        "transport": tree["transport"],
        "publisher_file_count": tree["file_count"],
        "publisher_tree_digest": tree["tree_digest"],
        "canonical_lane_extension": lane,
        "actor_model_count": len(selected_models),
        "actor_model_paths": tree["actor_model_paths"],
        "downloaded_file_count": len(raw_files),
        "byte_scope": "PINNED_ACTOR_MODEL_LANE_PLUS_EXACT_GLTF_DEPENDENCIES__DIRECT_FILE_ID_STREAM",
        "raw_file_count": len(raw_files),
        "raw_tree_digest": source_inventory.digest(
            sorted(
                (
                    path.relative_to(raw).as_posix(),
                    path.stat().st_size,
                    source_inventory.sha256_file(path),
                )
                for path in raw_files
            )
        ),
    }
    source_inventory.write_json(out / "ACQUISITION_RECEIPT.json", receipt)
    print(
        "POST139_DRIVE_ACTOR_LANE_PASS",
        source_id,
        lane,
        len(selected_models),
        len(raw_files),
        tree["file_count"],
        receipt["raw_tree_digest"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--pins", default=str(resilient.DEFAULT_PINS))
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    acquire(
        Path(args.registry).resolve(),
        Path(args.pins).resolve(),
        args.source_id,
        Path(args.out).resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
