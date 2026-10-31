"""Resilient acquisition wrapper for Post139 source recovery.

Transport is allowed to change; corpus identity is not. This wrapper keeps the
publisher tree complete while recovering only the bytes needed for structural
truth when a mixed Quaternius pack would otherwise waste Drive quota on static
props. Every recovered byte is re-hashed by the downstream inventory/audit/
authority/materialization chain before it can become training authority.
"""
from __future__ import annotations

import argparse
import html as html_lib
import http.cookiejar
import json
import posixpath
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from tools.corpus import post139_source_inventory_v1 as source_inventory

PINS_SCHEMA = "RealSaS.Post139.AcquisitionPins.v1"
DEFAULT_PINS = Path("canonical/POST139_ACQUISITION_PINS_V1.json")
_ORIGINAL_ACQUIRE_QUATERNIUS = source_inventory.acquire_quaternius
_ORIGINAL_KENNEY_OFFICIAL_ZIP = source_inventory.kenney_official_zip
_ACTIVE_PINS: dict[str, Any] = {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pins(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema") != PINS_SCHEMA:
        raise ValueError(f"POST139_ACQUISITION_PINS_SCHEMA_MISMATCH:{payload.get('schema')}")
    return payload


def _verified(out: Path) -> bool:
    receipt = out / "ACQUISITION_RECEIPT.json"
    raw = out / "raw"
    if not receipt.is_file() or not raw.is_dir():
        return False
    try:
        payload = _read_json(receipt)
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ACQUIRED_VERIFIED" and any(p.is_file() for p in raw.rglob("*"))


def _resilient_gdrive_item(gdown: Any, item: Any, raw: Path, attempts: int = 10) -> Path:
    target = raw / item.path
    target.parent.mkdir(parents=True, exist_ok=True)
    delays = [0, 3, 8, 15, 30, 60, 90, 120, 180, 240]
    last_error: Exception | None = None
    for attempt in range(attempts):
        if target.is_file() and target.stat().st_size > 0:
            return target
        if target.exists() and target.stat().st_size == 0:
            target.unlink(missing_ok=True)
        delay = delays[min(attempt, len(delays) - 1)]
        if delay:
            time.sleep(delay)
        try:
            result = gdown.download(
                id=item.id,
                output=str(target),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
            if result and target.is_file() and target.stat().st_size > 0:
                return target
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"POST139_RESILIENT_GDRIVE_FILE_FAILED:{item.id}:{item.path}:{last_error}")


def _actor_policy(row: dict[str, Any]) -> dict[str, Any] | None:
    policies = _ACTIVE_PINS.get("quaternius_actor_lane_by_source_id", {})
    policy = policies.get(row["id"])
    return policy if isinstance(policy, dict) else None


def _path_matches_actor_policy(path: str, policy: dict[str, Any]) -> bool:
    parts = [part.lower() for part in PurePosixPath(path).parts]
    include = [str(value).lower() for value in policy.get("include_path_tokens", [])]
    exclude = [str(value).lower() for value in policy.get("exclude_path_tokens", [])]
    included = bool(include) and any(token in part for token in include for part in parts)
    excluded = any(token in part for token in exclude for part in parts)
    return included and not excluded


def _gltf_dependency_paths(model_path: str, local_path: Path) -> list[str]:
    try:
        document = json.loads(local_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"POST139_ACTOR_GLTF_JSON_INVALID:{model_path}:{exc}") from exc
    parent = PurePosixPath(model_path).parent
    result: list[str] = []
    for collection in (document.get("buffers", []), document.get("images", [])):
        for row in collection:
            uri = row.get("uri") if isinstance(row, dict) else None
            if not isinstance(uri, str) or not uri or uri.startswith("data:"):
                continue
            parsed = urllib.parse.urlparse(uri)
            if parsed.scheme or uri.startswith("/"):
                raise RuntimeError(f"POST139_ACTOR_GLTF_EXTERNAL_URI_UNSAFE:{model_path}:{uri}")
            decoded = urllib.parse.unquote(uri).replace("\\", "/")
            normalized = posixpath.normpath((parent / decoded).as_posix())
            if normalized == ".." or normalized.startswith("../"):
                raise RuntimeError(f"POST139_ACTOR_GLTF_DEPENDENCY_ESCAPE:{model_path}:{uri}")
            result.append(normalized)
    return sorted(set(result))


def _acquire_quaternius_actor_lane(row: dict[str, Any], out: Path, raw: Path, receipt: dict[str, Any], policy: dict[str, Any]) -> None:
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
        raise RuntimeError(f"POST139_ACTOR_LANE_LIST_EMPTY:{row['id']}")

    publisher_rows: list[dict[str, Any]] = []
    extensions: Counter[str] = Counter()
    by_path: dict[str, Any] = {}
    by_casefold: dict[str, Any] = {}
    for item in listing:
        path = PurePosixPath(item.path).as_posix()
        extension = Path(path).suffix.lower() or "<none>"
        extensions[extension] += 1
        by_path[path] = item
        by_casefold[path.casefold()] = item
        publisher_rows.append({
            "file_id": item.id,
            "path": path,
            "extension": extension,
            "byte_state": "PUBLISHER_LISTED_NOT_ACQUIRED",
        })

    preferred = tuple(policy.get("preferred_formats", source_inventory.PREFERRED_AUDIT_FORMATS))
    lane = None
    selected_models: list[Any] = []
    for extension in preferred:
        candidates = [
            item for item in listing
            if Path(item.path).suffix.lower() == extension
            and _path_matches_actor_policy(PurePosixPath(item.path).as_posix(), policy)
        ]
        if candidates:
            lane = extension
            selected_models = sorted(candidates, key=lambda item: PurePosixPath(item.path).as_posix())
            break
    if lane is None:
        raise RuntimeError(f"POST139_ACTOR_LANE_NO_MODEL_FORMAT:{row['id']}")

    expected = int(policy.get("expected_model_count", 0))
    selected_paths = [PurePosixPath(item.path).as_posix() for item in selected_models]
    if expected and len(selected_models) != expected:
        raise RuntimeError(
            f"POST139_ACTOR_LANE_MODEL_COUNT:{row['id']}:expected={expected}:observed={len(selected_models)}:paths={selected_paths}"
        )

    selected_by_id: dict[str, Any] = {item.id: item for item in selected_models}
    for item in selected_models:
        model_path = PurePosixPath(item.path).as_posix()
        local_model = _resilient_gdrive_item(gdown, item, raw)
        if lane == ".gltf":
            for dependency_path in _gltf_dependency_paths(model_path, local_model):
                dependency = by_path.get(dependency_path) or by_casefold.get(dependency_path.casefold())
                if dependency is None:
                    raise RuntimeError(
                        f"POST139_ACTOR_LANE_DEPENDENCY_NOT_LISTED:{row['id']}:{model_path}:{dependency_path}"
                    )
                selected_by_id[dependency.id] = dependency

    # Download only exact dependencies referenced by the selected actor glTFs.
    for item in sorted(selected_by_id.values(), key=lambda value: PurePosixPath(value.path).as_posix()):
        _resilient_gdrive_item(gdown, item, raw)

    acquired_paths: dict[str, Path] = {}
    for item in selected_by_id.values():
        path = raw / item.path
        if path.is_file() and path.stat().st_size > 0:
            acquired_paths[item.id] = path
    for entry in publisher_rows:
        path = acquired_paths.get(entry["file_id"])
        if path is not None:
            entry.update({
                "byte_state": "ACQUIRED_HASHED",
                "size_bytes": path.stat().st_size,
                "sha256": source_inventory.sha256_file(path),
            })

    stable_tree_rows = [
        {"file_id": entry["file_id"], "path": entry["path"], "extension": entry["extension"]}
        for entry in publisher_rows
    ]
    publisher_tree = {
        "schema": source_inventory.SCHEMA_PUBLISHER_TREE,
        "source_id": row["id"],
        "transport": "QUATERNIUS_OFFICIAL_PUBLIC_DRIVE_ACTOR_LANE",
        "resolved_drive_folder_id": folder_id,
        "file_count": len(publisher_rows),
        "extension_counts": dict(sorted(extensions.items())),
        "canonical_lane_extension": lane,
        "actor_model_count": len(selected_models),
        "actor_model_paths": selected_paths,
        "acquired_file_count": len(acquired_paths),
        "files": publisher_rows,
    }
    publisher_tree["tree_digest"] = source_inventory.digest(stable_tree_rows)
    source_inventory.write_json(out / "PUBLISHER_TREE.json", publisher_tree)
    receipt.update({
        "status": "ACQUIRED_VERIFIED",
        "publisher_page_raw_url": page_url,
        "resolved_drive_folder_id": folder_id,
        "resolved_url": drive_url,
        "transport": "QUATERNIUS_OFFICIAL_PUBLIC_DRIVE_ACTOR_LANE",
        "publisher_file_count": len(publisher_rows),
        "publisher_tree_digest": publisher_tree["tree_digest"],
        "canonical_lane_extension": lane,
        "actor_model_count": len(selected_models),
        "actor_model_paths": selected_paths,
        "downloaded_file_count": len(acquired_paths),
        "byte_scope": "PINNED_ACTOR_MODEL_LANE_PLUS_EXACT_GLTF_DEPENDENCIES",
    })


def _response_text(opener: urllib.request.OpenerDirector, request: urllib.request.Request, timeout: int = 120) -> str:
    with opener.open(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _csrf_token(document: str) -> str:
    patterns = (
        r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)',
        r'csrf_token["\'][^>]*value=["\']([^"\']+)',
        r'value=["\']([^"\']+)["\'][^>]*name=["\']csrf_token["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, document, flags=re.IGNORECASE)
        if match:
            return html_lib.unescape(match.group(1))
    raise RuntimeError("POST139_ITCH_CSRF_NOT_FOUND")


def _upload_infos(document: str) -> list[tuple[str, str]]:
    matches = re.findall(
        r'<a[^>]*data-upload_id=["\'](\d+)["\'][^>]*>[\s\S]*?<strong[^>]*class=["\'][^"\']*\bname\b[^"\']*["\'][^>]*>([^<]+)</strong>',
        document,
        flags=re.IGNORECASE,
    )
    if not matches:
        blocks = re.findall(r'<a[^>]*data-upload_id=["\'](\d+)["\'][^>]*>([\s\S]*?)</a>', document, flags=re.IGNORECASE)
        for upload_id, block in blocks:
            text = re.sub(r"<[^>]+>", " ", block)
            text = " ".join(html_lib.unescape(text).split())
            if text:
                matches.append((upload_id, text))
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for upload_id, name in matches:
        if upload_id in seen:
            continue
        seen.add(upload_id)
        result.append((upload_id, " ".join(html_lib.unescape(name).split())))
    return result


def _post_json(opener: urllib.request.OpenerDirector, url: str, fields: dict[str, str], *, referer: str, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(fields).encode("utf-8"),
        headers={
            "User-Agent": source_inventory.USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": referer,
        },
        method="POST",
    )
    text = _response_text(opener, request, timeout=timeout)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"POST139_ITCH_JSON_INVALID:{url}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"POST139_ITCH_JSON_NOT_OBJECT:{url}")
    return payload


def _download_with_opener(opener: urllib.request.OpenerDirector, url: str, output: Path, *, referer: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": source_inventory.USER_AGENT, "Referer": referer})
    with opener.open(request, timeout=300) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("POST139_ITCH_ARCHIVE_EMPTY")


def _itch_game_url_from_quaternius_page(row: dict[str, Any]) -> tuple[str, str]:
    acq = row["acquisition"]
    owner, name = acq["publisher_repository"].split("/", 1)
    page_url = f"https://raw.githubusercontent.com/{owner}/{name}/main/{acq['page_path']}"
    document = source_inventory.fetch_text(page_url)
    urls = re.findall(r"https://quaternius\.itch\.io/[a-z0-9-]+", document, flags=re.IGNORECASE)
    urls = list(dict.fromkeys(url.rstrip("/')\";, ") for url in urls))
    if len(urls) != 1:
        users = re.findall(r'user\s*:\s*["\']([^"\']+)["\']', document, flags=re.IGNORECASE)
        games = re.findall(r'game\s*:\s*["\']([^"\']+)["\']', document, flags=re.IGNORECASE)
        if len(set(users)) == 1 and len(set(games)) == 1:
            urls = [f"https://{users[0]}.itch.io/{games[0]}"]
    if len(urls) != 1:
        raise ValueError(f"POST139_QUATERNIUS_ITCH_RESOLUTION:{row['id']}:{urls}")
    return urls[0], page_url


def _acquire_quaternius_itch(row: dict[str, Any], out: Path, raw: Path, receipt: dict[str, Any]) -> None:
    policy = _ACTIVE_PINS.get("quaternius_itch_fallback", {})
    if not policy.get("allowed"):
        raise RuntimeError("POST139_QUATERNIUS_ITCH_FALLBACK_DISABLED")
    game_url, publisher_page_url = _itch_game_url_from_quaternius_page(row)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    game_html = _response_text(opener, urllib.request.Request(game_url, headers={"User-Agent": source_inventory.USER_AGENT}))
    csrf = _csrf_token(game_html)
    initial_upload_ids = re.findall(r'data-upload_id=["\'](\d+)', game_html, flags=re.IGNORECASE)
    fields = {"csrf_token": csrf}
    if initial_upload_ids:
        fields["upload_id"] = initial_upload_ids[0]
    payload = _post_json(opener, f"{game_url}/download_url", fields, referer=game_url)
    download_page_url = payload.get("url")
    if not isinstance(download_page_url, str) or not download_page_url.startswith("https://"):
        raise RuntimeError(f"POST139_ITCH_DOWNLOAD_PAGE_NOT_RETURNED:{row['id']}")
    download_html = _response_text(
        opener,
        urllib.request.Request(download_page_url, headers={"User-Agent": source_inventory.USER_AGENT, "Referer": game_url}),
    )
    download_csrf = _csrf_token(download_html)
    infos = _upload_infos(download_html)
    required_token = str(policy.get("required_upload_name_token", "[Standard]")).lower()
    required_extension = str(policy.get("required_archive_extension", ".zip")).lower()
    rejects = [str(value).lower() for value in policy.get("reject_upload_name_tokens", ["[Pro]", "[Source]"])]
    candidates = [
        (upload_id, name) for upload_id, name in infos
        if required_token in name.lower()
        and name.lower().endswith(required_extension)
        and not any(token in name.lower() for token in rejects)
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"POST139_ITCH_STANDARD_UPLOAD_RESOLUTION:{row['id']}:{candidates}:all={infos}")
    upload_id, upload_name = candidates[0]
    file_payload = _post_json(
        opener,
        f"{game_url}/file/{upload_id}",
        {"csrf_token": download_csrf},
        referer=download_page_url,
    )
    cdn_url = file_payload.get("url")
    if not isinstance(cdn_url, str) or not cdn_url.startswith("https://"):
        raise RuntimeError(f"POST139_ITCH_CDN_URL_NOT_RETURNED:{row['id']}:{upload_id}")
    archive = out / "download.zip"
    _download_with_opener(opener, cdn_url, archive, referer=download_page_url)
    if not __import__("zipfile").is_zipfile(archive):
        raise RuntimeError(f"POST139_ITCH_ARCHIVE_NOT_ZIP:{row['id']}:{upload_name}")
    archive_sha = source_inventory.sha256_file(archive)
    archive_size = archive.stat().st_size
    source_inventory.safe_extract_zip(archive, raw)
    publisher_tree = source_inventory.local_publisher_tree(raw, row["id"], "QUATERNIUS_OFFICIAL_ITCH_STANDARD")
    source_inventory.write_json(out / "PUBLISHER_TREE.json", publisher_tree)
    receipt.update({
        "status": "ACQUIRED_VERIFIED",
        "publisher_page_raw_url": publisher_page_url,
        "resolved_url": game_url,
        "transport": "QUATERNIUS_OFFICIAL_ITCH_STANDARD",
        "itch_game_url": game_url,
        "itch_upload_id": upload_id,
        "itch_upload_name": upload_name,
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "publisher_file_count": publisher_tree["file_count"],
        "publisher_tree_digest": publisher_tree["tree_digest"],
        "downloaded_file_count": publisher_tree["file_count"],
        "byte_scope": "FREE_STANDARD_PUBLISHER_ARCHIVE_ALL_BYTES",
    })


def _resilient_quaternius(row: dict[str, Any], out: Path, raw: Path, receipt: dict[str, Any]) -> None:
    policy = _actor_policy(row)
    if policy is not None:
        return _acquire_quaternius_actor_lane(row, out, raw, receipt, policy)
    try:
        return _ORIGINAL_ACQUIRE_QUATERNIUS(row, out, raw, receipt)
    except ValueError as exc:
        if not str(exc).startswith("POST139_QUATERNIUS_DRIVE_RESOLUTION:"):
            raise
        print("POST139_QUATERNIUS_DRIVE_REPLACED_BY_ITCH", row["id"])
        return _acquire_quaternius_itch(row, out, raw, receipt)


def _resilient_kenney_official_zip(row: dict[str, Any]) -> tuple[str, str]:
    try:
        return _ORIGINAL_KENNEY_OFFICIAL_ZIP(row)
    except ValueError as exc:
        if not str(exc).startswith("POST139_KENNEY_ZIP_RESOLUTION:"):
            raise
        pins = _ACTIVE_PINS.get("kenney_direct_zip_by_source_id", {})
        pinned = pins.get(row["id"])
        if not isinstance(pinned, str) or not pinned.startswith("https://kenney.nl/media/pages/assets/") or not pinned.lower().endswith(".zip"):
            raise
        print("POST139_KENNEY_PINNED_PUBLISHER_ZIP", row["id"], pinned)
        return pinned, row["acquisition"]["page_url"]


def acquire_once(registry: Path, source_id: str, out: Path, *, clean: bool) -> None:
    source_inventory.download_gdrive_item = _resilient_gdrive_item
    source_inventory.acquire_quaternius = _resilient_quaternius
    source_inventory.kenney_official_zip = _resilient_kenney_official_zip
    args = argparse.Namespace(registry=str(registry), source_id=source_id, out=str(out), clean=clean)
    source_inventory.acquire(args)


def main() -> int:
    global _ACTIVE_PINS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pins", default=str(DEFAULT_PINS))
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--base-delay-seconds", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    registry = Path(args.registry).resolve()
    out = Path(args.out).resolve()
    _ACTIVE_PINS = _load_pins(Path(args.pins).resolve())
    if not args.force and _verified(out):
        print("POST139_RESILIENT_ACQUIRE_REUSE", args.source_id)
        return 0

    last_error: Exception | None = None
    for attempt in range(max(1, args.attempts)):
        if attempt:
            delay = args.base_delay_seconds * (2 ** (attempt - 1))
            print("POST139_RESILIENT_ACQUIRE_RETRY", args.source_id, attempt + 1, delay)
            time.sleep(delay)
        try:
            acquire_once(registry, args.source_id, out, clean=(attempt == 0))
            if not _verified(out):
                raise RuntimeError("POST139_RESILIENT_RECEIPT_NOT_VERIFIED")
            receipt = _read_json(out / "ACQUISITION_RECEIPT.json")
            print(
                "POST139_RESILIENT_ACQUIRE_PASS",
                args.source_id,
                receipt.get("acquisition_kind"),
                receipt.get("transport", receipt.get("acquisition_kind")),
                receipt.get("downloaded_file_count", "n/a"),
            )
            return 0
        except Exception as exc:
            last_error = exc
            print("POST139_RESILIENT_ACQUIRE_ATTEMPT_FAILED", args.source_id, attempt + 1, repr(exc))
    print("POST139_RESILIENT_ACQUIRE_FAILED", args.source_id, repr(last_error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
