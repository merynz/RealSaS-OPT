"""Resilient acquisition wrapper for Post139 source recovery.

This is the recovery transport authority used when an immutable source workspace
is missing from the original sweep.  It deliberately keeps semantic authority in
the inventory/audit/compiler/sealer chain while making publisher acquisition
robust to the two classes of source drift observed in the sweep:

* Google Drive files resume into the same workspace with bounded retry/backoff;
* newer Quaternius pages that replaced public Drive links with a free itch.io
  Standard upload are resolved through itch.io's public PWYW download flow;
* Kenney pages whose donation UI hides the final archive locator can use an
  explicit publisher-URL pin recorded in POST139_ACQUISITION_PINS_V1.json;
* zero-byte partials are removed before retry;
* already-complete workspaces are verified and reused;
* acquisition success is accepted only with an ``ACQUIRED_VERIFIED`` receipt.

No retry or fallback can mint corpus/training authority.  Every recovered byte is
hashed again by SourceInventory, the audit, CorpusAuthority and materialization
closure before a consumer can use it.
"""
from __future__ import annotations

import argparse
import html as html_lib
import http.cookiejar
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
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
            # id= is less brittle than rebuilding a uc URL and still supports resume.
            result = gdown.download(
                id=item.id,
                output=str(target),
                quiet=True,
                use_cookies=False,
                resume=True,
            )
            if result and target.is_file() and target.stat().st_size > 0:
                return target
        except Exception as exc:  # transport retry boundary
            last_error = exc
    raise RuntimeError(f"POST139_RESILIENT_GDRIVE_FILE_FAILED:{item.id}:{item.path}:{last_error}")


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
        # itch layouts occasionally put the name in a sibling span instead of strong.name.
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


def _post_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    fields: dict[str, str],
    *,
    referer: str,
    timeout: int = 120,
) -> dict[str, Any]:
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
        # The current publisher pages also expose the same slug through Itch.attachBuyButton.
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

    game_request = urllib.request.Request(game_url, headers={"User-Agent": source_inventory.USER_AGENT})
    game_html = _response_text(opener, game_request)
    csrf = _csrf_token(game_html)
    initial_upload_ids = re.findall(r'data-upload_id=["\'](\d+)', game_html, flags=re.IGNORECASE)
    fields = {"csrf_token": csrf}
    if initial_upload_ids:
        fields["upload_id"] = initial_upload_ids[0]
    download_url_payload = _post_json(opener, f"{game_url}/download_url", fields, referer=game_url)
    download_page_url = download_url_payload.get("url")
    if not isinstance(download_page_url, str) or not download_page_url.startswith("https://"):
        raise RuntimeError(f"POST139_ITCH_DOWNLOAD_PAGE_NOT_RETURNED:{row['id']}")

    page_request = urllib.request.Request(download_page_url, headers={"User-Agent": source_inventory.USER_AGENT, "Referer": game_url})
    download_html = _response_text(opener, page_request)
    download_csrf = _csrf_token(download_html)
    infos = _upload_infos(download_html)
    required_token = str(policy.get("required_upload_name_token", "[Standard]")).lower()
    required_extension = str(policy.get("required_archive_extension", ".zip")).lower()
    rejects = [str(value).lower() for value in policy.get("reject_upload_name_tokens", ["[Pro]", "[Source]"])]
    candidates = [
        (upload_id, name)
        for upload_id, name in infos
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
    acquired_count = sum(1 for path in raw.rglob("*") if path.is_file())
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
        "downloaded_file_count": acquired_count,
        "byte_scope": "FREE_STANDARD_PUBLISHER_ARCHIVE_ALL_BYTES",
    })


def _resilient_quaternius(row: dict[str, Any], out: Path, raw: Path, receipt: dict[str, Any]) -> None:
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
    # Patch transport/resolution only. SourceInventory still owns receipt/raw evidence production.
    source_inventory.download_gdrive_item = _resilient_gdrive_item
    source_inventory.acquire_quaternius = _resilient_quaternius
    source_inventory.kenney_official_zip = _resilient_kenney_official_zip
    args = argparse.Namespace(
        registry=str(registry),
        source_id=source_id,
        out=str(out),
        clean=clean,
    )
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
    parser.add_argument("--force", action="store_true", help="Discard an already verified workspace and reacquire")
    args = parser.parse_args()

    registry = Path(args.registry).resolve()
    pins_path = Path(args.pins).resolve()
    out = Path(args.out).resolve()
    _ACTIVE_PINS = _load_pins(pins_path)
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
            # First attempt starts clean to avoid stale evidence. Subsequent attempts preserve
            # successfully downloaded source bytes so Drive recovery is resumable.
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
