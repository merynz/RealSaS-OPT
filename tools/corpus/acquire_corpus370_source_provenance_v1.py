#!/usr/bin/env python3
from __future__ import annotations

"""Acquire exact official source-page and CC0 legalcode bytes for Corpus370.

This is provenance preparation, not family admission. Each distinct pack page is
fetched once, exact response bytes are persisted by SHA256, required pack identity
and CC0 markers are checked, and then 370 subjects are bound to those immutable
snapshots through their exact source_id. Live URLs are never consumed later as
scientific authority.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SCHEMA = "RealSaS.Corpus370SourceProvenanceIndex.v1"
SOURCES_SCHEMA = "RealSaS.Corpus370SourcePackProvenanceSources.v1"
ROSTER_SCHEMA = "RealSaS.Corpus370ExactSourceRoster.v1"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
USER_AGENT = "RealSaS-Corpus370-Provenance/1.0 (+research source audit)"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _publisher_host(publisher: str) -> str:
    return {
        "Quaternius": "quaternius.com",
        "Kenney": "kenney.nl",
        "KayKit": "kaylousberg.com",
    }[publisher]


def _fetch(url: str) -> tuple[bytes, str, int, dict[str, str]]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=45) as response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
        require(len(data) <= MAX_RESPONSE_BYTES, "CORPUS370_PROVENANCE_RESPONSE_TOO_LARGE")
        return data, response.geturl(), int(response.status), {str(k): str(v) for k, v in response.headers.items()}


def _persist(root: Path, role: str, data: bytes, suffix: str) -> dict:
    digest = sha256_bytes(data)
    path = root / "objects" / digest / (role + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "CORPUS370_PROVENANCE_EXISTING_BYTES_DRIFT")
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(data)
        require(sha256(temporary) == digest, "CORPUS370_PROVENANCE_STAGE_SHA")
        temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": len(data)}


def _validate_markers(data: bytes, markers: list[str], code: str) -> None:
    text = data.decode("utf-8", "replace").casefold()
    missing = [marker for marker in markers if marker.casefold() not in text]
    require(not missing, code + ":" + ",".join(missing))


def acquire(
    roster_path: Path,
    sources_path: Path,
    output_root: Path,
    *,
    fetcher: Callable[[str], tuple[bytes, str, int, dict[str, str]]] = _fetch,
) -> dict:
    roster_path = roster_path.resolve()
    sources_path = sources_path.resolve()
    output_root = output_root.resolve()
    roster = json.loads(roster_path.read_text())
    sources = json.loads(sources_path.read_text())
    require(roster.get("schema") == ROSTER_SCHEMA and int(roster.get("subject_count", -1)) == 370, "CORPUS370_PROVENANCE_ROSTER")
    require(sources.get("schema") == SOURCES_SCHEMA and sources.get("training_authorized") is False, "CORPUS370_PROVENANCE_SOURCES")
    rows = list(roster.get("subjects") or [])
    require(len(rows) == 370 and len({row["subject_id"] for row in rows}) == 370, "CORPUS370_PROVENANCE_ROSTER_SET")
    declared_packs = dict(sources.get("packs") or {})
    used_packs = {str(row.get("source_id") or "") for row in rows}
    require(used_packs == set(declared_packs), "CORPUS370_PROVENANCE_PACK_SET")

    license_cfg = dict(sources.get("license") or {})
    legal_url = str(license_cfg.get("legalcode_url") or "")
    require(_host(legal_url) == "creativecommons.org", "CORPUS370_PROVENANCE_LEGALCODE_DOMAIN")
    legal_bytes, legal_final, legal_status, legal_headers = fetcher(legal_url)
    require(legal_status == 200 and _host(legal_final) == "creativecommons.org", "CORPUS370_PROVENANCE_LEGALCODE_HTTP")
    _validate_markers(legal_bytes, list(license_cfg.get("required_legalcode_markers") or []), "CORPUS370_PROVENANCE_LEGALCODE_MARKER")
    legal_ref = _persist(output_root, "CC0_1_0_LEGALCODE", legal_bytes, ".html")

    pack_rows = {}
    for pack_id in sorted(declared_packs):
        cfg = dict(declared_packs[pack_id])
        publisher = str(cfg.get("publisher") or "")
        page_url = str(cfg.get("page_url") or "")
        require(_host(page_url) == _publisher_host(publisher), "CORPUS370_PROVENANCE_PAGE_DOMAIN:" + pack_id)
        page_bytes, final_url, status, headers = fetcher(page_url)
        require(status == 200, "CORPUS370_PROVENANCE_PAGE_HTTP:" + pack_id)
        require(_host(final_url) == _publisher_host(publisher), "CORPUS370_PROVENANCE_REDIRECT_DOMAIN:" + pack_id)
        markers = list(cfg.get("required_page_markers") or ())
        require(markers, "CORPUS370_PROVENANCE_PAGE_MARKERS_EMPTY:" + pack_id)
        _validate_markers(page_bytes, markers, "CORPUS370_PROVENANCE_PAGE_MARKER:" + pack_id)
        page_ref = _persist(output_root, "SOURCE_PAGE", page_bytes, ".html")
        pack_rows[pack_id] = {
            "source_pack": pack_id,
            "publisher": publisher,
            "requested_url": page_url,
            "final_url": final_url,
            "http_status": status,
            "page": page_ref,
            "page_markers": markers,
            "page_content_type": headers.get("Content-Type") or headers.get("content-type"),
            "license_name": license_cfg["name"],
            "license_ref": license_cfg["ref"],
            "license_legalcode_url": legal_final,
            "license_legalcode": legal_ref,
            "court_status": "PASS_OFFICIAL_PACK_PAGE_IDENTITY_AND_CC0_MARKERS__EXACT_BYTES_SEALED",
        }

    subject_rows = []
    for row in sorted(rows, key=lambda item: item["subject_id"]):
        pack_id = str(row["source_id"])
        pack = pack_rows[pack_id]
        source_sha = str(row.get("source_sha256") or "")
        require(len(source_sha) == 64, "CORPUS370_PROVENANCE_SOURCE_SHA")
        subject_rows.append(
            {
                "subject_id": row["subject_id"],
                "display_name": Path(str(row.get("source_relative_path") or row["subject_id"])).stem,
                "source_sha256": source_sha,
                "source_pack": pack_id,
                "publisher": pack["publisher"],
                "license_name": pack["license_name"],
                "license_ref": pack["license_ref"],
                "license_bytes": dict(pack["license_legalcode"]),
                "license_legalcode": dict(pack["license_legalcode"]),
                "source_page_snapshot": dict(pack["page"]),
                "source_url_or_locator": pack["final_url"],
                "pack_provenance_court_status": pack["court_status"],
                "training_authorized": False,
            }
        )

    result = {
        "schema": SCHEMA,
        "status": "PASS_370_SOURCE_PROVENANCE_INPUT_PREPARATION__FAMILY_ADMISSION_SEPARATE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "subject_count": len(subject_rows),
        "pack_count": len(pack_rows),
        "packs": pack_rows,
        "subjects": subject_rows,
        "license_legalcode": {
            **legal_ref,
            "requested_url": legal_url,
            "final_url": legal_final,
            "http_status": legal_status,
            "content_type": legal_headers.get("Content-Type") or legal_headers.get("content-type"),
        },
        "parents": {
            "exact_source_roster": {"path": str(roster_path), "sha256": sha256(roster_path)},
            "provenance_sources": {"path": str(sources_path), "sha256": sha256(sources_path)},
        },
        "authority": {
            "live_urls_consumed_by_training": False,
            "exact_snapshot_bytes_required_downstream": True,
            "family_authority_minted": False,
            "split_minted": False,
            "training_authorized": False,
        },
        "training_authorized": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "CORPUS370_SOURCE_PROVENANCE_INDEX_V1.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"schema": result["schema"], "subject_count": result["subject_count"], "pack_count": result["pack_count"], "status": result["status"]}, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-source-roster", type=Path, required=True)
    parser.add_argument("--provenance-sources", type=Path, default=Path(__file__).resolve().parents[2] / "canonical/CORPUS370_SOURCE_PACK_PROVENANCE_V1.json")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.exact_source_roster, args.provenance_sources, args.output_root)


if __name__ == "__main__":
    main()
