"""Recover a mixed Quaternius source through its official itch Standard archive.

This transport is intentionally narrow in authority:

* the complete official Standard archive is downloaded once and hashed;
* PUBLISHER_TREE.json is frozen from the complete extracted publisher archive;
* only the mechanically relevant actor lane plus exact glTF dependencies remains
  under raw/ for downstream SourceInventory/audit/materialization;
* training identity and admission remain owned by the Post139 compiler/sealer.

Use this when per-file public Drive transport is rate-limited.  It does not mint
corpus identity and it never selects Pro/Source/paid itch uploads.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any

from tools.corpus import post139_acquire_resilient_v1 as resilient
from tools.corpus import post139_source_inventory_v1 as source_inventory


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _choose_actor_models(raw: Path, policy: dict[str, Any]) -> tuple[str, list[Path]]:
    preferred = tuple(str(value).lower() for value in policy.get("preferred_formats", source_inventory.PREFERRED_AUDIT_FORMATS))
    files = sorted(path for path in raw.rglob("*") if path.is_file())
    for extension in preferred:
        matches = [
            path
            for path in files
            if path.suffix.lower() == extension
            and resilient._path_matches_actor_policy(path.relative_to(raw).as_posix(), policy)
        ]
        if matches:
            return extension, matches
    raise RuntimeError("POST139_ITCH_ACTOR_LANE_NO_MODEL_FORMAT")


def _dependency_path(raw: Path, model: Path, uri: str) -> Path:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme or uri.startswith("/"):
        raise RuntimeError(f"POST139_ITCH_ACTOR_EXTERNAL_URI_UNSAFE:{model}:{uri}")
    relative = urllib.parse.unquote(uri).replace("\\", "/")
    candidate = (model.parent / relative).resolve()
    raw_resolved = raw.resolve()
    if not candidate.is_relative_to(raw_resolved):
        raise RuntimeError(f"POST139_ITCH_ACTOR_DEPENDENCY_ESCAPE:{model}:{uri}")
    return candidate


def _gltf_dependencies(raw: Path, model: Path) -> list[Path]:
    document = json.loads(model.read_text(encoding="utf-8"))
    result: set[Path] = set()
    for collection in (document.get("buffers", []), document.get("images", [])):
        for row in collection:
            uri = row.get("uri") if isinstance(row, dict) else None
            if not isinstance(uri, str) or not uri or uri.startswith("data:"):
                continue
            dependency = _dependency_path(raw, model, uri)
            if not dependency.is_file():
                raise RuntimeError(
                    f"POST139_ITCH_ACTOR_DEPENDENCY_MISSING:{model.relative_to(raw).as_posix()}:{uri}"
                )
            result.add(dependency)
    return sorted(result)


def acquire(registry_path: Path, pins_path: Path, source_id: str, out: Path) -> None:
    registry = source_inventory.load_registry(registry_path)
    row = source_inventory.source_row(registry, source_id)
    if row.get("acquisition", {}).get("kind") != "QUATERNIUS_OFFICIAL_PAGE_DRIVE":
        raise RuntimeError(f"POST139_ITCH_ACTOR_SOURCE_KIND:{source_id}")

    pins = resilient._load_pins(pins_path)
    policy = pins.get("quaternius_actor_lane_by_source_id", {}).get(source_id)
    if not isinstance(policy, dict):
        raise RuntimeError(f"POST139_ITCH_ACTOR_POLICY_MISSING:{source_id}")

    if out.exists():
        shutil.rmtree(out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    resilient._ACTIVE_PINS = pins
    receipt: dict[str, Any] = {
        "schema": source_inventory.SCHEMA_RECEIPT,
        "source_id": row["id"],
        "source_family": row["source_family"],
        "pack": row["pack"],
        "role": row["role"],
        "license": row["license"],
        "provenance": row["provenance"],
        "acquisition_kind": row["acquisition"]["kind"],
        "status": "UNRESOLVED",
        "raw_root": str(raw),
    }

    # This writes the full official archive publisher tree before we prune raw/.
    resilient._acquire_quaternius_itch(row, out, raw, receipt)
    publisher_tree = _read_json(out / "PUBLISHER_TREE.json")
    archive_sha = receipt.get("archive_sha256")
    archive_size = receipt.get("archive_size_bytes")

    lane, models = _choose_actor_models(raw, policy)
    keep: set[Path] = set(models)
    if lane == ".gltf":
        for model in models:
            keep.update(_gltf_dependencies(raw, model))

    all_files = sorted(path for path in raw.rglob("*") if path.is_file())
    for path in all_files:
        if path not in keep:
            path.unlink()
    _remove_empty_dirs(raw)

    kept_files = sorted(path for path in raw.rglob("*") if path.is_file())
    if not kept_files:
        raise RuntimeError(f"POST139_ITCH_ACTOR_LANE_EMPTY:{source_id}")

    # The transport archive is evidence through its hash/size in the receipt; it is
    # deliberately not retained inside the immutable source workspace artifact.
    (out / "download.zip").unlink(missing_ok=True)

    receipt.update({
        "status": "ACQUIRED_VERIFIED",
        "transport": "QUATERNIUS_OFFICIAL_ITCH_STANDARD_ACTOR_LANE",
        "publisher_file_count": publisher_tree["file_count"],
        "publisher_tree_digest": publisher_tree["tree_digest"],
        "canonical_lane_extension": lane,
        "actor_model_count": len(models),
        "actor_model_paths": [path.relative_to(raw).as_posix() for path in models],
        "downloaded_file_count": len(kept_files),
        "byte_scope": "ITCH_STANDARD_ARCHIVE__PINNED_ACTOR_MODEL_LANE_PLUS_EXACT_GLTF_DEPENDENCIES",
        "archive_sha256": archive_sha,
        "archive_size_bytes": archive_size,
        "raw_file_count": len(kept_files),
        "raw_tree_digest": source_inventory.digest(
            sorted(
                (
                    path.relative_to(raw).as_posix(),
                    path.stat().st_size,
                    source_inventory.sha256_file(path),
                )
                for path in kept_files
            )
        ),
    })
    source_inventory.write_json(out / "ACQUISITION_RECEIPT.json", receipt)
    print(
        "POST139_ITCH_ACTOR_LANE_PASS",
        source_id,
        lane,
        len(models),
        len(kept_files),
        publisher_tree["file_count"],
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
