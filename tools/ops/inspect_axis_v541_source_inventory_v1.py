from __future__ import annotations

"""Read-only inventory for exact historical AXIS V5.4.1 runtime components.

This tool performs no training, packaging, artifact import, ModelRelease mutation,
channel promotion, inference, or Compiler execution. It only finds files with the
three exact historical component names and verifies their SHA-256 identities.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA = "RealSaS.AXISV541SourceInventory.v1"
EXPECTED = {
    "base_checkpoint": {
        "filename": "GEPPETTO_KNIGHT_CHECKPOINT.pt",
        "sha256": "1c124f876858f6f5b5fd0d4639fc7f2b596a25e962d19de8799ecd64ae59b08f",
    },
    "v53_progress": {
        "filename": "AXIS41_TREE_GRAMMAR_PROGRESS_V53.pt",
        "sha256": "5f527a51c676bf9ef2d82691f305c139b365e921867623b7338e0848c7dea55a",
    },
    "v541_composite": {
        "filename": "AXIS41_DISCRETE_XYZ_FROZEN_CAUSAL_COMPOSITE_V541.pt",
        "sha256": "4d484c325c3ae2a9a91df422a820cc5ccdf01c804adef4259ff5398cf30e12fb",
    },
}

PRUNE_DIRS = {
    ".git", ".cache", "node_modules", "__pycache__", ".venv", "venv",
    "$RECYCLE.BIN", "System Volume Information",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidate_roots(raw_roots: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in raw_roots:
        p = Path(raw).expanduser()
        key = str(p)
        if key in seen or not p.exists():
            continue
        seen.add(key)
        out.append(p)
    return out


def inventory(roots: list[Path]) -> dict[str, Any]:
    names = {spec["filename"]: key for key, spec in EXPECTED.items()}
    matches: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for root in roots:
        if root.is_file():
            files = [root]
            for path in files:
                component = names.get(path.name)
                if component is None:
                    continue
                resolved = str(path.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                actual = sha256_file(path)
                matches.append({
                    "component": component,
                    "path": resolved,
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": actual,
                    "exact_hash_match": actual == EXPECTED[component]["sha256"],
                })
            continue

        for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
            dirs[:] = [d for d in dirs if d not in PRUNE_DIRS]
            for filename in files:
                component = names.get(filename)
                if component is None:
                    continue
                path = Path(current) / filename
                resolved = str(path.resolve())
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                actual = sha256_file(path)
                matches.append({
                    "component": component,
                    "path": resolved,
                    "filename": filename,
                    "size_bytes": path.stat().st_size,
                    "sha256": actual,
                    "exact_hash_match": actual == EXPECTED[component]["sha256"],
                })

    exact = {
        component: [m for m in matches if m["component"] == component and m["exact_hash_match"]]
        for component in EXPECTED
    }
    found = sum(bool(v) for v in exact.values())
    if found == len(EXPECTED):
        status = "EXACT_ALL_COMPONENTS_FOUND_ON_HOST"
    elif found:
        status = "PARTIAL_EXACT_COMPONENTS_FOUND_ON_HOST"
    else:
        status = "NO_EXACT_COMPONENTS_FOUND_ON_HOST"

    return {
        "schema": SCHEMA,
        "status": status,
        "expected_components": EXPECTED,
        "roots_scanned": [str(r.resolve()) for r in roots],
        "matches": sorted(matches, key=lambda m: (m["component"], m["path"])),
        "exact_match_counts": {k: len(v) for k, v in exact.items()},
        "all_exact_components_present": found == len(EXPECTED),
        "read_only_inventory": True,
        "training_executed": False,
        "runtime_packaging_executed": False,
        "artifact_import_performed": False,
        "modelrelease_registered": False,
        "model_promotion_performed": False,
        "model_inference_executed": False,
        "compiler_capability_executed": False,
        "scientific_pass_minted": False,
        "product_authority_minted": False,
        "genericity_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    defaults = [
        "/home/monster/realsas_authority",
        "/home/monster",
        "/mnt/g",
        "/mnt/d/RealSaS",
        "/mnt/c/Users/monster/Google Drive",
        "/mnt/c/Users/monster/My Drive",
    ]
    roots = candidate_roots(args.root or defaults)
    receipt = inventory(roots)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
