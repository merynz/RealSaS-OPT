from __future__ import annotations

"""Resolve the exact current Mage Runtime-v4 smoke inputs from mounted artifact roots.

This resolver is deliberately filename-light and hash-strict. It never selects
"latest", never accepts partial SHA prefixes, and never opens giant product graph
JSON files merely to discover the current mechanical authority.

Selection rules:
- P1Q / foreground / component-assembly directories are anchored by exact manifest
  filename + exact SHA-256.
- corrected FIT2 surface, G22 skeleton, FIT2 skin, and V0..V7 cameras are selected by
  exact file SHA-256.
- only bounded JSON candidates are hashed for anonymous authority lookup.
- duplicate byte-identical aliases are permitted and reported; distinct hashes never
  compete for the same authority.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

SCHEMA = "RealSaS.MageRuntimeV4ExactInputResolution.v1"

EXPECTED = {
    "p1q_manifest": {
        "filename": "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json",
        "sha256": "c945377822561319c40e6e5774235ddb5a586cb230599c0939e100ccf32b14e4",
    },
    "foreground_manifest": {
        "filename": "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json",
        "sha256": "5a331c2a73f2fa8b425dacdde1d4890fa55873d01856fe1d65153b3b03d797bf",
    },
    "assembly_manifest": {
        "filename": "FIT2_COMPONENT_ASSEMBLY_MANIFEST.json",
        "sha256": "b678f04b7d4248709f22de8a6397ef498cb7d4c4b78ef1954f2a42cc9125592f",
    },
    "fit2_surface": {
        "filename": "CURRENT_FIT2_RIGGING_SURFACE_IR.json",
        "sha256": "170b8e4712fd78ef0721462f19f80ccb94eb008206fc6a7061908bfdc36f202f",
    },
    "skeleton": {
        "filename": None,
        "sha256": "319ae46d94450d97dd6f40f21d571ac9432968104e7a517af00ace442d05b9fe",
    },
    "fit2_skin": {
        "filename": None,
        "sha256": "722fdeb60fc32aa09e2296adbdc0e589de0379bfd49f6239f214330517f01ddd",
    },
}
CAMERA_SHA256 = (
    "73004e0654b576e0c51893af544e0af8fcc4e613ce07ea9884272285d55cd541",
    "bdc172a4aff332f956d1403e36b2f8684b68059fdc82f9efddf35d05a6d9b4d4",
    "3c2bbc44ef9005b4a545a3381205a5d6a92af15b4791c9075071b8cad02a1a6c",
    "24b2f115d908422d885f85e956fcc36ac78fd0c90b503f698caa62febc2b9c4d",
    "5bf00783d6509c2ca142e05ef705d5cdb5df17ad248b782d2fe8cf8a297bee39",
    "7ee3e50739318eeb122b5b0ec67260dd32e21d949398f48c408a6c239e5c89fe",
    "daa19fa58ff602977d64b720c4198956809855149d814df487c7762a963f1eec",
    "68f51fbfce4c31f94281e1569d74b44609435285668f8a8b1b278e76db6ea53f",
)

# Anonymous skeleton/skin/camera lookup should never hash giant CanonicalPuppetGraph
# files. The corrected surface is larger, but it is name-anchored above.
MAX_ANONYMOUS_JSON_BYTES = 64 * 1024 * 1024
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "node_modules",
    ".Trash",
    ".shortcut-targets-by-id",
}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _walk_files(roots: Iterable[Path]):
    seen = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file():
            key = str(root)
            if key not in seen:
                seen.add(key)
                yield root
            continue
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                rows = sorted(directory.iterdir(), key=lambda p: p.name)
            except (OSError, PermissionError):
                continue
            for path in rows:
                try:
                    if path.is_dir():
                        if path.name not in SKIP_DIR_NAMES:
                            stack.append(path)
                    elif path.is_file():
                        key = str(path.resolve())
                        if key not in seen:
                            seen.add(key)
                            yield path.resolve()
                except (OSError, PermissionError):
                    continue


def _name_anchors(files: tuple[Path, ...]):
    by_name: dict[str, list[Path]] = {}
    wanted = {
        row["filename"]
        for row in EXPECTED.values()
        if row.get("filename")
    }
    for path in files:
        if path.name in wanted:
            by_name.setdefault(path.name, []).append(path)
    return by_name


def _choose_exact(paths: Iterable[Path], expected_sha: str, label: str) -> tuple[Path, tuple[Path, ...]]:
    matches = []
    for path in paths:
        try:
            if _sha(path) == expected_sha:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        raise RuntimeError(f"MAGE_V4_INPUT_NOT_FOUND:{label}:{expected_sha}")
    matches = sorted(set(matches), key=lambda p: str(p))
    return matches[0], tuple(matches)


def resolve(search_roots: Iterable[Path]) -> dict:
    roots = tuple(Path(x).expanduser().resolve() for x in search_roots)
    files = tuple(_walk_files(roots))
    anchors = _name_anchors(files)

    resolved: dict[str, Path] = {}
    aliases: dict[str, tuple[Path, ...]] = {}

    for label in ("p1q_manifest", "foreground_manifest", "assembly_manifest", "fit2_surface"):
        spec = EXPECTED[label]
        path, rows = _choose_exact(
            anchors.get(str(spec["filename"]), ()),
            str(spec["sha256"]),
            label,
        )
        resolved[label] = path
        aliases[label] = rows

    anonymous_targets = {
        str(EXPECTED["skeleton"]["sha256"]): "skeleton",
        str(EXPECTED["fit2_skin"]["sha256"]): "fit2_skin",
        **{digest: f"camera_v{i}" for i, digest in enumerate(CAMERA_SHA256)},
    }
    remaining = set(anonymous_targets)
    digest_paths: dict[str, list[Path]] = {digest: [] for digest in remaining}
    for path in files:
        if not remaining:
            break
        if path.suffix.lower() != ".json":
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > MAX_ANONYMOUS_JSON_BYTES:
            continue
        digest = _sha(path)
        if digest in digest_paths:
            digest_paths[digest].append(path)
            remaining.discard(digest)

    if remaining:
        missing = {anonymous_targets[d]: d for d in sorted(remaining)}
        raise RuntimeError("MAGE_V4_ANONYMOUS_INPUTS_NOT_FOUND:" + json.dumps(missing, sort_keys=True))

    for digest, label in anonymous_targets.items():
        rows = tuple(sorted(set(digest_paths[digest]), key=lambda p: str(p)))
        if not rows:
            raise RuntimeError(f"MAGE_V4_INPUT_NOT_FOUND:{label}:{digest}")
        resolved[label] = rows[0]
        aliases[label] = rows

    p1q_dir = resolved["p1q_manifest"].parent
    foreground_dir = resolved["foreground_manifest"].parent
    assembly_dir = resolved["assembly_manifest"].parent
    cameras = tuple(resolved[f"camera_v{i}"] for i in range(8))

    # Directory completeness is part of admission: resolve only a manifest whose
    # referenced runtime source files are physically present.
    p1q_required = [
        p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_IR.json"
        for i in range(8)
    ] + [
        p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_SKIN_IR.json"
        for i in range(8)
    ] + [
        p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_APPEARANCE_IR.json"
        for i in range(8)
    ]
    foreground_required = [
        foreground_dir / f"V{i}_MAGE_PRODUCT_ATLAS.png"
        for i in range(8)
    ]
    assembly_required = [
        assembly_dir / "FIT2_QUALIFIED_COMPONENT_ASSEMBLY.json",
    ]
    missing_files = [
        str(path)
        for path in (*p1q_required, *foreground_required, *assembly_required)
        if not path.is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "MAGE_V4_RESOLVED_DIRECTORY_INCOMPLETE:"
            + json.dumps(missing_files[:32], sort_keys=True)
        )

    result = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED",
        "search_roots": [str(path) for path in roots],
        "p1q_dir": str(p1q_dir),
        "foreground_dir": str(foreground_dir),
        "assembly_dir": str(assembly_dir),
        "fit2_surface": str(resolved["fit2_surface"]),
        "skeleton": str(resolved["skeleton"]),
        "fit2_skin": str(resolved["fit2_skin"]),
        "cameras": [str(path) for path in cameras],
        "expected_hashes": {
            "p1q_manifest": EXPECTED["p1q_manifest"]["sha256"],
            "foreground_manifest": EXPECTED["foreground_manifest"]["sha256"],
            "assembly_manifest": EXPECTED["assembly_manifest"]["sha256"],
            "fit2_surface": EXPECTED["fit2_surface"]["sha256"],
            "skeleton": EXPECTED["skeleton"]["sha256"],
            "fit2_skin": EXPECTED["fit2_skin"]["sha256"],
            "cameras": list(CAMERA_SHA256),
        },
        "aliases": {
            label: [str(path) for path in rows]
            for label, rows in sorted(aliases.items())
        },
        "giant_product_graph_scanned": False,
        "anonymous_json_size_ceiling_bytes": MAX_ANONYMOUS_JSON_BYTES,
    }
    return result



def validate_resolution(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise RuntimeError("MAGE_V4_RESOLUTION_SCHEMA_INVALID")
    if value.get("status") != "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED":
        raise RuntimeError("MAGE_V4_RESOLUTION_STATUS_INVALID")

    expected_hashes = dict(value.get("expected_hashes") or {})
    canonical_expected = {
        "p1q_manifest": EXPECTED["p1q_manifest"]["sha256"],
        "foreground_manifest": EXPECTED["foreground_manifest"]["sha256"],
        "assembly_manifest": EXPECTED["assembly_manifest"]["sha256"],
        "fit2_surface": EXPECTED["fit2_surface"]["sha256"],
        "skeleton": EXPECTED["skeleton"]["sha256"],
        "fit2_skin": EXPECTED["fit2_skin"]["sha256"],
        "cameras": list(CAMERA_SHA256),
    }
    if expected_hashes != canonical_expected:
        raise RuntimeError("MAGE_V4_RESOLUTION_EXPECTED_HASH_POLICY_DRIFT")

    p1q_dir = Path(value["p1q_dir"]).expanduser().resolve()
    foreground_dir = Path(value["foreground_dir"]).expanduser().resolve()
    assembly_dir = Path(value["assembly_dir"]).expanduser().resolve()
    exact_paths = {
        "p1q_manifest": p1q_dir / EXPECTED["p1q_manifest"]["filename"],
        "foreground_manifest": foreground_dir / EXPECTED["foreground_manifest"]["filename"],
        "assembly_manifest": assembly_dir / EXPECTED["assembly_manifest"]["filename"],
        "fit2_surface": Path(value["fit2_surface"]).expanduser().resolve(),
        "skeleton": Path(value["skeleton"]).expanduser().resolve(),
        "fit2_skin": Path(value["fit2_skin"]).expanduser().resolve(),
    }
    for label, path in exact_paths.items():
        if not path.is_file():
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_PATH_MISSING:{label}:{path}")
        if _sha(path) != str(canonical_expected[label]):
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_SHA_DRIFT:{label}:{path}")

    cameras = tuple(Path(x).expanduser().resolve() for x in value.get("cameras") or ())
    if len(cameras) != 8:
        raise RuntimeError("MAGE_V4_RESOLUTION_CACHED_CAMERA_CARDINALITY")
    for index, (path, expected) in enumerate(zip(cameras, CAMERA_SHA256)):
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_CAMERA_DRIFT:V{index}:{path}")

    required = [
        *(p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_IR.json" for i in range(8)),
        *(p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_MESH_SKIN_IR.json" for i in range(8)),
        *(p1q_dir / f"P1Q_FIT2_CURRENT_V{i}_QUALIFIED_APPEARANCE_IR.json" for i in range(8)),
        *(foreground_dir / f"V{i}_MAGE_PRODUCT_ATLAS.png" for i in range(8)),
        assembly_dir / "FIT2_QUALIFIED_COMPONENT_ASSEMBLY.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "MAGE_V4_RESOLUTION_CACHED_DIRECTORY_INCOMPLETE:"
            + json.dumps(missing[:32], sort_keys=True)
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-if-valid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    cache_hit = False
    result = None
    if args.reuse_if_valid and output.is_file():
        try:
            result = validate_resolution(json.loads(output.read_text(encoding="utf-8")))
            cache_hit = True
        except Exception as exc:
            print("MAGE_RUNTIME_V4_EXACT_INPUT_RESOLUTION_CACHE_MISS:" + str(exc), flush=True)
    if result is None:
        result = resolve(Path(x) for x in args.search_root)
        validate_resolution(result)
        _write_json(output, result)
    print("MAGE_RUNTIME_V4_EXACT_INPUT_RESOLUTION_PASS")
    print("MAGE_RUNTIME_V4_EXACT_INPUT_RESOLUTION_CACHE_HIT=" + str(cache_hit).lower())
    print(json.dumps({
        "output": str(output),
        "p1q_dir": result["p1q_dir"],
        "foreground_dir": result["foreground_dir"],
        "assembly_dir": result["assembly_dir"],
        "fit2_surface": result["fit2_surface"],
        "skeleton": result["skeleton"],
        "fit2_skin": result["fit2_skin"],
        "cameras": result["cameras"],
    }, indent=2, sort_keys=True))
