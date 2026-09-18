from __future__ import annotations

"""Resolve exact current Mage Runtime-v4 inputs without teacher component artifacts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

SCHEMA = "RealSaS.MageRuntimeV4ExactInputResolution.v2"

P1Q_MANIFEST_NAME = "P1Q_FIT2_CURRENT_AUTHORITY_MATERIALIZATION_MANIFEST.json"
P1Q_STATUS = "PASS__FIT2_P1Q_CURRENT_AUTHORITY_V0_V7_FROZEN_FACE_POLICY"
EXPECTED_SURFACE_LINEAGE = "65319061d802c640717010dddf0fd71a66ee6bd2fd31f6e614386f4d2584d5da"
EXPECTED_SKELETON_LINEAGE = "69f05e4fdef65f2cd86fed66503911210e1dbc7212ad017d69bf3dcb7b0896a1"
EXPECTED_SKIN_LINEAGE = "eb96b398282e4e25cd6df662e7a02e8ff1afe881818127190979bddd2f6c4006"

EXPECTED = {
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
OBSERVATION_SHA256 = (
    "8e9875c16bba3047c8f2fc211b984f2399d1145f5720c7427c6fc03a3fec0616",
    "fa94283780d5e5f3ad3943bbffc1f0592a70fc362fdd03b94a1d1cb46889dc30",
    "777bf4f7c505b405d3a5d2a111b2f297949454f77cae7c33064bf02479d384a5",
    "2a771c91c1d1c0b75dab14f6e98b7905a8429bbf0c0a8dd690261f93f6476d86",
    "354bb239feb6c1a515917fee4efb42fb1e0cb9f173d0eb3191fb0901a02a6228",
    "158fc14a75aa69f6133f2ba3ec5df2ad146afc62f477d7d3b4a41ca2cb0e42f2",
    "e3b08836c187863819d8b8aaa53aef79eddfee167e1fabf3fb1dd8ec0484563a",
    "87d4ad46edffec3c6bff034d305194820288d3cc6ef9a9480ac2234fb56451bc",
)

MAX_ANONYMOUS_JSON_BYTES = 64 * 1024 * 1024
MAX_OBSERVATION_BYTES = 32 * 1024 * 1024
SKIP_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", ".Trash", ".shortcut-targets-by-id",
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


def _choose_exact(paths: Iterable[Path], expected_sha: str, label: str):
    matches = []
    for path in paths:
        try:
            if _sha(path) == expected_sha:
                matches.append(path)
        except OSError:
            continue
    if not matches:
        raise RuntimeError(f"MAGE_V4_INPUT_NOT_FOUND:{label}:{expected_sha}")
    matches = tuple(sorted(set(matches), key=lambda p: str(p)))
    return matches[0], matches


def _p1q_manifest_valid(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("status") != P1Q_STATUS:
        return False
    if payload.get("teacher_truth_used") is not False:
        return False
    if payload.get("source_component_truth_used") is not False:
        return False
    if payload.get("current_gsa_lineage_hash") != EXPECTED_SURFACE_LINEAGE:
        return False
    if payload.get("current_skeleton_lineage_hash") != EXPECTED_SKELETON_LINEAGE:
        return False
    if payload.get("current_skin_lineage_hash") != EXPECTED_SKIN_LINEAGE:
        return False
    rows = {int(row.get("view", -1)): row for row in payload.get("views") or ()}
    if set(rows) != set(range(8)):
        return False
    root = path.parent
    for view in range(8):
        files = dict(rows[view].get("files") or {})
        for key in ("mesh", "skin", "appearance"):
            name = str(files.get(key) or "")
            if not name or not (root / name).is_file():
                return False
    return True


def _choose_current_p1q(paths: Iterable[Path]):
    valid = tuple(
        sorted(
            {path.resolve() for path in paths if _p1q_manifest_valid(path)},
            key=lambda p: str(p),
        )
    )
    if not valid:
        raise RuntimeError("MAGE_V4_CURRENT_NO_TEACHER_P1Q_MANIFEST_NOT_FOUND")
    by_sha: dict[str, list[Path]] = {}
    for path in valid:
        by_sha.setdefault(_sha(path), []).append(path)
    if len(by_sha) != 1:
        raise RuntimeError(
            "MAGE_V4_CURRENT_P1Q_MANIFEST_AMBIGUOUS:"
            + json.dumps({digest: [str(p) for p in rows] for digest, rows in sorted(by_sha.items())})
        )
    digest = next(iter(by_sha))
    aliases = tuple(sorted(by_sha[digest], key=lambda p: str(p)))
    return aliases[0], aliases, digest


def resolve(search_roots: Iterable[Path]) -> dict:
    roots = tuple(Path(x).expanduser().resolve() for x in search_roots)
    files = tuple(_walk_files(roots))

    p1q_candidates = tuple(path for path in files if path.name == P1Q_MANIFEST_NAME)
    p1q_manifest, p1q_aliases, p1q_manifest_sha = _choose_current_p1q(p1q_candidates)

    surface_candidates = tuple(
        path for path in files
        if path.name == EXPECTED["fit2_surface"]["filename"]
    )
    fit2_surface, surface_aliases = _choose_exact(
        surface_candidates,
        EXPECTED["fit2_surface"]["sha256"],
        "fit2_surface",
    )

    json_targets = {
        EXPECTED["skeleton"]["sha256"]: "skeleton",
        EXPECTED["fit2_skin"]["sha256"]: "fit2_skin",
        **{digest: f"camera_v{i}" for i, digest in enumerate(CAMERA_SHA256)},
    }
    json_hits: dict[str, list[Path]] = {digest: [] for digest in json_targets}
    observation_targets = {
        digest: f"observation_v{i}" for i, digest in enumerate(OBSERVATION_SHA256)
    }
    observation_hits: dict[str, list[Path]] = {digest: [] for digest in observation_targets}

    for path in files:
        suffix = path.suffix.lower()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if suffix == ".json" and 0 < size <= MAX_ANONYMOUS_JSON_BYTES:
            digest = _sha(path)
            if digest in json_hits:
                json_hits[digest].append(path)
        elif suffix == ".png" and 0 < size <= MAX_OBSERVATION_BYTES:
            digest = _sha(path)
            if digest in observation_hits:
                observation_hits[digest].append(path)

    resolved = {"fit2_surface": fit2_surface, "p1q_manifest": p1q_manifest}
    aliases = {
        "fit2_surface": surface_aliases,
        "p1q_manifest": p1q_aliases,
    }
    for digest, label in json_targets.items():
        rows = tuple(sorted(set(json_hits[digest]), key=lambda p: str(p)))
        if not rows:
            raise RuntimeError(f"MAGE_V4_INPUT_NOT_FOUND:{label}:{digest}")
        resolved[label] = rows[0]
        aliases[label] = rows
    for digest, label in observation_targets.items():
        rows = tuple(sorted(set(observation_hits[digest]), key=lambda p: str(p)))
        if not rows:
            raise RuntimeError(f"MAGE_V4_INPUT_NOT_FOUND:{label}:{digest}")
        resolved[label] = rows[0]
        aliases[label] = rows

    p1q_dir = p1q_manifest.parent
    cameras = tuple(resolved[f"camera_v{i}"] for i in range(8))
    observations = tuple(resolved[f"observation_v{i}"] for i in range(8))

    result = {
        "schema": SCHEMA,
        "status": "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED",
        "search_roots": [str(path) for path in roots],
        "p1q_dir": str(p1q_dir),
        "p1q_manifest_sha256": p1q_manifest_sha,
        "fit2_surface": str(resolved["fit2_surface"]),
        "skeleton": str(resolved["skeleton"]),
        "fit2_skin": str(resolved["fit2_skin"]),
        "cameras": [str(path) for path in cameras],
        "observations": [str(path) for path in observations],
        "expected_hashes": {
            "fit2_surface": EXPECTED["fit2_surface"]["sha256"],
            "skeleton": EXPECTED["skeleton"]["sha256"],
            "fit2_skin": EXPECTED["fit2_skin"]["sha256"],
            "cameras": list(CAMERA_SHA256),
            "observations": list(OBSERVATION_SHA256),
        },
        "aliases": {
            label: [str(path) for path in rows]
            for label, rows in sorted(aliases.items())
        },
        "teacher_component_artifacts_resolved": False,
        "giant_product_graph_scanned": False,
        "anonymous_json_size_ceiling_bytes": MAX_ANONYMOUS_JSON_BYTES,
    }
    return result


def validate_resolution(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise RuntimeError("MAGE_V4_RESOLUTION_SCHEMA_INVALID")
    if value.get("status") != "PASS__EXACT_MAGE_RUNTIME_V4_INPUTS_RESOLVED":
        raise RuntimeError("MAGE_V4_RESOLUTION_STATUS_INVALID")
    if value.get("teacher_component_artifacts_resolved") is not False:
        raise RuntimeError("MAGE_V4_RESOLUTION_TEACHER_COMPONENT_ARTIFACT_FORBIDDEN")

    canonical_expected = {
        "fit2_surface": EXPECTED["fit2_surface"]["sha256"],
        "skeleton": EXPECTED["skeleton"]["sha256"],
        "fit2_skin": EXPECTED["fit2_skin"]["sha256"],
        "cameras": list(CAMERA_SHA256),
        "observations": list(OBSERVATION_SHA256),
    }
    if dict(value.get("expected_hashes") or {}) != canonical_expected:
        raise RuntimeError("MAGE_V4_RESOLUTION_EXPECTED_HASH_POLICY_DRIFT")

    p1q_dir = Path(value["p1q_dir"]).expanduser().resolve()
    p1q_manifest = p1q_dir / P1Q_MANIFEST_NAME
    if not p1q_manifest.is_file() or not _p1q_manifest_valid(p1q_manifest):
        raise RuntimeError("MAGE_V4_RESOLUTION_P1Q_POLICY_DRIFT")
    if _sha(p1q_manifest) != str(value.get("p1q_manifest_sha256") or ""):
        raise RuntimeError("MAGE_V4_RESOLUTION_P1Q_MANIFEST_SHA_DRIFT")

    exact_paths = {
        "fit2_surface": Path(value["fit2_surface"]).expanduser().resolve(),
        "skeleton": Path(value["skeleton"]).expanduser().resolve(),
        "fit2_skin": Path(value["fit2_skin"]).expanduser().resolve(),
    }
    for label, path in exact_paths.items():
        if not path.is_file() or _sha(path) != str(canonical_expected[label]):
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_SHA_DRIFT:{label}:{path}")

    cameras = tuple(Path(x).expanduser().resolve() for x in value.get("cameras") or ())
    observations = tuple(Path(x).expanduser().resolve() for x in value.get("observations") or ())
    if len(cameras) != 8 or len(observations) != 8:
        raise RuntimeError("MAGE_V4_RESOLUTION_CACHED_VIEW_CARDINALITY")
    for index, (path, expected) in enumerate(zip(cameras, CAMERA_SHA256)):
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_CAMERA_DRIFT:V{index}:{path}")
    for index, (path, expected) in enumerate(zip(observations, OBSERVATION_SHA256)):
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"MAGE_V4_RESOLUTION_CACHED_OBSERVATION_DRIFT:V{index}:{path}")
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
        "p1q_manifest_sha256": result["p1q_manifest_sha256"],
        "fit2_surface": result["fit2_surface"],
        "skeleton": result["skeleton"],
        "fit2_skin": result["fit2_skin"],
        "cameras": result["cameras"],
        "observations": result["observations"],
    }, indent=2, sort_keys=True))
