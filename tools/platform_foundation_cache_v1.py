#!/usr/bin/env python3
from __future__ import annotations

"""Provision exact, verified third-party runtime caches for product capabilities.

These paths are operational locators only. Scientific identity remains owned by
source-controlled capability implementations; the DINO/FlexiCubes loaders
re-verify exact revision/hash contracts again at consumption time.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DINO_AUTHORITY_SOURCE = ROOT / "models/iris/v2/dinov2_foundation_v2.py"
FLEXICUBES_AUTHORITY_SOURCE = ROOT / "models/iris/vnext/flexicubes_v1.py"
DINO_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
    "dinov2_vits14_pretrain.pth"
)


def _literal_constants(path: Path, names: tuple[str, ...]) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(value)
    missing = sorted(set(names) - set(values))
    if missing:
        raise RuntimeError("FOUNDATION_AUTHORITY_CONSTANT_MISSING:" + ",".join(missing))
    return values


def foundation_contracts() -> dict[str, dict[str, object]]:
    dino = _literal_constants(
        DINO_AUTHORITY_SOURCE,
        (
            "DINO_REPOSITORY",
            "DINO_SOURCE_REVISION",
            "DINO_S_WEIGHT_SHA256",
            "DINO_S_WEIGHT_SIZE_BYTES",
        ),
    )
    flex = _literal_constants(
        FLEXICUBES_AUTHORITY_SOURCE,
        ("FLEXICUBES_REPOSITORY", "FLEXICUBES_COMMIT"),
    )
    return {"dinov2": dino, "flexicubes": flex}


def runtime_foundation_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    foundation = root / "foundations"
    return {
        "root": foundation,
        "dino_source": foundation / "dinov2" / "source",
        "dino_weight": foundation / "dinov2" / "dinov2_vits14_pretrain.pth",
        "flexicubes_source": foundation / "flexicubes" / "source",
        "receipt": foundation / "FOUNDATION_CACHE_RECEIPT.json",
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _normalized_git_url(value: str) -> str:
    return value.strip().removesuffix("/").removesuffix(".git")


def _exact_git_checkout(destination: Path, repository_url: str, revision: str) -> None:
    destination = destination.resolve()
    if destination.exists():
        valid = False
        try:
            head = _git(destination, "rev-parse", "HEAD")
            remote = _git(destination, "config", "--get", "remote.origin.url")
            dirty = _git(destination, "status", "--porcelain")
            valid = (
                head == revision
                and _normalized_git_url(remote) == _normalized_git_url(repository_url)
                and not dirty
            )
        except (OSError, subprocess.CalledProcessError):
            valid = False
        if valid:
            return
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(destination)], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository_url],
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", revision],
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        check=True,
        timeout=60,
    )
    if _git(destination, "rev-parse", "HEAD") != revision:
        raise RuntimeError("FOUNDATION_GIT_REVISION_DRIFT")
    if _normalized_git_url(_git(destination, "config", "--get", "remote.origin.url")) != _normalized_git_url(repository_url):
        raise RuntimeError("FOUNDATION_GIT_REMOTE_DRIFT")
    if _git(destination, "status", "--porcelain"):
        raise RuntimeError("FOUNDATION_GIT_CHECKOUT_DIRTY")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_weight(path: Path, *, expected_sha256: str, expected_size: int) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(expected_size)
        and _sha256_file(path) == expected_sha256
    )


def _download_exact_weight(path: Path, *, expected_sha256: str, expected_size: int) -> None:
    if _verified_weight(path, expected_sha256=expected_sha256, expected_size=expected_size):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".staged", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        digest = hashlib.sha256()
        size = 0
        with urllib.request.urlopen(DINO_WEIGHT_URL, timeout=60) as response, temporary.open("wb") as out:
            while True:
                chunk = response.read(8 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > int(expected_size):
                    raise RuntimeError("DINO_WEIGHT_DOWNLOAD_OVERSIZE")
                digest.update(chunk)
                out.write(chunk)
        if size != int(expected_size):
            raise RuntimeError(f"DINO_WEIGHT_SIZE_DRIFT:{size}")
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("DINO_WEIGHT_SHA256_DRIFT")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def provision(root: str | Path) -> dict[str, object]:
    paths = runtime_foundation_paths(root)
    foundation_root = paths["root"]
    foundation_root.mkdir(parents=True, exist_ok=True)
    contracts = foundation_contracts()
    dino = contracts["dinov2"]
    flex = contracts["flexicubes"]

    dino_repo = f"https://github.com/{dino['DINO_REPOSITORY']}.git"
    _exact_git_checkout(paths["dino_source"], dino_repo, str(dino["DINO_SOURCE_REVISION"]))
    _download_exact_weight(
        paths["dino_weight"],
        expected_sha256=str(dino["DINO_S_WEIGHT_SHA256"]),
        expected_size=int(dino["DINO_S_WEIGHT_SIZE_BYTES"]),
    )
    _exact_git_checkout(
        paths["flexicubes_source"],
        str(flex["FLEXICUBES_REPOSITORY"]),
        str(flex["FLEXICUBES_COMMIT"]),
    )

    receipt = {
        "schema": "RealSaS.PlatformFoundationCacheReceipt.v1",
        "authority_class": "OPERATIONAL_CACHE_LOCATOR_ONLY",
        "scientific_authority_minted": False,
        "dinov2": {
            "source_revision": str(dino["DINO_SOURCE_REVISION"]),
            "weight_sha256": str(dino["DINO_S_WEIGHT_SHA256"]),
            "weight_size_bytes": int(dino["DINO_S_WEIGHT_SIZE_BYTES"]),
            "source_path": str(paths["dino_source"]),
            "weight_path": str(paths["dino_weight"]),
        },
        "flexicubes": {
            "source_revision": str(flex["FLEXICUBES_COMMIT"]),
            "source_path": str(paths["flexicubes_source"]),
        },
    }
    paths["receipt"].write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(provision(args.root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
