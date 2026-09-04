from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def load_family_source_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema == "RealSaS.FamilyFitManifest.v1":
        required = ("family_label", "asset_id", "source")
    elif schema == "RealSaS.FirstFamilyFitSpec.v1":
        # Historical compatibility only. Current workflows use FamilyFitManifest.
        required = ("family_label", "asset_id", "source")
    else:
        raise RuntimeError(f"unsupported family source manifest schema: {schema}")
    missing = [k for k in required if k not in payload]
    if missing:
        raise RuntimeError(f"family source manifest missing keys: {missing}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    manifest = ap.add_mutually_exclusive_group(required=True)
    manifest.add_argument("--family-manifest")
    manifest.add_argument("--spec")  # legacy alias
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--result-json", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.family_manifest or args.spec).resolve()
    family = load_family_source_manifest(manifest_path)
    src = family["source"]
    if src.get("provider") != "GIT":
        raise RuntimeError(f"unsupported source provider: {src.get('provider')}")

    cache_root = Path(args.cache_root).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((src["repository"] + "\n" + src["revision"]).encode()).hexdigest()[:16]
    repo = cache_root / f"git-{key}"
    if not (repo / ".git").is_dir():
        shutil.rmtree(repo, ignore_errors=True)
        subprocess.check_call(["git", "clone", "--filter=blob:none", "--no-checkout", src["repository"], str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "fetch", "--depth=1", "origin", src["revision"]])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--force", "--detach", src["revision"]])
    got_rev = run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    if got_rev != src["revision"]:
        raise RuntimeError(f"source revision drift: {got_rev} != {src['revision']}")

    source_file = repo / src["relative_file"]
    if not source_file.is_file():
        raise RuntimeError(f"source file missing: {source_file}")
    got_sha = sha256_file(source_file)
    if got_sha != src["sha256"]:
        raise RuntimeError(f"source SHA drift: {got_sha} != {src['sha256']}")

    result = {
        "schema": "RealSaS.MaterializedFamilySource.v1",
        "family_manifest_schema": family["schema"],
        "family_label": family["family_label"],
        "asset_id": family["asset_id"],
        "source_root": str(repo),
        "source_file": str(source_file),
        "source_sha256": got_sha,
        "source_revision": got_rev,
        "status": "PASS",
    }
    out = Path(args.result_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
