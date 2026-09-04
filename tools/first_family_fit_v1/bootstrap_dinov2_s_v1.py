from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

DINO_REPO = "https://github.com/facebookresearch/dinov2.git"
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
WEIGHT_NAME = "dinov2_vits14_pretrain.pth"
WEIGHT_BYTES = 88283115
WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    args = ap.parse_args()
    root = Path(args.cache_root).expanduser().resolve()
    source = root / "dinov2"
    weight = root / WEIGHT_NAME
    root.mkdir(parents=True, exist_ok=True)

    if not (source / ".git").is_dir():
        shutil.rmtree(source, ignore_errors=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", DINO_REPO, str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", "origin", DINO_REVISION], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", DINO_REVISION], check=True)
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != DINO_REVISION:
        raise RuntimeError(f"DINO revision drift: {head}")

    valid_weight = weight.is_file() and weight.stat().st_size == WEIGHT_BYTES and sha256_file(weight) == WEIGHT_SHA256
    if not valid_weight:
        tmp = weight.with_suffix(weight.suffix + ".part")
        tmp.unlink(missing_ok=True)
        with urllib.request.urlopen(WEIGHT_URL, timeout=600) as response, tmp.open("wb") as f:
            shutil.copyfileobj(response, f, length=8 << 20)
        if tmp.stat().st_size != WEIGHT_BYTES:
            raise RuntimeError(f"DINO weight size drift: {tmp.stat().st_size} != {WEIGHT_BYTES}")
        got = sha256_file(tmp)
        if got != WEIGHT_SHA256:
            raise RuntimeError(f"DINO weight SHA drift: {got} != {WEIGHT_SHA256}")
        tmp.replace(weight)

    print(f"source_dir={source}")
    print(f"source_head={head}")
    print(f"weight_path={weight}")
    print(f"weight_bytes={weight.stat().st_size}")
    print(f"weight_sha256={sha256_file(weight)}")
    print("status=PASS")


if __name__ == "__main__":
    main()
