from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

VERSION = "5.2.0"
RELEASE_DIR = "Blender5.2"
ARCHIVE = f"blender-{VERSION}-linux-x64.tar.xz"
CHECKSUM = f"blender-{VERSION}.sha256"
BASES = (
    f"https://mirror.blender.org/release/{RELEASE_DIR}",
    f"https://download.blender.org/release/{RELEASE_DIR}",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_first(urls: list[str], dst: Path, min_bytes: int) -> str:
    errors = []
    for url in urls:
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            tmp.unlink(missing_ok=True)
            with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=8 << 20)
            if tmp.stat().st_size < min_bytes:
                raise RuntimeError(f"download too small: {tmp.stat().st_size}")
            tmp.replace(dst)
            return url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError("all Blender downloads failed: " + " | ".join(errors[-4:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    args = ap.parse_args()
    cache = Path(args.cache_root).expanduser().resolve()
    home = cache / f"blender-{VERSION}-linux-x64"
    binary = home / "blender"
    if binary.is_file():
        version = subprocess.check_output([str(binary), "--version"], text=True, timeout=60).splitlines()[0]
        if f"Blender {VERSION}" not in version:
            raise RuntimeError(f"cached Blender version drift: {version}")
        print(binary)
        return

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / ARCHIVE
    checksum = cache / CHECKSUM
    download_first([f"{base}/{CHECKSUM}" for base in BASES], checksum, 64)
    expected = None
    for line in checksum.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1].lstrip("*")).name == ARCHIVE and re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            expected = parts[0].lower()
            break
    if not expected:
        raise RuntimeError("Blender checksum entry not found")
    if not archive.is_file() or sha256_file(archive) != expected:
        download_first([f"{base}/{ARCHIVE}" for base in BASES], archive, 100_000_000)
    got = sha256_file(archive)
    if got != expected:
        raise RuntimeError(f"Blender archive SHA mismatch: {got} != {expected}")

    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(cache)
    if not binary.is_file():
        raise RuntimeError("Blender binary missing after extraction")
    binary.chmod(binary.stat().st_mode | 0o111)
    version = subprocess.check_output([str(binary), "--version"], text=True, timeout=60).splitlines()[0]
    if f"Blender {VERSION}" not in version:
        raise RuntimeError(f"wrong Blender version: {version}")
    print(binary)


if __name__ == "__main__":
    main()
