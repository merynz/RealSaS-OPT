from __future__ import annotations
import argparse, hashlib, os, shutil, zipfile
from pathlib import Path

CHECKPOINT_SHA256 = "0e542d3bb9f01776b4af737dcadc7a02c45c31c440bb1b0dbdb35540638e6b18"
CANONICAL_SOURCE_ZIP_SHA256 = "230fac37ae52ae234437a017ef7ec88bfa865410d294fd840976a6c4e47ea65b"
V11_SOURCE_BUNDLE_SHA256 = "e39e3f383a7f9be58003aa4763c7cb4e31d30f26088d288771d0deb7db8a9aa5"
GFDR_V2_SOURCE_SHA256 = "ca22e3fd42e9c812632eb372544e2319ecf720f6dede1f4a60c8da14ff0cb8fa"
FROZEN_DEPS = Path(__file__).resolve().parent / "frozen_deps"
ACTIVITY_ROOT = Path("/mnt/data/activity_source")
ACTIVITY_PACKAGE = ACTIVITY_ROOT / "realsas_iris_sees_n1d_canonical"
CHECKPOINT_ALIAS = Path("/mnt/data/BEST_copy.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _safe_symlink(src: Path, dst: Path):
    src = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    elif dst.exists():
        raise RuntimeError(f"refusing to replace non-symlink runtime path: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _extract_source(source_zip: Path):
    ACTIVITY_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVITY_ROOT / ".realsas_n1d_extract_tmp"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile(source_zip) as z:
        z.extractall(tmp)
    extracted = tmp / "realsas_iris_sees_n1d_canonical"
    if not extracted.is_dir():
        raise RuntimeError("canonical source ZIP layout mismatch")
    if ACTIVITY_PACKAGE.exists() or ACTIVITY_PACKAGE.is_symlink():
        if ACTIVITY_PACKAGE.is_symlink() or ACTIVITY_PACKAGE.is_file(): ACTIVITY_PACKAGE.unlink()
        else: shutil.rmtree(ACTIVITY_PACKAGE)
    shutil.move(str(extracted), str(ACTIVITY_PACKAGE))
    shutil.rmtree(tmp)


def _stage_frozen_deps(v11_bundle: Path, gfdr_source: Path):
    if sha256_file(v11_bundle) != V11_SOURCE_BUNDLE_SHA256:
        raise RuntimeError("V11 source bundle SHA mismatch")
    if sha256_file(gfdr_source) != GFDR_V2_SOURCE_SHA256:
        raise RuntimeError("GFDR-V2 source SHA mismatch")
    FROZEN_DEPS.mkdir(parents=True, exist_ok=True)
    wanted = {
        "realsas_n1d_hybrid_v11_frozen_runner.py": "ddfd3c989138bb256c2c466e80b914bf89fef1796520d41acc734af0ff577407",
        "v8_base_frozen.py": "44fe83a56a588d4f0cb771351497090ca302b173a7c70619ff701fb0ed28b1f0",
        "v5_seed_geometry_frozen.py": "6185e0fe7b47fedd19c845aecc95d1b8c80a45c17fa683c9f5caedff4bd7d372",
    }
    with zipfile.ZipFile(v11_bundle) as z:
        for name, expected in wanted.items():
            data = z.read(name)
            got = hashlib.sha256(data).hexdigest()
            if got != expected:
                raise RuntimeError(f"frozen dependency SHA mismatch {name}: {got}")
            (FROZEN_DEPS / name).write_bytes(data)
    shutil.copyfile(gfdr_source, FROZEN_DEPS / "realsas_gfdr_v2.py")

def stage_corpus(corpus_root: Path, view_root: Path, families, episode: str):
    view_root.mkdir(parents=True, exist_ok=True)
    for family in families:
        padded = f"{int(family):05d}"
        src = corpus_root / padded
        A = src / "A"
        B = src / "B"
        if len(list(A.glob("*.png"))) != 8 or len(list(B.glob("*.png"))) != 8:
            raise RuntimeError(f"raster count mismatch for family {padded}")
        dst = view_root / str(int(family))
        dst.mkdir(parents=True, exist_ok=True)
        _safe_symlink(A, dst / "A")
        (dst / episode).mkdir(parents=True, exist_ok=True)
        _safe_symlink(B, dst / episode / "B")
    return view_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--canonical-source-zip", type=Path, required=True)
    ap.add_argument("--v11-source-bundle", type=Path, required=True)
    ap.add_argument("--gfdr-v2-source", type=Path, required=True)
    ap.add_argument("--corpus-root", type=Path, required=True,
                    help="padded family directories containing A/ and B/")
    ap.add_argument("--view-root", type=Path, required=True)
    ap.add_argument("--families", nargs="+", type=int, required=True)
    ap.add_argument("--episode", default="e00")
    a = ap.parse_args()
    got_ck = sha256_file(a.checkpoint); got_src = sha256_file(a.canonical_source_zip)
    if got_ck != CHECKPOINT_SHA256: raise RuntimeError(f"checkpoint SHA mismatch: {got_ck}")
    if got_src != CANONICAL_SOURCE_ZIP_SHA256: raise RuntimeError(f"canonical source ZIP SHA mismatch: {got_src}")
    _extract_source(a.canonical_source_zip)
    _stage_frozen_deps(a.v11_source_bundle, a.gfdr_v2_source)
    _safe_symlink(a.checkpoint, CHECKPOINT_ALIAS)
    stage_corpus(a.corpus_root, a.view_root, a.families, a.episode)
    print(f"RUNTIME_PREP_PASS view_root={a.view_root}")


if __name__ == "__main__":
    main()
