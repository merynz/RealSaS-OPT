from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw

ASSET_ID = "asset_fbc8d57f848df78bd953fbb5"
CANDIDATE_ID = "kaykit_cc0:93b401d56e2547316f8841a4"
SOURCE_SHA256 = "93b401d56e2547316f8841a4ac0086011c64d79185c9cb7623567c2dbf125c7d"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(zip_path: Path, dst: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if not str(target).startswith(str(dst.resolve()) + str(Path("/"))):
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        zf.extractall(dst)


def validate_authority(root: Path, manifest: dict) -> None:
    if manifest["asset_id"] != ASSET_ID or manifest["selected_candidate_id"] != CANDIDATE_ID:
        raise RuntimeError("authority manifest identity mismatch")
    for rel, expected in manifest["files"].items():
        p = root / rel
        if not p.is_file():
            raise RuntimeError(f"authority file missing: {rel}")
        if p.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"authority byte-size drift: {rel}")
        got = sha256_file(p)
        if got != expected["sha256"]:
            raise RuntimeError(f"authority SHA drift: {rel}: {got} != {expected['sha256']}")
    with np.load(root / "primary_geometry.npz", allow_pickle=False) as z:
        if np.asarray(z["vertices"]).shape != (2937, 3):
            raise RuntimeError("Mage vertex shape drift")
        if np.asarray(z["faces"]).shape != (5683, 3):
            raise RuntimeError("Mage face shape drift")
        if np.asarray(z["skin"]).shape != (2937, 41):
            raise RuntimeError("Mage skin shape drift")
    for v in range(8):
        with np.load(root / f"V{v}" / "raster_authority.npz", allow_pickle=False) as z:
            res = np.asarray(z["resolution"]).reshape(-1).tolist()
            if res != [1024, 1024]:
                raise RuntimeError(f"V{v} resolution drift: {res}")
            origin = str(np.asarray(z["origin"]).reshape(-1)[0])
            if origin != "TOP_LEFT":
                raise RuntimeError(f"V{v} origin drift: {origin}")


def make_contact_sheet(observation_root: Path, out_path: Path) -> None:
    labels = ["S", "SE", "E", "NE", "N", "NW", "W", "SW"]
    thumbs = []
    for v, label in enumerate(labels):
        with Image.open(observation_root / f"V{v}" / "RGBA.png") as im:
            rgba = im.convert("RGBA")
            canvas = Image.new("RGBA", (520, 560), (30, 30, 30, 255))
            thumb = ImageOps.contain(rgba, (500, 500), method=Image.Resampling.LANCZOS)
            canvas.alpha_composite(thumb, ((520 - thumb.width) // 2, 10 + (500 - thumb.height) // 2))
            draw = ImageDraw.Draw(canvas)
            draw.text((250, 525), label, fill=(255, 255, 255, 255), anchor="mm")
            thumbs.append(canvas.convert("RGB"))
    sheet = Image.new("RGB", (4 * 520, 2 * 560), (20, 20, 20))
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i % 4) * 520, (i // 4) * 560))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority-zip", required=True)
    ap.add_argument("--authority-manifest", required=True)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--source-package", required=True)
    ap.add_argument("--blender", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    zip_path = Path(args.authority_zip).resolve()
    manifest_path = Path(args.authority_manifest).resolve()
    source_file = Path(args.source_file).resolve()
    source_package = Path(args.source_package).resolve()
    blender = Path(args.blender).resolve()
    repo = Path(args.repo_root).resolve()
    work = Path(args.work_root).resolve()
    out = Path(args.out_root).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    zip_sha = sha256_file(zip_path)
    if zip_sha != manifest["zip_sha256"]:
        raise RuntimeError(f"authority ZIP SHA drift: {zip_sha} != {manifest['zip_sha256']}")
    if sha256_file(source_file) != SOURCE_SHA256:
        raise RuntimeError("Mage source FBX SHA drift")
    if not blender.is_file():
        raise RuntimeError("Blender binary missing")

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    safe_extract(zip_path, work)
    authority = work / "mage_authority"
    validate_authority(authority, manifest)

    appearance_npz = work / "appearance" / "source_appearance.npz"
    appearance_json = work / "appearance" / "source_appearance.json"
    appearance_npz.parent.mkdir(parents=True, exist_ok=True)
    extractor = repo / "experiments" / "fit_observation_v1" / "blender_extract_source_appearance_v1.py"
    binder = repo / "experiments" / "fit_observation_v1" / "master_aligned_source_appearance_v1.py"
    if not extractor.is_file() or not binder.is_file():
        raise RuntimeError("appearance apparatus source missing")

    cp = subprocess.run(
        [str(blender), "-b", "--factory-startup", "--python", str(extractor), "--", str(source_file), str(appearance_npz), str(appearance_json)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    (work / "BLENDER_APPEARANCE_EXTRACT.log").write_text(cp.stdout, encoding="utf-8")
    if cp.returncode != 0 or not appearance_npz.is_file() or not appearance_json.is_file():
        raise RuntimeError(f"Blender appearance extraction failed rc={cp.returncode}")

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(binder),
        "--asset-id", ASSET_ID,
        "--candidate-id", CANDIDATE_ID,
        "--source-sha256", SOURCE_SHA256,
        "--source-file", str(source_file),
        "--source-package", str(source_package),
        "--canonical-geometry", str(authority / "primary_geometry.npz"),
        "--appearance-npz", str(appearance_npz),
        "--appearance-json", str(appearance_json),
        "--canonical-render-root", str(authority),
        "--output-root", str(out),
        "--min-texture-coverage", "0.05",
    ]
    cp2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    (work / "MASTER_ALIGNED_APPEARANCE.log").write_text(cp2.stdout, encoding="utf-8")
    if cp2.returncode != 0:
        raise RuntimeError(f"master-aligned appearance failed rc={cp2.returncode}: {cp2.stdout[-3000:]}")

    obs_manifest_path = out / "PREFIT_OBSERVATION_MANIFEST_V1.json"
    obs_manifest = json.loads(obs_manifest_path.read_text(encoding="utf-8"))
    if not obs_manifest.get("exact_eight_views") or obs_manifest.get("raster_authority") != "MASTER_SOURCE_TEXTURED_RGBA":
        raise RuntimeError("observation manifest contract drift")
    for v in range(8):
        p = out / f"V{v}" / "RGBA.png"
        with Image.open(p) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"V{v} RGBA contract drift: {im.mode}/{im.size}")

    contact = out / "MAGE_TEXTURED_8VIEW_CONTACT_SHEET.jpg"
    make_contact_sheet(out, contact)
    result = {
        "schema": "RealSaS.FirstFamily.MageObservationRun.v1",
        "asset_id": ASSET_ID,
        "candidate_id": CANDIDATE_ID,
        "authority_zip_sha256": zip_sha,
        "source_fbx_sha256": SOURCE_SHA256,
        "observation_manifest_sha256": sha256_file(obs_manifest_path),
        "contact_sheet_sha256": sha256_file(contact),
        "textured_foreground_fraction": obs_manifest["textured_foreground_fraction"],
        "source_reextract_vertex_max_abs_err": obs_manifest["source_reextract_vertex_max_abs_err"],
        "source_reextract_faces_equal": obs_manifest["source_reextract_faces_equal"],
        "status": "PASS",
    }
    (out / "FIRST_FAMILY_OBSERVATION_RESULT_V1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
