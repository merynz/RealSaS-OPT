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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(path: Path) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "RealSaS.FirstFamilyFitSpec.v1":
        raise RuntimeError("first-family spec schema drift")
    if spec["observation"].get("views") != 8 or spec["observation"].get("native_resolution") != 1024:
        raise RuntimeError("first-family observation contract drift")
    return spec


def safe_extract(zip_path: Path, dst: Path) -> None:
    dst = dst.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if target != dst and not str(target).startswith(str(dst) + str(Path("/"))):
                raise RuntimeError(f"unsafe zip member: {member.filename}")
        zf.extractall(dst)


def validate_authority(root: Path, manifest: dict, spec: dict) -> None:
    if manifest.get("schema") != "RealSaS.FirstFamilyAuthorityPack.v1":
        raise RuntimeError("authority manifest schema drift")
    if manifest["asset_id"] != spec["asset_id"] or manifest["selected_candidate_id"] != spec["candidate_id"]:
        raise RuntimeError("authority manifest identity mismatch")
    if manifest["build_id"] != spec["authority"]["build_id"]:
        raise RuntimeError("authority build id mismatch")
    for rel, expected in manifest["files"].items():
        p = root / rel
        if not p.is_file():
            raise RuntimeError(f"authority file missing: {rel}")
        if p.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"authority byte-size drift: {rel}")
        got = sha256_file(p)
        if got != expected["sha256"]:
            raise RuntimeError(f"authority SHA drift: {rel}: {got} != {expected['sha256']}")
    if sha256_file(root / "primary_geometry.npz") != spec["authority"]["primary_geometry_sha256"]:
        raise RuntimeError("primary geometry identity drift")
    with np.load(root / "primary_geometry.npz", allow_pickle=False) as z:
        expected_v = tuple(manifest["primary_geometry"]["vertices"])
        expected_f = tuple(manifest["primary_geometry"]["faces"])
        expected_s = tuple(manifest["primary_geometry"]["skin"])
        if np.asarray(z["vertices"]).shape != expected_v:
            raise RuntimeError("primary vertex shape drift")
        if np.asarray(z["faces"]).shape != expected_f:
            raise RuntimeError("primary face shape drift")
        if np.asarray(z["skin"]).shape != expected_s:
            raise RuntimeError("primary skin shape drift")
    for v in range(8):
        with np.load(root / f"V{v}" / "raster_authority.npz", allow_pickle=False) as z:
            res = np.asarray(z["resolution"]).reshape(-1).tolist()
            if res != [1024, 1024]:
                raise RuntimeError(f"V{v} resolution drift: {res}")
            origin = str(np.asarray(z["origin"]).reshape(-1)[0])
            if origin != "TOP_LEFT":
                raise RuntimeError(f"V{v} origin drift: {origin}")


def stage_from_master(master_root: Path, dst: Path, manifest: dict, spec: dict) -> Path:
    src = master_root / "master" / "assets" / spec["asset_id"]
    if not src.is_dir():
        raise RuntimeError(f"Master asset directory missing: {src}")
    authority = dst / "family_authority"
    authority.mkdir(parents=True, exist_ok=True)
    for rel in manifest["files"]:
        source = src / rel if rel == "primary_geometry.npz" else src / "renders" / rel
        target = authority / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            raise RuntimeError(f"Master authority source missing: {source}")
        shutil.copy2(source, target)
    validate_authority(authority, manifest, spec)
    return authority


def stage_authority(*, zip_path: Path | None, master_root: Path | None, work: Path, manifest: dict, spec: dict) -> tuple[Path, dict]:
    if (zip_path is None) == (master_root is None):
        raise RuntimeError("exactly one of authority ZIP or Master root must be supplied")
    if master_root is not None:
        authority = stage_from_master(master_root, work, manifest, spec)
        return authority, {"mode": "DIRECT_MASTER_CORPUS", "master_root": str(master_root), "asset_id": spec["asset_id"]}
    assert zip_path is not None
    zip_sha = sha256_file(zip_path)
    if zip_sha != spec["authority"]["fallback_zip_sha256"]:
        raise RuntimeError(f"authority ZIP SHA drift: {zip_sha} != {spec['authority']['fallback_zip_sha256']}")
    safe_extract(zip_path, work)
    dirs = [p for p in work.iterdir() if p.is_dir()]
    candidates = [p for p in dirs if (p / "primary_geometry.npz").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"authority ZIP must expose exactly one authority root, found {len(candidates)}")
    authority = candidates[0]
    validate_authority(authority, manifest, spec)
    return authority, {"mode": "PREPACKAGED_ZIP_FALLBACK", "authority_zip": str(zip_path), "authority_zip_sha256": zip_sha}


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
    ap.add_argument("--spec", required=True)
    authority = ap.add_mutually_exclusive_group(required=True)
    authority.add_argument("--authority-zip")
    authority.add_argument("--master-root")
    ap.add_argument("--authority-manifest", required=True)
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--source-package", required=True)
    ap.add_argument("--blender", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    spec = load_spec(Path(args.spec).resolve())
    manifest = json.loads(Path(args.authority_manifest).resolve().read_text(encoding="utf-8"))
    source_file = Path(args.source_file).resolve()
    source_package = Path(args.source_package).resolve()
    blender = Path(args.blender).resolve()
    repo = Path(args.repo_root).resolve()
    work = Path(args.work_root).resolve()
    out = Path(args.out_root).resolve()

    if sha256_file(source_file) != spec["source"]["sha256"]:
        raise RuntimeError("source asset SHA drift")
    if not blender.is_file():
        raise RuntimeError("Blender binary missing")

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    authority_root, authority_source = stage_authority(
        zip_path=Path(args.authority_zip).resolve() if args.authority_zip else None,
        master_root=Path(args.master_root).resolve() if args.master_root else None,
        work=work,
        manifest=manifest,
        spec=spec,
    )

    appearance_npz = work / "appearance" / "source_appearance.npz"
    appearance_json = work / "appearance" / "source_appearance.json"
    appearance_npz.parent.mkdir(parents=True, exist_ok=True)
    extractor = repo / "experiments" / "fit_observation_v1" / "blender_extract_source_appearance_v1.py"
    binder = repo / "experiments" / "fit_observation_v1" / "master_aligned_source_appearance_v1.py"
    if not extractor.is_file() or not binder.is_file():
        raise RuntimeError("appearance apparatus source missing")

    cp = subprocess.run([str(blender), "-b", "--factory-startup", "--python", str(extractor), "--", str(source_file), str(appearance_npz), str(appearance_json)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    (work / "BLENDER_APPEARANCE_EXTRACT.log").write_text(cp.stdout, encoding="utf-8")
    if cp.returncode != 0 or not appearance_npz.is_file() or not appearance_json.is_file():
        raise RuntimeError(f"Blender appearance extraction failed rc={cp.returncode}")

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(binder), "--asset-id", spec["asset_id"], "--candidate-id", spec["candidate_id"], "--source-sha256", spec["source"]["sha256"], "--source-file", str(source_file), "--source-package", str(source_package), "--canonical-geometry", str(authority_root / "primary_geometry.npz"), "--appearance-npz", str(appearance_npz), "--appearance-json", str(appearance_json), "--canonical-render-root", str(authority_root), "--output-root", str(out), "--min-texture-coverage", str(spec["observation"]["min_texture_coverage"])]
    cp2 = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    (work / "MASTER_ALIGNED_APPEARANCE.log").write_text(cp2.stdout, encoding="utf-8")
    if cp2.returncode != 0:
        raise RuntimeError(f"master-aligned appearance failed rc={cp2.returncode}: {cp2.stdout[-5000:]}")

    obs_manifest_path = out / "PREFIT_OBSERVATION_MANIFEST_V1.json"
    obs_manifest = json.loads(obs_manifest_path.read_text(encoding="utf-8"))
    if obs_manifest.get("asset_id") != spec["asset_id"]:
        raise RuntimeError("observation asset identity drift")
    if not obs_manifest.get("exact_eight_views") or obs_manifest.get("raster_authority") != spec["observation"]["raster_authority"]:
        raise RuntimeError("observation manifest contract drift")
    for v in range(8):
        p = out / f"V{v}" / "RGBA.png"
        with Image.open(p) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise RuntimeError(f"V{v} RGBA contract drift: {im.mode}/{im.size}")

    contact = out / "TEXTURED_8VIEW_CONTACT_SHEET.jpg"
    make_contact_sheet(out, contact)
    result = {"schema": "RealSaS.FirstFamilyObservationRun.v1", "family_label": spec["family_label"], "asset_id": spec["asset_id"], "candidate_id": spec["candidate_id"], "authority_source": authority_source, "source_sha256": spec["source"]["sha256"], "observation_manifest_sha256": sha256_file(obs_manifest_path), "contact_sheet_sha256": sha256_file(contact), "textured_foreground_fraction": obs_manifest["textured_foreground_fraction"], "source_geometry_alignment": obs_manifest["source_geometry_alignment"], "status": "PASS"}
    (out / "FIRST_FAMILY_OBSERVATION_RESULT_V1.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
