from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

ALLOWED_RESOLUTIONS = (256, 512, 1024)
VIEWS = 8
STAGE_SCHEMA = "RealSaS.IRISSinglePoseV2.Stage.v2"


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_dependency_hashes(src: Path, input_resolution: int) -> dict[str, str]:
    rels = [Path("primary_geometry.npz")]
    for v in range(VIEWS):
        base = Path("renders") / f"V{v}"
        rels.extend([base / "raster_authority.npz", base / "camera.json"])
        for style in ("cel_clean", "ink_cel"):
            rels.append(base / f"{style}.png")
            if input_resolution in (256, 512):
                rels.append(base / f"{style}_512.png")
    out = {}
    for rel in rels:
        path = src / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        out[rel.as_posix()] = sha256_file(path)
    return out


def _output_hashes(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "STAGE.json"):
        out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def _marker_reusable(final: Path, marker: dict, source_fingerprint: str, input_resolution: int) -> bool:
    if marker.get("schema") != STAGE_SCHEMA:
        return False
    if marker.get("input_resolution") != input_resolution or marker.get("physical_firewall") is not True:
        return False
    if marker.get("source_fingerprint") != source_fingerprint:
        return False
    expected = marker.get("staged_sha256")
    if not isinstance(expected, dict) or not expected:
        return False
    for rel, digest in expected.items():
        path = final / rel
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def _validate_native(native: Path):
    with Image.open(native) as im:
        if im.size != (1024, 1024) or im.mode != "RGBA":
            raise RuntimeError(f"bad native authority {native}: {im.mode} {im.size}")


def _stage_image(src_view: Path, dst: Path, style: str, input_resolution: int):
    native = src_view / f"{style}.png"
    _validate_native(native)
    if input_resolution == 1024:
        with Image.open(native) as im:
            im.save(dst)
        return {"source": native.name, "transform": "identity", "source_resolution": 1024}

    deriv512 = src_view / f"{style}_512.png"
    with Image.open(deriv512) as im:
        if im.size != (512, 512) or im.mode != "RGBA":
            raise RuntimeError(f"bad 512 derivative {deriv512}: {im.mode} {im.size}")
        if input_resolution == 512:
            im.save(dst)
            return {"source": deriv512.name, "transform": "canonical_512_derivative", "source_resolution": 512}
        im = im.resize((256, 256), resample=Image.Resampling.BILINEAR)
        im.save(dst)
    return {"source": deriv512.name, "transform": "PIL_RGBA_BILINEAR_512_to_256", "source_resolution": 512}


def stage_asset(master_root: Path, out_root: Path, asset_id: str, input_resolution: int):
    if input_resolution not in ALLOWED_RESOLUTIONS:
        raise ValueError(input_resolution)
    src = master_root / "master" / "assets" / asset_id
    final = out_root / "assets" / asset_id
    marker_path = final / "STAGE.json"

    source_dependencies = _source_dependency_hashes(src, input_resolution)
    source_fingerprint = canonical_hash(
        {
            "schema": STAGE_SCHEMA,
            "asset_id": asset_id,
            "input_resolution": input_resolution,
            "source_sha256": source_dependencies,
        }
    )
    if marker_path.exists():
        marker = json.load(open(marker_path, encoding="utf-8"))
        if marker.get("asset_id") == asset_id and _marker_reusable(final, marker, source_fingerprint, input_resolution):
            return marker

    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=asset_id + ".partial.", dir=final.parent))
    try:
        with np.load(src / "primary_geometry.npz", allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], np.float32)
            faces = np.asarray(z["faces"], np.int32)
        # Physical firewall: only geometry allow-list survives into IRIS staging.
        np.savez(tmp / "primary_geometry.npz", vertices=vertices, faces=faces)
        views = []
        for v in range(VIEWS):
            sv = src / "renders" / f"V{v}"
            dv = tmp / "renders" / f"V{v}"
            dv.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sv / "raster_authority.npz", dv / "raster_authority.npz")
            shutil.copyfile(sv / "camera.json", dv / "camera.json")
            cam = json.load(open(dv / "camera.json", encoding="utf-8"))
            styles = {}
            for style in ("cel_clean", "ink_cel"):
                styles[style] = _stage_image(sv, dv / f"{style}_input.png", style, input_resolution)
            views.append({"view": v, "yaw_deg": float(cam["yaw_deg"]), "styles": styles})
        yaws = np.asarray([x["yaw_deg"] for x in views], np.float32)
        if not np.allclose(yaws, np.arange(8, dtype=np.float32) * 45.0, atol=1e-4):
            raise RuntimeError(f"noncanonical yaws {yaws.tolist()}")
        with np.load(tmp / "primary_geometry.npz", allow_pickle=False) as z:
            if set(z.files) != {"vertices", "faces"}:
                raise RuntimeError(f"firewall failed: {z.files}")
        staged_sha256 = _output_hashes(tmp)
        meta = {
            "schema": STAGE_SCHEMA,
            "asset_id": asset_id,
            "input_resolution": input_resolution,
            "authority_resolution": 1024,
            "physical_firewall": True,
            "geometry_fields": ["vertices", "faces"],
            "source_sha256": source_dependencies,
            "source_fingerprint": source_fingerprint,
            "staged_sha256": staged_sha256,
            "views": views,
        }
        atomic_json(tmp / "STAGE.json", meta)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        return meta
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--input-resolution", type=int, choices=ALLOWED_RESOLUTIONS, required=True)
    ap.add_argument("--splits", default="FIT,TUNE")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out)
    seed = json.load(open(a.seed_manifest, encoding="utf-8"))
    splits = {x.strip().upper() for x in a.splits.split(",") if x.strip()}
    if splits & {"CAL", "DEV", "EXTERNAL_HOLDOUT"}:
        raise RuntimeError("sealed split requested")
    rows = [r for r in seed["records"] if r["split"] in splits]
    rows = rows[: a.limit] if a.limit else rows
    result = []
    for i, r in enumerate(rows, 1):
        meta = stage_asset(root, out, r["asset_id"], a.input_resolution)
        result.append(
            {
                "asset_id": r["asset_id"],
                "split": r["split"],
                "asset_dir": str(out / "assets" / r["asset_id"]),
                "source_fingerprint": meta["source_fingerprint"],
            }
        )
        if i % 5 == 0 or i == len(rows):
            print(f"[stage-v2] {i}/{len(rows)}", flush=True)
    atomic_json(
        out / "STAGE_MANIFEST.json",
        {
            "schema": "RealSaS.IRISSinglePoseV2.StageManifest.v2",
            "input_resolution": a.input_resolution,
            "authority_resolution": 1024,
            "record_count": len(result),
            "records": result,
        },
    )


if __name__ == "__main__":
    main()
