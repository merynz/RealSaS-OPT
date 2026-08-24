from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

VIEWS = 8
STAGE_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStage.v1"
STAGE_MANIFEST_SCHEMA = "RealSaS.IRISSinglePoseV2.RepresentationStageManifest.v1"
STAGE_PROFILE = "representation_authority_geometry_only"
AUTHORITY_RESOLUTION = 1024


def atomic_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def array_hash(name: str, arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(name.encode("utf-8") + b"\0")
    h.update(a.dtype.str.encode("ascii") + b"\0")
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode("ascii") + b"\0")
    h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def copy_and_hash(src: Path, dst: Path, chunk: int = 8 << 20) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            block = fin.read(chunk)
            if not block:
                break
            h.update(block)
            fout.write(block)
    return h.hexdigest()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def output_hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "STAGE.json")
    }


def marker_reusable(final: Path, marker: dict) -> bool:
    if marker.get("schema") != STAGE_SCHEMA or marker.get("stage_profile") != STAGE_PROFILE:
        return False
    if marker.get("physical_firewall") is not True or marker.get("images_staged") is not False:
        return False
    expected = marker.get("staged_sha256")
    if not isinstance(expected, dict) or not expected:
        return False
    for rel, digest in expected.items():
        p = final / rel
        if not p.is_file() or sha256_file(p) != digest:
            return False
    return True


def stage_asset(master_root: Path, out_root: Path, asset_id: str) -> dict:
    src = master_root / "master" / "assets" / asset_id
    final = out_root / "assets" / asset_id
    marker_path = final / "STAGE.json"

    # Reuse is only of a completed atomic local stage inside the same opened run.
    # The marker already preserves the exact legal source hashes used to create it.
    if marker_path.is_file():
        marker = json.load(marker_path.open(encoding="utf-8"))
        if marker.get("asset_id") == asset_id and marker_reusable(final, marker):
            return marker

    if not src.is_dir():
        raise FileNotFoundError(src)

    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=asset_id + ".representation.partial.", dir=final.parent))
    try:
        # Only legal geometry arrays are admitted. Hidden rig/mechanics fields are not
        # copied and are deliberately excluded from the representation-gate fingerprint.
        with np.load(src / "primary_geometry.npz", allow_pickle=False) as z:
            vertices = np.asarray(z["vertices"], np.float32)
            faces = np.asarray(z["faces"], np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise RuntimeError(f"bad vertices {asset_id}: {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise RuntimeError(f"bad faces {asset_id}: {faces.shape}")
        if not np.isfinite(vertices).all():
            raise RuntimeError(f"nonfinite vertices {asset_id}")
        if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise RuntimeError(f"face index out of bounds {asset_id}")

        np.savez(tmp / "primary_geometry.npz", vertices=vertices, faces=faces)
        source_sha256: dict[str, str] = {
            "primary_geometry.vertices": array_hash("vertices", vertices),
            "primary_geometry.faces": array_hash("faces", faces),
        }
        views = []
        for v in range(VIEWS):
            sv = src / "renders" / f"V{v}"
            dv = tmp / "renders" / f"V{v}"
            dv.mkdir(parents=True, exist_ok=True)

            raster_src = sv / "raster_authority.npz"
            camera_src = sv / "camera.json"
            raster_sha = copy_and_hash(raster_src, dv / "raster_authority.npz")
            camera_sha = copy_and_hash(camera_src, dv / "camera.json")
            source_sha256[f"renders/V{v}/raster_authority.npz"] = raster_sha
            source_sha256[f"renders/V{v}/camera.json"] = camera_sha

            cam = json.load((dv / "camera.json").open(encoding="utf-8"))
            yaw = float(cam["yaw_deg"])
            if abs(yaw - 45.0 * v) > 1e-4:
                raise RuntimeError(f"noncanonical yaw {asset_id} V{v}: {yaw}")
            with np.load(dv / "raster_authority.npz", allow_pickle=False) as ra:
                required = {"resolution", "pixel_linear_index", "triangle_id", "barycentric_uv"}
                missing = required - set(ra.files)
                if missing:
                    raise RuntimeError(f"raster authority missing {asset_id} V{v}: {sorted(missing)}")
                res = int(np.asarray(ra["resolution"]).reshape(-1)[0])
                if res != AUTHORITY_RESOLUTION:
                    raise RuntimeError(f"authority resolution drift {asset_id} V{v}: {res}")
                pix = np.asarray(ra["pixel_linear_index"], np.int64)
                if len(pix) == 0 or np.any(pix[1:] <= pix[:-1]):
                    raise RuntimeError(f"bad raster pixels {asset_id} V{v}")
            views.append({"view": v, "yaw_deg": yaw, "raster_pixels": int(len(pix))})

        # Representation Authority does not consume RGB. Assert that no image was
        # accidentally staged into this profile.
        staged_images = list(tmp.rglob("*.png")) + list(tmp.rglob("*.jpg")) + list(tmp.rglob("*.jpeg"))
        if staged_images:
            raise RuntimeError(f"image leaked into representation-only stage: {staged_images[:3]}")

        staged_sha256 = output_hashes(tmp)
        source_fingerprint = canonical_hash({
            "schema": STAGE_SCHEMA,
            "stage_profile": STAGE_PROFILE,
            "asset_id": asset_id,
            "authority_resolution": AUTHORITY_RESOLUTION,
            "legal_source_sha256": source_sha256,
        })
        meta = {
            "schema": STAGE_SCHEMA,
            "stage_profile": STAGE_PROFILE,
            "asset_id": asset_id,
            "authority_resolution": AUTHORITY_RESOLUTION,
            "input_resolution": AUTHORITY_RESOLUTION,
            "physical_firewall": True,
            "geometry_fields": ["vertices", "faces"],
            "images_staged": False,
            "source_dependency_policy": "legal vertices/faces arrays + exact raster_authority bytes + exact camera bytes; RGB and hidden rig/mechanics fields excluded",
            "source_sha256": source_sha256,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Representation-only source stage: geometry+raster+camera, no RGB")
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="FIT,TUNE")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    seed = json.load(open(args.seed_manifest, encoding="utf-8"))
    splits = {x.strip().upper() for x in args.splits.split(",") if x.strip()}
    if splits & {"CAL", "DEV", "EXTERNAL_HOLDOUT"}:
        raise RuntimeError("sealed split requested")
    if not splits <= {"FIT", "TUNE"}:
        raise RuntimeError(f"unsupported/open split set: {sorted(splits)}")
    rows = [r for r in seed["records"] if str(r["split"]).upper() in splits]
    rows = rows[: args.limit] if args.limit else rows

    result = []
    for i, r in enumerate(rows, 1):
        aid = r["asset_id"]
        meta = stage_asset(root, out, aid)
        result.append({
            "asset_id": aid,
            "split": str(r["split"]).upper(),
            "asset_dir": str(out / "assets" / aid),
            "source_fingerprint": meta["source_fingerprint"],
        })
        if i % 5 == 0 or i == len(rows):
            print(f"[repr-stage] {i}/{len(rows)}", flush=True)

    atomic_json(out / "STAGE_MANIFEST.json", {
        "schema": STAGE_MANIFEST_SCHEMA,
        "stage_profile": STAGE_PROFILE,
        "authority_resolution": AUTHORITY_RESOLUTION,
        "input_resolution": AUTHORITY_RESOLUTION,
        "images_staged": False,
        "record_count": len(result),
        "records": result,
    })


if __name__ == "__main__":
    main()
