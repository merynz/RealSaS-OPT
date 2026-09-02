from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class MasterViewBindingV1:
    view_index: int
    image_relpath: str
    image_sha256: str
    camera_relpath: str
    camera_sha256: str
    raster_relpath: str
    raster_sha256: str
    yaw_deg: float
    resolution: tuple[int, int]


@dataclass(frozen=True)
class MasterFamilyManifestV1:
    asset_id: str
    image_filename: str
    geometry_relpath: str
    geometry_sha256: str
    views: tuple[MasterViewBindingV1, ...]
    file_hashes: tuple[tuple[str, str], ...]
    manifest_hash: str
    schema_version: str = "RealSaS.MasterFamilyManifest.v1"

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_camera(path: Path, view_index: int) -> float:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("contract") != "realsas.level_orthographic_z_orbit.v1":
        raise ValueError(f"CAMERA_CONTRACT_DRIFT:{path}")
    yaw = float(obj.get("yaw_deg", float("nan")))
    if not np.isfinite(yaw) or abs(yaw - 45.0 * view_index) > 1e-6:
        raise ValueError(f"CAMERA_VIEW_ORDER_DRIFT:{path}:{yaw}")
    for key in ("forward", "right", "screen_up"):
        vec = np.asarray(obj.get(key), dtype=np.float64)
        if vec.shape != (3,) or not np.isfinite(vec).all():
            raise ValueError(f"CAMERA_BASIS_INVALID:{path}:{key}")
    half_extent = float(obj.get("half_extent", 0.0))
    if not np.isfinite(half_extent) or half_extent <= 0.0:
        raise ValueError(f"CAMERA_HALF_EXTENT_INVALID:{path}")
    return yaw


def _validate_raster(path: Path) -> tuple[int, int]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pixel_linear_index", "triangle_id", "barycentric_uv", "resolution"}
        if not required.issubset(data.files):
            raise ValueError(f"RASTER_FIELDS_MISSING:{path}:{sorted(required - set(data.files))}")
        pixel = np.asarray(data["pixel_linear_index"], dtype=np.int64).reshape(-1)
        tri = np.asarray(data["triangle_id"], dtype=np.int64).reshape(-1)
        uv = np.asarray(data["barycentric_uv"], dtype=np.float64)
        raw_res = np.asarray(data["resolution"], dtype=np.int64).reshape(-1)
    if len(raw_res) == 1:
        resolution = (int(raw_res[0]), int(raw_res[0]))
    elif len(raw_res) >= 2:
        resolution = (int(raw_res[0]), int(raw_res[1]))
    else:
        raise ValueError(f"RASTER_RESOLUTION_MISSING:{path}")
    if resolution != (1024, 1024):
        raise ValueError(f"RASTER_RESOLUTION_DRIFT:{path}:{resolution}")
    if tri.shape != pixel.shape or uv.shape != (len(pixel), 2):
        raise ValueError(f"RASTER_SHAPE_DRIFT:{path}")
    if len(pixel) == 0 or pixel.min() < 0 or pixel.max() >= 1024 * 1024:
        raise ValueError(f"RASTER_PIXEL_RANGE_DRIFT:{path}")
    if len(np.unique(pixel)) != len(pixel):
        raise ValueError(f"RASTER_DUPLICATE_PIXEL:{path}")
    return resolution


def _validate_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        if image.size != (1024, 1024):
            raise ValueError(f"MASTER_IMAGE_RESOLUTION_DRIFT:{path}:{image.size}")
        if image.mode != "RGBA":
            raise ValueError(f"MASTER_IMAGE_MODE_DRIFT:{path}:{image.mode}")
        return tuple(map(int, image.size))


def build_master_family_manifest_v1(
    asset_dir: str | Path,
    *,
    image_filename: str,
    expected_file_hashes: Mapping[str, str] | None = None,
    expected_manifest_hash: str | None = None,
) -> MasterFamilyManifestV1:
    """Bind one Master family to exact observation/camera/raster/geometry bytes.

    `image_filename` is intentionally explicit. The builder never silently chooses
    cel-clean, textured sibling, or any other appearance authority.
    """
    asset_dir = Path(asset_dir)
    if not asset_dir.is_dir() or not image_filename or Path(image_filename).name != image_filename:
        raise ValueError("explicit existing asset_dir and basename image_filename required")
    asset_id = asset_dir.name
    geometry = asset_dir / "primary_geometry.npz"
    if not geometry.is_file():
        raise FileNotFoundError(f"MASTER_GEOMETRY_MISSING:{geometry}")
    hashes: dict[str, str] = {}
    geometry_rel = "primary_geometry.npz"
    hashes[geometry_rel] = sha256_file(geometry)
    views = []
    for view_index in range(8):
        view_dir = asset_dir / "renders" / f"V{view_index}"
        image = view_dir / image_filename
        camera = view_dir / "camera.json"
        raster = view_dir / "raster_authority.npz"
        for required in (image, camera, raster):
            if not required.is_file():
                raise FileNotFoundError(f"MASTER_MEMBER_MISSING:{required}")
        image_size = _validate_image(image)
        yaw = _validate_camera(camera, view_index)
        raster_size = _validate_raster(raster)
        if image_size != raster_size:
            raise ValueError(f"MASTER_IMAGE_RASTER_RESOLUTION_MISMATCH:{view_dir}")
        image_rel = f"renders/V{view_index}/{image_filename}"
        camera_rel = f"renders/V{view_index}/camera.json"
        raster_rel = f"renders/V{view_index}/raster_authority.npz"
        hashes[image_rel] = sha256_file(image)
        hashes[camera_rel] = sha256_file(camera)
        hashes[raster_rel] = sha256_file(raster)
        views.append(MasterViewBindingV1(
            view_index=view_index,
            image_relpath=image_rel,
            image_sha256=hashes[image_rel],
            camera_relpath=camera_rel,
            camera_sha256=hashes[camera_rel],
            raster_relpath=raster_rel,
            raster_sha256=hashes[raster_rel],
            yaw_deg=yaw,
            resolution=raster_size,
        ))
    expected = dict(expected_file_hashes or {})
    unknown_expected = set(expected) - set(hashes)
    if unknown_expected:
        raise ValueError(f"EXPECTED_HASH_PATH_UNKNOWN:{sorted(unknown_expected)}")
    mismatched = {path: (expected[path], hashes[path]) for path in expected if expected[path] != hashes[path]}
    if mismatched:
        raise ValueError(f"MASTER_MEMBER_HASH_DRIFT:{mismatched}")
    payload = {
        "schema_version": "RealSaS.MasterFamilyManifest.v1",
        "asset_id": asset_id,
        "image_filename": image_filename,
        "geometry_relpath": geometry_rel,
        "geometry_sha256": hashes[geometry_rel],
        "views": [asdict(view) for view in views],
        "file_hashes": sorted(hashes.items()),
    }
    manifest_hash = _canonical_hash(payload)
    if expected_manifest_hash is not None and manifest_hash != expected_manifest_hash:
        raise ValueError(f"MASTER_MANIFEST_HASH_DRIFT:{expected_manifest_hash}:{manifest_hash}")
    return MasterFamilyManifestV1(
        asset_id=asset_id,
        image_filename=image_filename,
        geometry_relpath=geometry_rel,
        geometry_sha256=hashes[geometry_rel],
        views=tuple(views),
        file_hashes=tuple(sorted(hashes.items())),
        manifest_hash=manifest_hash,
    )
