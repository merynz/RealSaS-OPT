from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Iterable
import numpy as np


def _hash(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _vec3(v, name: str) -> tuple[float, float, float]:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite vec3")
    return tuple(map(float, a))


@dataclass(frozen=True)
class OrthographicCameraV2:
    view_index: int
    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    forward: tuple[float, float, float]
    half_extent: float
    resolution: int = 1024
    schema_version: str = "RealSaS.OrthographicCamera.v2"

    def __post_init__(self) -> None:
        if self.view_index not in range(8):
            raise ValueError("view_index must be 0..7")
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))
        object.__setattr__(self, "right", _vec3(self.right, "right"))
        object.__setattr__(self, "up", _vec3(self.up, "up"))
        object.__setattr__(self, "forward", _vec3(self.forward, "forward"))
        if not math.isfinite(float(self.half_extent)) or self.half_extent <= 0:
            raise ValueError("half_extent must be positive")
        if int(self.resolution) != 1024:
            raise ValueError("IRIS V2 production observation contract requires native 1024 resolution")
        basis = np.asarray([self.right, self.up, self.forward], np.float64)
        norms = np.linalg.norm(basis, axis=1)
        gram = basis @ basis.T
        if np.max(np.abs(norms - 1.0)) > 1e-5 or np.max(np.abs(gram - np.eye(3))) > 1e-5:
            raise ValueError("camera basis must be orthonormal")

    @property
    def camera_hash(self) -> str:
        return _hash(asdict(self))

    def ray_origin_for_grid(self, grid_xy):
        g = np.asarray(grid_xy, np.float64)
        o = np.asarray(self.origin, np.float64)
        r = np.asarray(self.right, np.float64)
        u = np.asarray(self.up, np.float64)
        return o + g[..., 0, None] * self.half_extent * r - g[..., 1, None] * self.half_extent * u

    def point_for_grid_depth(self, grid_xy, depth):
        d = np.asarray(depth, np.float64)
        f = np.asarray(self.forward, np.float64)
        return self.ray_origin_for_grid(grid_xy) + d[..., None] * f

    def project_grid(self, points):
        p = np.asarray(points, np.float64) - np.asarray(self.origin, np.float64)
        gx = (p @ np.asarray(self.right, np.float64)) / self.half_extent
        gy = -(p @ np.asarray(self.up, np.float64)) / self.half_extent
        return np.stack([gx, gy], axis=-1)

    def depth_for_point(self, points):
        p = np.asarray(points, np.float64) - np.asarray(self.origin, np.float64)
        return p @ np.asarray(self.forward, np.float64)

    def grid_to_pixel_center(self, grid_xy):
        g = np.asarray(grid_xy, np.float64)
        x = (g[..., 0] + 1.0) * 0.5 * self.resolution - 0.5
        y = (g[..., 1] + 1.0) * 0.5 * self.resolution - 0.5
        return np.stack([x, y], axis=-1)


@dataclass(frozen=True)
class ObservationContractV2:
    asset_id: str
    cameras: tuple[OrthographicCameraV2, ...]
    raster_hashes: tuple[str, ...]
    rgba_shape: tuple[int, int, int, int] = (8, 1024, 1024, 4)
    raster_authority: str = "MASTER_SOURCE_TEXTURED_RGBA"
    schema_version: str = "RealSaS.ObservationContract.v2"

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id required")
        if tuple(c.view_index for c in self.cameras) != tuple(range(8)):
            raise ValueError("exact ordered camera set 0..7 required")
        if len(self.raster_hashes) != 8 or any(not x for x in self.raster_hashes):
            raise ValueError("exact eight raster hashes required")
        if tuple(self.rgba_shape) != (8, 1024, 1024, 4):
            raise ValueError("native 8x1024x1024 RGBA required; implicit resize forbidden")

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        payload["camera_hashes"] = [c.camera_hash for c in self.cameras]
        return _hash(payload)


def validate_rgba_array(images, contract: ObservationContractV2) -> np.ndarray:
    arr = np.asarray(images)
    if arr.shape != contract.rgba_shape:
        raise ValueError(f"observation raster shape drift:{arr.shape}")
    if arr.dtype not in (np.uint8, np.float16, np.float32, np.float64):
        raise ValueError("unsupported RGBA dtype")
    if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
        raise ValueError("non-finite RGBA")
    return arr


def freeze_contract(asset_id: str, cameras: Iterable[OrthographicCameraV2], raster_hashes: Iterable[str]) -> ObservationContractV2:
    # Preserve caller pairing between ordered camera and raster hash. The constructor
    # fail-closes if cameras are not exactly ordered 0..7.
    return ObservationContractV2(asset_id, tuple(cameras), tuple(raster_hashes))


@dataclass(frozen=True)
class ObservationFileManifestV2:
    asset_id: str
    rgba_paths: tuple[str, ...]
    camera_paths: tuple[str, ...]
    rgba_sha256: tuple[str, ...]
    camera_sha256: tuple[str, ...]
    raster_authority: str = "MASTER_SOURCE_TEXTURED_RGBA"
    schema_version: str = "RealSaS.ObservationFileManifest.v2"

    def __post_init__(self) -> None:
        if any(len(x) != 8 for x in (self.rgba_paths, self.camera_paths, self.rgba_sha256, self.camera_sha256)):
            raise ValueError("observation file manifest requires exact eight paths/hashes")
        if any(not str(x) for seq in (self.rgba_paths, self.camera_paths, self.rgba_sha256, self.camera_sha256) for x in seq):
            raise ValueError("empty observation manifest field")


def _sha256_path(path) -> str:
    from pathlib import Path
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def camera_from_renderer_json_v2(path, expected_view: int) -> OrthographicCameraV2:
    from pathlib import Path
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    yaw = payload.get("yaw_deg")
    if yaw is not None and abs(float(yaw) - 45.0 * expected_view) > 1e-6:
        raise ValueError("camera yaw/order drift")
    if "screen_up" in payload:
        up = payload["screen_up"]
    elif "up" in payload:
        up = payload["up"]
    else:
        raise ValueError("camera screen_up missing")
    origin = payload.get("origin", (0.0, 0.0, 0.0))
    return OrthographicCameraV2(
        expected_view,
        tuple(origin),
        tuple(payload["right"]),
        tuple(up),
        tuple(payload["forward"]),
        float(payload["half_extent"]),
        int(payload.get("resolution", 1024) if not isinstance(payload.get("resolution", 1024), (list, tuple)) else payload["resolution"][0]),
    )


def load_observation_manifest_v2(manifest: ObservationFileManifestV2):
    """Load exactly the paths frozen by DATA-01; never guess a render variant."""
    from PIL import Image
    images = []
    cameras = []
    for v in range(8):
        if _sha256_path(manifest.rgba_paths[v]) != manifest.rgba_sha256[v]:
            raise ValueError(f"rgba hash drift:V{v}")
        if _sha256_path(manifest.camera_paths[v]) != manifest.camera_sha256[v]:
            raise ValueError(f"camera hash drift:V{v}")
        with Image.open(manifest.rgba_paths[v]) as im:
            if im.mode != "RGBA" or im.size != (1024, 1024):
                raise ValueError(f"native RGBA contract drift:V{v}:{im.mode}:{im.size}")
            images.append(np.asarray(im, dtype=np.uint8))
        cameras.append(camera_from_renderer_json_v2(manifest.camera_paths[v], v))
    contract = ObservationContractV2(manifest.asset_id, tuple(cameras), manifest.rgba_sha256, raster_authority=manifest.raster_authority)
    return np.stack(images, axis=0), contract
