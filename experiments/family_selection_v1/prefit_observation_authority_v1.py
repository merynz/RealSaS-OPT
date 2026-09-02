from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


SCHEMA = "RealSaS.PrefitObservationAuthority.v1"
ALLOWED_AUTHORITIES = frozenset({
    "TEXTURED_RGBA_WITH_SOURCE_APPEARANCE",
    "MASTER_SOURCE_TEXTURED_RGBA",
})
FORBIDDEN_RASTER_TOKENS = (
    "cel_clean",
    "ink_cel",
    "toon_sibling",
    "geometry_control",
    "geometry-isolation",
)
EXPECTED_YAWS = tuple(range(0, 360, 45))


def _sha256_path(path: str | Path) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_hash(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _assert_textured_path(path: str | Path) -> None:
    value = str(path).replace("\\", "/").lower()
    bad = [token for token in FORBIDDEN_RASTER_TOKENS if token in value]
    if bad:
        raise ValueError(f"NON_PRODUCT_RASTER_FORBIDDEN:{bad}:{path}")


@dataclass(frozen=True)
class ObservationViewEvidenceV1:
    view_index: int
    yaw_deg: int
    rgba_path: str
    rgba_sha256: str
    camera_path: str
    camera_sha256: str


@dataclass(frozen=True)
class PrefitObservationAuthorityV1:
    schema: str
    asset_id: str
    raster_authority: str
    native_resolution: int
    views: tuple[ObservationViewEvidenceV1, ...]
    exact_eight_views: bool
    exact_camera_raster_authority: bool
    explicit_source_textured_rgba: bool
    observation_authority_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def verify_prefit_observation_authority_v1(
    *,
    asset_id: str,
    raster_authority: str,
    views: Iterable[ObservationViewEvidenceV1],
) -> PrefitObservationAuthorityV1:
    if not asset_id:
        raise ValueError("asset_id required")
    if raster_authority not in ALLOWED_AUTHORITIES:
        raise ValueError(f"SOURCE_TEXTURED_RGBA_AUTHORITY_REQUIRED:{raster_authority}")

    rows = tuple(views)
    if len(rows) != 8:
        raise ValueError(f"EXACT_EIGHT_VIEWS_REQUIRED:{len(rows)}")
    if tuple(row.view_index for row in rows) != tuple(range(8)):
        raise ValueError("VIEW_ORDER_MUST_BE_EXACT_0_TO_7")
    if tuple(int(row.yaw_deg) for row in rows) != EXPECTED_YAWS:
        raise ValueError("YAW_ORDER_MUST_BE_EXACT_0_45_90_135_180_225_270_315")

    verified: list[ObservationViewEvidenceV1] = []
    for row in rows:
        _assert_textured_path(row.rgba_path)
        rgba_path = Path(row.rgba_path)
        camera_path = Path(row.camera_path)
        if not rgba_path.is_file():
            raise FileNotFoundError(f"RGBA_MISSING:V{row.view_index}:{rgba_path}")
        if not camera_path.is_file():
            raise FileNotFoundError(f"CAMERA_MISSING:V{row.view_index}:{camera_path}")
        if _sha256_path(rgba_path) != row.rgba_sha256:
            raise ValueError(f"RGBA_SHA_DRIFT:V{row.view_index}")
        if _sha256_path(camera_path) != row.camera_sha256:
            raise ValueError(f"CAMERA_SHA_DRIFT:V{row.view_index}")
        with Image.open(rgba_path) as im:
            if im.size != (1024, 1024):
                raise ValueError(f"NATIVE_1024_REQUIRED:V{row.view_index}:{im.size}")
            if im.mode != "RGBA":
                raise ValueError(f"RGBA_MODE_REQUIRED:V{row.view_index}:{im.mode}")
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        yaw = camera.get("yaw_deg")
        if yaw is None or abs(float(yaw) - float(row.yaw_deg)) > 1e-6:
            raise ValueError(f"CAMERA_YAW_DRIFT:V{row.view_index}")
        verified.append(row)

    hash_payload = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "raster_authority": raster_authority,
        "native_resolution": 1024,
        "views": [asdict(row) for row in verified],
        "exact_eight_views": True,
        "exact_camera_raster_authority": True,
        "explicit_source_textured_rgba": True,
    }
    return PrefitObservationAuthorityV1(
        schema=SCHEMA,
        asset_id=asset_id,
        raster_authority=raster_authority,
        native_resolution=1024,
        views=tuple(verified),
        exact_eight_views=True,
        exact_camera_raster_authority=True,
        explicit_source_textured_rgba=True,
        observation_authority_sha256=_canonical_hash(hash_payload),
    )


def load_and_verify_manifest_v1(path: str | Path) -> PrefitObservationAuthorityV1:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != "RealSaS.PrefitObservationManifest.v1":
        raise ValueError("observation manifest schema mismatch")
    rows = tuple(ObservationViewEvidenceV1(**row) for row in obj.get("views", ()))
    return verify_prefit_observation_authority_v1(
        asset_id=str(obj.get("asset_id", "")),
        raster_authority=str(obj.get("raster_authority", "")),
        views=rows,
    )
