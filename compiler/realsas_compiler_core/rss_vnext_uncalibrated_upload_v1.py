from __future__ import annotations

"""Lossless pre-canonicalization two-image intake: scene, pose, camera UNKNOWN.

This is upstream of existing RSSvNextTwoViewSourceObservationV1, which requires
two measured antipodal 13D cameras. An arbitrary user upload cannot be
silently cast into that calibrated source schema. StdGEN (CVPR 2025) applies
learned A-pose canonicalization and multiview diffusion after preparing the
reference image. Our generic two-view LoRA candidate will need jointly
conditioned pose/camera estimation; this source envelope preserves both input
files until trained canonicalization exists.

Original uploaded file bytes take precedence over any derived resize, matte,
future diffusion view or 3D inference. View names are USER CLAIMS only.
No scene segmentation, camera estimation, anti-podal baseline, canonical pose,
source depth, foreground ownership, or product authority is manufactured.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256

SCHEMA = "RealSaS.RSSvNext.UncalibratedTwoImageSource.v1"
CONDITIONING_SCHEMA = "RealSaS.RSSvNext.DerivedTwoImageConditioning.v1"
_ALLOWED_FORMATS = frozenset(("PNG", "JPEG", "WEBP"))
_ALLOWED_CLAIMS = frozenset(("UNKNOWN", "FRONT", "BACK", "OTHER"))
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_ORIGINAL_PIXELS = 16_777_216
_MAX_SOURCE_ARCHIVE = 70 * 1024 * 1024
_ARCHIVE_NAMES = frozenset(("manifest.json", "source_0", "source_1"))


def _fail(reason: str) -> None:
    raise ValueError("RSS_VNEXT_RAW_PAIR_" + reason)


def _hex64(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _inspect_image(blob: bytes) -> dict[str, Any]:
    from PIL import Image
    if type(blob) is not bytes or not 1 <= len(blob) <= _MAX_IMAGE_BYTES:
        _fail("IMAGE_BYTES_BUDGET_OR_TYPE")
    try:
        with Image.open(io.BytesIO(blob)) as image:
            codec = str(image.format)
            if codec not in _ALLOWED_FORMATS:
                _fail("UNSUPPORTED_IMAGE_CODEC")
            h, w = image.height, image.width
            if min(h, w) < 14 or max(h, w) > 4096 or h * w > _MAX_ORIGINAL_PIXELS:
                _fail("ORIGINAL_IMAGE_PIXEL_BUDGET")
            alpha_declared = (
                "A" in image.getbands()
                or "transparency" in image.info
            )
            image.load()
            return {
                "format": codec,
                "width": w,
                "height": h,
                "encoded_mode": str(image.mode),
                "source_alpha_present": bool(alpha_declared),
                "source_foreground_segmented": False,
                "source_camera_calibrated": False,
                "source_pose_known": False,
                "original_bytes_sha256": sha256(blob).hexdigest(),
                "original_bytes": len(blob),
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("RSS_VNEXT_RAW_PAIR_IMAGE_DECODE_INVALID") from exc


@dataclass(frozen=True)
class RSSvNextUncalibratedTwoImageSourceV1:
    original_files: tuple[bytes, bytes]
    source_rights_claim_sha256: str
    view_claims: tuple[str, str] = ("UNKNOWN", "UNKNOWN")
    schema: str = SCHEMA

    def validate(self) -> None:
        if (
            self.schema != SCHEMA
            or type(self.original_files) is not tuple
            or len(self.original_files) != 2
            or type(self.view_claims) is not tuple
            or len(self.view_claims) != 2
            or any(type(c) is not str or c not in _ALLOWED_CLAIMS
                   for c in self.view_claims)
            or not _hex64(self.source_rights_claim_sha256)
        ):
            _fail("TWO_IMAGE_OR_USER_CLAIM_SCHEMA")
        for raw in self.original_files:
            _inspect_image(raw)

    @property
    def source_images(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.validate()
        return tuple(_inspect_image(blob) for blob in self.original_files)

    @property
    def source_binding_sha256(self) -> str:
        return content_sha256({
            "schema": SCHEMA,
            "source_images": self.source_images,
            "view_claims": list(self.view_claims),
            "source_rights_claim_sha256": self.source_rights_claim_sha256,
            "source_cameras_known": False,
            "source_pose_canonicalized": False,
            "foreground_mask_qualified": False,
            "original_observation_count": 2,
            "generated_view_count": 0,
        })

    def manifest(self) -> dict[str, Any]:
        data = {
            "schema": SCHEMA,
            "source_images": list(self.source_images),
            "source_rights_claim_sha256": self.source_rights_claim_sha256,
            "view_claims": list(self.view_claims),
            "source_camera_state": "UNKNOWN_NO_ORIGINAL_CAMERA_CALIBRATION",
            "source_pose_state": "UNKNOWN_NO_CANONICAL_POSE",
            "foreground_mask_state": "UNKNOWN_OPAQUE_SCENE_ALLOWED",
            "original_observation_count": 2,
            "generated_view_count": 0,
            "is_product_ready": False,
            "is_canonical_mesh_qualified": False,
            "source_rights_verified": False,
            "calibrated_rss_source_artifact": False,
            "source_binding_sha256": self.source_binding_sha256,
        }
        return data


def _fixed_zipinfo(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    return info


def encode_uncalibrated_two_image_source_v1(
    source: RSSvNextUncalibratedTwoImageSourceV1,
) -> bytes:
    if type(source) is not RSSvNextUncalibratedTwoImageSourceV1:
        _fail("TYPED_INPUT_REQUIRED")
    source.validate()
    stream = io.BytesIO()
    manifest = json.dumps(
        source.manifest(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")
    with ZipFile(stream, "w") as archive:
        for name, data in (
            ("manifest.json", manifest),
            ("source_0", source.original_files[0]),
            ("source_1", source.original_files[1]),
        ):
            archive.writestr(_fixed_zipinfo(name), data)
    result = stream.getvalue()
    if len(result) > _MAX_SOURCE_ARCHIVE:
        _fail("CAS_ARCHIVE_BYTES_BUDGET")
    return result


def decode_uncalibrated_two_image_source_v1(
    packed: bytes,
) -> RSSvNextUncalibratedTwoImageSourceV1:
    if type(packed) is not bytes or not 1 <= len(packed) <= _MAX_SOURCE_ARCHIVE:
        _fail("CAS_ARCHIVE_BYTE_BUDGET_OR_TYPE")
    try:
        with ZipFile(io.BytesIO(packed), "r") as archive:
            entries = archive.infolist()
            if (
                len(entries) != 3
                or {item.filename for item in entries} != _ARCHIVE_NAMES
                or any(
                    item.file_size > (_MAX_IMAGE_BYTES if item.filename != "manifest.json" else 8192)
                    for item in entries
                )
                or sum(item.file_size for item in entries) > 2 * _MAX_IMAGE_BYTES + 8192
                or any(item.compress_type != ZIP_STORED for item in entries)
            ):
                _fail("CAS_ARCHIVE_FIELDS_OR_RESOURCE_BUDGET")
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            files = (archive.read("source_0"), archive.read("source_1"))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("RSS_VNEXT_RAW_PAIR_CAS_ARCHIVE_PARSE_FAILED") from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema") != SCHEMA
        or type(manifest.get("view_claims")) is not list
        or len(manifest["view_claims"]) != 2
        or not _hex64(manifest.get("source_binding_sha256"))
    ):
        _fail("CAS_MANIFEST_SCHEMA_INVALID")
    source = RSSvNextUncalibratedTwoImageSourceV1(
        original_files=files,
        source_rights_claim_sha256=manifest["source_rights_claim_sha256"],
        view_claims=tuple(manifest["view_claims"]),
    )
    source.validate()
    if source.manifest() != manifest:
        _fail("CAS_MANIFEST_BYTES_OR_AUTHORITY_DRIFT")
    return source


@dataclass(frozen=True)
class RSSvNextDerivedConditioningPairV1:
    rgba_uint8: np.ndarray  # [2,4,H,W], derived PREVIEW, not original bytes
    original_footprint: np.ndarray  # [2,H,W], valid resampled source footprint
    source_binding_sha256: str
    pixel_transform_rows: tuple[dict[str, Any], dict[str, Any]]
    conditioning_sha256: str
    schema: str = CONDITIONING_SCHEMA
    authority: str = "DERIVED_RESAMPLED_CONDITIONING_NOT_ORIGINAL_SOURCE"


def derive_uncalibrated_pair_conditioning_v1(
    source: RSSvNextUncalibratedTwoImageSourceV1,
    *,
    target_square_size: int = 512,
) -> RSSvNextDerivedConditioningPairV1:
    """Pad, NEVER crop/force matte/estimate camera; preserve exact raw CAS.

    A valid-footprint mask marks image extent, NOT character segmentation.
    Transparent padding does not license generated anatomy or 3D depth.
    EXIF orientation is applied only to this separately hashed derivative.
    """
    from PIL import Image, ImageOps
    source.validate()
    if type(target_square_size) is not int or not 64 <= target_square_size <= 1024:
        _fail("CONDITIONING_TARGET_SIZE_INVALID")
    images = []
    masks = []
    rows = []
    for index, blob in enumerate(source.original_files):
        with Image.open(io.BytesIO(blob)) as raw:
            oriented = ImageOps.exif_transpose(raw)
            rgba = oriented.convert("RGBA")
        orig_w, orig_h = rgba.size
        scale = min(target_square_size / orig_w, target_square_size / orig_h)
        dst_w = max(1, min(target_square_size, round(orig_w * scale)))
        dst_h = max(1, min(target_square_size, round(orig_h * scale)))
        resized = rgba.resize((dst_w, dst_h), resample=Image.Resampling.LANCZOS)
        left = (target_square_size - dst_w) // 2
        top = (target_square_size - dst_h) // 2
        canvas = Image.new("RGBA", (target_square_size, target_square_size), (0,0,0,0))
        canvas.paste(resized, (left, top))
        view = np.transpose(np.asarray(canvas, dtype=np.uint8), (2,0,1)).copy()
        mask = np.zeros((target_square_size, target_square_size), dtype=np.bool_)
        mask[top:top+dst_h, left:left+dst_w] = True
        rows.append({
            "original_view_index": index,
            "original_file_sha256": sha256(blob).hexdigest(),
            "oriented_width": orig_w,
            "oriented_height": orig_h,
            "resampled_width": dst_w,
            "resampled_height": dst_h,
            "letterbox_left": left,
            "letterbox_top": top,
            "scale_x": dst_w/orig_w,
            "scale_y": dst_h/orig_h,
            "interpolation": "LANCZOS_DERIVED_NOT_ORIGINAL",
            "foreground_extracted": False,
            "pose_estimated": False,
            "camera_estimated": False,
        })
        images.append(view)
        masks.append(mask)
    pixels = np.stack(images).astype(np.uint8)
    footprints = np.stack(masks)
    pixels.setflags(write=False)
    footprints.setflags(write=False)
    digest = content_sha256({
        "schema": CONDITIONING_SCHEMA,
        "source_binding_sha256": source.source_binding_sha256,
        "pixel_transform_rows": rows,
        "derived_rgba_sha256": sha256(pixels.tobytes(order="C")).hexdigest(),
        "derived_footprint_sha256": sha256(footprints.tobytes(order="C")).hexdigest(),
        "authority": "DERIVED_RESAMPLED_CONDITIONING_NOT_ORIGINAL_SOURCE",
    })
    return RSSvNextDerivedConditioningPairV1(
        rgba_uint8=pixels, original_footprint=footprints,
        source_binding_sha256=source.source_binding_sha256,
        pixel_transform_rows=tuple(rows),
        conditioning_sha256=digest,
    )


__all__ = [
    "SCHEMA", "CONDITIONING_SCHEMA",
    "RSSvNextUncalibratedTwoImageSourceV1",
    "RSSvNextDerivedConditioningPairV1",
    "encode_uncalibrated_two_image_source_v1",
    "decode_uncalibrated_two_image_source_v1",
    "derive_uncalibrated_pair_conditioning_v1",
]
