from __future__ import annotations

"""Pure reference-raster convention for V2 geometry/coverage gates."""

from dataclasses import asdict, dataclass

from .hashing import content_sha256

REFERENCE_RASTER_SCHEMA = "RealSaS.ReferenceRasterContract.v1"


@dataclass(frozen=True)
class ReferenceRasterContractV1:
    pixel_center: str = "HALF_INTEGER_CENTER"
    triangle_fill_rule: str = "TOP_LEFT"
    uv_origin: str = "TOP_LEFT"
    texture_filter: str = "BILINEAR"
    wrap_mode: str = "CLAMP_TO_EDGE"
    alpha_encoding: str = "STRAIGHT"
    color_space: str = "SRGB_RGBA8"
    blend_mode: str = "SOURCE_OVER"
    depth_compare: str = "LESS_EQUAL_WITH_STABLE_TIE_BREAK"
    depth_space: str = "POSED_CAMERA_FORWARD_DEPTH_SMALLER_IS_NEARER"
    face_culling: str = "DISABLED_UNLESS_EXPLICIT_ATTACHMENT_POLICY"
    per_frame_affine_refit_forbidden: bool = True
    schema_version: str = REFERENCE_RASTER_SCHEMA

    @property
    def contract_hash(self) -> str:
        return content_sha256(asdict(self))
