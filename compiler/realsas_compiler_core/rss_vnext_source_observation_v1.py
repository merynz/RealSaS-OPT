from __future__ import annotations

"""Lossless, two-observation RSS.vNext canonical input contract.

This is a canonical representation of *provided source bytes*, not generated
views, camera estimation, a rights qualification, or mesh/source truth. It is
separate from the historical eight-view SourceObservation.v1 used by the
research/Shape v2 lane; silently relabeling one as the other is forbidden.
"""

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.TwoViewSourceObservation.v1"
VIEW_IDS = ("FRONT", "BACK")
MAX_BYTES = 128 * 1024 * 1024
PROVENANCE = "ORIGINAL_STRAIGHT_RGBA_UINT8_NO_RESAMPLING"
NORMALIZATION = "RGBA_UINT8_DIV_255_FLOAT32_NO_COMPOSITE_NO_RECENTER"


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _digest_valid(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


@dataclass(frozen=True, eq=False)
class RSSvNextTwoViewSourceObservationV1:
    """Both original RGBA images and their measured 13D orthographic cameras."""

    native_rgba: np.ndarray  # [2,4,H,W], exact uint8
    camera_features: np.ndarray  # [2,13], exact float32 in common frame
    source_rights_claim_sha256: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        rgba = np.asarray(self.native_rgba)
        camera = np.asarray(self.camera_features)
        if self.schema != SCHEMA:
            raise QualificationError("RSS_VNEXT_SOURCE_SCHEMA_INVALID")
        if (
            rgba.ndim != 4 or rgba.shape[:2] != (2, 4)
            or rgba.dtype != np.uint8 or
            min(rgba.shape[2:]) < 14 or max(rgba.shape[2:]) > 2048
        ):
            raise QualificationError("RSS_VNEXT_SOURCE_RGBA_SHAPE_OR_DTYPE_INVALID")
        if rgba.nbytes + 104 > MAX_BYTES:
            raise QualificationError("RSS_VNEXT_SOURCE_RESOURCE_BUDGET_EXCEEDED")
        if camera.shape != (2, 13) or camera.dtype != np.float32 or not np.isfinite(camera).all():
            raise QualificationError("RSS_VNEXT_SOURCE_CAMERA_SHAPE_OR_DTYPE_INVALID")
        if not _digest_valid(self.source_rights_claim_sha256):
            raise QualificationError("RSS_VNEXT_SOURCE_RIGHTS_CLAIM_HASH_REQUIRED")
        if any(not np.any(rgba[i, 3] > 0) for i in (0, 1)):
            raise QualificationError("RSS_VNEXT_SOURCE_EMPTY_OBSERVED_ALPHA")
        for i in (0, 1):
            right, up, forward = (
                camera[i, 3:6], camera[i, 6:9], camera[i, 9:12]
            )
            basis = np.stack((right, up, forward), axis=0).astype(np.float64)
            if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-3, rtol=0):
                raise QualificationError("RSS_VNEXT_SOURCE_CAMERA_BASIS_NON_ORTHONORMAL")
            if np.linalg.det(basis) < 0.99:
                raise QualificationError("RSS_VNEXT_SOURCE_CAMERA_HANDEDNESS_DRIFT")
            if not math.isfinite(float(camera[i, 12])) or camera[i, 12] <= 0:
                raise QualificationError("RSS_VNEXT_SOURCE_CAMERA_EXTENT_INVALID")
        if float(np.dot(camera[0, 9:12], camera[1, 9:12])) > -0.95:
            raise QualificationError("RSS_VNEXT_SOURCE_FRONT_BACK_CAMERA_OPPOSITION_REQUIRED")
        native = np.array(rgba, order="C", copy=True)
        cameras = np.array(camera, order="C", copy=True)
        native.setflags(write=False)
        cameras.setflags(write=False)
        object.__setattr__(self, "native_rgba", native)
        object.__setattr__(self, "camera_features", cameras)

    @property
    def view_pixel_sha256(self) -> tuple[str, str]:
        return tuple(_digest_array(self.native_rgba[i]) for i in (0, 1))

    @property
    def camera_sha256(self) -> str:
        return _digest_array(self.camera_features)

    @property
    def source_binding_sha256(self) -> str:
        return content_sha256({
            "schema": self.schema,
            "view_ids": VIEW_IDS,
            "view_pixel_sha256": self.view_pixel_sha256,
            "camera_sha256": self.camera_sha256,
            "source_rights_claim_sha256": self.source_rights_claim_sha256,
            "provenance": PROVENANCE,
            "normalization": NORMALIZATION,
            "native_resolution_hw": list(self.native_rgba.shape[-2:]),
            "qualification_granted": False,
        })

    def to_model_tensors(self, *, device: str = "cpu") -> tuple[Any, Any]:
        """Deterministic value-preserving normalization. Never synthesize rays."""
        import torch

        rgba = torch.from_numpy(self.native_rgba.copy()).unsqueeze(0).to(
            device=device, dtype=torch.float32
        ).div_(255.0)
        camera = torch.from_numpy(self.camera_features.copy()).unsqueeze(0).to(
            device=device, dtype=torch.float32
        )
        return rgba.contiguous(), camera.contiguous()

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_binding_sha256": self.source_binding_sha256,
            "view_ids": list(VIEW_IDS),
            "view_pixel_sha256": list(self.view_pixel_sha256),
            "camera_sha256": self.camera_sha256,
            "source_rights_claim_sha256": self.source_rights_claim_sha256,
            "normalization": NORMALIZATION,
            "provenance": PROVENANCE,
            "qualification_status": "UNQUALIFIED_SOURCE_BYTES_CANONICALIZED",
            "source_or_model_rights_verified": False,
            "auxiliary_views_observed": False,
        }


def canonicalize_two_view_source_v1(
    native_rgba: np.ndarray,
    camera_features: np.ndarray,
    *,
    source_rights_claim_sha256: str,
) -> RSSvNextTwoViewSourceObservationV1:
    """Strictly typed source bytes. Never round/crop/recenter/transpose implicitly."""
    return RSSvNextTwoViewSourceObservationV1(
        native_rgba=native_rgba,
        camera_features=camera_features,
        source_rights_claim_sha256=source_rights_claim_sha256,
    )


__all__ = [
    "RSSvNextTwoViewSourceObservationV1",
    "canonicalize_two_view_source_v1",
    "SCHEMA", "VIEW_IDS", "MAX_BYTES", "PROVENANCE", "NORMALIZATION",
]
