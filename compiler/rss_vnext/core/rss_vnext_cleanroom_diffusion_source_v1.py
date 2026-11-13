from __future__ import annotations

"""RealSaS-inspired, independently authored two-source diffusion boundary.

CVPR 2025 the CVPR 2025 reference §3.1: SD + ReferenceNet canonicalizes arbitrary pose;
Era3D-style joint multi-view RGB/normal diffusion predicts missing views.
Here TWO *unmodified* customer frames are retained. Generated canonical
frames, generated camera labels, hidden surfaces, and normals have no original
observation authority. No unseen result is upgraded by a SHA256 receipt.

These types precede the calibrated, antipodal RSSvNextTwoViewSourceObservation
interface. They intentionally cannot be passed to its constructor.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import math
from typing import Any

import numpy as np

from .hashing import content_sha256
from .types import QualificationError


RAW_SCHEMA = "RealSaS.RSSvNext.RawTwoSceneObservations.v1"
PREDICTION_SCHEMA = "RealSaS.RSSvNext.CleanroomDiffusionPrediction.v1"
MAX_RAW_BYTES = 128 * 1024 * 1024
STAGE_CANONICAL = "ARBITRARY_TWO_SCENES_TO_CANONICAL_POSE"
STAGE_MULTIVIEW = "CANONICAL_POSE_TO_CAMERA_CONDITIONED_RGBA_NORMAL"
OUTPUT_AUTHORITY = "LEARNED_GENERATION_NOT_ORIGINAL_OBSERVATION"
POSE_POLICY = "MORPHOLOGY_CONDITIONAL_NEUTRAL_POSE_NOT_HUMANOID_A_POSE"
VIEW_HINTS = frozenset({"UNKNOWN", "FRONT_LIKE", "BACK_LIKE", "OTHER"})
_MAX_PREDICTED_BYTES = 128 * 1024 * 1024


def _reject(code: str) -> None:
    raise QualificationError("RSS_VNEXT_CLEANROOM_" + code)


def _digest_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}|{array.shape}".encode("ascii")
    return sha256(header + b"\0" + array.tobytes(order="C")).hexdigest()


def _sha(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(
        ch in "0123456789abcdef" for ch in value
    )


def _valid_camera(values: np.ndarray) -> bool:
    if values.shape != (13,) or not np.isfinite(values).all():
        return False
    basis = np.stack([values[3:6], values[6:9], values[9:12]], axis=0)
    return (
        values[12] > 0
        and np.allclose(basis @ basis.T, np.eye(3), atol=1.e-3, rtol=0.)
        and np.linalg.det(basis) >= .99
    )


@dataclass(frozen=True, eq=False)
class RSSvNextRawTwoSceneObservationsV1:
    """Two original RGB/RGBA user uploads. Camera/foreground may be unknown.

    Frames are independently sized CHW uint8; no fake alpha, crop, backdrop,
    default camera, pose correction or semantic decomposition is performed.
    """
    frames_chw: tuple[np.ndarray, np.ndarray]
    view_hints: tuple[str, str]
    source_rights_claim_sha256: str
    schema: str = RAW_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RAW_SCHEMA or not _sha(self.source_rights_claim_sha256):
            _reject("RAW_SCHEMA_OR_RIGHTS_CLAIM")
        if type(self.frames_chw) is not tuple or len(self.frames_chw) != 2:
            _reject("EXACTLY_TWO_RAW_SCENES_REQUIRED")
        if type(self.view_hints) is not tuple or len(self.view_hints) != 2 or any(
            hint not in VIEW_HINTS for hint in self.view_hints
        ):
            _reject("RAW_VIEW_HINT_INVALID")
        source = []
        total = 0
        for image in self.frames_chw:
            if (
                type(image) is not np.ndarray or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[0] not in (3, 4)
                or min(image.shape[-2:]) < 14 or max(image.shape[-2:]) > 2048
            ):
                _reject("RAW_IMAGE_DOMAIN")
            total += image.nbytes
            clone = np.array(image, order="C", copy=True)
            clone.setflags(write=False)
            source.append(clone)
        if total > MAX_RAW_BYTES:
            _reject("RAW_IMAGE_RESOURCE_BUDGET")
        object.__setattr__(self, "frames_chw", tuple(source))

    @property
    def image_sha256(self) -> tuple[str, str]:
        return tuple(_digest_array(im) for im in self.frames_chw)

    @property
    def source_binding_sha256(self) -> str:
        return content_sha256({
            "schema": RAW_SCHEMA, "source_view_count": 2,
            "image_sha256": self.image_sha256,
            "view_hints_are_not_measured_cameras": self.view_hints,
            "source_rights_claim_sha256": self.source_rights_claim_sha256,
            "camera_geometry_observed": False,
            "foreground_segmentation_observed": False,
            "source_rights_verified": False,
        })


@dataclass(frozen=True, eq=False)
class RSSvNextCleanroomDiffusionPredictionV1:
    """Hash-bound generated candidate. No field carries original RGB authority.

    stage 1: canonical RGBA [2,4,H,W] plus proposed cameras, no normals.
    stage 2: multiview RGBA [V,4,H,W], normals [V,3,H,W], proposed cameras.
    Camera proposals are coordinate labels, NOT measured source cameras.
    """
    source_binding_sha256: str
    generator_checkpoint_sha256: str
    stage: str
    generated_rgba: np.ndarray
    target_camera_features: np.ndarray
    parent_canonical_prediction_sha256: str | None
    generated_normals: np.ndarray | None
    generated_confidence: np.ndarray  # [V,1,H,W], purely learned
    generation_seed: int
    generated_view_count: int
    prediction_hash: str = ""
    schema: str = PREDICTION_SCHEMA
    authority: str = OUTPUT_AUTHORITY
    target_pose_policy: str = POSE_POLICY
    source_pixel_authority: bool = False
    camera_calibration_observed: bool = False
    source_surface_or_depth_authority: bool = False
    canonical_mesh_authority: bool = False
    qualified_3d_character: bool = False


def _check_prediction_shape(
    value: RSSvNextCleanroomDiffusionPredictionV1,
) -> None:
    if (
        type(value) is not RSSvNextCleanroomDiffusionPredictionV1
        or value.schema != PREDICTION_SCHEMA
        or value.authority != OUTPUT_AUTHORITY or value.target_pose_policy != POSE_POLICY
        or any((
            value.source_pixel_authority, value.camera_calibration_observed,
            value.source_surface_or_depth_authority, value.canonical_mesh_authority,
            value.qualified_3d_character,
        ))
        or not all(type(x) is bool for x in (
            value.source_pixel_authority, value.camera_calibration_observed,
            value.source_surface_or_depth_authority, value.canonical_mesh_authority,
            value.qualified_3d_character,
        ))
        or not _sha(value.source_binding_sha256)
        or not _sha(value.generator_checkpoint_sha256)
        or type(value.generation_seed) is not int or not 0 <= value.generation_seed < 2**63
        or type(value.generated_view_count) is not int
    ):
        _reject("PREDICTION_AUTHORITY_OR_METADATA")
    stage = value.stage
    count = value.generated_view_count
    if stage == STAGE_CANONICAL:
        if count != 2 or value.generated_normals is not None or value.parent_canonical_prediction_sha256 is not None:
            _reject("CANONICAL_STAGE_DOMAIN")
    elif stage == STAGE_MULTIVIEW:
        if count not in (6, 8) or not _sha(value.parent_canonical_prediction_sha256):
            _reject("MULTIVIEW_STAGE_DOMAIN")
    else:
        _reject("PREDICTION_STAGE_INVALID")
    rgba = value.generated_rgba
    confidence = value.generated_confidence
    cams = value.target_camera_features
    if (
        type(rgba) is not np.ndarray or rgba.dtype != np.uint8 or rgba.ndim != 4
        or rgba.shape[:2] != (count, 4)
        or min(rgba.shape[2:]) < 14 or max(rgba.shape[2:]) > 2048
        or type(confidence) is not np.ndarray or confidence.dtype != np.float32
        or confidence.shape != (count, 1, *rgba.shape[-2:])
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0.) | (confidence > 1.))
        or type(cams) is not np.ndarray or cams.dtype != np.float32
        or cams.shape != (count, 13) or not all(_valid_camera(cam) for cam in cams)
    ):
        _reject("GENERATED_IMAGES_CAMERA_OR_UNCERTAINTY")
    total = rgba.nbytes + cams.nbytes + confidence.nbytes
    if stage == STAGE_MULTIVIEW:
        normals = value.generated_normals
        if (
            type(normals) is not np.ndarray or normals.dtype != np.float32
            or normals.shape != (count, 3, *rgba.shape[-2:])
            or not np.isfinite(normals).all() or np.any(np.abs(normals) > 1.00001)
        ):
            _reject("MULTIVIEW_GENERATED_NORMAL_DOMAIN")
        total += normals.nbytes
    if total > _MAX_PREDICTED_BYTES:
        _reject("GENERATED_IMAGE_RESOURCE_BUDGET")


def _prediction_payload(value: RSSvNextCleanroomDiffusionPredictionV1) -> dict[str, Any]:
    payload = asdict(value)
    for key in ("generated_rgba", "target_camera_features", "generated_confidence"):
        payload[key] = _digest_array(getattr(value, key))
    if value.generated_normals is not None:
        payload["generated_normals"] = _digest_array(value.generated_normals)
    payload.pop("prediction_hash")
    return payload


def validate_cleanroom_diffusion_prediction_v1(
    value: RSSvNextCleanroomDiffusionPredictionV1,
    *,
    source: RSSvNextRawTwoSceneObservationsV1,
    parent_canonical: RSSvNextCleanroomDiffusionPredictionV1 | None = None,
) -> None:
    _check_prediction_shape(value)
    if type(source) is not RSSvNextRawTwoSceneObservationsV1 or (
        value.source_binding_sha256 != source.source_binding_sha256
    ):
        _reject("PREDICTION_ORIGINAL_SOURCE_LINEAGE_DRIFT")
    if value.stage == STAGE_MULTIVIEW:
        if parent_canonical is None:
            _reject("MULTIVIEW_REQUIRES_CANONICAL_PARENT")
        _check_prediction_shape(parent_canonical)
        if (
            parent_canonical.stage != STAGE_CANONICAL
            or parent_canonical.source_binding_sha256 != source.source_binding_sha256
            or parent_canonical.prediction_hash != content_sha256(_prediction_payload(parent_canonical))
            or value.parent_canonical_prediction_sha256 != parent_canonical.prediction_hash
        ):
            _reject("MULTIVIEW_CANONICAL_PARENT_DRIFT")
    elif parent_canonical is not None:
        _reject("CANONICAL_CANNOT_HAVE_GENERATED_PARENT")
    if value.prediction_hash != content_sha256(_prediction_payload(value)):
        _reject("PREDICTION_CONTENT_DRIFT")


def seal_cleanroom_diffusion_prediction_v1(
    *,
    source: RSSvNextRawTwoSceneObservationsV1,
    generator_checkpoint_sha256: str,
    stage: str,
    generated_rgba: np.ndarray,
    target_camera_features: np.ndarray,
    generated_confidence: np.ndarray,
    generation_seed: int,
    parent_canonical: RSSvNextCleanroomDiffusionPredictionV1 | None = None,
    generated_normals: np.ndarray | None = None,
) -> RSSvNextCleanroomDiffusionPredictionV1:
    """Seal model output; does not assert that a trained model ran."""
    if type(source) is not RSSvNextRawTwoSceneObservationsV1:
        _reject("RAW_SOURCE_REQUIRED")
    value = RSSvNextCleanroomDiffusionPredictionV1(
        source_binding_sha256=source.source_binding_sha256,
        generator_checkpoint_sha256=generator_checkpoint_sha256,
        stage=stage,
        generated_rgba=np.array(generated_rgba, copy=True),
        target_camera_features=np.array(target_camera_features, copy=True),
        parent_canonical_prediction_sha256=(
            None if parent_canonical is None else parent_canonical.prediction_hash
        ),
        generated_normals=(
            None if generated_normals is None else np.array(generated_normals, copy=True)
        ),
        generated_confidence=np.array(generated_confidence, copy=True),
        generation_seed=generation_seed,
        generated_view_count=len(generated_rgba),
    )
    _check_prediction_shape(value)
    for key in ("generated_rgba", "target_camera_features", "generated_confidence", "generated_normals"):
        array = getattr(value, key)
        if array is not None:
            array.setflags(write=False)
    value = replace(value, prediction_hash=content_sha256(_prediction_payload(value)))
    validate_cleanroom_diffusion_prediction_v1(
        value, source=source, parent_canonical=parent_canonical,
    )
    return value


__all__ = [
    "RAW_SCHEMA", "PREDICTION_SCHEMA", "STAGE_CANONICAL", "STAGE_MULTIVIEW",
    "RSSvNextRawTwoSceneObservationsV1",
    "RSSvNextCleanroomDiffusionPredictionV1",
    "seal_cleanroom_diffusion_prediction_v1",
    "validate_cleanroom_diffusion_prediction_v1",
]
