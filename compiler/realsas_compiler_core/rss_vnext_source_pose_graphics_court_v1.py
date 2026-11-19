from __future__ import annotations

"""Source-registered native Graphics pixel/face-owner proof after Mechanics.

This is a strictly observational Compiler court, not a generated teacher or
a source appearance model. It compares independently qualified original or
artist-approved *known* source pixels to the actual native C++ renderer output
from one final canonical Mechanics frame. An unregistered virtual preview
camera, student face hypotheses, synthetic six-view construction, or unknown
newly revealed surface may never become positive source evidence.

A passing frame is one *local* known-pixel witness, never a ProductRevision or
complete hidden-surface/temporal/physics proof. Full subject/clip coverage and
product admission remain the responsibility of product.proof.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from typing import Any, Mapping

import numpy as np

from .hashing import content_sha256
from .types import QualificationError

SCHEMA = "RealSaS.RSSvNext.SourcePoseGraphicsCourt.v1"
SOURCE_ADMISSION = "SOURCE_2D_NATIVE_CAMERA_AND_FACE_OWNER_INDEPENDENTLY_QUALIFIED"
NATIVE_ADMISSION = "CPP_NATIVE_FINAL_MECHANICS_RGBA_AND_OWNER_REPLAY_QUALIFIED"
SOURCE_REST = "ORIGINAL_SOURCE_REST_RGBA"
SOURCE_MOTION = "ARTIST_VALIDATED_POSTMOTION_RGBA"
MAX_SIDE = 2048
MAX_KNOWN_PIXELS = 2_000_000


def _sha(value: object) -> bool:
    return (
        type(value) is str and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    header = json.dumps({
        "dtype": arr.dtype.str,
        "shape": list(arr.shape),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(header + b"\0" + arr.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class SourcePoseGraphicsThresholdsV1:
    max_known_rgb_mae: float
    max_known_alpha_mae: float
    min_known_silhouette_iou: float
    min_known_face_owner_accuracy: float
    max_known_unknown_occlusion_fraction: float = 0.0
    max_known_inferred_color_fraction: float = 0.0

    def validate(self) -> None:
        values = asdict(self)
        if any(type(v) is not float or not math.isfinite(v) for v in values.values()):
            raise QualificationError("RSS_SOURCE_GRAPHICS_THRESHOLDS_FINITE_FLOAT_REQUIRED")
        if (
            not 0.0 <= self.max_known_rgb_mae <= 0.10
            or not 0.0 <= self.max_known_alpha_mae <= 0.10
            or not .90 <= self.min_known_silhouette_iou <= 1.0
            or not .90 <= self.min_known_face_owner_accuracy <= 1.0
            or not 0.0 <= self.max_known_unknown_occlusion_fraction <= .02
            or not 0.0 <= self.max_known_inferred_color_fraction <= .05
        ):
            raise QualificationError("RSS_SOURCE_GRAPHICS_THRESHOLD_PRODUCT_BOUNDS")

    def policy_sha256(self) -> str:
        return content_sha256({
            "schema":"RealSaS.RSSvNext.SourcePoseGraphicsThresholdPolicy.v1",
            **asdict(self),
        })


@dataclass(frozen=True)
class SourcePoseGraphicsCourtV1:
    canonical_mesh_lineage_hash: str
    canonical_rig_lineage_hash: str
    canonical_skin_lineage_hash: str
    compiled_graphics_hash: str
    canonical_motion_lineage_hash: str
    original_source_asset_sha256: str
    original_source_file_sha256: str
    source_registration_sha256: str
    registered_native_camera_sha256: str
    native_pose_bytes_sha256: str
    native_renderer_binary_sha256: str
    source_witness_key: str
    renderer_witness_key: str
    source_view_index: int
    frame_index: int
    frame_time_seconds: float
    source_target_kind: str
    known_pixel_count: int
    known_foreground_pixel_count: int
    known_background_pixel_count: int
    known_rgb_mae: float
    known_alpha_mae: float
    known_silhouette_iou: float
    known_face_owner_accuracy: float
    known_unknown_occlusion_fraction: float
    known_inferred_color_fraction: float
    target_rgba_sha256: str
    target_face_owner_sha256: str
    target_known_mask_sha256: str
    native_rgba_sha256: str
    native_face_owner_sha256: str
    native_color_authority_sha256: str
    native_unknown_occlusion_sha256: str
    frozen_threshold_policy_sha256: str
    source_registered_pixels_passed: bool
    product_release_admission: bool = False
    hidden_surface_color_proven: bool = False
    unobserved_motion_quality_proven: bool = False
    schema_version: str = SCHEMA
    court_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_pose_graphics_court_hash_v1(value: SourcePoseGraphicsCourtV1) -> str:
    value_dict = value.to_dict()
    value_dict.pop("court_hash", None)
    return content_sha256(value_dict)


def reopen_source_pose_graphics_court_v1(
    payload: Mapping[str, Any],
    *,
    canonical_mesh_lineage_hash: str,
    canonical_rig_lineage_hash: str,
    canonical_skin_lineage_hash: str,
    compiled_graphics_hash: str,
    canonical_motion_lineage_hash: str,
) -> SourcePoseGraphicsCourtV1:
    if not isinstance(payload, Mapping):
        raise QualificationError("RSS_SOURCE_GRAPHICS_COURT_WIRE_REQUIRED")
    try:
        court = SourcePoseGraphicsCourtV1(**dict(payload))
    except (TypeError, ValueError) as exc:
        raise QualificationError("RSS_SOURCE_GRAPHICS_COURT_WIRE_INVALID") from exc
    if (
        court.schema_version != SCHEMA
        or type(court.source_registered_pixels_passed) is not bool
        or court.product_release_admission is not False
        or court.hidden_surface_color_proven is not False
        or court.unobserved_motion_quality_proven is not False
        or court.court_hash != source_pose_graphics_court_hash_v1(court)
        or any(not _sha(value) for value in (
            court.canonical_mesh_lineage_hash, court.canonical_rig_lineage_hash,
            court.canonical_skin_lineage_hash, court.compiled_graphics_hash,
            court.canonical_motion_lineage_hash, court.original_source_asset_sha256,
            court.original_source_file_sha256, court.source_registration_sha256,
            court.registered_native_camera_sha256, court.native_pose_bytes_sha256,
            court.native_renderer_binary_sha256, court.target_rgba_sha256,
            court.target_face_owner_sha256, court.target_known_mask_sha256,
            court.native_rgba_sha256, court.native_face_owner_sha256,
            court.native_color_authority_sha256,
            court.native_unknown_occlusion_sha256, court.frozen_threshold_policy_sha256,
        ))
        or court.source_view_index not in (0, 1)
        or type(court.frame_index) is not int or court.frame_index < 0
        or court.source_target_kind not in (SOURCE_REST, SOURCE_MOTION)
        or type(court.frame_time_seconds) is not float
        or not math.isfinite(court.frame_time_seconds)
        or court.frame_time_seconds < 0
        or not court.source_witness_key or not court.renderer_witness_key
        or any(type(v) is not int or v < 0 for v in (
            court.known_pixel_count, court.known_foreground_pixel_count,
            court.known_background_pixel_count,
        ))
        or court.known_pixel_count != (
            court.known_foreground_pixel_count + court.known_background_pixel_count
        )
        or any(type(v) is not float or not math.isfinite(v) or not 0<=v<=1
               for v in (
                   court.known_rgb_mae, court.known_alpha_mae,
                   court.known_silhouette_iou, court.known_face_owner_accuracy,
                   court.known_unknown_occlusion_fraction,
                   court.known_inferred_color_fraction,
               ))
        or court.canonical_mesh_lineage_hash != canonical_mesh_lineage_hash
        or court.canonical_rig_lineage_hash != canonical_rig_lineage_hash
        or court.canonical_skin_lineage_hash != canonical_skin_lineage_hash
        or court.compiled_graphics_hash != compiled_graphics_hash
        or court.canonical_motion_lineage_hash != canonical_motion_lineage_hash
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_COURT_REQUALIFICATION_DRIFT")
    return court


def evaluate_registered_source_pose_graphics_v1(
    *,
    target_rgba_u8: np.ndarray,
    target_face_owner_i32: np.ndarray,
    target_known_mask: np.ndarray,
    native_rgba_u8: np.ndarray,
    native_front_face_i32: np.ndarray,
    native_color_authority_u8: np.ndarray,
    native_unknown_occlusion_u8: np.ndarray,
    native_face_count: int,
    original_source_asset_sha256: str,
    original_source_file_sha256: str,
    canonical_mesh_lineage_hash: str,
    canonical_rig_lineage_hash: str,
    canonical_skin_lineage_hash: str,
    compiled_graphics_hash: str,
    canonical_motion_lineage_hash: str,
    source_view_index: int,
    frame_index: int,
    source_target_kind: str,
    source_registration_sha256: str,
    source_witness_key: str,
    renderer_witness_key: str,
    native_renderer_binary_sha256: str,
    live_pose_receipt: Mapping[str, Any],
    thresholds: SourcePoseGraphicsThresholdsV1,
    trusted_source_registry: Mapping[str, Mapping[str, Any]],
    trusted_native_renderer_registry: Mapping[str, Mapping[str, Any]],
) -> SourcePoseGraphicsCourtV1:
    """Score independently registered source/artist pixels vs actual C++ buffers.

    The renderer registry must attest actual execution of the exact final
    Mechanics pose bytes; the source registry is independent and cannot be a
    generated pseudo-view or self-labeled model pixel. Unknown is never BG.
    """
    if not isinstance(thresholds, SourcePoseGraphicsThresholdsV1):
        raise QualificationError("RSS_SOURCE_GRAPHICS_FROZEN_POLICY_REQUIRED")
    thresholds.validate()
    arrays = (
        target_rgba_u8, target_face_owner_i32, target_known_mask,
        native_rgba_u8, native_front_face_i32, native_color_authority_u8,
        native_unknown_occlusion_u8,
    )
    if not all(isinstance(x, np.ndarray) for x in arrays):
        raise QualificationError("RSS_SOURCE_GRAPHICS_PIXEL_ARRAYS_REQUIRED")
    target, target_owner, known, rendered, front, auth, unknown = arrays
    h,w = known.shape if known.ndim == 2 else (0,0)
    if (
        not 16 <= h <= MAX_SIDE or not 16 <= w <= MAX_SIDE
        or target.shape != (h,w,4) or rendered.shape != (h,w,4)
        or target_owner.shape != (h,w) or front.shape != (h,w)
        or auth.shape != (h,w) or unknown.shape != (h,w)
        or target.dtype != np.uint8 or rendered.dtype != np.uint8
        or target_owner.dtype != np.int32 or front.dtype != np.int32
        or known.dtype != np.bool_ or auth.dtype != np.uint8
        or unknown.dtype != np.uint8
        or not np.isin(auth, [0,1,2]).all()
        or not np.isin(unknown,[0,1]).all()
        or type(native_face_count) is not int or native_face_count < 1
        or np.any((front < -1) | (front >= native_face_count))
        or np.any((target_owner < -1) | (target_owner >= native_face_count))
        or type(source_view_index) is not int or source_view_index not in (0,1)
        or type(frame_index) is not int or frame_index < 0
        or source_target_kind not in (SOURCE_REST,SOURCE_MOTION)
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_NATIVE_PIXEL_DOMAIN")
    n_known = int(np.count_nonzero(known))
    if not 0 < n_known <= MAX_KNOWN_PIXELS:
        raise QualificationError("RSS_SOURCE_GRAPHICS_KNOWN_COVERAGE_REQUIRED")
    fg = known & (target[:,:,3] != 0)
    bg = known & ~fg
    n_fg, n_bg = int(np.count_nonzero(fg)), int(np.count_nonzero(bg))
    if (
        n_fg < 1 or n_bg < 1
        or np.any(target_owner[fg] < 0)
        or np.any(target_owner[bg] != -1)
        or any(not _sha(value) for value in (
            original_source_asset_sha256, original_source_file_sha256,
            canonical_mesh_lineage_hash, canonical_rig_lineage_hash,
            canonical_skin_lineage_hash, compiled_graphics_hash,
            canonical_motion_lineage_hash, source_registration_sha256,
            native_renderer_binary_sha256,
        ))
        or type(source_witness_key) is not str or not source_witness_key
        or type(renderer_witness_key) is not str or not renderer_witness_key
        or source_witness_key == renderer_witness_key
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_SOURCE_ADDRESS_OR_MASK_INVALID")
    if (
        not isinstance(live_pose_receipt, Mapping)
        or live_pose_receipt.get("schema")
            != "RealSaS.RSSvNext.LiveMechanicsNativePoseTransport.v1"
        or live_pose_receipt.get("authority")
            != "LIVE_MECHANICS_FINAL_SURFACE_NOT_SOURCE_OR_PRODUCT_PROOF"
        or live_pose_receipt.get("source_color_or_visibility_qualified") is not False
        or live_pose_receipt.get("product_release_admission") is not False
        or live_pose_receipt.get("mesh_lineage_hash") != canonical_mesh_lineage_hash
        or live_pose_receipt.get("rig_lineage_hash") != canonical_rig_lineage_hash
        or live_pose_receipt.get("skin_lineage_hash") != canonical_skin_lineage_hash
        or live_pose_receipt.get("resolution") != w
        or h != w
        or type(live_pose_receipt.get("frames")) is not list
        or not 0 <= frame_index < len(live_pose_receipt["frames"])
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_LIVE_MECHANICS_RECEIPT_DRIFT")
    live_frame = live_pose_receipt["frames"][frame_index]
    if (
        type(live_frame) is not dict
        or type(live_frame.get("time_seconds")) not in (int,float)
        or not math.isfinite(float(live_frame["time_seconds"]))
        or live_frame.get("frame_index") != frame_index
        or not _sha(live_frame.get("final_world_vertex_positions_sha256"))
        or not _sha(live_pose_receipt.get("native_pose_bytes_sha256"))
        or not _sha(live_pose_receipt.get("camera_policy_sha256"))
        or source_target_kind == SOURCE_REST
            and live_frame.get("xpbd_surface_state_hash") is not None
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_FINAL_FRAME_DRIFT")
    time = float(live_frame["time_seconds"])
    if source_target_kind == SOURCE_MOTION and time <= 0.0:
        raise QualificationError("RSS_SOURCE_GRAPHICS_POSTMOTION_TIME_REQUIRED")
    registry_source = trusted_source_registry.get(source_witness_key)
    registry_native = trusted_native_renderer_registry.get(renderer_witness_key)
    target_sha, owner_sha, known_sha = map(_array_sha,(target,target_owner,known))
    native_sha,front_sha,auth_sha,unknown_sha = map(
        _array_sha,(rendered,front,auth,unknown)
    )
    common = {
        "canonical_mesh_lineage_hash":canonical_mesh_lineage_hash,
        "canonical_rig_lineage_hash":canonical_rig_lineage_hash,
        "canonical_skin_lineage_hash":canonical_skin_lineage_hash,
        "compiled_graphics_hash":compiled_graphics_hash,
        "canonical_motion_lineage_hash":canonical_motion_lineage_hash,
        "source_view_index":source_view_index,
        "frame_index":frame_index,
        "frame_time_seconds":time,
        "native_pose_bytes_sha256":live_pose_receipt["native_pose_bytes_sha256"],
        "registered_native_camera_sha256":
            live_pose_receipt["camera_policy_sha256"],
    }
    expected_source = {
        **common,
        "admission_status":SOURCE_ADMISSION,
        "source_witness_key":source_witness_key,
        "original_source_asset_sha256":original_source_asset_sha256,
        "original_source_file_sha256":original_source_file_sha256,
        "source_registration_sha256":source_registration_sha256,
        "source_target_kind":source_target_kind,
        "target_rgba_sha256":target_sha,
        "target_face_owner_sha256":owner_sha,
        "target_known_mask_sha256":known_sha,
        "frozen_threshold_policy_sha256":thresholds.policy_sha256(),
        "original_two_source_observations_only":True,
        "source_camera_registration_independently_verified":True,
        "native_face_owner_transport_independently_verified":True,
        "source_known_mask_has_explicit_unknown":True,
        "generated_six_views_used_as_source_truth":False,
        "virtual_preview_camera_misrepresented_as_source":False,
        "source_issuer_role":"INDEPENDENT_SOURCE_WITNESS_AUTHORITY",
    }
    if source_target_kind == SOURCE_MOTION:
        expected_source.update({
            "artist_verified_postmotion_rgba_and_visibility":True,
        })
    expected_renderer = {
        **common,
        "admission_status":NATIVE_ADMISSION,
        "renderer_witness_key":renderer_witness_key,
        "native_renderer_binary_sha256":native_renderer_binary_sha256,
        "native_rgba_sha256":native_sha,
        "native_face_owner_sha256":front_sha,
        "native_color_authority_sha256":auth_sha,
        "native_unknown_occlusion_sha256":unknown_sha,
        "final_world_vertex_positions_sha256":
            live_frame["final_world_vertex_positions_sha256"],
        "native_cpp_executed_for_exact_pose":True,
        "native_render_owner_map_independently_captured":True,
        "native_unknown_occlusion_sidecar_captured":True,
        "renderer_issuer_role":"INDEPENDENT_CPP_RENDER_EXECUTION",
        "source_rgb_teacher_was_renderer_output":False,
    }
    if (
        not isinstance(registry_source,Mapping)
        or not isinstance(registry_native,Mapping)
        or any(registry_source.get(k)!=v for k,v in expected_source.items())
        or any(registry_native.get(k)!=v for k,v in expected_renderer.items())
    ):
        raise QualificationError("RSS_SOURCE_GRAPHICS_INDEPENDENT_WITNESS_DRIFT")

    predicted_fg = rendered[:,:,3] != 0
    union = int(np.count_nonzero(known & ((target[:,:,3]!=0)|predicted_fg)))
    intersection = int(np.count_nonzero(known & ((target[:,:,3]!=0)&predicted_fg)))
    iou = float(intersection/union) if union else 0.0
    rgb_diff = np.abs(target[:,:,:3][fg].astype(np.int16)-
                      rendered[:,:,:3][fg].astype(np.int16))
    alpha_diff = np.abs(target[:,:,3][known].astype(np.int16)-
                        rendered[:,:,3][known].astype(np.int16))
    rgb_mae = float(rgb_diff.mean()/255.)
    alpha_mae = float(alpha_diff.mean()/255.)
    owner_accuracy = float(np.count_nonzero(front[fg]==target_owner[fg])/n_fg)
    unknown_fraction = float(np.count_nonzero(unknown[known])/n_known)
    inferred_fraction = float(np.count_nonzero(auth[fg]!=1)/n_fg)
    passed = (
        rgb_mae <= thresholds.max_known_rgb_mae
        and alpha_mae <= thresholds.max_known_alpha_mae
        and iou >= thresholds.min_known_silhouette_iou
        and owner_accuracy >= thresholds.min_known_face_owner_accuracy
        and unknown_fraction <= thresholds.max_known_unknown_occlusion_fraction
        and inferred_fraction <= thresholds.max_known_inferred_color_fraction
    )
    value = SourcePoseGraphicsCourtV1(
        canonical_mesh_lineage_hash, canonical_rig_lineage_hash,
        canonical_skin_lineage_hash, compiled_graphics_hash,
        canonical_motion_lineage_hash, original_source_asset_sha256,
        original_source_file_sha256, source_registration_sha256,
        live_pose_receipt["camera_policy_sha256"],
        live_pose_receipt["native_pose_bytes_sha256"],
        native_renderer_binary_sha256,source_witness_key,
        renderer_witness_key,source_view_index,frame_index,time,
        source_target_kind,n_known,n_fg,n_bg,
        rgb_mae,alpha_mae,iou,owner_accuracy,unknown_fraction,inferred_fraction,
        target_sha,owner_sha,known_sha,native_sha,front_sha,auth_sha,unknown_sha,
        thresholds.policy_sha256(),bool(passed),
    )
    return replace(value,court_hash=source_pose_graphics_court_hash_v1(value))


__all__ = [
    "SCHEMA","SOURCE_ADMISSION","NATIVE_ADMISSION",
    "SOURCE_REST","SOURCE_MOTION",
    "SourcePoseGraphicsThresholdsV1","SourcePoseGraphicsCourtV1",
    "source_pose_graphics_court_hash_v1","reopen_source_pose_graphics_court_v1",
    "evaluate_registered_source_pose_graphics_v1",
]
