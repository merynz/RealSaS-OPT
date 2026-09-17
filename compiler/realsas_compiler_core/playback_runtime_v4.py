from __future__ import annotations

"""Compact Runtime-v4 diagnostic contract.

P0/P1 owns representation only: immutable canonical attachment assets are stored once,
view-specific camera/appearance state is carried by overlays, and canonical posed XYZ
is stored once per attachment per frame. Scientific ownership remains upstream.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3ClipInterval,
    RuntimeV3FrameComposition,
    ReferenceRasterContractV1,
    RuntimeV3Slot,
    RuntimeV3VisibilityPolicy,
    TopologyClass,
)
from compiler.realsas_compiler_core.types import QualificationError

RUNTIME_V4_CONTRACT_SCHEMA = "RealSaS.RuntimeV4PlaybackContract.v1"
RUNTIME_V4_FRAME_SCHEMA = "RealSaS.RuntimeV4Frame.v1"
RUNTIME_V4_CLIP_SCHEMA = "RealSaS.RuntimeV4Clip.v1"
DEFAULT_VIEWS = tuple(f"V{i}" for i in range(8))

_PROVENANCE_TO_CODE = {
    AppearanceProvenance.DIRECT_SOURCE: 0,
    AppearanceProvenance.OTHER_VIEW_SOURCE: 1,
    AppearanceProvenance.UNDER_RIGID_SOURCE: 2,
    AppearanceProvenance.UNSEEN: 3,
    AppearanceProvenance.COMPLETION: 4,
}
_CODE_TO_PROVENANCE = {value: key for key, value in _PROVENANCE_TO_CODE.items()}


def _array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii"))
    h.update(b"|")
    h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def _f32_xyz(value, *, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) < 3 or not np.isfinite(arr).all():
        raise QualificationError(label)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _u32_triangles(value, *, vertex_count: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0:
        raise QualificationError(label)
    if not np.issubdtype(raw.dtype, np.integer):
        raise QualificationError(label)
    if np.any(raw < 0) or np.any(raw >= int(vertex_count)):
        raise QualificationError(f"{label}_INDEX_OUT_OF_RANGE")
    return np.ascontiguousarray(raw, dtype=np.uint32)


def _f32_uv(value, *, vertex_count: int, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (int(vertex_count), 2) or not np.isfinite(arr).all():
        raise QualificationError(label)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise QualificationError(f"{label}_OUTSIDE_UNIT_DOMAIN")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _provenance_codes(value, *, face_count: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (int(face_count),):
        raise QualificationError(label)
    if raw.dtype.kind in "OUS":
        try:
            codes = np.fromiter(
                (_PROVENANCE_TO_CODE[AppearanceProvenance(str(x))] for x in raw.tolist()),
                dtype=np.uint8,
                count=face_count,
            )
        except (ValueError, KeyError) as exc:
            raise QualificationError(label) from exc
    else:
        codes = np.asarray(raw, dtype=np.int16)
        if np.any(codes < 0) or np.any(codes > 4):
            raise QualificationError(label)
        codes = codes.astype(np.uint8, copy=False)
    return np.ascontiguousarray(codes, dtype=np.uint8)


def _donor_indices(value, *, face_count: int, view_count: int, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int16)
    if arr.shape != (int(face_count),):
        raise QualificationError(label)
    if np.any(arr < -1) or np.any(arr >= int(view_count)):
        raise QualificationError(label)
    return np.ascontiguousarray(arr, dtype=np.int16)


@dataclass(frozen=True, eq=False)
class RuntimeV4AttachmentAsset:
    asset_id: str
    slot_id: str
    attachment_id: str
    attachment_kind: AttachmentKind
    topology_class: TopologyClass
    rest_xyz: np.ndarray
    triangles: np.ndarray
    sealed_source_hash: str
    _asset_hash_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        xyz = np.ascontiguousarray(self.rest_xyz, dtype=np.float32).copy()
        tri = np.ascontiguousarray(self.triangles, dtype=np.uint32).copy()
        xyz.setflags(write=False)
        tri.setflags(write=False)
        object.__setattr__(self, "rest_xyz", xyz)
        object.__setattr__(self, "triangles", tri)

    def canonical_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        xyz = _f32_xyz(self.rest_xyz, label="RUNTIME_V4_ASSET_REST_XYZ_INVALID")
        tri = _u32_triangles(
            self.triangles,
            vertex_count=len(xyz),
            label="RUNTIME_V4_ASSET_TRIANGLES_INVALID",
        )
        return xyz, tri

    @property
    def vertex_count(self) -> int:
        return int(np.asarray(self.rest_xyz).shape[0])

    @property
    def face_count(self) -> int:
        return int(np.asarray(self.triangles).shape[0])

    @property
    def asset_hash(self) -> str:
        cached = self._asset_hash_cache
        if cached is not None:
            return cached
        xyz, tri = self.canonical_arrays()
        value = content_sha256({
            "schema": "RealSaS.RuntimeV4AttachmentAsset.v1",
            "asset_id": self.asset_id,
            "slot_id": self.slot_id,
            "attachment_id": self.attachment_id,
            "attachment_kind": self.attachment_kind.value,
            "topology_class": self.topology_class.value,
            "sealed_source_hash": self.sealed_source_hash,
            "rest_xyz_sha256": _array_sha256(xyz),
            "triangles_sha256": _array_sha256(tri),
            "vertex_count": len(xyz),
            "face_count": len(tri),
        })
        object.__setattr__(self, "_asset_hash_cache", value)
        return value


@dataclass(frozen=True, eq=False)
class RuntimeV4ViewAssetOverlay:
    asset_id: str
    uv: np.ndarray
    provenance_codes: np.ndarray
    donor_view_indices: np.ndarray

    def __post_init__(self) -> None:
        uv = np.ascontiguousarray(self.uv, dtype=np.float32).copy()
        provenance = np.ascontiguousarray(self.provenance_codes, dtype=np.uint8).copy()
        donors = np.ascontiguousarray(self.donor_view_indices, dtype=np.int16).copy()
        uv.setflags(write=False); provenance.setflags(write=False); donors.setflags(write=False)
        object.__setattr__(self, "uv", uv)
        object.__setattr__(self, "provenance_codes", provenance)
        object.__setattr__(self, "donor_view_indices", donors)


@dataclass(frozen=True, eq=False)
class RuntimeV4ViewOverlay:
    view_id: str
    view_index: int
    camera: CameraProjectionV3
    assets: tuple[RuntimeV4ViewAssetOverlay, ...]


@dataclass(frozen=True, eq=False)
class RuntimeV4PlaybackContract:
    slots: tuple[RuntimeV3Slot, ...]
    assets: tuple[RuntimeV4AttachmentAsset, ...]
    views: tuple[RuntimeV4ViewOverlay, ...]
    raster: ReferenceRasterContractV1 = field(default_factory=ReferenceRasterContractV1)
    visibility: RuntimeV3VisibilityPolicy = field(default_factory=RuntimeV3VisibilityPolicy)
    allow_completion: bool = False
    schema_version: str = RUNTIME_V4_CONTRACT_SCHEMA


@dataclass(frozen=True, eq=False)
class RuntimeV4Frame:
    time_seconds: float
    canonical_posed_xyz_by_asset: Mapping[str, np.ndarray]
    composition_by_view: Mapping[str, RuntimeV3FrameComposition]
    schema_version: str = RUNTIME_V4_FRAME_SCHEMA


@dataclass(frozen=True, eq=False)
class RuntimeV4Clip:
    clip_id: str
    display_name: str
    intent: str
    duration_seconds: float
    fps: float
    loop: bool
    frames: tuple[RuntimeV4Frame, ...]
    runtime_qualified: bool = False
    schema_version: str = RUNTIME_V4_CLIP_SCHEMA


def _validate_camera(camera: CameraProjectionV3, *, view_id: str, view_index: int) -> None:
    if camera.view_id != view_id or int(camera.view_index) != int(view_index):
        raise QualificationError("RUNTIME_V4_CAMERA_VIEW_IDENTITY_DRIFT")
    for label, value in (
        ("origin", camera.origin),
        ("right", camera.right),
        ("screen_up", camera.screen_up),
        ("forward", camera.forward),
    ):
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise QualificationError(f"RUNTIME_V4_CAMERA_{label.upper()}_INVALID")
    if not math.isfinite(float(camera.half_extent)) or float(camera.half_extent) <= 0:
        raise QualificationError("RUNTIME_V4_CAMERA_HALF_EXTENT_INVALID")
    if int(camera.resolution) <= 0:
        raise QualificationError("RUNTIME_V4_CAMERA_RESOLUTION_INVALID")


def _validate_frame_composition(
    contract: RuntimeV4PlaybackContract,
    frame: RuntimeV4Frame,
    *,
    required_view_ids: tuple[str, ...],
) -> None:
    slot_ids = tuple(slot.slot_id for slot in contract.slots)
    slot_set = set(slot_ids)
    allowed_by_slot: dict[str, set[str]] = {slot_id: set() for slot_id in slot_ids}
    for asset in contract.assets:
        allowed_by_slot[asset.slot_id].add(asset.attachment_id)

    if set(frame.composition_by_view) != set(required_view_ids):
        raise QualificationError("RUNTIME_V4_FRAME_COMPOSITION_VIEW_SET_MISMATCH")
    for view_id in required_view_ids:
        composition = frame.composition_by_view[view_id]
        if composition.view_id != view_id:
            raise QualificationError("RUNTIME_V4_FRAME_COMPOSITION_VIEW_ID_DRIFT")
        order = tuple(composition.draw_order_slot_ids)
        if len(order) != len(slot_ids) or set(order) != slot_set:
            raise QualificationError("RUNTIME_V4_DRAW_ORDER_MUST_BE_EXACT_SLOT_PERMUTATION")
        if set(composition.active_attachment_by_slot) != slot_set:
            raise QualificationError("RUNTIME_V4_ACTIVE_ATTACHMENT_SLOT_SET_MISMATCH")
        for slot_id, attachment_id in composition.active_attachment_by_slot.items():
            if attachment_id is not None and attachment_id not in allowed_by_slot[slot_id]:
                raise QualificationError("RUNTIME_V4_ACTIVE_ATTACHMENT_NOT_ALLOWED_FOR_SLOT")
        order_index = {slot_id: i for i, slot_id in enumerate(order)}
        for interval in composition.clip_intervals:
            if not isinstance(interval, RuntimeV3ClipInterval):
                raise QualificationError("RUNTIME_V4_CLIP_INTERVAL_TYPE_INVALID")
            if interval.start_slot_id not in slot_set or interval.end_slot_id not in slot_set:
                raise QualificationError("RUNTIME_V4_CLIP_INTERVAL_SLOT_UNKNOWN")
            if order_index[interval.start_slot_id] > order_index[interval.end_slot_id]:
                raise QualificationError("RUNTIME_V4_CLIP_INTERVAL_ORDER_INVALID")
            if composition.active_attachment_by_slot[interval.start_slot_id] != interval.clip_attachment_id:
                raise QualificationError("RUNTIME_V4_CLIP_INTERVAL_ATTACHMENT_NOT_ACTIVE")


def validate_playback_runtime_v4_contract(
    contract: RuntimeV4PlaybackContract,
    *,
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
) -> str:
    view_ids = tuple(map(str, required_view_ids))
    if not view_ids or len(view_ids) != len(set(view_ids)):
        raise QualificationError("RUNTIME_V4_REQUIRED_VIEW_SET_INVALID")
    if contract.schema_version != RUNTIME_V4_CONTRACT_SCHEMA:
        raise QualificationError("RUNTIME_V4_SCHEMA_DRIFT")
    if not contract.slots or not contract.assets:
        raise QualificationError("RUNTIME_V4_SLOT_OR_ASSET_SET_EMPTY")

    slot_ids = [slot.slot_id for slot in contract.slots]
    if len(slot_ids) != len(set(slot_ids)):
        raise QualificationError("RUNTIME_V4_SLOT_ID_DUPLICATE")
    setup_orders = [int(slot.setup_order) for slot in contract.slots]
    if len(setup_orders) != len(set(setup_orders)):
        raise QualificationError("RUNTIME_V4_SLOT_SETUP_ORDER_DUPLICATE")
    slot_set = set(slot_ids)

    asset_ids = [asset.asset_id for asset in contract.assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise QualificationError("RUNTIME_V4_ASSET_ID_DUPLICATE")
    attachment_ids = [asset.attachment_id for asset in contract.assets]
    if len(attachment_ids) != len(set(attachment_ids)):
        raise QualificationError("RUNTIME_V4_ATTACHMENT_ID_DUPLICATE")

    asset_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    asset_meta: list[dict] = []
    for asset in contract.assets:
        if not asset.asset_id.strip() or not asset.attachment_id.strip() or asset.slot_id not in slot_set:
            raise QualificationError("RUNTIME_V4_ASSET_IDENTITY_INVALID")
        if len(asset.sealed_source_hash) != 64:
            raise QualificationError("RUNTIME_V4_ASSET_SOURCE_HASH_MUST_BE_SHA256")
        xyz, tri = asset.canonical_arrays()
        asset_arrays[asset.asset_id] = (xyz, tri)
        asset_meta.append({
            "asset_id": asset.asset_id,
            "slot_id": asset.slot_id,
            "attachment_id": asset.attachment_id,
            "kind": asset.attachment_kind.value,
            "topology_class": asset.topology_class.value,
            "asset_hash": asset.asset_hash,
        })

    if tuple(view.view_id for view in contract.views) != view_ids:
        raise QualificationError("RUNTIME_V4_VIEW_ORDER_MISMATCH")
    overlay_hashes: list[str] = []
    for expected_index, view in enumerate(contract.views):
        if int(view.view_index) != expected_index:
            raise QualificationError("RUNTIME_V4_VIEW_INDEX_DRIFT")
        _validate_camera(view.camera, view_id=view.view_id, view_index=expected_index)
        if tuple(row.asset_id for row in view.assets) != tuple(asset_ids):
            raise QualificationError("RUNTIME_V4_VIEW_ASSET_ORDER_MISMATCH")
        per_asset_hashes: list[dict] = []
        for row in view.assets:
            xyz, tri = asset_arrays[row.asset_id]
            uv = _f32_uv(
                row.uv,
                vertex_count=len(xyz),
                label="RUNTIME_V4_VIEW_UV_INVALID",
            )
            provenance = _provenance_codes(
                row.provenance_codes,
                face_count=len(tri),
                label="RUNTIME_V4_VIEW_PROVENANCE_INVALID",
            )
            donors = _donor_indices(
                row.donor_view_indices,
                face_count=len(tri),
                view_count=len(view_ids),
                label="RUNTIME_V4_VIEW_DONOR_INVALID",
            )
            unseen = provenance == _PROVENANCE_TO_CODE[AppearanceProvenance.UNSEEN]
            completion = provenance == _PROVENANCE_TO_CODE[AppearanceProvenance.COMPLETION]
            direct = provenance == _PROVENANCE_TO_CODE[AppearanceProvenance.DIRECT_SOURCE]
            source_bound = ~(unseen | completion)
            if np.any(donors[unseen] != -1):
                raise QualificationError("RUNTIME_V4_UNSEEN_MUST_REMAIN_UNBOUND")
            if np.any(completion) and not contract.allow_completion:
                raise QualificationError("RUNTIME_V4_COMPLETION_NOT_ALLOWED_BY_PRODUCT_POLICY")
            if np.any(donors[source_bound] < 0):
                raise QualificationError("RUNTIME_V4_SOURCE_PROVENANCE_REQUIRES_DONOR")
            if np.any(donors[direct] != expected_index):
                raise QualificationError("RUNTIME_V4_DIRECT_SOURCE_DONOR_MUST_MATCH_TARGET_VIEW")
            per_asset_hashes.append({
                "asset_id": row.asset_id,
                "uv_sha256": _array_sha256(uv),
                "provenance_sha256": _array_sha256(provenance),
                "donor_sha256": _array_sha256(donors),
            })
        overlay_hashes.append(content_sha256({
            "schema": "RealSaS.RuntimeV4ViewOverlay.v1",
            "view_id": view.view_id,
            "view_index": view.view_index,
            "camera": {
                "origin": view.camera.origin,
                "right": view.camera.right,
                "screen_up": view.camera.screen_up,
                "forward": view.camera.forward,
                "half_extent": view.camera.half_extent,
                "resolution": view.camera.resolution,
            },
            "assets": per_asset_hashes,
        }))

    # Preserve the v3 raster/visibility policy as the behavioral authority.
    if not bool(contract.visibility.body_depth_test):
        raise QualificationError("RUNTIME_V4_BODY_DEPTH_TEST_REQUIRED")

    return content_sha256({
        "schema": RUNTIME_V4_CONTRACT_SCHEMA,
        "asset_meta": asset_meta,
        "view_ids": view_ids,
        "view_overlay_hashes": overlay_hashes,
        "raster_contract_hash": contract.raster.contract_hash,
        "allow_completion": bool(contract.allow_completion),
    })


def validate_runtime_v4_clip(
    contract: RuntimeV4PlaybackContract,
    clip: RuntimeV4Clip,
    *,
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
) -> None:
    view_ids = tuple(map(str, required_view_ids))
    if not clip.clip_id.strip() or not math.isfinite(float(clip.duration_seconds)) or clip.duration_seconds <= 0:
        raise QualificationError("RUNTIME_V4_CLIP_ID_OR_DURATION_INVALID")
    if not math.isfinite(float(clip.fps)) or clip.fps <= 0 or not clip.frames:
        raise QualificationError("RUNTIME_V4_CLIP_FPS_OR_FRAMES_INVALID")
    times = np.asarray([frame.time_seconds for frame in clip.frames], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise QualificationError("RUNTIME_V4_FRAME_TIMES_INVALID")
    if abs(float(times[0])) > 1.0e-8 or abs(float(times[-1]) - float(clip.duration_seconds)) > 1.0e-6:
        raise QualificationError("RUNTIME_V4_CLIP_MUST_INCLUDE_ENDPOINTS")

    asset_ids = tuple(asset.asset_id for asset in contract.assets)
    counts = {asset.asset_id: asset.vertex_count for asset in contract.assets}
    for frame in clip.frames:
        if set(frame.canonical_posed_xyz_by_asset) != set(asset_ids):
            raise QualificationError("RUNTIME_V4_FRAME_ASSET_SET_MISMATCH")
        for asset_id in asset_ids:
            arr = np.asarray(frame.canonical_posed_xyz_by_asset[asset_id], dtype=np.float32)
            if arr.shape != (counts[asset_id], 3) or not np.isfinite(arr).all():
                raise QualificationError("RUNTIME_V4_FRAME_POSED_XYZ_INVALID")
        _validate_frame_composition(contract, frame, required_view_ids=view_ids)


def provenance_code(provenance: AppearanceProvenance) -> int:
    return _PROVENANCE_TO_CODE[provenance]


def provenance_from_code(code: int) -> AppearanceProvenance:
    try:
        return _CODE_TO_PROVENANCE[int(code)]
    except KeyError as exc:
        raise QualificationError("RUNTIME_V4_UNKNOWN_PROVENANCE_CODE") from exc
