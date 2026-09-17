from __future__ import annotations

"""Shared-canonical full-surface bridge for Runtime-v4 P0/P1."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    CameraProjectionV3,
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    RuntimeV4AttachmentAsset,
    RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay,
    provenance_code,
)
from compiler.realsas_compiler_core.types import QualificationError

FULL_SURFACE_RUNTIME_BRIDGE_V4_SCHEMA = "RealSaS.FullSurfaceRuntimeBridge.v4"


@dataclass(frozen=True, eq=False)
class FullSurfaceRuntimeBridgeV4:
    asset: RuntimeV4AttachmentAsset
    views: tuple[RuntimeV4ViewOverlay, ...]
    bridge_hash: str
    vertex_count: int
    face_count: int
    view_ids: tuple[str, ...]
    schema_version: str = FULL_SURFACE_RUNTIME_BRIDGE_V4_SCHEMA


def build_full_surface_runtime_asset_v4(
    canonical_vertices,
    canonical_faces,
    cameras: Mapping[str, Mapping],
    uv_by_view: Mapping[str, np.ndarray],
    provenance_by_view: Mapping[str, Sequence[int | str | AppearanceProvenance]],
    donor_view_index_by_view: Mapping[str, Sequence[int]],
    *,
    asset_id: str,
    slot_id: str,
    attachment_id: str,
    sealed_source_hash: str,
    attachment_kind: AttachmentKind = AttachmentKind.DEFORMABLE_BODY,
    topology_class: TopologyClass = TopologyClass.STATIC,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> FullSurfaceRuntimeBridgeV4:
    p = np.asarray(canonical_vertices, dtype=np.float32)
    f = np.asarray(canonical_faces)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise QualificationError("FULL_SURFACE_V4_CANONICAL_VERTICES_INVALID")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or not np.issubdtype(f.dtype, np.integer):
        raise QualificationError("FULL_SURFACE_V4_CANONICAL_FACES_INVALID")
    if np.any(f < 0) or np.any(f >= len(p)):
        raise QualificationError("FULL_SURFACE_V4_CANONICAL_FACE_INDEX_OUT_OF_RANGE")
    if len(sealed_source_hash) != 64:
        raise QualificationError("FULL_SURFACE_V4_SEALED_SOURCE_HASH_MUST_BE_SHA256")

    view_ids = tuple(map(str, required_view_ids))
    required = set(view_ids)
    for label, mapping in (
        ("CAMERA", cameras),
        ("UV", uv_by_view),
        ("PROVENANCE", provenance_by_view),
        ("DONOR", donor_view_index_by_view),
    ):
        if set(map(str, mapping.keys())) != required:
            raise QualificationError(f"FULL_SURFACE_V4_{label}_VIEW_SET_MISMATCH")

    asset = RuntimeV4AttachmentAsset(
        asset_id=str(asset_id),
        slot_id=str(slot_id),
        attachment_id=str(attachment_id),
        attachment_kind=attachment_kind,
        topology_class=topology_class,
        rest_xyz=np.ascontiguousarray(p, dtype=np.float32),
        triangles=np.ascontiguousarray(f, dtype=np.uint32),
        sealed_source_hash=str(sealed_source_hash),
    )

    views: list[RuntimeV4ViewOverlay] = []
    for view_index, view_id in enumerate(view_ids):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        uv = np.asarray(uv_by_view[view_id], dtype=np.float32)
        if uv.shape != (len(p), 2):
            raise QualificationError("FULL_SURFACE_V4_UV_SHAPE_MISMATCH")
        raw_prov = provenance_by_view[view_id]
        codes = []
        for value in raw_prov:
            if isinstance(value, AppearanceProvenance):
                codes.append(provenance_code(value))
            elif isinstance(value, str):
                codes.append(provenance_code(AppearanceProvenance(value)))
            else:
                codes.append(int(value))
        provenance = np.asarray(codes, dtype=np.uint8)
        donors = np.asarray(donor_view_index_by_view[view_id], dtype=np.int16)
        if provenance.shape != (len(f),) or donors.shape != (len(f),):
            raise QualificationError("FULL_SURFACE_V4_FACE_OVERLAY_SHAPE_MISMATCH")
        views.append(RuntimeV4ViewOverlay(
            view_id=view_id,
            view_index=view_index,
            camera=camera,
            assets=(RuntimeV4ViewAssetOverlay(
                asset_id=str(asset_id),
                uv=np.ascontiguousarray(uv, dtype=np.float32),
                provenance_codes=np.ascontiguousarray(provenance, dtype=np.uint8),
                donor_view_indices=np.ascontiguousarray(donors, dtype=np.int16),
            ),),
        ))

    bridge_hash = content_sha256({
        "schema": FULL_SURFACE_RUNTIME_BRIDGE_V4_SCHEMA,
        "asset_hash": asset.asset_hash,
        "view_ids": list(view_ids),
        "camera_contract": [
            {
                "view_id": view.view_id,
                "origin": view.camera.origin,
                "right": view.camera.right,
                "screen_up": view.camera.screen_up,
                "forward": view.camera.forward,
                "half_extent": view.camera.half_extent,
                "resolution": view.camera.resolution,
            }
            for view in views
        ],
    })
    return FullSurfaceRuntimeBridgeV4(
        asset=asset,
        views=tuple(views),
        bridge_hash=bridge_hash,
        vertex_count=len(p),
        face_count=len(f),
        view_ids=view_ids,
    )


def project_runtime_v4_canonical_xyz_for_view(
    canonical_posed_xyz,
    camera: CameraProjectionV3,
) -> np.ndarray:
    """Exact v4 runtime projection used by the v3<->v4 parity gate."""
    return np.ascontiguousarray(project_points_xyz_v3(canonical_posed_xyz, camera), dtype=np.float32)
