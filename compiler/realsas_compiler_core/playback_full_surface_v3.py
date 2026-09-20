from __future__ import annotations

"""Generic full-surface -> Runtime-v3 projection bridge.

This module is deliberately independent of the historical coverage-selected mesh
pipeline. Product Runtime-v3 geometry authority is the complete canonical surface
provided by the caller. Visibility is resolved later by posed camera depth.

No subject-specific policy belongs here.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.camera_geometry_v2 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3AppearancePatch,
    RuntimeV3Mesh,
    RuntimeV3Vertex,
    TopologyClass,
)
from compiler.realsas_compiler_core.types import QualificationError


FULL_SURFACE_RUNTIME_BRIDGE_SCHEMA = "RealSaS.FullSurfaceRuntimeBridge.v3"
FULL_SURFACE_CAMERA_PROJECTION_SCHEMA = "RealSaS.FullSurfaceCameraProjection.v3"


@dataclass(frozen=True)
class FaceAppearanceAuthorityV3:
    provenance: AppearanceProvenance
    donor_view_index: int | None
    atlas_id: str | None
    completion_method: str | None = None


@dataclass(frozen=True)
class FullSurfaceRuntimeBridgeV3:
    meshes: tuple[RuntimeV3Mesh, ...]
    topology_hash: str
    canonical_vertex_sha256: str
    canonical_face_sha256: str
    camera_hash: str
    vertex_count: int
    face_count: int
    view_ids: tuple[str, ...]
    schema_version: str = FULL_SURFACE_RUNTIME_BRIDGE_SCHEMA


def _array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(b"|")
    h.update("x".join(map(str, arr.shape)).encode("ascii"))
    h.update(b"|")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _unit(value, *, label: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise QualificationError(f"FULL_SURFACE_V3_{label}_INVALID")
    n = float(np.linalg.norm(v))
    if n <= 1.0e-12:
        raise QualificationError(f"FULL_SURFACE_V3_{label}_DEGENERATE")
    return v / n


def qualify_camera_v3(camera: Mapping, *, view_id: str, view_index: int) -> CameraProjectionV3:
    """Freeze one orthographic camera into the Runtime-v3 half-pixel convention.

    Screen-space pixel centers live at (x+0.5, y+0.5). Unlike the historical
    dense bridge, no -0.5 legacy raster offset is emitted here.
    """

    try:
        raw_index = int(camera.get("view_index", view_index))
        origin = np.asarray(camera["origin"], dtype=np.float64)
        right = _unit(camera["right"], label="CAMERA_RIGHT")
        up = _unit(camera["screen_up"], label="CAMERA_UP")
        forward = _unit(camera["forward"], label="CAMERA_FORWARD")
        half = float(camera["half_extent"])
        resolution = int(camera["resolution"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_CONTRACT_INCOMPLETE") from exc

    if raw_index != int(view_index):
        raise QualificationError("FULL_SURFACE_V3_CAMERA_VIEW_INDEX_DRIFT")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise QualificationError("FULL_SURFACE_V3_CAMERA_ORIGIN_INVALID")
    if not math.isfinite(half) or half <= 0.0 or resolution <= 0:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_SCALE_INVALID")
    if abs(float(np.dot(right, up))) > 1.0e-6:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_SCREEN_BASIS_NOT_ORTHOGONAL")
    if abs(float(np.dot(right, forward))) > 1.0e-6 or abs(float(np.dot(up, forward))) > 1.0e-6:
        raise QualificationError("FULL_SURFACE_V3_CAMERA_FORWARD_NOT_ORTHOGONAL")

    return CameraProjectionV3(
        view_id=str(view_id),
        view_index=int(view_index),
        origin=tuple(map(float, origin)),
        right=tuple(map(float, right)),
        screen_up=tuple(map(float, up)),
        forward=tuple(map(float, forward)),
        half_extent=half,
        resolution=resolution,
    )


def project_points_xyz_v3(points_xyz, camera: CameraProjectionV3) -> np.ndarray:
    """Project canonical/posed 3D points to Runtime-v3 screen XYZ.

    X/Y use the frozen orthographic camera. Z is camera-forward distance from the
    same posed 3D point and is therefore suitable for the Runtime-v3 depth test.
    Smaller Z is nearer. No affine fit is performed here or per frame.
    """

    p = np.asarray(points_xyz, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise QualificationError("FULL_SURFACE_V3_POINTS_INVALID")
    origin = np.asarray(camera.origin, dtype=np.float64)
    right = np.asarray(camera.right, dtype=np.float64)
    up = np.asarray(camera.screen_up, dtype=np.float64)
    forward = np.asarray(camera.forward, dtype=np.float64)
    d = p - origin[None, :]
    gx = (d @ right) / float(camera.half_extent)
    gy = -(d @ up) / float(camera.half_extent)
    # R1 screen convention: pixel centers are 0.5, 1.5, ...
    x = (gx + 1.0) * 0.5 * float(camera.resolution)
    y = (gy + 1.0) * 0.5 * float(camera.resolution)
    z = d @ forward
    out = np.stack((x, y, z), axis=1)
    if not np.isfinite(out).all():
        raise QualificationError("FULL_SURFACE_V3_PROJECTED_NONFINITE")
    return out


def _validate_surface(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise QualificationError("FULL_SURFACE_V3_CANONICAL_VERTICES_INVALID")
    if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0:
        raise QualificationError("FULL_SURFACE_V3_CANONICAL_FACES_INVALID")
    if np.any(f < 0) or np.any(f >= len(p)):
        raise QualificationError("FULL_SURFACE_V3_CANONICAL_FACE_INDEX_OUT_OF_RANGE")
    return p, f


def _validate_uv_by_view(
    uv_by_view: Mapping[str, np.ndarray],
    *,
    view_ids: tuple[str, ...],
    vertex_count: int,
) -> dict[str, np.ndarray]:
    if set(map(str, uv_by_view.keys())) != set(view_ids):
        raise QualificationError("FULL_SURFACE_V3_UV_VIEW_SET_MISMATCH")
    out: dict[str, np.ndarray] = {}
    for view_id in view_ids:
        uv = np.asarray(uv_by_view[view_id], dtype=np.float64)
        if uv.shape != (vertex_count, 2) or not np.isfinite(uv).all():
            raise QualificationError(f"FULL_SURFACE_V3_UV_PAYLOAD_INVALID:{view_id}")
        if np.any(uv < 0.0) or np.any(uv > 1.0):
            raise QualificationError(f"FULL_SURFACE_V3_UV_OUTSIDE_UNIT_DOMAIN:{view_id}")
        out[view_id] = uv
    return out


def build_full_surface_runtime_meshes_v3(
    canonical_vertices,
    canonical_faces,
    cameras: Mapping[str, Mapping],
    uv_by_view: Mapping[str, np.ndarray],
    *,
    slot_id: str,
    attachment_id: str,
    attachment_kind: AttachmentKind = AttachmentKind.DEFORMABLE_BODY,
    topology_class: TopologyClass = TopologyClass.STATIC,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> FullSurfaceRuntimeBridgeV3:
    """Materialize exact full topology as one Runtime-v3 mesh variant per view.

    UVs are intentionally caller-owned. Geometry projection must never invent
    appearance authority or fake texels merely to make a runtime package render.
    """

    if not str(slot_id).strip() or not str(attachment_id).strip():
        raise QualificationError("FULL_SURFACE_V3_SLOT_ATTACHMENT_REQUIRED")
    p, f = _validate_surface(canonical_vertices, canonical_faces)
    view_ids = tuple(map(str, required_view_ids))
    if not view_ids or len(view_ids) != len(set(view_ids)):
        raise QualificationError("FULL_SURFACE_V3_REQUIRED_VIEW_SET_INVALID")
    if set(map(str, cameras.keys())) != set(view_ids):
        raise QualificationError("FULL_SURFACE_V3_CAMERA_VIEW_SET_MISMATCH")
    uv_map = _validate_uv_by_view(uv_by_view, view_ids=view_ids, vertex_count=len(p))

    qualified_cameras: list[CameraProjectionV3] = []
    meshes: list[RuntimeV3Mesh] = []
    exact_faces = tuple(tuple(map(int, tri)) for tri in f.tolist())
    for view_index, view_id in enumerate(view_ids):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        qualified_cameras.append(camera)
        xyz = project_points_xyz_v3(p, camera)
        uv = uv_map[view_id]
        vertices = tuple(
            RuntimeV3Vertex(
                float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                float(uv[i, 0]), float(uv[i, 1]),
            )
            for i in range(len(p))
        )
        meshes.append(RuntimeV3Mesh(
            view_id=view_id,
            mesh_id=f"{view_id}:{attachment_id}",
            slot_id=str(slot_id),
            attachment_id=str(attachment_id),
            attachment_kind=attachment_kind,
            topology_class=topology_class,
            vertices=vertices,
            triangles=exact_faces,
        ))

    vertex_sha = _array_sha256(p)
    face_sha = _array_sha256(f)
    camera_payload = [
        {
            "view_id": c.view_id,
            "view_index": c.view_index,
            "origin": c.origin,
            "right": c.right,
            "screen_up": c.screen_up,
            "forward": c.forward,
            "half_extent": c.half_extent,
            "resolution": c.resolution,
        }
        for c in qualified_cameras
    ]
    camera_hash = content_sha256({"schema": FULL_SURFACE_CAMERA_PROJECTION_SCHEMA, "cameras": camera_payload})
    topology_hash = content_sha256({
        "schema": FULL_SURFACE_RUNTIME_BRIDGE_SCHEMA,
        "canonical_vertex_sha256": vertex_sha,
        "canonical_face_sha256": face_sha,
        "camera_hash": camera_hash,
        "vertex_count": len(p),
        "face_count": len(f),
        "view_ids": view_ids,
        "slot_id": str(slot_id),
        "attachment_id": str(attachment_id),
        "attachment_kind": attachment_kind.value,
        "topology_class": topology_class.value,
    })
    return FullSurfaceRuntimeBridgeV3(
        meshes=tuple(meshes),
        topology_hash=topology_hash,
        canonical_vertex_sha256=vertex_sha,
        canonical_face_sha256=face_sha,
        camera_hash=camera_hash,
        vertex_count=len(p),
        face_count=len(f),
        view_ids=view_ids,
    )


def project_posed_full_surface_frames_v3(
    posed_canonical_vertices,
    cameras: Mapping[str, Mapping],
    *,
    attachment_id: str,
    expected_vertex_count: int,
    required_view_ids: Sequence[str] = tuple(f"V{i}" for i in range(8)),
) -> dict[str, tuple[tuple[float, float, float], ...]]:
    """Project one posed canonical surface into every Runtime-v3 view variant."""

    p = np.asarray(posed_canonical_vertices, dtype=np.float64)
    if p.shape != (int(expected_vertex_count), 3) or not np.isfinite(p).all():
        raise QualificationError("FULL_SURFACE_V3_POSED_VERTEX_PAYLOAD_INVALID")
    view_ids = tuple(map(str, required_view_ids))
    if set(map(str, cameras.keys())) != set(view_ids):
        raise QualificationError("FULL_SURFACE_V3_POSED_CAMERA_VIEW_SET_MISMATCH")
    out: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for view_index, view_id in enumerate(view_ids):
        camera = qualify_camera_v3(cameras[view_id], view_id=view_id, view_index=view_index)
        xyz = project_points_xyz_v3(p, camera)
        out[f"{view_id}:{attachment_id}"] = tuple(tuple(map(float, row)) for row in xyz)
    return out


def build_face_appearance_patches_v3(
    *,
    mesh: RuntimeV3Mesh,
    face_authorities: Sequence[FaceAppearanceAuthorityV3],
) -> tuple[RuntimeV3AppearancePatch, ...]:
    """Convert exact per-face appearance authority into coherent runtime patches.

    This groups only faces with identical authority. It does not choose donors.
    R2 owns donor visibility qualification; this function only preserves it.
    """

    if len(face_authorities) != len(mesh.triangles):
        raise QualificationError("FULL_SURFACE_V3_APPEARANCE_FACE_COUNT_MISMATCH")
    grouped: dict[tuple, list[int]] = {}
    for face_index, authority in enumerate(face_authorities):
        provenance = authority.provenance
        donor = authority.donor_view_index
        atlas = authority.atlas_id
        completion = authority.completion_method
        if provenance == AppearanceProvenance.UNSEEN:
            if donor is not None or atlas is not None or completion is not None:
                raise QualificationError("FULL_SURFACE_V3_UNSEEN_MUST_REMAIN_UNBOUND")
        elif provenance == AppearanceProvenance.COMPLETION:
            if not completion:
                raise QualificationError("FULL_SURFACE_V3_COMPLETION_METHOD_REQUIRED")
        else:
            if donor is None or atlas is None:
                raise QualificationError("FULL_SURFACE_V3_SOURCE_APPEARANCE_REQUIRES_DONOR_AND_ATLAS")
        key = (provenance.value, donor, atlas, completion)
        grouped.setdefault(key, []).append(int(face_index))

    patches: list[RuntimeV3AppearancePatch] = []
    for patch_index, key in enumerate(sorted(grouped, key=lambda x: tuple("" if v is None else str(v) for v in x))):
        provenance_value, donor, atlas, completion = key
        patches.append(RuntimeV3AppearancePatch(
            patch_id=f"{mesh.mesh_id}:AP{patch_index:04d}",
            mesh_id=mesh.mesh_id,
            face_indices=tuple(grouped[key]),
            provenance=AppearanceProvenance(provenance_value),
            donor_view_index=None if donor is None else int(donor),
            atlas_id=None if atlas is None else str(atlas),
            completion_method=None if completion is None else str(completion),
        ))
    return tuple(patches)
