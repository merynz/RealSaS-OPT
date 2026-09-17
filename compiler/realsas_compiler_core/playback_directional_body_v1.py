from __future__ import annotations

"""Generic source-backed directional BODY representation for Runtime-v4.

Scientific geometry remains upstream authority. Product BODY drawables are one
view-local mesh attachment per required source direction. Every drawable vertex is
already bound to canonical mechanics by a qualified mesh skin; this module only
packages that authority for playback and never invents appearance or geometry.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.motion_3d_v1 import apply_lbs_matrix_v1
from compiler.realsas_compiler_core.playback_full_surface_v3 import CameraProjectionV3
from compiler.realsas_compiler_core.playback_runtime_v3 import (
    AppearanceProvenance,
    AttachmentKind,
    RuntimeV3FrameComposition,
    RuntimeV3Slot,
    TopologyClass,
)
from compiler.realsas_compiler_core.playback_runtime_v4 import (
    DEFAULT_VIEWS,
    RuntimeV4AttachmentAsset,
    RuntimeV4Clip,
    RuntimeV4Frame,
    RuntimeV4PlaybackContract,
    RuntimeV4ViewAssetOverlay,
    RuntimeV4ViewOverlay,
    provenance_code,
    validate_playback_runtime_v4_contract,
    validate_runtime_v4_clip,
)
from compiler.realsas_compiler_core.types import QualificationError

DIRECTIONAL_BODY_SCHEMA = "RealSaS.DirectionalBodyAsset.v1"


@dataclass(frozen=True, eq=False)
class DirectionalBodyAssetV1:
    view_id: str
    view_index: int
    runtime_asset: RuntimeV4AttachmentAsset
    uv: np.ndarray
    weight_matrix: np.ndarray
    joint_ids: tuple[str, ...]
    mesh_lineage_hash: str
    mesh_skin_lineage_hash: str
    source_texture_sha256: str
    source_alpha_recall: float
    precision_inside_alpha: float
    schema_version: str = DIRECTIONAL_BODY_SCHEMA


def _mesh_runtime_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    vertices = tuple(mesh.vertices)
    if len(vertices) < 3:
        raise QualificationError("DIRECTIONAL_BODY_MESH_VERTEX_SET_TOO_SMALL")
    vertex_ids = tuple(str(v.canonical_mesh_vertex_id) for v in vertices)
    if len(set(vertex_ids)) != len(vertex_ids):
        raise QualificationError("DIRECTIONAL_BODY_MESH_VERTEX_ID_DUPLICATE")
    index = {vertex_id: i for i, vertex_id in enumerate(vertex_ids)}
    xyz = np.asarray([v.P for v in vertices], dtype=np.float64)
    if xyz.shape != (len(vertices), 3) or not np.isfinite(xyz).all():
        raise QualificationError("DIRECTIONAL_BODY_MESH_XYZ_INVALID")
    triangles = []
    for face in mesh.faces:
        if len(face) != 3:
            raise QualificationError("DIRECTIONAL_BODY_REQUIRES_TRIANGULATED_MESH")
        try:
            tri = tuple(index[str(vertex_id)] for vertex_id in face)
        except KeyError as exc:
            raise QualificationError("DIRECTIONAL_BODY_FACE_REFERENCES_UNKNOWN_VERTEX") from exc
        if len(set(tri)) != 3:
            raise QualificationError("DIRECTIONAL_BODY_DEGENERATE_FACE_INDEX")
        triangles.append(tri)
    if not triangles:
        raise QualificationError("DIRECTIONAL_BODY_FACE_SET_EMPTY")
    return (
        np.ascontiguousarray(xyz, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.uint32),
    )


def _mesh_uv(mesh, *, source_width: int, source_height: int) -> np.ndarray:
    width = int(source_width)
    height = int(source_height)
    if width <= 1 or height <= 1:
        raise QualificationError("DIRECTIONAL_BODY_SOURCE_DIMENSIONS_INVALID")
    uv = []
    for vertex in mesh.vertices:
        md = dict(getattr(vertex, "metadata", {}) or {})
        xy = md.get("source_raster_xy", md.get("raster_xy"))
        if xy is None or len(xy) != 2:
            raise QualificationError("DIRECTIONAL_BODY_VERTEX_SOURCE_RASTER_XY_REQUIRED")
        x, y = float(xy[0]), float(xy[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise QualificationError("DIRECTIONAL_BODY_VERTEX_SOURCE_RASTER_XY_NONFINITE")
        u = x / float(width - 1)
        v = y / float(height - 1)
        if u < -1.0e-7 or u > 1.0 + 1.0e-7 or v < -1.0e-7 or v > 1.0 + 1.0e-7:
            raise QualificationError("DIRECTIONAL_BODY_VERTEX_SOURCE_RASTER_XY_OUT_OF_BOUNDS")
        uv.append((min(1.0, max(0.0, u)), min(1.0, max(0.0, v))))
    return np.ascontiguousarray(uv, dtype=np.float32)


def _weight_matrix(mesh, mesh_skin, joint_ids: tuple[str, ...]) -> np.ndarray:
    rows = {str(row.canonical_mesh_vertex_id): row for row in mesh_skin.rows}
    joint_index = {joint_id: i for i, joint_id in enumerate(joint_ids)}
    if len(joint_index) != len(joint_ids) or not joint_ids:
        raise QualificationError("DIRECTIONAL_BODY_JOINT_ORDER_INVALID")
    out = np.zeros((len(mesh.vertices), len(joint_ids)), dtype=np.float64)
    for vertex_index, vertex in enumerate(mesh.vertices):
        vertex_id = str(vertex.canonical_mesh_vertex_id)
        row = rows.get(vertex_id)
        if row is None:
            raise QualificationError("DIRECTIONAL_BODY_MESH_SKIN_ROW_MISSING")
        for joint_id, weight in row.influences:
            joint_id = str(joint_id)
            if joint_id not in joint_index:
                raise QualificationError("DIRECTIONAL_BODY_MESH_SKIN_JOINT_UNKNOWN")
            out[vertex_index, joint_index[joint_id]] = float(weight)
    if not np.isfinite(out).all() or np.any(out < -1.0e-12):
        raise QualificationError("DIRECTIONAL_BODY_WEIGHT_MATRIX_INVALID")
    if not np.allclose(out.sum(axis=1), 1.0, atol=1.0e-9, rtol=0.0):
        raise QualificationError("DIRECTIONAL_BODY_WEIGHT_MATRIX_SIMPLEX_DRIFT")
    return np.ascontiguousarray(out, dtype=np.float64)


def build_directional_body_asset_v1(
    mesh,
    mesh_skin,
    *,
    view_id: str,
    view_index: int,
    joint_ids: Sequence[str],
    source_width: int,
    source_height: int,
    source_texture_sha256: str,
    slot_id: str = "BODY_UNDERLAY",
) -> DirectionalBodyAssetV1:
    view_id = str(view_id)
    view_index = int(view_index)
    if int(mesh.view_index) != view_index:
        raise QualificationError("DIRECTIONAL_BODY_MESH_VIEW_INDEX_DRIFT")
    if len(str(source_texture_sha256)) != 64:
        raise QualificationError("DIRECTIONAL_BODY_SOURCE_TEXTURE_HASH_INVALID")
    xyz, tri = _mesh_runtime_arrays(mesh)
    uv = _mesh_uv(mesh, source_width=source_width, source_height=source_height)
    joints = tuple(map(str, joint_ids))
    weights = _weight_matrix(mesh, mesh_skin, joints)

    report = dict(getattr(mesh, "qualification_report", {}) or {})
    residual = dict(report.get("candidate_residual_report", {}) or {})
    recall = float(report.get("source_alpha_recall", residual.get("source_alpha_recall", -1.0)))
    precision = float(report.get("precision_inside_alpha", residual.get("precision_inside_alpha", -1.0)))
    if not math.isfinite(recall) or not math.isfinite(precision) or recall < 0.0 or precision < 0.0:
        raise QualificationError("DIRECTIONAL_BODY_SOURCE_COVERAGE_EVIDENCE_REQUIRED")

    mesh_lineage_hash = str(mesh.mesh_lineage_hash)
    mesh_skin_lineage_hash = str(mesh_skin.mesh_skin_lineage_hash)
    if len(mesh_lineage_hash) != 64 or len(mesh_skin_lineage_hash) != 64:
        raise QualificationError("DIRECTIONAL_BODY_LINEAGE_HASH_INVALID")
    sealed_source_hash = content_sha256({
        "schema": DIRECTIONAL_BODY_SCHEMA,
        "view_id": view_id,
        "view_index": view_index,
        "mesh_lineage_hash": mesh_lineage_hash,
        "mesh_skin_lineage_hash": mesh_skin_lineage_hash,
        "source_texture_sha256": str(source_texture_sha256),
    })
    attachment_id = f"{slot_id}__{view_id}"
    runtime_asset = RuntimeV4AttachmentAsset(
        asset_id=f"{attachment_id}__ASSET",
        slot_id=str(slot_id),
        attachment_id=attachment_id,
        attachment_kind=AttachmentKind.DEFORMABLE_BODY,
        topology_class=TopologyClass.STATIC,
        rest_xyz=xyz,
        triangles=tri,
        sealed_source_hash=sealed_source_hash,
    )
    return DirectionalBodyAssetV1(
        view_id=view_id,
        view_index=view_index,
        runtime_asset=runtime_asset,
        uv=uv,
        weight_matrix=weights,
        joint_ids=joints,
        mesh_lineage_hash=mesh_lineage_hash,
        mesh_skin_lineage_hash=mesh_skin_lineage_hash,
        source_texture_sha256=str(source_texture_sha256),
        source_alpha_recall=recall,
        precision_inside_alpha=precision,
    )


def build_directional_body_runtime_v4_contract_v1(
    assets: Sequence[DirectionalBodyAssetV1],
    cameras: Mapping[str, CameraProjectionV3],
    *,
    root_bone_id: str,
    slot_id: str = "BODY_UNDERLAY",
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
) -> RuntimeV4PlaybackContract:
    """Build one active source-backed BODY attachment per target direction.

    Each directional asset has a real source owner. Non-owner overlays bind that same
    asset to OTHER_VIEW_SOURCE from its owner view; they are never mislabeled UNSEEN.
    """

    view_ids = tuple(map(str, required_view_ids))
    rows = tuple(assets)
    by_view = {row.view_id: row for row in rows}
    if len(by_view) != len(rows) or set(by_view) != set(view_ids):
        raise QualificationError("DIRECTIONAL_BODY_ASSET_VIEW_SET_MISMATCH")
    if set(cameras) != set(view_ids):
        raise QualificationError("DIRECTIONAL_BODY_CAMERA_VIEW_SET_MISMATCH")
    ordered = tuple(by_view[v] for v in view_ids)
    if any(row.view_index != i for i, row in enumerate(ordered)):
        raise QualificationError("DIRECTIONAL_BODY_ASSET_VIEW_INDEX_DRIFT")
    if len({row.joint_ids for row in ordered}) != 1:
        raise QualificationError("DIRECTIONAL_BODY_JOINT_ORDER_DRIFT")

    runtime_assets = tuple(row.runtime_asset for row in ordered)
    views = []
    for target_index, target_view_id in enumerate(view_ids):
        overlays = []
        for owner_index, row in enumerate(ordered):
            face_count = row.runtime_asset.face_count
            provenance = (
                provenance_code(AppearanceProvenance.DIRECT_SOURCE)
                if target_index == owner_index
                else provenance_code(AppearanceProvenance.OTHER_VIEW_SOURCE)
            )
            overlays.append(RuntimeV4ViewAssetOverlay(
                asset_id=row.runtime_asset.asset_id,
                uv=row.uv,
                provenance_codes=np.full(face_count, provenance, dtype=np.uint8),
                donor_view_indices=np.full(face_count, owner_index, dtype=np.int16),
            ))
        views.append(RuntimeV4ViewOverlay(
            view_id=target_view_id,
            view_index=target_index,
            camera=cameras[target_view_id],
            assets=tuple(overlays),
        ))

    contract = RuntimeV4PlaybackContract(
        slots=(RuntimeV3Slot(
            str(slot_id),
            str(root_bone_id),
            0,
            ordered[0].runtime_asset.attachment_id,
        ),),
        assets=runtime_assets,
        views=tuple(views),
        allow_completion=False,
    )
    validate_playback_runtime_v4_contract(contract, required_view_ids=view_ids)
    return contract


def build_directional_body_runtime_v4_clip_v1(
    contract: RuntimeV4PlaybackContract,
    assets: Sequence[DirectionalBodyAssetV1],
    d1_clip,
    *,
    display_name: str,
    intent: str,
    nominal_fps: float,
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
    runtime_qualified: bool = False,
) -> RuntimeV4Clip:
    """Bake all directional drawables from one canonical D1 mechanical clip."""

    view_ids = tuple(map(str, required_view_ids))
    by_view = {row.view_id: row for row in assets}
    if set(by_view) != set(view_ids):
        raise QualificationError("DIRECTIONAL_BODY_CLIP_ASSET_VIEW_SET_MISMATCH")
    joint_ids = tuple(map(str, d1_clip.joint_ids))
    if any(row.joint_ids != joint_ids for row in by_view.values()):
        raise QualificationError("DIRECTIONAL_BODY_CLIP_JOINT_ORDER_DRIFT")
    if len(d1_clip.times) != len(d1_clip.poses) or not d1_clip.poses:
        raise QualificationError("DIRECTIONAL_BODY_CLIP_FRAME_CARDINALITY_DRIFT")

    slot_id = contract.slots[0].slot_id
    frames = []
    for time_seconds, pose in zip(d1_clip.times, d1_clip.poses):
        posed_by_asset = {}
        for view_id in view_ids:
            row = by_view[view_id]
            posed_by_asset[row.runtime_asset.asset_id] = np.ascontiguousarray(
                apply_lbs_matrix_v1(
                    row.runtime_asset.rest_xyz,
                    row.weight_matrix,
                    pose.skin_matrices,
                ),
                dtype=np.float32,
            )
        composition = {
            view_id: RuntimeV3FrameComposition(
                view_id=view_id,
                draw_order_slot_ids=(slot_id,),
                active_attachment_by_slot={
                    slot_id: by_view[view_id].runtime_asset.attachment_id
                },
            )
            for view_id in view_ids
        }
        frames.append(RuntimeV4Frame(
            time_seconds=float(time_seconds),
            canonical_posed_xyz_by_asset=posed_by_asset,
            composition_by_view=composition,
        ))

    clip = RuntimeV4Clip(
        clip_id=str(d1_clip.clip_id),
        display_name=str(display_name),
        intent=str(intent),
        duration_seconds=float(d1_clip.duration_seconds),
        fps=float(nominal_fps),
        loop=bool(d1_clip.loop),
        frames=tuple(frames),
        runtime_qualified=bool(runtime_qualified),
    )
    validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)
    return clip


__all__ = [
    "DIRECTIONAL_BODY_SCHEMA",
    "DirectionalBodyAssetV1",
    "build_directional_body_asset_v1",
    "build_directional_body_runtime_v4_contract_v1",
    "build_directional_body_runtime_v4_clip_v1",
]
