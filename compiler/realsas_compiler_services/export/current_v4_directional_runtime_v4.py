from __future__ import annotations

"""Qualified directional product/motion bake -> Runtime-v4 view-local assembly.

Each product direction is already a 2D/2.5D artist-facing representation. Runtime-v4
stores every (view, component) as a separate attachment asset and activates only the
five owner-view attachments for that target direction. Qualification-owned bake XY
is lifted through the exact inverse orthographic camera map onto one equal-depth
plane. Projecting that plane through the owner camera is algebraically identical to
the bake XY, while equal depth delegates inter-component ordering to the explicit
semantic slot order.

No solver, model, deformation evaluator, completion, or hidden-surface inference runs
here. This is an export bridge over already-qualified product renderables and motion
bakes.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import math
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
from compiler.realsas_compiler_services.export.current_v4_runtime_v2 import (
    RuntimeTexturePayloadV1,
)
from compiler.realsas_compiler_services.export.runtime_v3 import RuntimeV3TexturePayload
from compiler.realsas_compiler_services.proof.motion_bake import (
    QualificationOwnedMotionBakeIR,
    assert_motion_bake_binding,
)

DIRECTIONAL_ASSEMBLY_RUNTIME_V4_SCHEMA = "RealSaS.DirectionalAssemblyToRuntimeV4.v1"
BODY_COMPONENT_ID = "BODY_UNDERLAY"


@dataclass(frozen=True)
class RuntimeV4DirectionalAssemblyProjectionV1:
    contract: RuntimeV4PlaybackContract
    clips: tuple[RuntimeV4Clip, ...]
    textures: tuple[RuntimeV3TexturePayload, ...]
    runtime_source_indices_by_asset: Mapping[str, tuple[int, ...]]
    owner_view_by_asset: Mapping[str, int]
    component_id_by_asset: Mapping[str, str]
    body_component_id: str
    projection_hash: str
    schema_version: str = DIRECTIONAL_ASSEMBLY_RUNTIME_V4_SCHEMA


def _camera_map(
    cameras: Mapping[str, Mapping | CameraProjectionV3],
    view_ids: tuple[str, ...],
) -> dict[str, CameraProjectionV3]:
    if set(cameras) != set(view_ids):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_CAMERA_VIEW_SET_MISMATCH")
    out = {}
    for index, view_id in enumerate(view_ids):
        raw = cameras[view_id]
        if isinstance(raw, CameraProjectionV3):
            camera = raw
            if camera.view_id != view_id or int(camera.view_index) != index:
                raise QualificationError("DIRECTIONAL_ASSEMBLY_CAMERA_IDENTITY_DRIFT")
        else:
            camera = qualify_camera_v3(raw, view_id=view_id, view_index=index)
        out[view_id] = camera
    return out


def _inverse_project_xy(
    xy,
    camera: CameraProjectionV3,
    *,
    camera_forward_depth: float = 0.0,
) -> np.ndarray:
    """Exact inverse of Runtime-v3/v4 orthographic XY projection at fixed depth."""

    points = np.asarray(xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise QualificationError("DIRECTIONAL_ASSEMBLY_SCREEN_XY_INVALID")
    depth = float(camera_forward_depth)
    if not math.isfinite(depth):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_DEPTH_NONFINITE")

    resolution = float(camera.resolution)
    gx = 2.0 * points[:, 0] / resolution - 1.0
    gy = 1.0 - 2.0 * points[:, 1] / resolution
    origin = np.asarray(camera.origin, dtype=np.float64)
    right = np.asarray(camera.right, dtype=np.float64)
    up = np.asarray(camera.screen_up, dtype=np.float64)
    forward = np.asarray(camera.forward, dtype=np.float64)
    xyz = (
        origin[None, :]
        + (gx * float(camera.half_extent))[:, None] * right[None, :]
        + (gy * float(camera.half_extent))[:, None] * up[None, :]
        + depth * forward[None, :]
    )
    replay = project_points_xyz_v3(xyz, camera)
    if not np.allclose(replay[:, :2], points, atol=2.0e-5, rtol=0.0):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_INVERSE_PROJECTION_REPLAY_FAIL")
    if not np.allclose(replay[:, 2], depth, atol=1.0e-7, rtol=0.0):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_INVERSE_DEPTH_REPLAY_FAIL")
    return np.ascontiguousarray(xyz, dtype=np.float32)


def _texture_rows(
    texture_bindings: Iterable[RuntimeTexturePayloadV1],
    view_ids: tuple[str, ...],
) -> tuple[RuntimeV3TexturePayload, ...]:
    rows = tuple(sorted(texture_bindings, key=lambda row: int(row.view_index)))
    if tuple(int(row.view_index) for row in rows) != tuple(range(len(view_ids))):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_TEXTURE_VIEW_SET_MISMATCH")
    out = []
    for index, row in enumerate(rows):
        if not row.image_relpath or len(str(row.image_sha256)) != 64:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_TEXTURE_IDENTITY_INVALID")
        if int(row.width) <= 0 or int(row.height) <= 0:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_TEXTURE_DIMENSIONS_INVALID")
        out.append(RuntimeV3TexturePayload(
            view_id=view_ids[index],
            texture_path=str(row.image_relpath),
            texture_sha256=str(row.image_sha256),
            texture_crc32=int(row.image_crc32),
            width=int(row.width),
            height=int(row.height),
        ))
    return tuple(out)


def _native_uv(corner, texture: RuntimeTexturePayloadV1) -> tuple[float, float]:
    if corner.authority_class != "OBSERVED_LOCAL":
        raise QualificationError("DIRECTIONAL_ASSEMBLY_NONLOCAL_APPEARANCE_NOT_QUALIFIED")
    if int(corner.donor_view_index) != int(texture.view_index):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_APPEARANCE_DONOR_VIEW_DRIFT")
    x, y = map(float, corner.donor_raster_xy)
    if not all(math.isfinite(v) for v in (x, y)):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_DONOR_RASTER_NONFINITE")
    width, height = int(texture.width), int(texture.height)
    if x < -1e-7 or x > width - 1 + 1e-7 or y < -1e-7 or y > height - 1 + 1e-7:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_DONOR_RASTER_OUT_OF_BOUNDS")
    u = 0.0 if width == 1 else x / float(width - 1)
    v = 0.0 if height == 1 else y / float(height - 1)
    return (
        float(min(1.0, max(0.0, u))),
        float(min(1.0, max(0.0, v))),
    )


def _asset_from_component(
    direction,
    component,
    texture: RuntimeTexturePayloadV1,
    rest_xy,
    camera: CameraProjectionV3,
    *,
    body_component_id: str,
):
    view_index = int(direction.view_index)
    view_id = f"V{view_index}"
    if int(component.view_index) != view_index:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_COMPONENT_VIEW_DRIFT")
    if int(component.appearance.target_view_index) != view_index:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_APPEARANCE_TARGET_VIEW_DRIFT")
    if component.appearance.atlas_payload_hash != texture.atlas_payload_hash:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_APPEARANCE_ATLAS_DRIFT")

    vertices = tuple(component.mesh.vertices)
    canonical_index = {str(v.canonical_mesh_vertex_id): i for i, v in enumerate(vertices)}
    if len(canonical_index) != len(vertices):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_DUPLICATE_CANONICAL_VERTEX")
    rest = np.asarray(rest_xy, dtype=np.float64)
    if rest.shape != (len(vertices), 2) or not np.isfinite(rest).all():
        raise QualificationError("DIRECTIONAL_ASSEMBLY_REST_BAKE_VERTEX_DRIFT")

    corner_map = {
        (int(row.face_index), int(row.corner_index)): row
        for row in component.appearance.corner_bindings
    }
    expected_corners = sum(len(face) for face in component.mesh.faces)
    if len(corner_map) != expected_corners:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_APPEARANCE_COVERAGE_INCOMPLETE")

    source_indices: list[int] = []
    runtime_uv: list[tuple[float, float]] = []
    runtime_index: dict[tuple[int, float, float], int] = {}
    triangles: list[tuple[int, int, int]] = []

    for face_index, face in enumerate(component.mesh.faces):
        if len(face) != 3:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_REQUIRES_TRIANGULATED_MESH")
        tri = []
        for corner_index, vertex_id in enumerate(face):
            source_index = canonical_index.get(str(vertex_id))
            if source_index is None:
                raise QualificationError("DIRECTIONAL_ASSEMBLY_FACE_UNKNOWN_VERTEX")
            corner = corner_map.get((face_index, corner_index))
            if corner is None:
                raise QualificationError("DIRECTIONAL_ASSEMBLY_APPEARANCE_CORNER_MISSING")
            u, v = _native_uv(corner, texture)
            key = (int(source_index), float(u), float(v))
            index = runtime_index.get(key)
            if index is None:
                index = len(source_indices)
                runtime_index[key] = index
                source_indices.append(int(source_index))
                runtime_uv.append((u, v))
            tri.append(index)
        triangles.append(tuple(tri))

    runtime_rest_xy = rest[np.asarray(source_indices, dtype=np.int64)]
    rest_xyz = _inverse_project_xy(runtime_rest_xy, camera, camera_forward_depth=0.0)
    triangles_arr = np.ascontiguousarray(triangles, dtype=np.uint32)
    component_id = str(component.component_id)
    attachment_id = f"{component_id}__{view_id}"
    asset_id = f"{attachment_id}__ASSET"
    kind = (
        AttachmentKind.DEFORMABLE_BODY
        if component_id == str(body_component_id)
        else AttachmentKind.RIGID_COMPONENT
    )
    sealed_source_hash = content_sha256({
        "schema": DIRECTIONAL_ASSEMBLY_RUNTIME_V4_SCHEMA,
        "component_state_hash": str(component.component_state_hash),
        "mesh_lineage_hash": str(component.mesh.mesh_lineage_hash),
        "mesh_skin_lineage_hash": str(component.mesh_skin.mesh_skin_lineage_hash),
        "appearance_lineage_hash": str(component.appearance.appearance_lineage_hash),
        "owner_view": view_index,
        "texture_sha256": str(texture.image_sha256),
        "runtime_vertex_source_indices": source_indices,
        "runtime_uv": runtime_uv,
    })
    asset = RuntimeV4AttachmentAsset(
        asset_id=asset_id,
        slot_id=component_id,
        attachment_id=attachment_id,
        attachment_kind=kind,
        topology_class=TopologyClass.STATIC,
        rest_xyz=rest_xyz,
        triangles=triangles_arr,
        sealed_source_hash=sealed_source_hash,
    )
    return (
        asset,
        np.ascontiguousarray(runtime_uv, dtype=np.float32),
        tuple(source_indices),
    )


def project_directional_product_bakes_to_runtime_v4(
    *,
    product,
    motion_bakes: Iterable[QualificationOwnedMotionBakeIR],
    texture_bindings: Iterable[RuntimeTexturePayloadV1],
    cameras: Mapping[str, Mapping | CameraProjectionV3],
    required_view_ids: Sequence[str] = DEFAULT_VIEWS,
    body_component_id: str = BODY_COMPONENT_ID,
    runtime_qualified: bool = False,
) -> RuntimeV4DirectionalAssemblyProjectionV1:
    view_ids = tuple(map(str, required_view_ids))
    body_component_id = str(body_component_id)
    if not body_component_id:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_BODY_COMPONENT_ID_REQUIRED")
    if view_ids != tuple(f"V{i}" for i in range(len(view_ids))):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_VIEW_IDS_MUST_BE_DENSE_V_INDEXED")
    camera_by_view = _camera_map(cameras, view_ids)
    texture_input = tuple(sorted(texture_bindings, key=lambda row: int(row.view_index)))
    textures = _texture_rows(texture_input, view_ids)
    texture_by_view = {int(row.view_index): row for row in texture_input}

    directions = tuple(sorted(
        product.directional_renderables.directions,
        key=lambda row: int(row.view_index),
    ))
    if tuple(int(row.view_index) for row in directions) != tuple(range(len(view_ids))):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_PRODUCT_VIEW_SET_MISMATCH")

    bakes = tuple(sorted(motion_bakes, key=lambda row: str(row.clip_id)))
    if not bakes:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_REQUIRES_MOTION_BAKE")
    for bake in bakes:
        if bake.source_product_state_hash != product.product_state_hash:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_STALE_BAKE_PRODUCT")
        assert_motion_bake_binding(
            bake,
            source_product_state_hash=product.product_state_hash,
            proof_plan_hash=bake.proof_plan_hash,
        )
    reference = bakes[0]
    ref_rest = dict(reference.rest_mesh_vertices_by_id)

    # Stable slot semantics come from product setup order, never filename order.
    component_rows = {}
    for direction in directions:
        for component in direction.components:
            cid = str(component.component_id)
            prior = component_rows.setdefault(
                cid,
                (int(component.setup_order), str(getattr(component.mesh_skin, "metadata", {}).get("parent_joint_id", ""))),
            )
            if prior[0] != int(component.setup_order):
                raise QualificationError("DIRECTIONAL_ASSEMBLY_SETUP_ORDER_DRIFT")
    slot_order = tuple(sorted(component_rows, key=lambda cid: (component_rows[cid][0], cid)))
    if not slot_order or slot_order[0] != body_component_id:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_BODY_COMPONENT_MUST_BE_FIRST_SLOT")
    if tuple(component_rows[cid][0] for cid in slot_order) != tuple(sorted(component_rows[cid][0] for cid in slot_order)):
        raise QualificationError("DIRECTIONAL_ASSEMBLY_SLOT_ORDER_INVALID")

    roots = [
        str(j.canonical_joint_id)
        for j in product.mechanical_state.skeleton.joints
        if j.parent_canonical_id is None
    ]
    if len(roots) != 1:
        raise QualificationError("DIRECTIONAL_ASSEMBLY_REQUIRES_ONE_ROOT")
    root_joint = roots[0]
    slots = tuple(
        RuntimeV3Slot(
            slot_id=cid,
            bone_id=(component_rows[cid][1] or root_joint),
            setup_order=index,
            default_attachment_id=f"{cid}__V0",
        )
        for index, cid in enumerate(slot_order)
    )

    assets = []
    uv_by_asset = {}
    source_indices_by_asset = {}
    owner_by_asset = {}
    component_by_asset = {}
    asset_id_by_view_component = {}
    for direction in directions:
        view_index = int(direction.view_index)
        view_id = view_ids[view_index]
        components = {str(c.component_id): c for c in direction.components}
        if set(components) != set(slot_order):
            raise QualificationError("DIRECTIONAL_ASSEMBLY_COMPONENT_SET_DRIFT_ACROSS_VIEWS")
        for cid in slot_order:
            component = components[cid]
            mesh_id = f"{view_id}:{cid}"
            rest_xy = ref_rest.get(mesh_id)
            if rest_xy is None:
                raise QualificationError(f"DIRECTIONAL_ASSEMBLY_BAKE_REST_MESH_MISSING:{mesh_id}")
            asset, uv, source_indices = _asset_from_component(
                direction,
                component,
                texture_by_view[view_index],
                rest_xy,
                camera_by_view[view_id],
                body_component_id=body_component_id,
            )
            assets.append(asset)
            uv_by_asset[asset.asset_id] = uv
            source_indices_by_asset[asset.asset_id] = source_indices
            owner_by_asset[asset.asset_id] = view_index
            component_by_asset[asset.asset_id] = cid
            asset_id_by_view_component[(view_index, cid)] = asset.asset_id

    assets = tuple(assets)
    overlays = []
    for target_index, target_view_id in enumerate(view_ids):
        view_assets = []
        for asset in assets:
            owner = int(owner_by_asset[asset.asset_id])
            provenance = (
                provenance_code(AppearanceProvenance.DIRECT_SOURCE)
                if owner == target_index
                else provenance_code(AppearanceProvenance.OTHER_VIEW_SOURCE)
            )
            view_assets.append(RuntimeV4ViewAssetOverlay(
                asset_id=asset.asset_id,
                uv=uv_by_asset[asset.asset_id],
                provenance_codes=np.full(asset.face_count, provenance, dtype=np.uint8),
                donor_view_indices=np.full(asset.face_count, owner, dtype=np.int16),
            ))
        overlays.append(RuntimeV4ViewOverlay(
            view_id=target_view_id,
            view_index=target_index,
            camera=camera_by_view[target_view_id],
            assets=tuple(view_assets),
        ))

    contract = RuntimeV4PlaybackContract(
        slots=slots,
        assets=assets,
        views=tuple(overlays),
        allow_completion=False,
    )
    contract_hash = validate_playback_runtime_v4_contract(
        contract,
        required_view_ids=view_ids,
    )

    clips = []
    for bake in bakes:
        if bake.rest_mesh_vertices_by_id != reference.rest_mesh_vertices_by_id:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_BAKE_REST_DRIFT")
        if bake.triangles_by_mesh_id != reference.triangles_by_mesh_id:
            raise QualificationError("DIRECTIONAL_ASSEMBLY_BAKE_TOPOLOGY_DRIFT")
        frames = []
        for frame in bake.frames:
            mesh_xy = dict(frame.mesh_vertices_by_id)
            orders = dict(frame.render_order_by_view)
            posed = {}
            for asset in assets:
                owner = int(owner_by_asset[asset.asset_id])
                cid = component_by_asset[asset.asset_id]
                mesh_id = f"{view_ids[owner]}:{cid}"
                points = np.asarray(mesh_xy.get(mesh_id, ()), dtype=np.float64)
                source_indices = source_indices_by_asset[asset.asset_id]
                if points.ndim != 2 or points.shape[1:] != (2,) or (
                    source_indices and max(source_indices) >= len(points)
                ):
                    raise QualificationError(
                        f"DIRECTIONAL_ASSEMBLY_FRAME_MESH_DRIFT:{mesh_id}"
                    )
                runtime_xy = points[np.asarray(source_indices, dtype=np.int64)]
                posed[asset.asset_id] = _inverse_project_xy(
                    runtime_xy,
                    camera_by_view[view_ids[owner]],
                    camera_forward_depth=0.0,
                )

            composition = {}
            for target_index, target_view_id in enumerate(view_ids):
                raw_order = tuple(map(str, orders.get(target_view_id, ())))
                expected_mesh_ids = {
                    f"{target_view_id}:{cid}" for cid in slot_order
                }
                if set(raw_order) != expected_mesh_ids or len(raw_order) != len(expected_mesh_ids):
                    raise QualificationError(
                        "DIRECTIONAL_ASSEMBLY_BAKE_DRAW_ORDER_NOT_EXACT_COMPONENT_PERMUTATION"
                    )
                draw_slots = tuple(mesh_id.split(":", 1)[1] for mesh_id in raw_order)
                active = {
                    cid: contract.assets[
                        next(
                            i for i, asset in enumerate(contract.assets)
                            if asset.asset_id == asset_id_by_view_component[(target_index, cid)]
                        )
                    ].attachment_id
                    for cid in slot_order
                }
                composition[target_view_id] = RuntimeV3FrameComposition(
                    view_id=target_view_id,
                    draw_order_slot_ids=draw_slots,
                    active_attachment_by_slot=active,
                )
            frames.append(RuntimeV4Frame(
                time_seconds=float(frame.time_seconds),
                canonical_posed_xyz_by_asset=posed,
                composition_by_view=composition,
            ))
        clip = RuntimeV4Clip(
            clip_id=str(bake.clip_id),
            display_name=str(bake.clip_id),
            intent=str(bake.clip_id).upper(),
            duration_seconds=float(bake.duration_seconds),
            fps=float(bake.fps),
            loop=bool(bake.loop),
            frames=tuple(frames),
            runtime_qualified=bool(runtime_qualified),
        )
        validate_runtime_v4_clip(contract, clip, required_view_ids=view_ids)
        clips.append(clip)

    payload = {
        "schema": DIRECTIONAL_ASSEMBLY_RUNTIME_V4_SCHEMA,
        "source_product_state_hash": product.product_state_hash,
        "playback_contract_hash": contract_hash,
        "clip_bake_hashes": {bake.clip_id: bake.bake_hash for bake in bakes},
        "view_ids": list(view_ids),
        "slot_order": list(slot_order),
        "body_component_id": body_component_id,
        "asset_count": len(assets),
        "runtime_vertex_source_indices": {
            key: list(value) for key, value in sorted(source_indices_by_asset.items())
        },
        "owner_view_by_asset": dict(sorted(owner_by_asset.items())),
        "component_id_by_asset": dict(sorted(component_by_asset.items())),
        "representation": "VIEW_LOCAL_DIRECTIONAL_ATTACHMENTS__QUALIFICATION_BAKE_XY__EQUAL_DEPTH_SEMANTIC_ORDER",
        "solver_replay": False,
        "completion_used": False,
    }
    return RuntimeV4DirectionalAssemblyProjectionV1(
        contract=contract,
        clips=tuple(clips),
        textures=textures,
        runtime_source_indices_by_asset=source_indices_by_asset,
        owner_view_by_asset=owner_by_asset,
        component_id_by_asset=component_by_asset,
        body_component_id=body_component_id,
        projection_hash=content_sha256(payload),
    )


__all__ = [
    "BODY_COMPONENT_ID",
    "DIRECTIONAL_ASSEMBLY_RUNTIME_V4_SCHEMA",
    "RuntimeV4DirectionalAssemblyProjectionV1",
    "project_directional_product_bakes_to_runtime_v4",
]
