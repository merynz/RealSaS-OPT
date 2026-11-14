from __future__ import annotations

"""Real canonical Mechanics + Graphics + UV -> existing RealSaS .rss archive.

This is an INTERACTIVE PREVIEW artifact while independent product Proof still
fails; it is NOT a ProductRevision. Save/reopen preserves the exact canonical
mesh/rig/skin and authored control frames, and recomputes their real LBS poses
rather than trusting pose.bin to be an animation source.
"""

from collections import OrderedDict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, canonical_mesh_from_dict_v1,
    validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigIRV2, canonical_rig_from_dict_v2,
    validate_canonical_rig_domain_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, canonical_skin_from_dict_v2,
    validate_canonical_skin_domain_v2,
)
from compiler.realsas_compiler_core.graphics_runtime_v1 import (
    RuntimeGraphicsV1, runtime_graphics_hash_v1, validate_runtime_graphics_v1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    ControlTransformIRV1, solve_control_pose_v1, deform_mesh_lbs_v1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    qualify_camera_v3, project_points_xyz_v3,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    encode_runtime_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
)
from .rss_vnext_native_uv_transport_v1 import encode_native_rss_vnext_uv_v1

SCHEMA = "RealSaS.RSSvNext.NativeInteractivePuppetPreview.v1"
_REQUIRED = frozenset({
    "manifest.txt", "mesh.json", "rig.json", "skin.json",
    "graphics.bin", "texture.bin", "pose.bin", "controls.json",
})
_MAX_PREVIEW_BYTES = 160 * 1024 * 1024


def _json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _make_pose_binary(
    mesh: CanonicalMeshIR, rig: CanonicalRigIRV2, skin: CanonicalSkinIRV2,
    frames: Sequence[Sequence[ControlTransformIRV1]], camera: Mapping,
) -> tuple[bytes, list[str]]:
    camera_state = qualify_camera_v3(
        camera, view_id="V0", view_index=0,
    )
    if camera_state.resolution > 2048 or len(frames) not in range(1, 513):
        raise RuntimeError("RSS_PREVIEW_CAMERA_OR_FRAME_RESOURCE_BUDGET")
    v_ids = tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)
    v_index = {value: index for index, value in enumerate(v_ids)}
    if len(v_index) != len(v_ids) or len(mesh.faces) > 4096:
        raise RuntimeError("RSS_PREVIEW_CANONICAL_ADDRESS_BUDGET")
    faces = tuple(tuple(v_index[k] for k in face) for face in mesh.faces)
    data = bytearray(b"RSPOSE01")
    data.extend(struct.pack("<IIIIII", 1, len(v_ids), len(faces),
                            len(frames), camera_state.resolution,
                            camera_state.resolution))
    for face in faces:
        data.extend(struct.pack("<III", *face))
    program_hashes = []
    for controls in frames:
        if len(controls) > 256 or any(type(x) is not ControlTransformIRV1 for x in controls):
            raise RuntimeError("RSS_PREVIEW_CONTROL_FRAME_DOMAIN")
        pose = solve_control_pose_v1(rig, controls)
        result = deform_mesh_lbs_v1(mesh, rig, skin, pose)
        if tuple(result.canonical_mesh_vertex_ids) != v_ids:
            raise RuntimeError("RSS_PREVIEW_MECHANICS_VERTEX_REORDER")
        positions = np.asarray(result.positions, dtype=np.float64)
        projected = project_points_xyz_v3(positions, camera_state)
        if not np.isfinite(projected).all() or np.any(projected[:, 2] <= 0):
            raise RuntimeError("RSS_PREVIEW_VIRTUAL_CAMERA_CLIPS_MESH")
        data.extend(projected.astype("<f8", copy=False).tobytes(order="C"))
        program_hashes.append(pose.control_program_hash)
    return bytes(data), program_hashes


def select_rss_vnext_native_preview_camera_v1(
    mesh: CanonicalMeshIR, *, resolution: int = 128,
) -> dict[str, Any]:
    """Choose an honest virtual preview angle from the *same* canonical mesh.

    Maximize aggregate triangle projected area among three orthogonal axes.
    This is a deterministic display heuristic; it is NOT the camera of either
    uploaded image and cannot establish source visibility or back-side color.
    """
    validate_canonical_mesh_domain_v1(mesh)
    if type(resolution) is not int or not 16 <= resolution <= 2048:
        raise RuntimeError("RSS_PREVIEW_VIRTUAL_RESOLUTION_INVALID")
    xyz = np.asarray([v.P for v in mesh.vertices], dtype=np.float64)
    index = {v.canonical_mesh_vertex_id: i for i,v in enumerate(mesh.vertices)}
    faces = np.asarray([[index[k] for k in f] for f in mesh.faces], dtype=np.int64)
    if len(faces)>4096 or len(xyz)>65536:
        raise RuntimeError("RSS_PREVIEW_VIRTUAL_MESH_BUDGET")
    axes = (
        ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
        ((1.,0.,0.),(0.,0.,1.),(0.,1.,0.)),
        ((0.,1.,0.),(0.,0.,1.),(1.,0.,0.)),
    )
    center = (xyz.min(axis=0)+xyz.max(axis=0))*.5
    best = None
    for right,up,forward in axes:
        projected = np.stack(
            (xyz @ np.asarray(right), xyz @ np.asarray(up)), axis=-1
        )
        triangles = projected[faces]
        area = float(np.abs(
            (triangles[:,1,0]-triangles[:,0,0]) *
            (triangles[:,2,1]-triangles[:,0,1]) -
            (triangles[:,1,1]-triangles[:,0,1]) *
            (triangles[:,2,0]-triangles[:,0,0])
        ).sum())
        extents = projected.max(axis=0)-projected.min(axis=0)
        # Prefer larger area. Ties pick XY, then XZ, then YZ consistently.
        if best is None or area > best[0] + 1e-12:
            best = (area,right,up,forward,extents)
    assert best is not None
    area,right,up,forward,extents = best
    if area <= 1e-12:
        raise RuntimeError("RSS_PREVIEW_NO_NONDEGENERATE_VIRTUAL_VIEW")
    diagonal = float(np.linalg.norm(xyz.max(axis=0)-xyz.min(axis=0)))
    distance = max(2.,diagonal*3.)
    origin = center-np.asarray(forward)*distance
    return {
        "view_index":0,
        "origin":tuple(float(x) for x in origin),
        "right":right, "screen_up":up, "forward":forward,
        "half_extent":max(float(extents.max())*.65,1e-3),
        "resolution":resolution,
        "source_camera_observed":False,
        "selection_policy":"MAX_PROJECTED_CANONICAL_TRIANGLE_AREA_V1",
    }


def prepare_rss_vnext_virtual_preview_graphics_v1(
    graphics: RuntimeGraphicsV1, *,
    canonical_mesh_hash: str, face_count: int,
) -> RuntimeGraphicsV1:
    """Demote an unregistered *virtual* camera to conservative geometry visibility.

    View 0's learned visibility mask and partial order describe one indexed
    candidate view. A virtual axis selected from a mesh is NOT that view.
    Reusing those rules would suppress real posed triangles or invert depth.
    Only the preview-only view 0 is widened to all canonical fragments and
    depth sorting; the other seven candidate schedules are preserved, but
    cannot be selected by this single-camera archive.
    """
    validate_runtime_graphics_v1(
        graphics, canonical_mesh_hash=canonical_mesh_hash,
        face_count=face_count,
    )
    if len(graphics.views) != 8:
        raise RuntimeError("RSS_PREVIEW_GRAPHICS_VIEW_COVERAGE_REQUIRED")
    n = len(graphics.fragment_ids)
    universal_view = tuple(range(n))
    views = tuple(
        replace(
            view,
            visible_fragment_indices=universal_view,
            topological_fragment_order=universal_view,
            order_edges=(),
        ) if view.view_index == 0 else view
        for view in graphics.views
    )
    preview = replace(
        graphics,
        views=views,
        runtime_graphics_hash="",
        metadata={
            **graphics.metadata,
            "virtual_camera_view_is_registered": False,
            "virtual_preview_source_view_visibility_reused": False,
            "virtual_preview_draw_policy": "ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY",
        },
    )
    preview = replace(
        preview, runtime_graphics_hash=runtime_graphics_hash_v1(preview),
    )
    validate_runtime_graphics_v1(
        preview, canonical_mesh_hash=canonical_mesh_hash,
        face_count=face_count,
    )
    return preview


def compile_rss_vnext_native_preview_package_v1(
    *, mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    graphics: RuntimeGraphicsV1,
    uv_atlas_cas: bytes,
    virtual_camera: Mapping[str, Any],
    control_frames: Sequence[Sequence[ControlTransformIRV1]],
) -> tuple[bytes, dict[str, Any]]:
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    if (
        rig.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.deform_structure_hash != rig.deform_structure_hash
    ):
        raise RuntimeError("RSS_PREVIEW_MESH_RIG_SKIN_LINEAGE_DRIFT")
    validate_runtime_graphics_v1(
        graphics, canonical_mesh_hash=mesh.mesh_lineage_hash,
        face_count=len(mesh.faces),
    )
    if len(graphics.views)!=8 or len(control_frames)>512:
        raise RuntimeError("RSS_PREVIEW_GRAPHICS_VIEW_DOMAIN")
    texture_bytes = encode_native_rss_vnext_uv_v1(
        uv_atlas_cas=uv_atlas_cas,
        canonical_mesh_lineage_hash=mesh.mesh_lineage_hash,
    )
    preview_graphics = prepare_rss_vnext_virtual_preview_graphics_v1(
        graphics, canonical_mesh_hash=mesh.mesh_lineage_hash,
        face_count=len(mesh.faces),
    )
    graphics_bytes = encode_runtime_graphics_v1(preview_graphics)
    camera = dict(virtual_camera)
    camera["view_index"] = 0
    pose_bytes, pose_hashes = _make_pose_binary(
        mesh, rig, skin, control_frames, camera,
    )
    controls_payload = {
        "schema": "RealSaS.RSSvNext.NativePreviewControls.v1",
        "virtual_camera_is_original_source_observation": False,
        "virtual_camera": camera,
        "control_frames": [
            [asdict(row) for row in rows] for rows in control_frames
        ],
        "mechanics_pose_hashes": pose_hashes,
        "motion_source": "USER_AUTHORED_CONTROLS_REAL_LBS_NOT_GENERATED_MOTION",
        "product_revision_minted": False,
    }
    entries: OrderedDict[str,bytes] = OrderedDict([
        ("mesh.json", _json(mesh.to_dict())),
        ("rig.json", _json(rig.to_dict())),
        ("skin.json", _json(skin.to_dict())),
        ("graphics.bin", graphics_bytes),
        ("texture.bin", texture_bytes),
        ("pose.bin", pose_bytes),
        ("controls.json", _json(controls_payload)),
    ])
    manifest = {
        "schema": SCHEMA,
        "admission": "DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION",
        "product_revision_minted": "0",
        "original_source_camera_observed": "0",
        "source_rgb_surface_ownership_qualified": "0",
        "pose_conditioned_visibility_source_qualified": "0",
        "native_texture_preview_rendering_enabled": "1",
        "canonical_mesh_lineage_hash": mesh.mesh_lineage_hash,
        "canonical_rig_lineage_hash": rig.rig_lineage_hash,
        "canonical_skin_lineage_hash": skin.skin_lineage_hash,
        "graphics_runtime_hash": preview_graphics.runtime_graphics_hash,
        "source_compiled_graphics_runtime_hash": graphics.runtime_graphics_hash,
        "virtual_preview_camera_registered_to_graphics_view": "0",
        "virtual_preview_draw_policy": "ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY",
        "frame_count": str(len(control_frames)),
        "view_index": "0",
        "color_source_layer_policy": "EXPLICIT_HINT_ONLY",
        "texture_layer_draw_order_authority": "POSED_DEPTH_ONLY_FOR_UNREGISTERED_VIRTUAL_CAMERA",
    }
    for name, blob in entries.items():
        manifest["sha256:" + name] = _sha(blob)
    entries["manifest.txt"] = (
        "\n".join(f"{key}={value}" for key, value in sorted(manifest.items()))
        + "\n"
    ).encode("utf-8")
    packed = _entry_bytes(entries)
    if len(packed) > _MAX_PREVIEW_BYTES:
        raise RuntimeError("RSS_PREVIEW_ARCHIVE_BUDGET")
    return packed, manifest


def reopen_rss_vnext_native_preview_package_v1(path: Path) -> dict[str, Any]:
    """Reopen authoring state AND recompute every stored pose from actual LBS."""
    if path.stat().st_size > _MAX_PREVIEW_BYTES:
        raise RuntimeError("RSS_PREVIEW_REOPEN_ARCHIVE_BUDGET")
    archive = read_rss_v2(path)
    if set(archive) != _REQUIRED:
        raise RuntimeError("RSS_PREVIEW_REOPEN_ENTRIES")
    lines = archive["manifest.txt"].decode("utf-8").splitlines()
    manifest = {}
    for line in lines:
        if "=" not in line:
            raise RuntimeError("RSS_PREVIEW_REOPEN_MANIFEST_LINE")
        key, value = line.split("=", 1)
        if key in manifest:
            raise RuntimeError("RSS_PREVIEW_REOPEN_DUPLICATE_MANIFEST")
        manifest[key] = value
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("admission") != "DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION"
        or manifest.get("product_revision_minted") != "0"
        or manifest.get("original_source_camera_observed") != "0"
        or manifest.get("source_rgb_surface_ownership_qualified") != "0"
    ):
        raise RuntimeError("RSS_PREVIEW_REOPEN_FALSE_PRODUCT_AUTHORITY")
    for name in _REQUIRED - {"manifest.txt"}:
        if manifest.get("sha256:" + name) != _sha(archive[name]):
            raise RuntimeError("RSS_PREVIEW_REOPEN_ENTRY_SHA256")
    mesh = canonical_mesh_from_dict_v1(json.loads(archive["mesh.json"]))
    rig = canonical_rig_from_dict_v2(json.loads(archive["rig.json"]))
    skin = canonical_skin_from_dict_v2(json.loads(archive["skin.json"]))
    data = json.loads(archive["controls.json"])
    if (
        data.get("schema") != "RealSaS.RSSvNext.NativePreviewControls.v1"
        or data.get("virtual_camera_is_original_source_observation") is not False
        or data.get("product_revision_minted") is not False
        or type(data.get("control_frames")) is not list
        or len(data["control_frames"]) != int(manifest["frame_count"])
    ):
        raise RuntimeError("RSS_PREVIEW_REOPEN_CONTROL_STATE_DRIFT")
    frames = [
        [ControlTransformIRV1(
            canonical_control_id=str(row["canonical_control_id"]),
            translation=tuple(row["translation"]),
            rotation_quat_wxyz=tuple(row["rotation_quat_wxyz"]),
            scale=tuple(row["scale"]),
        ) for row in group]
        for group in data["control_frames"]
    ]
    pose_bytes, hashes = _make_pose_binary(
        mesh, rig, skin, frames, data["virtual_camera"],
    )
    if (
        pose_bytes != archive["pose.bin"]
        or hashes != data["mechanics_pose_hashes"]
        or manifest["canonical_mesh_lineage_hash"] != mesh.mesh_lineage_hash
        or manifest["canonical_rig_lineage_hash"] != rig.rig_lineage_hash
        or manifest["canonical_skin_lineage_hash"] != skin.skin_lineage_hash
    ):
        raise RuntimeError("RSS_PREVIEW_REOPEN_MECHANICS_POSE_REPLAY_DRIFT")
    return {
        "manifest": manifest, "mesh": mesh, "rig": rig, "skin": skin,
        "control_frames": tuple(tuple(group) for group in frames),
        "virtual_camera": data["virtual_camera"],
        "pose_bytes": pose_bytes,
        "graphics_bytes": archive["graphics.bin"],
        "texture_bytes": archive["texture.bin"],
        "product_revision_minted": False,
    }


def edit_rss_vnext_native_preview_controls_v1(
    *, source_package_path: Path,
    control_frames: Sequence[Sequence[ControlTransformIRV1]],
) -> tuple[bytes, dict[str, Any]]:
    """Save a new actual Mechanics+LBS edit without replacing puppet identity.

    The existing source .rss is fully reopened and independently replayed
    BEFORE edits are accepted. All unchanged canonical Mesh/Rig/Skin,
    Graphics and UV bytes are kept identical. The ONLY mutable artifacts are
    controls.json, pose.bin, and manifest.txt. This remains a PREVIEW, never
    a ProductRevision/verified source color.
    """
    state = reopen_rss_vnext_native_preview_package_v1(source_package_path)
    entries = read_rss_v2(source_package_path)
    frames = tuple(tuple(group) for group in control_frames)
    pose_bytes, pose_hashes = _make_pose_binary(
        state["mesh"], state["rig"], state["skin"],
        frames, state["virtual_camera"],
    )
    old_controls = json.loads(entries["controls.json"])
    if (
        old_controls.get("schema") != "RealSaS.RSSvNext.NativePreviewControls.v1"
        or old_controls.get("product_revision_minted") is not False
        or old_controls.get("virtual_camera_is_original_source_observation") is not False
    ):
        raise RuntimeError("RSS_PREVIEW_EDIT_AUTHORITY_MISMATCH")
    old_controls["control_frames"] = [
        [asdict(row) for row in group] for group in frames
    ]
    old_controls["mechanics_pose_hashes"] = pose_hashes
    old_controls["edit_source_control_state_sha256"] = _sha(entries["controls.json"])
    old_controls["edit_operation"] = "CANONICAL_CONTROLS_RECOMPUTED_MECHANICS_LBS"
    entries["controls.json"] = _json(old_controls)
    entries["pose.bin"] = pose_bytes
    manifest = dict(state["manifest"])
    manifest["sha256:controls.json"] = _sha(entries["controls.json"])
    manifest["sha256:pose.bin"] = _sha(pose_bytes)
    manifest["frame_count"] = str(len(frames))
    entries["manifest.txt"] = (
        "\n".join(f"{key}={value}" for key, value in sorted(manifest.items()))
        + "\n"
    ).encode("utf-8")
    packed = _entry_bytes(entries)
    if len(packed) > _MAX_PREVIEW_BYTES:
        raise RuntimeError("RSS_PREVIEW_EDIT_ARCHIVE_BUDGET")
    return packed, manifest


__all__ = [
    "SCHEMA", "compile_rss_vnext_native_preview_package_v1",
    "reopen_rss_vnext_native_preview_package_v1",
    "select_rss_vnext_native_preview_camera_v1",
    "prepare_rss_vnext_virtual_preview_graphics_v1",
    "edit_rss_vnext_native_preview_controls_v1",
]
