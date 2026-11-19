from __future__ import annotations

"""Live Mechanics -> native 2D Graphics pose transport (no source-pixel claim).

The existing editor already computes LBS, optional joint XPBD and optional
surface XPBD on the SAME canonical mesh. The normal source-preview .rss
packager deliberately re-solves only LBS for reproducible authoring replay.
This bridge streams the editor's *actual final* vertex positions to the
RSPOSE01/02 ABI consumed by the standalone native RGBA renderer. It does
not alter the persistent .rss package, mint ProductProof or substitute
synthetic source appearance.

The caller must obtain the frames from ControllablePuppetSessionV1.frame().
Hashes prevent cross-revision joins. An XPBD trajectory must be reproduced
by the owning Mechanics runtime before receiving product authority.
"""

from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from compiler.realsas_compiler_core.canonical_mesh_ir_v1 import (
    CanonicalMeshIR, validate_canonical_mesh_domain_v1,
)
from compiler.realsas_compiler_core.canonical_rig_ir_v2 import (
    CanonicalRigIRV2, validate_canonical_rig_domain_v2,
)
from compiler.realsas_compiler_core.canonical_skin_ir_v2 import (
    CanonicalSkinIRV2, validate_canonical_skin_domain_v2,
)
from compiler.realsas_compiler_core.controllable_puppet_runtime_v1 import (
    PuppetRuntimeFrameV1,
)
from compiler.realsas_compiler_core.mechanics_control_solver_v1 import (
    deform_mesh_lbs_v1,
)
from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    qualify_camera_v3, project_points_xyz_v3,
)
from .rss_vnext_native_preview_package_v1 import (
    reopen_rss_vnext_native_preview_package_v1,
)

SCHEMA = "RealSaS.RSSvNext.LiveMechanicsNativePoseTransport.v1"
AUTHORITY = "LIVE_MECHANICS_FINAL_SURFACE_NOT_SOURCE_OR_PRODUCT_PROOF"


def encode_rss_vnext_live_mechanics_native_pose_v1(
    *,
    mesh: CanonicalMeshIR,
    rig: CanonicalRigIRV2,
    skin: CanonicalSkinIRV2,
    frames: Sequence[PuppetRuntimeFrameV1],
    frame_times_seconds: Sequence[float],
    cameras: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[bytes, dict[str, Any]]:
    """Encode final LBS/XPBD frames using the native player's pose ABI.

    Unlike the saved .rss preview, this is an ephemeral render stream. The
    result must be paired with the *same mesh/face IDs*, Graphics binary and
    UV atlas. Native RGB/alpha still needs a separate source appearance court.
    """
    validate_canonical_mesh_domain_v1(mesh)
    validate_canonical_rig_domain_v2(rig)
    validate_canonical_skin_domain_v2(skin)
    if (
        rig.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.mesh_lineage_hash != mesh.mesh_lineage_hash
        or skin.deform_structure_hash != rig.deform_structure_hash
    ):
        raise ValueError("RSS_LIVE_GRAPHICS_MESH_RIG_SKIN_LINEAGE_DRIFT")
    if (
        not 1 <= len(frames) <= 512
        or len(frame_times_seconds) != len(frames)
        or len(mesh.faces) > 4096
        or len(mesh.vertices) > 65536
    ):
        raise ValueError("RSS_LIVE_GRAPHICS_FRAME_OR_FACE_BUDGET")
    times = tuple(float(t) for t in frame_times_seconds)
    if (
        not all(math.isfinite(t) and t >= 0 for t in times)
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValueError("RSS_LIVE_GRAPHICS_FRAME_TIME_INVALID")
    is_multi = not isinstance(cameras, Mapping)
    if is_multi:
        if not isinstance(cameras, (tuple, list)) or len(cameras) != 8:
            raise ValueError("RSS_LIVE_GRAPHICS_CAMERA_CARDINALITY")
        views = tuple(
            qualify_camera_v3(c, view_id=f"V{i}", view_index=i)
            for i, c in enumerate(cameras)
        )
    else:
        views = (qualify_camera_v3(cameras, view_id="V0", view_index=0),)
    resolution = views[0].resolution
    if resolution > 2048 or any(v.resolution != resolution for v in views):
        raise ValueError("RSS_LIVE_GRAPHICS_CAMERA_RESOLUTION_DRIFT")
    canonical_ids = tuple(v.canonical_mesh_vertex_id for v in mesh.vertices)
    indexing = {v: i for i, v in enumerate(canonical_ids)}
    if len(indexing) != len(canonical_ids):
        raise ValueError("RSS_LIVE_GRAPHICS_DUPLICATE_VERTEX_ID")
    faces = tuple(tuple(indexing[v] for v in tri) for tri in mesh.faces)
    nviews = len(views)
    expected = (36 if is_multi else 32) + len(faces) * 12 + (
        len(frames) * nviews * len(canonical_ids) * 24
    )
    if expected > 64 * 1024 * 1024:
        raise ValueError("RSS_LIVE_GRAPHICS_NATIVE_POSE_BUDGET")
    out = bytearray(b"RSPOSE02" if is_multi else b"RSPOSE01")
    if is_multi:
        out.extend(struct.pack(
            "<IIIIIII", 2, len(canonical_ids), len(faces),
            len(frames), nviews, resolution, resolution,
        ))
    else:
        out.extend(struct.pack(
            "<IIIIII", 1, len(canonical_ids), len(faces),
            len(frames), resolution, resolution,
        ))
    for face in faces:
        out.extend(struct.pack("<III", *face))
    evidence = []
    revision = None
    for index, (frame, time_seconds) in enumerate(zip(frames, times)):
        if not isinstance(frame, PuppetRuntimeFrameV1):
            raise ValueError("RSS_LIVE_GRAPHICS_FRAME_TYPE")
        deform = frame.deformed
        pose = frame.pose
        if (
            tuple(deform.canonical_mesh_vertex_ids) != canonical_ids
            or deform.mesh_lineage_hash != mesh.mesh_lineage_hash
            or deform.skin_lineage_hash != skin.skin_lineage_hash
            or deform.rig_lineage_hash != rig.rig_lineage_hash
            or pose.rig_lineage_hash != rig.rig_lineage_hash
            or pose.control_program_hash != rig.control_program_hash
            or not frame.product_revision_id
            or (revision is not None and frame.product_revision_id != revision)
        ):
            raise ValueError("RSS_LIVE_GRAPHICS_FRAME_PARENT_OR_VERTEX_DRIFT")
        if revision is None:
            revision = frame.product_revision_id
        xyz = np.asarray(deform.positions, dtype=np.float64)
        if xyz.shape != (len(canonical_ids), 3) or not np.isfinite(xyz).all():
            raise ValueError("RSS_LIVE_GRAPHICS_FRAME_SURFACE_INVALID")
        if frame.xpbd_surface_state is None:
            # Non-surface frames are fully replayable from pose + primary skin.
            replay = deform_mesh_lbs_v1(mesh, rig, skin, pose)
            if not np.allclose(
                xyz, np.asarray(replay.positions, dtype=np.float64),
                rtol=0., atol=1e-9,
            ):
                raise ValueError("RSS_LIVE_GRAPHICS_LBS_REPLAY_DRIFT")
        for camera in views:
            projected = project_points_xyz_v3(xyz, camera)
            if (
                projected.shape != xyz.shape
                or not np.isfinite(projected).all()
                or np.any(projected[:, 2] <= 0)
            ):
                raise ValueError("RSS_LIVE_GRAPHICS_PROJECTED_FRAME_INVALID")
            out.extend(projected.astype("<f8", copy=False).tobytes(order="C"))
        evidence.append({
            "frame_index": index,
            "time_seconds": time_seconds,
            "final_world_vertex_positions_sha256":
                sha256(xyz.astype("<f8", copy=False).tobytes()).hexdigest(),
            "xpbd_surface_state_hash": (
                frame.xpbd_surface_state.state_hash
                if frame.xpbd_surface_state is not None else None
            ),
            "secondary_state_present": frame.xpbd_secondary_state is not None,
            "pose_control_program_hash": pose.control_program_hash,
        })
    if len(out) != expected:
        raise RuntimeError("RSS_LIVE_GRAPHICS_NATIVE_POSE_LAYOUT_DRIFT")
    payload = bytes(out)
    camera_receipt_sha256 = sha256(json.dumps(
        [dict(c) for c in cameras] if is_multi else dict(cameras),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    receipt = {
        "schema": SCHEMA,
        "camera_policy_sha256": camera_receipt_sha256,
        "authority": AUTHORITY,
        "product_revision_id": revision,
        "mesh_lineage_hash": mesh.mesh_lineage_hash,
        "rig_lineage_hash": rig.rig_lineage_hash,
        "skin_lineage_hash": skin.skin_lineage_hash,
        "native_pose_bytes_sha256": sha256(payload).hexdigest(),
        "native_pose_bytes_count": len(payload),
        "frame_count": len(frames),
        "view_count": nviews,
        "resolution": resolution,
        "source_color_or_visibility_qualified": False,
        "independent_xpbd_behavior_proof": False,
        "product_release_admission": False,
        "frames": evidence,
    }
    return payload, receipt



def bind_rss_vnext_live_pose_to_native_preview_v1(
    *,
    template_rss_path: Path,
    live_pose_bytes: bytes,
    live_pose_receipt: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Create a raw-mode native renderer input from a reopened template.

    The authored .rss remains untouched; reopening it MUST still reproduce
    its LBS controls. The live frame stream substitutes only the ephemeral
    pose.bin passed to the C++ player's raw input mode:
      player graphics.bin texture.bin pose.bin frame view source_view out.rgba

    Source visibility/style and XPBD behavior remain unqualified. The template
    is never silently rewritten as an independently replayable product.
    """
    state = reopen_rss_vnext_native_preview_package_v1(template_rss_path)
    m = state["manifest"]
    r = live_pose_receipt
    if (
        not isinstance(live_pose_bytes, bytes)
        or r.get("schema") != SCHEMA
        or r.get("authority") != AUTHORITY
        or r.get("product_release_admission") is not False
        or r.get("source_color_or_visibility_qualified") is not False
        or r.get("mesh_lineage_hash") != m["canonical_mesh_lineage_hash"]
        or r.get("rig_lineage_hash") != m["canonical_rig_lineage_hash"]
        or r.get("skin_lineage_hash") != m["canonical_skin_lineage_hash"]
        or r.get("native_pose_bytes_sha256")
            != sha256(live_pose_bytes).hexdigest()
        or r.get("native_pose_bytes_count") != len(live_pose_bytes)
        or r.get("view_count") != int(m["native_view_count"])
        or not isinstance(r.get("resolution"),int)
        or not 16 <= r["resolution"] <= 2048
        or not isinstance(r.get("frame_count"),int)
        or not 1 <= r["frame_count"] <= 512
    ):
        raise ValueError("RSS_LIVE_GRAPHICS_TEMPLATE_OR_RECEIPT_DRIFT")
    cameras = state["virtual_camera"]
    expected_camera_sha = sha256(json.dumps(
        [dict(c) for c in cameras]
        if isinstance(cameras,(tuple,list)) else dict(cameras),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    if r.get("camera_policy_sha256") != expected_camera_sha:
        raise ValueError("RSS_LIVE_GRAPHICS_TEMPLATE_CAMERA_POLICY_DRIFT")
    magic = b"RSPOSE02" if r["view_count"] == 8 else b"RSPOSE01"
    if not live_pose_bytes.startswith(magic):
        raise ValueError("RSS_LIVE_GRAPHICS_NATIVE_POSE_SCHEMA_DRIFT")
    # Native player is responsible for its own header/count validation.
    # Bind to the immutable, fully reopened Graphics+UV payload.
    assets = {
        "graphics.bin":state["graphics_bytes"],
        "texture.bin":state["texture_bytes"],
        "pose.bin":live_pose_bytes,
    }
    transport_receipt = {
        "schema":"RealSaS.RSSvNext.LiveNativeGraphicsInput.v1",
        "authority":AUTHORITY,
        "template_schema":m["schema"],
        "template_pose_remains_authoring_lbs":True,
        "native_source_color_qualified":False,
        "product_release_admission":False,
        "graphics_runtime_hash":m["graphics_runtime_hash"],
        "pose_sha256":sha256(live_pose_bytes).hexdigest(),
        "texture_sha256":sha256(assets["texture.bin"]).hexdigest(),
        "graphics_sha256":sha256(assets["graphics.bin"]).hexdigest(),
    }
    return assets, transport_receipt

__all__ = [
    "SCHEMA", "AUTHORITY",
    "encode_rss_vnext_live_mechanics_native_pose_v1",
    "bind_rss_vnext_live_pose_to_native_preview_v1",
]
