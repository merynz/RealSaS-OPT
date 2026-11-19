from __future__ import annotations

"""Render a canonical user edit with face-preserving, source-UNQUALIFIED UV.

The base source .rss is immutable. .rsedit owns the new canonical Mesh/Rig/Skin
and Mechanics pose. This adapter transports the old indexed UV candidate and
Graphics fragment addresses only because the edit preserves canonical face
topology. It removes source RGB taps rather than letting an edited projection
claim original pixel authority. Rebinding is not new source truth, a teacher,
or a ProductRevision.
"""

from collections import OrderedDict
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import struct

from compiler.realsas_compiler_core.graphics_runtime_v1 import (
    RuntimeGraphicsV1, RuntimeGraphicsViewV1,
    runtime_graphics_hash_v1, validate_runtime_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_graphics_binary_v1 import (
    encode_runtime_graphics_v1,
)
from compiler.realsas_compiler_core.runtime_package_v2 import (
    _entry_bytes, read_rss_v2,
)
from .rss_vnext_canonical_authoring_revision_v1 import (
    reopen_rss_vnext_canonical_user_edit_revision_v1,
)
from .rss_vnext_native_preview_package_v1 import (
    SCHEMA_EIGHT, _make_pose_binary,
    reopen_rss_vnext_native_preview_package_v1,
    select_rss_vnext_eight_way_virtual_cameras_v1,
)

SCHEMA = "RealSaS.RSSvNext.UserEditedGraphicsPreviewRebind.v1"
AUTHORITY = "USER_EDITED_FACE_UV_CANDIDATE_NOT_SOURCE_APPEARANCE"
_MAX_BYTES = 160 * 1024 * 1024


def _sha(blob: bytes) -> str:
    return sha256(blob).hexdigest()


def _json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, size: int) -> bytes:
        if size < 0 or size > len(self.data)-self.pos:
            raise RuntimeError("RSS_EDIT_GRAPHICS_BINARY_TRUNCATED")
        result = self.data[self.pos:self.pos+size]
        self.pos += size
        return result

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def f64(self) -> float:
        value = struct.unpack("<d", self.take(8))[0]
        if not math.isfinite(value):
            raise RuntimeError("RSS_EDIT_GRAPHICS_NONFINITE")
        return value

    def text(self) -> str:
        count = self.u32()
        if not 1 <= count <= 4096:
            raise RuntimeError("RSS_EDIT_GRAPHICS_TEXT_BUDGET")
        try:
            return self.take(count).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("RSS_EDIT_GRAPHICS_TEXT_INVALID") from exc


def _rebind_graphics(
    original: bytes, *, old_mesh_hash: str, new_mesh_hash: str,
    original_runtime_hash: str, face_count: int,
    revision_id: str,
) -> tuple[bytes, str]:
    """Decode actual RSGFX001, preserve indexed ownership, demote all views."""
    reader = _Reader(original)
    if reader.take(8) != b"RSGFX001" or reader.u32() != 1:
        raise RuntimeError("RSS_EDIT_GRAPHICS_ABI_INVALID")
    faces, fragments, views, attachments = (
        reader.u32(), reader.u32(), reader.u32(), reader.u32(),
    )
    if (
        faces != face_count or not 1 <= fragments <= faces
        or views != 8 or attachments > faces * 8
        or faces > 4096
    ):
        raise RuntimeError("RSS_EDIT_GRAPHICS_DOMAIN_INVALID")
    old_mesh = reader.text()
    compiled = reader.text()
    old_runtime = reader.text()
    if old_mesh != old_mesh_hash or old_runtime != original_runtime_hash:
        raise RuntimeError("RSS_EDIT_GRAPHICS_ORIGINAL_PARENT_DRIFT")

    face_owners = tuple(reader.u32() for _ in range(faces))
    fragment_ids = []
    supports = []
    for _ in range(fragments):
        fragment_ids.append(reader.text())
        supports.append(reader.u32())
    for index in range(views):
        if reader.u32() != index:
            raise RuntimeError("RSS_EDIT_GRAPHICS_VIEW_ADDRESS_DRIFT")
        visible_count = reader.u32()
        if visible_count > fragments:
            raise RuntimeError("RSS_EDIT_GRAPHICS_VISIBLE_COUNT")
        visible = [reader.u32() for _ in range(visible_count)]
        order_count = reader.u32()
        if order_count != fragments:
            raise RuntimeError("RSS_EDIT_GRAPHICS_ORDER_COUNT")
        order = [reader.u32() for _ in range(order_count)]
        edge_count = reader.u32()
        if edge_count > fragments * fragments:
            raise RuntimeError("RSS_EDIT_GRAPHICS_EDGE_COUNT")
        edges = [(reader.u32(), reader.u32()) for _ in range(edge_count)]
        if (len(set(visible)) != len(visible)
            or any(x >= fragments for x in visible)
            or set(order) != set(range(fragments))
            or any(a >= fragments or b >= fragments or a == b for a,b in edges)):
            raise RuntimeError("RSS_EDIT_GRAPHICS_VIEW_CONTENT")
    rows = []
    for _ in range(attachments):
        fragment = reader.u32()
        parent = reader.text()
        face = reader.u32()
        bary = tuple(reader.f64() for _ in range(3))
        confidence = reader.f64()
        rows.append({
            "fragment_index": fragment,
            "parent_mechanical_domain_id": parent,
            "anchor_face_index": face,
            "barycentric_anchor": bary,
            "confidence": confidence,
            "evidence_refs": (),
        })
    if reader.pos != len(original):
        raise RuntimeError("RSS_EDIT_GRAPHICS_TRAILING")
    # The input is bound to a fully reopened original RSS and its SHA-pinned
    # graphics.bin. Nothing from its original source-view selection is reused.
    universal = tuple(range(fragments))
    graph = RuntimeGraphicsV1(
        canonical_mesh_binding_hash=new_mesh_hash,
        compiled_graphics_binding_hash=compiled,
        face_fragment_index=face_owners,
        fragment_ids=tuple(fragment_ids),
        fragment_support_band_pixels=tuple(supports),
        views=tuple(
            RuntimeGraphicsViewV1(index, universal, universal, ())
            for index in range(8)
        ),
        attachments=tuple(rows),
        runtime_graphics_hash="",
        metadata={
            "source_visibility_coverage_complete": False,
            "pose_conditioned_order_execution": False,
            "source_visibility_requires_independent_ray_receipt": True,
            "user_edited_graphics_preview_schema": SCHEMA,
            "authority": AUTHORITY,
            "authoring_revision_id": revision_id,
            "original_runtime_graphics_hash": original_runtime_hash,
            "original_graphics_bytes_sha256": _sha(original),
            "original_mesh_lineage_hash": old_mesh_hash,
            "virtual_preview_source_view_visibility_reused": False,
            "virtual_preview_draw_policy":
                "ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY",
        },
    )
    graph = replace(graph, runtime_graphics_hash=runtime_graphics_hash_v1(graph))
    validate_runtime_graphics_v1(
        graph, canonical_mesh_hash=new_mesh_hash, face_count=faces,
    )
    return encode_runtime_graphics_v1(graph), graph.runtime_graphics_hash


def _rebind_uv_candidate(
    original: bytes, *, old_mesh_hash: str, new_mesh_hash: str,
    face_count: int,
) -> bytes:
    """Retain face-local UV and inferred H(x); erase both SOURCE color layers."""
    if len(original) < 232 or original[:8] not in (b"RSUVT001", b"RSUVT002"):
        raise RuntimeError("RSS_EDIT_UV_ABI_INVALID")
    version,width,height,faces,channels = struct.unpack_from("<IIIII", original, 8)
    if (
        version != (2 if original[:8] == b"RSUVT002" else 1)
        or channels != 3 or faces != face_count or faces > 4096
        or not 1 <= width <= 2048 or not 1 <= height <= 2048
        or len(new_mesh_hash) != 64
        or struct.unpack_from("<I",original,28)[0] != 64
        or original[32:96].decode("ascii") != old_mesh_hash
        or struct.unpack_from("<I",original,96)[0] != 64
        or struct.unpack_from("<I",original,164)[0] != 64
    ):
        raise RuntimeError("RSS_EDIT_UV_PARENT_OR_DOMAIN")
    pixels = width * height
    source_rgba_start = 232 + faces * 3 * 2 * 4
    source_rgba_end = source_rgba_start + 2 * pixels * 4
    source_mask_end = source_rgba_end + 2 * pixels
    inferred_rgba_end = source_mask_end + pixels * 4
    inferred_mask_end = inferred_rgba_end + pixels
    conflict_end = inferred_mask_end + pixels
    minimum_end = conflict_end + 2 * pixels
    if minimum_end > len(original):
        raise RuntimeError("RSS_EDIT_UV_BINARY_TRUNCATED")
    rebound = bytearray(original)
    rebound[32:96] = new_mesh_hash.encode("ascii")
    rebound[source_rgba_start:source_mask_end] = bytes(
        source_mask_end-source_rgba_start
    )
    rebound[inferred_mask_end:conflict_end] = bytes(pixels)
    # A source-view hint now produces only authority=2 inferred color,
    # never authority=1 observed source color on moved geometry.
    return bytes(rebound)


def compile_rss_vnext_edited_graphics_preview_v1(
    *, source_rss_path: Path, revision_path: Path,
) -> tuple[bytes, dict]:
    edit = reopen_rss_vnext_canonical_user_edit_revision_v1(
        revision_path=revision_path, source_rss_path=source_rss_path,
    )
    source = edit["source_preview"]
    original = read_rss_v2(source_rss_path)
    manifest = dict(source["manifest"])
    mesh, rig, skin = edit["mesh"], edit["rig"], edit["skin"]
    graphics, graphics_hash = _rebind_graphics(
        original["graphics.bin"],
        old_mesh_hash=source["mesh"].mesh_lineage_hash,
        new_mesh_hash=mesh.mesh_lineage_hash,
        original_runtime_hash=manifest["graphics_runtime_hash"],
        face_count=len(mesh.faces),
        revision_id=edit["authoring_revision_id"],
    )
    texture = _rebind_uv_candidate(
        original["texture.bin"],
        old_mesh_hash=source["mesh"].mesh_lineage_hash,
        new_mesh_hash=mesh.mesh_lineage_hash,
        face_count=len(mesh.faces),
    )
    controls = json.loads(original["controls.json"])
    eight_way = manifest["schema"] == SCHEMA_EIGHT
    cameras = source["virtual_camera"]
    if eight_way:
        cameras = select_rss_vnext_eight_way_virtual_cameras_v1(
            mesh, resolution=cameras[0]["resolution"],
        )
        controls["virtual_cameras"] = list(cameras)
    pose, pose_hashes = _make_pose_binary(
        mesh,rig,skin,source["control_frames"],cameras,
    )
    controls["mechanics_pose_hashes"] = pose_hashes
    controls["product_revision_minted"] = False
    controls["virtual_camera_is_original_source_observation"] = False
    changes = {
        "mesh.json":_json(mesh.to_dict()),
        "rig.json":_json(rig.to_dict()),
        "skin.json":_json(skin.to_dict()),
        "graphics.bin":graphics,
        "texture.bin":texture,
        "pose.bin":pose,
        "controls.json":_json(controls),
    }
    manifest.update({
        "canonical_mesh_lineage_hash":mesh.mesh_lineage_hash,
        "canonical_rig_lineage_hash":rig.rig_lineage_hash,
        "canonical_skin_lineage_hash":skin.skin_lineage_hash,
        "graphics_runtime_hash":graphics_hash,
        "source_rgb_surface_ownership_qualified":"0",
        "pose_conditioned_visibility_source_qualified":"0",
        "original_source_camera_observed":"0",
        "product_revision_minted":"0",
        "virtual_preview_camera_registered_to_graphics_view":"0",
        "graphics_uv_binding_qualified":"0",
        "user_edited_graphics_preview_schema":SCHEMA,
        "user_edited_graphics_authority":AUTHORITY,
        "user_authoring_revision_id":edit["authoring_revision_id"],
        "user_authoring_source_rss_sha256":_sha(source_rss_path.read_bytes()),
        "original_graphics_bytes_sha256":_sha(original["graphics.bin"]),
        "original_texture_bytes_sha256":_sha(original["texture.bin"]),
        "user_edited_source_color_layers_erased":"1",
        "user_edited_visibility_original_views_discarded":"1",
        "source_uv_requalification_pass":"0",
        "product_release_admission":"0",
        "virtual_preview_draw_policy":"ALL_CANONICAL_FRAGMENTS_POSED_DEPTH_ONLY",
        "texture_layer_draw_order_authority":
            "POSED_DEPTH_ONLY_FOR_UNREGISTERED_VIRTUAL_CAMERA",
    })
    for name,body in changes.items():
        manifest["sha256:"+name] = _sha(body)
    entries = OrderedDict(
        (key, changes[key]) if key in changes else (key, value)
        for key,value in original.items() if key != "manifest.txt"
    )
    entries["manifest.txt"] = (
        "\n".join(f"{key}={val}" for key,val in sorted(manifest.items()))+"\n"
    ).encode("utf-8")
    packed = _entry_bytes(entries)
    if len(packed)>_MAX_BYTES:
        raise RuntimeError("RSS_EDIT_GRAPHICS_PREVIEW_ARCHIVE_BUDGET")
    receipt = {
        "schema":SCHEMA,
        "authority":AUTHORITY,
        "source_rss_sha256":_sha(source_rss_path.read_bytes()),
        "authoring_revision_id":edit["authoring_revision_id"],
        "source_graphics_sha256":_sha(original["graphics.bin"]),
        "source_uv_sha256":_sha(original["texture.bin"]),
        "edited_graphics_sha256":_sha(graphics),
        "edited_uv_sha256":_sha(texture),
        "edited_pose_sha256":_sha(pose),
        "edited_preview_sha256":_sha(packed),
        "face_topology_preserved":True,
        "source_rgb_layers_removed":True,
        "source_color_or_visibility_qualified":False,
        "original_source_camera_registered":False,
        "product_release_admission":False,
    }
    return packed,receipt


def reopen_rss_vnext_edited_graphics_preview_v1(
    *, preview_path: Path, source_rss_path: Path, revision_path: Path,
) -> dict:
    if preview_path.stat().st_size > _MAX_BYTES:
        raise RuntimeError("RSS_EDIT_GRAPHICS_PREVIEW_BUDGET")
    actual = preview_path.read_bytes()
    expected,receipt = compile_rss_vnext_edited_graphics_preview_v1(
        source_rss_path=source_rss_path, revision_path=revision_path,
    )
    if actual != expected:
        raise RuntimeError("RSS_EDIT_GRAPHICS_PREVIEW_REPLAY_DRIFT")
    state = reopen_rss_vnext_native_preview_package_v1(preview_path)
    if (
        state["manifest"].get("user_edited_graphics_authority") != AUTHORITY
        or state["manifest"].get("source_uv_requalification_pass") != "0"
        or state["product_revision_minted"] is not False
    ):
        raise RuntimeError("RSS_EDIT_GRAPHICS_PREVIEW_FALSE_AUTHORITY")
    return {"preview":state,"receipt":receipt}


__all__ = [
    "SCHEMA","AUTHORITY",
    "compile_rss_vnext_edited_graphics_preview_v1",
    "reopen_rss_vnext_edited_graphics_preview_v1",
]
