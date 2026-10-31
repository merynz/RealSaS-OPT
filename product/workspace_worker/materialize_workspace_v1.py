from __future__ import annotations

"""Deterministic post-seal ProductRevision -> Studio compatibility projection.

This module creates no product authority. Canonical Mesh/Rig/Skin/Mechanics/
Graphics/Motion and PASS Proof remain the authority. SourceObservation is a
non-authoritative immutable workspace support input used only for camera/pixel
materialization. The emitted CanonicalPuppetGraph.v3 is explicitly a derived
compatibility view for the current Studio/Living Compile reader.
"""

import argparse
import binascii
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping
import zipfile
import zlib

import numpy as np

from compiler.realsas_compiler_core.playback_full_surface_v3 import (
    project_points_xyz_v3,
    qualify_camera_v3,
)
from compiler.realsas_compiler_services.platform_worker.source_observation_contract_v1 import (
    decode_source_observation_arrays_v1,
)


REQUEST_SCHEMA = "RealSaS.ProductWorkspaceMaterializationRequest.v1"
BUNDLE_SCHEMA = "RealSaS.ProductWorkspaceBundleZip.v1"
PROJECTION_SCHEMA = "RealSaS.ModularProductStudioProjection.v1"

AUTHORITY_TYPES = {
    "mesh:canonical": ("RealSaS.CanonicalMeshIR", "v1", "RealSaS.CanonicalMeshIR.v1"),
    "rig:canonical": ("RealSaS.CanonicalRigIR", "v2", "RealSaS.CanonicalRigIR.v2"),
    "skin:canonical": ("RealSaS.CanonicalSkinIR", "v2", "RealSaS.CanonicalSkinIR.v2"),
    "mechanics:canonical": ("RealSaS.CanonicalMechanicsIR", "v1", "RealSaS.CanonicalMechanicsIR.v1"),
    "graphics:canonical": ("RealSaS.CanonicalGraphicsIR", "v1", "RealSaS.CanonicalGraphicsIR.v1"),
    "motion:canonical": ("RealSaS.CanonicalMotionIR", "v2", "RealSaS.CanonicalMotionIR.v2"),
    "proof:product_pass": ("RealSaS.ProductProofBundleIR", "v1", "RealSaS.ProductProofBundleIR.v1"),
}
SUPPORT_TYPES = {
    "support:source_observation": ("RealSaS.SourceObservation", "v1"),
}


class WorkspaceMaterializationError(RuntimeError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise WorkspaceMaterializationError(f"WORKSPACE_JSON_INVALID:{path.name}") from exc
    if not isinstance(value, dict):
        raise WorkspaceMaterializationError(f"WORKSPACE_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _schema(value: Mapping[str, Any]) -> str:
    return str(value.get("schema_version") or value.get("schema") or "")


def _by_role(rows: Any, *, scope: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise WorkspaceMaterializationError(f"WORKSPACE_{scope}_ARTIFACT_LIST_REQUIRED")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise WorkspaceMaterializationError(f"WORKSPACE_{scope}_ARTIFACT_ROW_INVALID")
        role = str(row.get("role") or "")
        if not role or role in out:
            raise WorkspaceMaterializationError(f"WORKSPACE_{scope}_ROLE_DUPLICATE_OR_EMPTY")
        if str(row.get("scope") or "").upper() != scope:
            raise WorkspaceMaterializationError(f"WORKSPACE_{scope}_SCOPE_DRIFT:{role}")
        path = Path(str(row.get("path") or ""))
        if not path.is_file():
            raise WorkspaceMaterializationError(f"WORKSPACE_ARTIFACT_FILE_MISSING:{role}")
        out[role] = row
    return out


def _validate_artifact_contracts(
    authority: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(authority) != set(AUTHORITY_TYPES):
        raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_ROLE_SET_MISMATCH")
    if set(support) != set(SUPPORT_TYPES):
        raise WorkspaceMaterializationError("WORKSPACE_SUPPORT_ROLE_SET_MISMATCH")
    parsed: dict[str, dict[str, Any]] = {}
    for role, expected in AUTHORITY_TYPES.items():
        row = authority[role]
        if str(row.get("artifact_type") or "") != expected[0] or str(row.get("schema_version") or "") != expected[1]:
            raise WorkspaceMaterializationError(f"WORKSPACE_ARTIFACT_CONTRACT_DRIFT:{role}")
        value = _load_json(Path(str(row["path"])))
        if _schema(value) != expected[2]:
            raise WorkspaceMaterializationError(f"WORKSPACE_PAYLOAD_SCHEMA_DRIFT:{role}")
        parsed[role] = value
    for role, expected in SUPPORT_TYPES.items():
        row = support[role]
        if str(row.get("artifact_type") or "") != expected[0] or str(row.get("schema_version") or "") != expected[1]:
            raise WorkspaceMaterializationError(f"WORKSPACE_SUPPORT_CONTRACT_DRIFT:{role}")
    return parsed


def _validate_lineage(parsed: Mapping[str, Mapping[str, Any]]) -> None:
    mesh = parsed["mesh:canonical"]
    rig = parsed["rig:canonical"]
    skin = parsed["skin:canonical"]
    mechanics = parsed["mechanics:canonical"]
    graphics = parsed["graphics:canonical"]
    motion = parsed["motion:canonical"]
    proof = parsed["proof:product_pass"]

    mesh_hash = str(mesh.get("mesh_lineage_hash") or "")
    rig_hash = str(rig.get("rig_lineage_hash") or "")
    skin_hash = str(skin.get("skin_lineage_hash") or "")
    mechanics_hash = str(mechanics.get("mechanics_lineage_hash") or "")
    control_hash = str(rig.get("control_program_hash") or "")
    deform_hash = str(rig.get("deform_structure_hash") or "")
    if not all(len(x) >= 16 for x in (mesh_hash, rig_hash, skin_hash, mechanics_hash, control_hash, deform_hash)):
        raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_LINEAGE_REQUIRED")
    if str(rig.get("mesh_lineage_hash") or "") != mesh_hash:
        raise WorkspaceMaterializationError("WORKSPACE_RIG_MESH_LINEAGE_DRIFT")
    if str(skin.get("mesh_lineage_hash") or "") != mesh_hash or str(skin.get("deform_structure_hash") or "") != deform_hash:
        raise WorkspaceMaterializationError("WORKSPACE_SKIN_PARENT_LINEAGE_DRIFT")
    if str(mechanics.get("mesh_lineage_hash") or "") != mesh_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MECHANICS_MESH_LINEAGE_DRIFT")
    if str(mechanics.get("rig_lineage_hash") or "") != rig_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MECHANICS_RIG_LINEAGE_DRIFT")
    if str(mechanics.get("skin_lineage_hash") or "") != skin_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MECHANICS_SKIN_LINEAGE_DRIFT")
    if str(graphics.get("mesh_lineage_hash") or "") != mesh_hash:
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_MESH_LINEAGE_DRIFT")
    if str(graphics.get("skin_lineage_hash") or "") != skin_hash:
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_SKIN_LINEAGE_DRIFT")
    if str(graphics.get("mechanics_lineage_hash") or "") != mechanics_hash:
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_MECHANICS_LINEAGE_DRIFT")
    if str(motion.get("rig_lineage_hash") or "") != rig_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MOTION_RIG_LINEAGE_DRIFT")
    if str(motion.get("control_program_hash") or "") != control_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MOTION_CONTROL_PROGRAM_DRIFT")
    if str(motion.get("mechanics_lineage_hash") or "") != mechanics_hash:
        raise WorkspaceMaterializationError("WORKSPACE_MOTION_MECHANICS_LINEAGE_DRIFT")
    if str(proof.get("overall_status") or "") != "PASS":
        raise WorkspaceMaterializationError("WORKSPACE_PRODUCT_PROOF_NOT_PASS")
    source_hash = str(proof.get("source_product_state_hash") or "")
    if len(source_hash) != 64:
        raise WorkspaceMaterializationError("WORKSPACE_PRODUCT_STATE_HASH_INVALID")


def _camera_rows(camera_bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = camera_bundle.get("cameras")
    if not isinstance(rows, list) or len(rows) != 8:
        raise WorkspaceMaterializationError("WORKSPACE_CAMERA_SET_INVALID")
    ordered = sorted(rows, key=lambda row: int(row.get("view_index", -1)))
    if [int(row.get("view_index", -1)) for row in ordered] != list(range(8)):
        raise WorkspaceMaterializationError("WORKSPACE_CAMERA_VIEW_SET_INVALID")
    return ordered


def _project(P: Iterable[float], cameras: list[Mapping[str, Any]]) -> list[list[Any]]:
    point = np.asarray([list(map(float, P))], dtype=np.float64)
    if point.shape != (1, 3) or not np.isfinite(point).all():
        raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_POINT_INVALID")
    rows: list[list[Any]] = []
    for index, raw in enumerate(cameras):
        camera = qualify_camera_v3(raw, view_id=f"V{index}", view_index=index)
        xyz = project_points_xyz_v3(point, camera)[0]
        rows.append([index, [float(xyz[0]), float(xyz[1])]])
    return rows


def _surface_node(surface_id: str, P: Iterable[float], cameras: list[Mapping[str, Any]], *, kind: str) -> dict[str, Any]:
    return {
        "surface_id": surface_id,
        "P": list(map(float, P)),
        "raster_bindings": _project(P, cameras),
        "metadata": {
            "workspace_projection_kind": kind,
            "derived_compatibility_cache": True,
            "canonical_geometry_mutated": False,
        },
    }


def _topological_back_to_front(visible: list[str], edges: list[Mapping[str, Any]]) -> list[str]:
    nodes = list(dict.fromkeys(map(str, visible)))
    if len(nodes) != len(visible):
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_VISIBLE_FRAGMENT_DUPLICATE")
    node_set = set(nodes)
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in edges:
        front = str(edge.get("front_fragment_id") or "")
        back = str(edge.get("back_fragment_id") or "")
        if front not in node_set or back not in node_set or front == back:
            continue
        if front not in outgoing[back]:
            outgoing[back].add(front)
            indegree[front] += 1
    ready = sorted(node for node in nodes if indegree[node] == 0)
    ordered: list[str] = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(nodes):
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_DRAW_ORDER_CYCLE")
    return ordered


def _graphics_fragments(mesh: Mapping[str, Any], graphics: Mapping[str, Any]) -> tuple[dict[str, list[int]], dict[int, list[str]]]:
    presentation = graphics.get("presentation") or {}
    fragments = presentation.get("fragments") or []
    faces = list(mesh.get("faces") or [])
    by_fragment: dict[str, list[int]] = {}
    covered: set[int] = set()
    for row in fragments:
        fragment_id = str(row.get("fragment_id") or "")
        indices = [int(value) for value in row.get("face_indices") or []]
        if not fragment_id or fragment_id in by_fragment:
            raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_FRAGMENT_ID_INVALID")
        if any(index < 0 or index >= len(faces) for index in indices):
            raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_FRAGMENT_FACE_INVALID")
        by_fragment[fragment_id] = indices
        covered.update(indices)
    if not by_fragment or covered != set(range(len(faces))):
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_FRAGMENT_COVERAGE_DRIFT")

    order_by_view: dict[int, list[str]] = {}
    views = presentation.get("views") or []
    for row in views:
        view = int(row.get("view_index", -1))
        if view < 0 or view > 7 or view in order_by_view:
            raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_VIEW_INVALID")
        visible = list(map(str, row.get("visible_fragment_ids") or []))
        unknown = set(visible) - set(by_fragment)
        if unknown:
            raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_VISIBLE_FRAGMENT_UNKNOWN")
        order_by_view[view] = _topological_back_to_front(visible, list(row.get("order_edges") or []))
    if set(order_by_view) != set(range(8)):
        raise WorkspaceMaterializationError("WORKSPACE_GRAPHICS_VIEW_COVERAGE_DRIFT")
    return by_fragment, order_by_view


def _component_payload(
    fragment_id: str,
    face_indices: list[int],
    mesh: Mapping[str, Any],
    skin: Mapping[str, Any],
) -> dict[str, Any]:
    faces = list(mesh.get("faces") or [])
    selected_faces = [list(map(str, faces[index])) for index in face_indices]
    vertex_ids = {vertex_id for face in selected_faces for vertex_id in face}
    vertices_by_id = {str(row.get("canonical_mesh_vertex_id") or ""): row for row in mesh.get("vertices") or []}
    skin_by_id = {str(row.get("canonical_mesh_vertex_id") or ""): row for row in skin.get("rows") or []}
    if any(vertex_id not in vertices_by_id or vertex_id not in skin_by_id for vertex_id in vertex_ids):
        raise WorkspaceMaterializationError("WORKSPACE_FRAGMENT_VERTEX_OR_SKIN_MISSING")
    vertices = []
    rows = []
    for vertex_id in sorted(vertex_ids):
        source = vertices_by_id[vertex_id]
        vertices.append({
            "canonical_mesh_vertex_id": vertex_id,
            "P": list(map(float, source.get("P") or ())),
            "support_binding": {
                "mode": "WORKSPACE_CANONICAL_VERTEX_IDENTITY",
                "coefficients": [[vertex_id, 1.0]],
                "metadata": {"derived_compatibility_projection": True},
            },
            "metadata": {
                "source_component_id": str(source.get("component_id") or ""),
                "canonical_support_binding_preserved_in_authority_artifact": True,
            },
        })
        skin_row = skin_by_id[vertex_id]
        rows.append({
            "canonical_mesh_vertex_id": vertex_id,
            "influences": [[str(joint), float(weight)] for joint, weight in skin_row.get("influences") or []],
        })
    core = {
        "component_id": fragment_id,
        "mesh": {"vertices": vertices, "faces": selected_faces},
        "mesh_skin": {"rows": rows, "transfer_method": "CANONICAL_SKIN_IR_V2"},
        "metadata": {
            "graphics_fragment_id": fragment_id,
            "derived_from_canonical_graphics": True,
            "full_silhouette_substrate": len(face_indices) == len(faces),
        },
    }
    core["component_state_hash"] = _sha_bytes(_canonical_json_bytes(core))
    return core


def _compatibility_product(
    parsed: Mapping[str, Mapping[str, Any]],
    camera_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    mesh = parsed["mesh:canonical"]
    rig = parsed["rig:canonical"]
    skin = parsed["skin:canonical"]
    graphics = parsed["graphics:canonical"]
    motion = parsed["motion:canonical"]
    proof = parsed["proof:product_pass"]
    cameras = _camera_rows(camera_bundle)

    surface_nodes: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for vertex in mesh.get("vertices") or []:
        vertex_id = str(vertex.get("canonical_mesh_vertex_id") or "")
        if not vertex_id or vertex_id in used_ids:
            raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_VERTEX_ID_INVALID")
        used_ids.add(vertex_id)
        surface_nodes.append(_surface_node(vertex_id, vertex.get("P") or (), cameras, kind="CANONICAL_MESH_VERTEX"))
    for joint in rig.get("joints") or []:
        joint_id = str(joint.get("canonical_joint_id") or "")
        if not joint_id or joint_id in used_ids:
            raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_JOINT_ID_COLLISION")
        used_ids.add(joint_id)
        surface_nodes.append(_surface_node(joint_id, joint.get("P") or (), cameras, kind="CANONICAL_DEFORM_JOINT"))
    for control in rig.get("controls") or []:
        control_id = str(control.get("canonical_control_id") or "")
        if not control_id or control_id in used_ids:
            raise WorkspaceMaterializationError("WORKSPACE_CANONICAL_CONTROL_ID_COLLISION")
        used_ids.add(control_id)
        surface_nodes.append(_surface_node(control_id, control.get("rest_P") or (), cameras, kind="CANONICAL_ANIMATOR_CONTROL"))

    joints = [{
        "canonical_joint_id": str(row.get("canonical_joint_id") or ""),
        "position": list(map(float, row.get("P") or ())),
        "parent_canonical_id": row.get("parent_canonical_joint_id"),
        "support_surface_ids": [str(row.get("canonical_joint_id") or "")],
        "metadata": {
            "canonical_support_evidence_ids": list(map(str, row.get("support_evidence_ids") or [])),
            "derived_compatibility_projection": True,
        },
    } for row in rig.get("joints") or []]

    controls = [{
        "canonical_control_id": str(row.get("canonical_control_id") or ""),
        "control_kind": str(row.get("control_kind") or ""),
        "target_canonical_joint_ids": list(map(str, row.get("target_canonical_joint_ids") or [])),
        "parent_canonical_control_id": row.get("parent_canonical_control_id"),
        "rest_P": list(map(float, row.get("rest_P") or ())),
        "support_surface_ids": [str(row.get("canonical_control_id") or "")],
        "metadata": dict(row.get("metadata") or {}),
    } for row in rig.get("controls") or []]

    by_fragment, order_by_view = _graphics_fragments(mesh, graphics)
    component_cache = {
        fragment_id: _component_payload(fragment_id, face_indices, mesh, skin)
        for fragment_id, face_indices in by_fragment.items()
    }
    directions = []
    for view in range(8):
        directions.append({
            "view_index": view,
            "camera_binding_hash": _sha_bytes(_canonical_json_bytes(cameras[view])),
            "components": [component_cache[fragment_id] for fragment_id in order_by_view[view]],
            "graphics_draw_order_back_to_front": order_by_view[view],
            "metadata": {"canonical_graphics_presentation_authority_preserved": True},
        })

    motion_clip = {
        "clip_id": str(motion.get("canonical_clip_id") or ""),
        "clip_kind": str(motion.get("intent") or "custom"),
        "duration_sec": float(motion.get("duration_seconds") or 0.0),
        "loop": bool(motion.get("loop", False)),
        "control_tracks": list(motion.get("control_tracks") or []),
        "metadata": {
            "display_name": str(motion.get("display_name") or motion.get("canonical_clip_id") or "Clip"),
            "editable_motion_space": "ANIMATOR_CONTROLS",
            "direct_joint_track_authority": False,
            "canonical_motion_lineage_hash": str(motion.get("motion_lineage_hash") or ""),
        },
    }
    return {
        "schema_version": "RealSaS.CanonicalPuppetGraph.v3",
        "product_state_hash": str(proof["source_product_state_hash"]),
        "representation_class": "DIRECTIONAL_2D_2P5D_PUPPET",
        "full_3d_reconstruction_authority": False,
        "mechanical_state": {
            "surface": {
                "surface_nodes": surface_nodes,
                "metadata": {
                    "authority": "DERIVED_COMPATIBILITY_CACHE",
                    "canonical_geometry_mutated": False,
                    "projection_schema": PROJECTION_SCHEMA,
                },
            },
            "skeleton": {"joints": joints, "deform_root_joint_ids": list(rig.get("deform_root_joint_ids") or [])},
            "control_program": {
                "controls": controls,
                "chains": list(rig.get("chains") or []),
                "constraints": list(rig.get("constraints") or []),
                "attachment_anchors": list(rig.get("attachment_anchors") or []),
                "control_program_hash": str(rig.get("control_program_hash") or ""),
            },
        },
        "directional_renderables": {"directions": directions},
        "motion_state": {"clips": [motion_clip]},
        "metadata": {
            "authority": "DERIVED_POST_SEAL_WORKSPACE_PROJECTION",
            "projection_schema": PROJECTION_SCHEMA,
            "product_revision_authority_unchanged": True,
            "new_mesh_identity_minted": False,
            "new_rig_identity_minted": False,
            "new_appearance_authority_minted": False,
            "source_pixels_are_workspace_support": True,
        },
    }


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def _encode_rgba_png(chw: np.ndarray) -> bytes:
    value = np.asarray(chw, dtype=np.uint8)
    if value.shape[0] != 4 or value.ndim != 3:
        raise WorkspaceMaterializationError("WORKSPACE_SOURCE_RGBA_SHAPE_INVALID")
    rgba = np.moveaxis(value, 0, -1)
    height, width, channels = rgba.shape
    if channels != 4:
        raise WorkspaceMaterializationError("WORKSPACE_SOURCE_RGBA_CHANNEL_INVALID")
    scanlines = b"".join(b"\x00" + rgba[y].tobytes(order="C") for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(scanlines, 9)) + _png_chunk(b"IEND", b"")


def _zip_bytes(entries: Mapping[str, bytes]) -> bytes:
    from io import BytesIO
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name])
    return stream.getvalue()


def materialize(request_path: Path) -> Path:
    request = _load_json(request_path)
    if str(request.get("schema_version") or "") != REQUEST_SCHEMA:
        raise WorkspaceMaterializationError("WORKSPACE_REQUEST_SCHEMA_DRIFT")
    authority_flags = request.get("authority") or {}
    if authority_flags.get("source_product_revision_is_immutable") is not True:
        raise WorkspaceMaterializationError("WORKSPACE_IMMUTABLE_REVISION_REQUIRED")
    if authority_flags.get("workspace_is_derived_not_product_authority") is not True:
        raise WorkspaceMaterializationError("WORKSPACE_DERIVED_AUTHORITY_BOUNDARY_REQUIRED")
    if authority_flags.get("support_artifacts_are_non_authoritative") is not True:
        raise WorkspaceMaterializationError("WORKSPACE_SUPPORT_AUTHORITY_BOUNDARY_REQUIRED")

    authority = _by_role(request.get("artifacts"), scope="AUTHORITY")
    support = _by_role(request.get("support_artifacts"), scope="SUPPORT")
    parsed = _validate_artifact_contracts(authority, support)
    _validate_lineage(parsed)

    source_row = support["support:source_observation"]
    source_bytes = Path(str(source_row["path"])).read_bytes()
    source = decode_source_observation_arrays_v1(source_bytes)
    product = _compatibility_product(parsed, source.camera_bundle)
    proof_bytes = Path(str(authority["proof:product_pass"]["path"])).read_bytes()

    authority_manifest = []
    for role in sorted(authority):
        row = authority[role]
        path = Path(str(row["path"]))
        authority_manifest.append({
            "role": role,
            "artifact_id": str(row.get("artifact_id") or ""),
            "artifact_type": str(row.get("artifact_type") or ""),
            "schema_version": str(row.get("schema_version") or ""),
            "content_sha256": _sha_file(path),
            "scope": "AUTHORITY",
        })
    support_manifest = []
    for role in sorted(support):
        row = support[role]
        path = Path(str(row["path"]))
        support_manifest.append({
            "role": role,
            "artifact_id": str(row.get("artifact_id") or ""),
            "artifact_type": str(row.get("artifact_type") or ""),
            "schema_version": str(row.get("schema_version") or ""),
            "content_sha256": _sha_file(path),
            "scope": "SUPPORT",
            "product_authority": False,
        })

    product_bytes = _canonical_json_bytes(product)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "compile_id": str(request.get("compile_id") or ""),
        "product_revision_id": str(request.get("product_revision_id") or ""),
        "product_state_hash": str(product["product_state_hash"]),
        "projection_schema": PROJECTION_SCHEMA,
        "projection_content_sha256": _sha_bytes(product_bytes),
        "authority_artifacts": authority_manifest,
        "support_artifacts": support_manifest,
        "authority": {
            "product_revision_is_authority": True,
            "workspace_projection_is_authority": False,
            "support_artifacts_are_authority": False,
            "canonical_geometry_mutated": False,
            "canonical_identity_minted": False,
            "proof_preserved_byte_exact": True,
        },
    }

    entries: dict[str, bytes] = {
        "bundle_manifest.json": _canonical_json_bytes(manifest),
        "puppet/canonical_puppet_graph_v3.json": product_bytes,
        "proof/product_proof_bundle_ir.json": proof_bytes,
    }
    rgba = np.asarray(source.native_rgba, dtype=np.uint8)
    for view in range(8):
        entries[f"source/view_{view}.png"] = _encode_rgba_png(rgba[0, view])
    entries["source/camera_bundle.json"] = _canonical_json_bytes(source.camera_bundle)

    output = Path(str((request.get("required_output") or {}).get("path") or ""))
    if not output.is_absolute():
        output = (request_path.parent / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _zip_bytes(entries)
    output.write_bytes(payload)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    output = materialize(Path(args.request).resolve())
    print(json.dumps({"schema_version": BUNDLE_SCHEMA, "bundle_path": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
