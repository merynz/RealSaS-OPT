"""Fail-closed Master Corpus source-scene observation (NOT truth admission).

Stage order: source hash -> raw scene hierarchy/mesh domains -> native pose ->
character axes -> native render -> semantic review -> truth -> admission.

This producer is intentionally import-free for GLB/OBJ source metadata. It
never orients an arbitrary artist character by source +Z, treats object names
as student features, interprets mesh-domain count as drawable identity or
claims native FBX geometry without pinned Blender.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.Corpus4008SourceSceneObservation.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition, reason):
    if not condition:
        raise ValueError("CORPUS4008_SOURCE_SCENE_" + reason)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(4 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def parse_glb(path):
    payload = path.read_bytes()
    require(len(payload) >= 20 and payload[:4] == b"glTF", "GLB_SIGNATURE")
    version, length = struct.unpack_from("<II", payload, 4)
    require(version == 2 and length == len(payload), "GLB_HEADER")
    chunks, pos = [], 12
    while pos < len(payload):
        require(pos + 8 <= len(payload), "GLB_CHUNK_HEADER")
        size, kind = struct.unpack_from("<I4s", payload, pos)
        pos += 8
        require(pos + size <= len(payload), "GLB_CHUNK_TRUNCATED")
        chunks.append((kind, payload[pos:pos + size]))
        pos += size
    require(bool(chunks) and chunks[0][0] == b"JSON", "GLB_FIRST_CHUNK_JSON")
    doc = json.loads(chunks[0][1])
    require(doc.get("asset", {}).get("version") == "2.0", "GLTF_VERSION")
    return doc


def scene_from_glb(doc):
    nodes = doc.get("nodes", [])
    skins = doc.get("skins", [])
    meshes = doc.get("meshes", [])
    joints = set()
    for skin in skins:
        for i in skin.get("joints", []):
            require(isinstance(i, int) and 0 <= i < len(nodes), "BAD_JOINT")
            joints.add(i)
    parents = {}
    for i, node in enumerate(nodes):
        for c in node.get("children", []):
            require(isinstance(c, int) and 0 <= c < len(nodes), "BAD_CHILD")
            require(c not in parents, "TWO_NODE_PARENTS")
            parents[c] = i
    observed = []
    rigid_joint_children = []
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh = node["mesh"]
        require(isinstance(mesh, int) and 0 <= mesh < len(meshes), "MESH_REFERENCE")
        skin_idx = node.get("skin")
        require(skin_idx is None or (isinstance(skin_idx, int) and 0 <= skin_idx < len(skins)), "SKIN_REFERENCE")
        lineage, p, seen = [], parents.get(i), {i}
        parent_joint = None
        while p is not None:
            require(p not in seen, "NODE_CYCLE")
            seen.add(p)
            lineage.append(p)
            if p in joints and parent_joint is None:
                parent_joint = p
            p = parents.get(p)
        observed.append({
            "mesh_node_index": i,
            "mesh_index": mesh,
            "skin_index": skin_idx,
            "source_node_name_teacher_only": node.get("name"),
            "source_parent_node_path": lineage,
        })
        if skin_idx is None and parent_joint is not None:
            rigid_joint_children.append({
                "mesh_node_index": i,
                "mesh_index": mesh,
                "parent_joint_node_index": parent_joint,
                "source_node_name_teacher_only": node.get("name"),
                "source_parent_joint_name_teacher_only": nodes[parent_joint].get("name"),
                "link_observation": "BONE_PARENTED_UNSKINNED_MESH",
                "motion_consequence_courted": False,
                "semantic_attachment_type": "UNKNOWN",
            })
    source_animations = [{
        "animation_index": i, "source_action_name_teacher_only": a.get("name"),
        "channel_count": len(a.get("channels", []))
    } for i, a in enumerate(doc.get("animations", []))]
    return {
        "scene_node_count": len(nodes),
        "mesh_count": len(meshes),
        "mesh_primitive_count": sum(len(m.get("primitives", [])) for m in meshes),
        "mesh_node_count": len(observed),
        "skinned_mesh_node_count": sum(x["skin_index"] is not None for x in observed),
        "skin_count": len(skins),
        "source_skin_joint_counts": [len(s.get("joints", [])) for s in skins],
        "native_animation_count": len(source_animations),
        "source_animation_inventory_teacher_only": source_animations,
        "bone_parented_unskinned_mesh_count": len(rigid_joint_children),
        "bone_parented_unskinned_meshes": rigid_joint_children,
        "mesh_nodes": observed,
        "material_count": len(doc.get("materials", [])),
        "image_count": len(doc.get("images", [])),
        "source_external_image_count": sum(
            "uri" in x and not x["uri"].startswith("data:")
            for x in doc.get("images", [])
        ),
        "source_nodes_privileged_not_student": True,
        "attachment_transform_consequence_courted": False,
    }


def scene_from_obj(path):
    # Only original source object/group and topology counts; no Blender parity.
    vertices = 0
    faces = 0
    domains = Counter()
    boundary = defaultdict(Counter)
    active_domain = "default"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            require(len(parts) >= 4 and all(math.isfinite(float(x)) for x in parts[1:4]), "NONFINITE_OBJ_VERTEX")
            vertices += 1
        elif parts[0] in ("g", "o"):
            active_domain = " ".join(parts[1:]) or "unnamed"
        elif parts[0] == "f":
            ids = []
            for token in parts[1:]:
                raw = token.split("/")[0]
                require(raw != "", "OBJ_VERTEX_INDEX_MISSING")
                idx = int(raw)
                actual = idx-1 if idx > 0 else vertices+idx
                require(0 <= actual < vertices, "OBJ_BAD_FACE_INDEX")
                ids.append(actual)
            require(len(ids) >= 3, "OBJ_SHORT_FACE")
            for k in range(1,len(ids)-1):
                tri = (ids[0], ids[k], ids[k+1])
                domains[active_domain] += 1
                faces += 1
                for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
                    boundary[active_domain][(min(a,b),max(a,b))] += 1
    require(vertices > 0 and faces > 0, "EMPTY_OBJ")
    return {
        "source_vertex_count": vertices,
        "source_triangle_count": faces,
        "source_domain_count": len(domains),
        "source_domains": [{
            "name_teacher_only": key,
            "triangle_count": count,
            "boundary_edge_count": sum(v == 1 for v in boundary[key].values()),
            "nonmanifold_edge_count": sum(v > 2 for v in boundary[key].values()),
            "watertight_by_shared_source_index": bool(boundary[key]) and all(v == 2 for v in boundary[key].values()),
        } for key,count in sorted(domains.items())],
        "rig_in_native_obj": False,
        "skin_in_native_obj": False,
        "animation_in_native_obj": False,
        "source_group_is_semantic_component": False,
        "source_domain_is_graphics_drawable": False,
    }


def build(path, *, source_format, expected_sha256, canonical_asset_id, candidate_id):
    require(source_format in {"glb", "obj", "fbx", "blend"}, "FORMAT")
    require(isinstance(expected_sha256, str) and HEX64.fullmatch(expected_sha256), "EXPECTED_SHA")
    require(path.is_file(), "SOURCE_MISSING")
    actual = sha256(path)
    require(actual == expected_sha256, "SOURCE_SHA_MISMATCH")
    require(isinstance(candidate_id, str) and ":" in candidate_id, "CANDIDATE_ID")
    suffix = candidate_id.split(":", 1)[1]
    require(expected_sha256.startswith(suffix), "SOURCE_CANDIDATE_TOKEN_DRIFT")
    item = {
        "schema": SCHEMA,
        "canonical_asset_id": canonical_asset_id,
        "candidate_id": candidate_id,
        "source_sha256": actual,
        "source_size_bytes": path.stat().st_size,
        "source_format": source_format,
        "source_bytes_verified": True,
        "source_scene": None,
        "source_anatomical_up": "UNKNOWN",
        "source_anatomical_front": "UNKNOWN",
        "source_pose_class": "UNKNOWN",
        "character_vs_prop": "UNKNOWN",
        "morphology": "UNKNOWN",
        "semantic_component_class_courted": False,
        "native_render_courted": False,
        "stylized_render_courted": False,
        "eight_view_camera_courted": False,
        "geometry_training_authorized": False,
        "semantic_training_authorized": False,
        "graphics_training_authorized": False,
        "rig_skin_training_authorized": False,
        "training_authorized": False,
        "source_names_student_features": False,
        "mesh_island_is_drawable": False,
    }
    if source_format == "glb":
        item["source_scene"] = scene_from_glb(parse_glb(path))
        item["source_scene_status"] = "GLB_NATIVE_HIERARCHY_OBSERVED__NO_CHARACTER_OR_RENDER_COURT"
    elif source_format == "obj":
        item["source_scene"] = scene_from_obj(path)
        item["source_scene_status"] = "OBJ_SOURCE_DOMAINS_OBSERVED__NO_CHARACTER_OR_RENDER_COURT"
    else:
        item["source_scene_status"] = "PINNED_BLENDER_SOURCE_EVALUATOR_REQUIRED"
    return item


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--format", choices=("glb","obj","fbx","blend"),required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--canonical-asset-id", required=True)
    p.add_argument("--candidate-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    receipt = build(a.source, source_format=a.format, expected_sha256=a.expected_source_sha256,
                    canonical_asset_id=a.canonical_asset_id, candidate_id=a.candidate_id)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status":receipt["source_scene_status"],"candidate":a.candidate_id,
                      "training_authorized":False},sort_keys=True))


if __name__ == "__main__":
    main()
