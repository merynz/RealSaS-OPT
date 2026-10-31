"""Canonical Blender scene extractor for Post139 structural subject fingerprints.

Run only through Blender, for example:
  blender --background --factory-startup --python-exit-code 1 \
    --python tools/corpus/blender_post139_source_structure_v1.py -- \
    --source Character.fbx --out source_structure.json

The structural subject fingerprint intentionally excludes appearance. That lets
color/material-only variants collapse when geometry/topology/rig/weights are
identical, while appearance remains separately fingerprinted and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SOURCE_STRUCTURE_SCHEMA = "RealSaS.Post139.SourceStructure.v1"


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(value, digits=7):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("POST139_BLENDER_NONFINITE")
    value = round(value, digits)
    return 0.0 if value == -0.0 else value


def args_after_double_dash():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.images, bpy.data.actions):
        # Orphans are removed after import extraction; do not mutate while iterating later.
        pass


def import_source(path):
    suffix = path.suffix.lower()
    if suffix in {".gltf", ".glb"}:
        # Blender's glTF importer otherwise creates an Icosphere helper in the
        # glTF_not_exported collection for bone display. That helper is not source
        # geometry and would poison geometry/rig normalization and cross-format hashes.
        bpy.ops.import_scene.gltf(filepath=str(path), disable_bone_shape=True)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    else:
        raise ValueError(f"POST139_BLENDER_FORMAT_UNSUPPORTED:{suffix}")


def world_vertices(obj):
    matrix = obj.matrix_world
    return [matrix @ vertex.co for vertex in obj.data.vertices]


def global_normalization(mesh_objects):
    points = []
    for obj in mesh_objects:
        points.extend(world_vertices(obj))
    if not points:
        return Vector((0.0, 0.0, 0.0)), 1.0
    center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    rms = math.sqrt(sum((point - center).length_squared for point in points) / len(points))
    return center, rms if rms > 1e-12 else 1.0


def geometry_records(mesh_objects, center, scale):
    records = []
    for obj in mesh_objects:
        points = world_vertices(obj)
        radial = sorted(q((point - center).length / scale, 6) for point in points)
        edge_lengths = []
        degree = [0] * len(points)
        for edge in obj.data.edges:
            a, b = edge.vertices
            degree[a] += 1
            degree[b] += 1
            edge_lengths.append(q((points[a] - points[b]).length / scale, 6))
        edge_lengths.sort()
        face_sizes = sorted(len(poly.vertices) for poly in obj.data.polygons)
        tri_count = sum(max(0, len(poly.vertices) - 2) for poly in obj.data.polygons)
        records.append(
            {
                "vertex_count": len(points),
                "edge_count": len(obj.data.edges),
                "polygon_count": len(obj.data.polygons),
                "triangle_count": tri_count,
                "radial_distribution_hash": digest(radial),
                "edge_length_distribution_hash": digest(edge_lengths),
                "vertex_degree_distribution": sorted(degree),
                "face_size_distribution": face_sizes,
            }
        )
    return sorted(records, key=canonical_bytes)


def rig_records(armatures, scale):
    records = []
    for obj in armatures:
        matrix = obj.matrix_world
        for bone in obj.data.bones:
            head = matrix @ bone.head_local
            tail = matrix @ bone.tail_local
            records.append(
                {
                    "name": bone.name,
                    "parent": bone.parent.name if bone.parent else None,
                    "length": q((tail - head).length / scale, 6),
                    "use_deform": bool(bone.use_deform),
                }
            )
    return sorted(records, key=lambda row: (row["name"], row["parent"] or "", row["length"]))


def skin_records(mesh_objects):
    records = []
    skinned_vertices = 0
    for obj in mesh_objects:
        group_names = {group.index: group.name for group in obj.vertex_groups}
        vertex_signatures = []
        for vertex in obj.data.vertices:
            influences = []
            for group in vertex.groups:
                if group.weight <= 0.0:
                    continue
                influences.append((group_names.get(group.group, f"group_{group.group}"), q(group.weight, 6)))
            influences.sort()
            if influences:
                skinned_vertices += 1
                vertex_signatures.append(influences)
        records.append(
            {
                "vertex_count": len(obj.data.vertices),
                "skinned_vertex_count": len(vertex_signatures),
                "influence_multiset_hash": digest(sorted(vertex_signatures, key=canonical_bytes)),
            }
        )
    return sorted(records, key=canonical_bytes), skinned_vertices


def image_record(image):
    path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    row = {
        "name": image.name,
        "size": [int(image.size[0]), int(image.size[1])],
        "source": image.source,
    }
    if path and path.is_file():
        row["file_name"] = path.name
        row["sha256"] = file_hash(path)
    elif getattr(image, "packed_file", None) is not None:
        try:
            row["packed_sha256"] = hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
        except Exception:
            row["packed_sha256"] = "UNAVAILABLE"
    return row


def appearance_records(mesh_objects):
    materials = []
    material_names = set()
    has_vertex_color = False
    for obj in mesh_objects:
        color_attributes = getattr(obj.data, "color_attributes", None)
        has_vertex_color = has_vertex_color or bool(color_attributes and len(color_attributes))
        for slot in obj.material_slots:
            if slot.material:
                material_names.add(slot.material.name)
    for name in sorted(material_names):
        material = bpy.data.materials.get(name)
        if material is None:
            continue
        materials.append(
            {
                "name": material.name,
                "diffuse_color": [q(v, 6) for v in material.diffuse_color],
                "blend_method": getattr(material, "blend_method", None),
                "use_nodes": bool(material.use_nodes),
            }
        )
    images = sorted((image_record(image) for image in bpy.data.images), key=canonical_bytes)
    return {"materials": materials, "images": images, "has_vertex_color": has_vertex_color}


def animation_records():
    records = []
    for action in bpy.data.actions:
        frame_range = list(action.frame_range)
        records.append(
            {
                "name": action.name,
                "frame_start": q(frame_range[0], 4),
                "frame_end": q(frame_range[1], 4),
                "fcurve_count": len(action.fcurves),
            }
        )
    return sorted(records, key=lambda row: row["name"])


def main():
    args = args_after_double_dash()
    source = Path(args.source).resolve()
    output = Path(args.out).resolve()
    if not source.is_file():
        raise SystemExit(f"POST139_BLENDER_SOURCE_MISSING:{source}")
    clear_scene()
    import_source(source)
    bpy.context.scene.frame_set(0)
    armatures = sorted((obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"), key=lambda o: o.name)
    for armature in armatures:
        armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    mesh_objects = sorted((obj for obj in bpy.context.scene.objects if obj.type == "MESH"), key=lambda o: o.name)
    if not mesh_objects:
        raise SystemExit("POST139_BLENDER_NO_MESH")
    center, scale = global_normalization(mesh_objects)
    geometry = geometry_records(mesh_objects, center, scale)
    rig = rig_records(armatures, scale)
    skin, skinned_vertices = skin_records(mesh_objects)
    appearance = appearance_records(mesh_objects)
    animations = animation_records()
    geometry_hash = digest(geometry)
    rig_hash = digest(rig)
    skin_hash = digest(skin)
    structural_hash = digest({"geometry_topology": geometry_hash, "rig_rest": rig_hash, "skin_weights": skin_hash})
    payload = {
        "schema": SOURCE_STRUCTURE_SCHEMA,
        "extractor": {
            "name": "BLENDER_CANONICAL_SOURCE_STRUCTURE_V1",
            "blender_version": bpy.app.version_string,
            "canonical_space": "IMPORT_NORMALIZED_RIGID_AND_SCALE_INVARIANT_V1",
            "cross_format_dedup_ready": True,
        },
        "source": {
            "path": str(source),
            "name": source.name,
            "extension": source.suffix.lower(),
            "sha256": file_hash(source),
            "size_bytes": source.stat().st_size,
        },
        "metrics": {
            "mesh_count": len(mesh_objects),
            "vertex_count": sum(len(obj.data.vertices) for obj in mesh_objects),
            "edge_count": sum(len(obj.data.edges) for obj in mesh_objects),
            "polygon_count": sum(len(obj.data.polygons) for obj in mesh_objects),
            "armature_count": len(armatures),
            "joint_count": len(rig),
            "skinned_vertex_count": skinned_vertices,
            "animation_count": len(animations),
            "material_count": len(appearance["materials"]),
            "image_count": len(appearance["images"]),
        },
        "capabilities": {
            "geometry": bool(geometry),
            "appearance": bool(appearance["materials"] or appearance["images"] or appearance["has_vertex_color"]),
            "rig": bool(rig),
            "skin": skinned_vertices > 0,
            "animation": bool(animations),
        },
        "fingerprints": {
            "geometry_topology": geometry_hash,
            "rig_rest": rig_hash,
            "skin_weights": skin_hash,
            "structural_subject": structural_hash,
            "appearance": digest(appearance),
            "animation_inventory": digest(animations),
        },
        "inventory": {
            "geometry": geometry,
            "rig": rig,
            "skin": skin,
            "appearance": appearance,
            "animations": animations,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("POST139_SOURCE_STRUCTURE_PASS", structural_hash)


if __name__ == "__main__":
    main()
