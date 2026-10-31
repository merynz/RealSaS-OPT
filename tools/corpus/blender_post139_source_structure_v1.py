"""Canonical Blender scene extractor for Post139 structural subject fingerprints.

The extractor records both compact legacy descriptors and strong V2 canonical
fingerprints.  V2 binds topology to normalized welded vertex positions, skin
weights to their spatial vertices, and rest bones to normalized head/tail
positions.  Appearance remains outside structural subject identity so pure
recolors do not inflate the subject denominator.
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

# Keep the additive evidence envelope compatible with the existing audit reader.
# The extractor/canonical-space fields below carry the V2 semantic version.
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


def qvec(vector, digits=6):
    return tuple(q(value, digits) for value in vector)


def args_after_double_dash():
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_source(path):
    suffix = path.suffix.lower()
    if suffix in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True, automatic_bone_orientation=False)
    elif suffix == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    else:
        raise ValueError(f"POST139_BLENDER_FORMAT_UNSUPPORTED:{suffix}")


def importer_display_helpers(armatures):
    helpers = set()
    for armature in armatures:
        for pose_bone in armature.pose.bones:
            if pose_bone.custom_shape is not None:
                helpers.add(pose_bone.custom_shape)
    collection = bpy.data.collections.get("glTF_not_exported")
    if collection is not None:
        for obj in collection.all_objects:
            if obj.type == "MESH":
                helpers.add(obj)
    return helpers


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


def normalized_point(point, center, scale):
    return qvec((point - center) / scale, 6)


def geometry_records_legacy(mesh_objects, center, scale):
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
        records.append({
            "vertex_count": len(points),
            "edge_count": len(obj.data.edges),
            "polygon_count": len(obj.data.polygons),
            "triangle_count": tri_count,
            "radial_distribution_hash": digest(radial),
            "edge_length_distribution_hash": digest(edge_lengths),
            "vertex_degree_distribution": sorted(degree),
            "face_size_distribution": face_sizes,
        })
    return sorted(records, key=canonical_bytes)


def geometry_canonical_surface(mesh_objects, center, scale):
    per_object_positions = {}
    unique_positions = set()
    for obj in mesh_objects:
        points = [normalized_point(point, center, scale) for point in world_vertices(obj)]
        per_object_positions[obj.name] = points
        unique_positions.update(points)
    positions = sorted(unique_positions)
    position_index = {value: index for index, value in enumerate(positions)}
    triangles = []
    for obj in mesh_objects:
        obj.data.calc_loop_triangles()
        points = per_object_positions[obj.name]
        for tri in obj.data.loop_triangles:
            indices = sorted(position_index[points[index]] for index in tri.vertices)
            if len(set(indices)) == 3:
                triangles.append(tuple(indices))
    triangles = sorted(set(triangles))
    return {
        "normalization": "GLOBAL_VERTEX_CENTROID_RMS__QUANTIZED_1E-6",
        "weld_policy": "IDENTICAL_NORMALIZED_POSITION_AFTER_QUANTIZATION",
        "welded_vertex_count": len(positions),
        "triangle_count": len(triangles),
        "vertex_position_hash": digest(positions),
        "triangle_connectivity_hash": digest(triangles),
        "surface_hash": digest({"positions": positions, "triangles": triangles}),
    }


def rig_records(armatures, center, scale):
    records = []
    hierarchy = []
    for obj in armatures:
        matrix = obj.matrix_world
        for bone in obj.data.bones:
            head = matrix @ bone.head_local
            tail = matrix @ bone.tail_local
            row = {
                "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "length": q((tail - head).length / scale, 6),
                "head": normalized_point(head, center, scale),
                "tail": normalized_point(tail, center, scale),
                "use_deform": bool(bone.use_deform),
            }
            records.append(row)
            hierarchy.append({"name": row["name"], "parent": row["parent"], "use_deform": row["use_deform"]})
    records = sorted(records, key=lambda row: (row["name"], row["parent"] or "", row["head"], row["tail"]))
    hierarchy = sorted(hierarchy, key=lambda row: (row["name"], row["parent"] or "", row["use_deform"]))
    return records, hierarchy


def skin_records(mesh_objects, center, scale):
    legacy_records = []
    spatial_rows = []
    skinned_vertices = 0
    for obj in mesh_objects:
        group_names = {group.index: group.name for group in obj.vertex_groups}
        points = world_vertices(obj)
        vertex_signatures = []
        for vertex, point in zip(obj.data.vertices, points):
            influences = []
            for group in vertex.groups:
                if group.weight <= 0.0:
                    continue
                influences.append((group_names.get(group.group, f"group_{group.group}"), q(group.weight, 6)))
            influences.sort()
            if influences:
                skinned_vertices += 1
                vertex_signatures.append(influences)
                spatial_rows.append({"position": normalized_point(point, center, scale), "influences": influences})
        legacy_records.append({
            "vertex_count": len(obj.data.vertices),
            "skinned_vertex_count": len(vertex_signatures),
            "influence_multiset_hash": digest(sorted(vertex_signatures, key=canonical_bytes)),
        })
    spatial_rows = sorted(spatial_rows, key=canonical_bytes)
    spatial_unique = []
    previous = None
    for row in spatial_rows:
        encoded = canonical_bytes(row)
        if encoded != previous:
            spatial_unique.append(row)
        previous = encoded
    return sorted(legacy_records, key=canonical_bytes), skinned_vertices, {
        "binding_policy": "NORMALIZED_POSITION_PLUS_NAMED_WEIGHT_VECTOR",
        "spatial_binding_count": len(spatial_unique),
        "spatial_binding_hash": digest(spatial_unique),
    }


def image_record(image):
    path = Path(bpy.path.abspath(image.filepath)) if image.filepath else None
    row = {"name": image.name, "size": [int(image.size[0]), int(image.size[1])], "source": image.source}
    if path and path.is_file():
        row["file_name"] = path.name
        row["sha256"] = file_hash(path)
    elif getattr(image, "packed_file", None) is not None:
        try:
            row["packed_sha256"] = hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
        except Exception:
            row["packed_sha256"] = "UNAVAILABLE"
    return row


def material_record(material):
    row = {
        "name": material.name,
        "diffuse_color": [q(v, 6) for v in material.diffuse_color],
        "blend_method": getattr(material, "blend_method", None),
        "use_nodes": bool(material.use_nodes),
    }
    if material.use_nodes and material.node_tree:
        nodes = []
        for node in material.node_tree.nodes:
            record = {"type": node.bl_idname, "name": node.name}
            if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
                record["image"] = node.image.name
            nodes.append(record)
        row["node_inventory"] = sorted(nodes, key=canonical_bytes)
    return row


def appearance_records(mesh_objects):
    material_names = set()
    has_vertex_color = False
    object_bindings = []
    for obj in mesh_objects:
        color_attributes = getattr(obj.data, "color_attributes", None)
        has_vertex_color = has_vertex_color or bool(color_attributes and len(color_attributes))
        slots = []
        for slot in obj.material_slots:
            if slot.material:
                material_names.add(slot.material.name)
                slots.append(slot.material.name)
            else:
                slots.append(None)
        object_bindings.append({"object": obj.name, "material_slots": slots})
    materials = [material_record(bpy.data.materials[name]) for name in sorted(material_names) if bpy.data.materials.get(name) is not None]
    images = sorted((image_record(image) for image in bpy.data.images), key=canonical_bytes)
    return {
        "materials": materials,
        "images": images,
        "object_material_bindings": sorted(object_bindings, key=canonical_bytes),
        "has_vertex_color": has_vertex_color,
    }


def animation_records():
    records = []
    for action in bpy.data.actions:
        frame_range = list(action.frame_range)
        channels = []
        total_keyframes = 0
        for curve in action.fcurves:
            keyframes = len(curve.keyframe_points)
            total_keyframes += keyframes
            channels.append({"data_path": curve.data_path, "array_index": int(curve.array_index), "keyframe_count": keyframes})
        channels = sorted(channels, key=canonical_bytes)
        records.append({
            "name": action.name,
            "frame_start": q(frame_range[0], 4),
            "frame_end": q(frame_range[1], 4),
            "fcurve_count": len(action.fcurves),
            "keyframe_count": total_keyframes,
            "channel_inventory_hash": digest(channels),
            "channels": channels,
        })
    return sorted(records, key=lambda row: row["name"])


def object_records(armatures):
    armature_names = {obj.name for obj in armatures}
    records = []
    for obj in sorted(bpy.context.scene.objects, key=lambda item: item.name):
        row = {
            "name": obj.name,
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
            "parent_type": obj.parent_type if obj.parent else None,
            "parent_bone": obj.parent_bone or None,
        }
        if obj.type == "MESH":
            armature_targets = [modifier.object.name for modifier in obj.modifiers if modifier.type == "ARMATURE" and modifier.object]
            row.update({
                "vertex_count": len(obj.data.vertices),
                "edge_count": len(obj.data.edges),
                "polygon_count": len(obj.data.polygons),
                "vertex_group_count": len(obj.vertex_groups),
                "material_slots": [slot.material.name if slot.material else None for slot in obj.material_slots],
                "armature_targets": sorted(armature_targets),
                "bone_parented": bool(obj.parent and obj.parent.name in armature_names and obj.parent_type == "BONE"),
            })
        records.append(row)
    return records


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
    ignored_helpers = importer_display_helpers(armatures)
    mesh_objects = sorted((obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in ignored_helpers), key=lambda o: o.name)
    if not mesh_objects:
        raise SystemExit("POST139_BLENDER_NO_MESH")

    center, scale = global_normalization(mesh_objects)
    geometry_legacy = geometry_records_legacy(mesh_objects, center, scale)
    geometry_surface = geometry_canonical_surface(mesh_objects, center, scale)
    rig, rig_hierarchy = rig_records(armatures, center, scale)
    skin_legacy, skinned_vertices, skin_spatial = skin_records(mesh_objects, center, scale)
    appearance = appearance_records(mesh_objects)
    animations = animation_records()
    objects = object_records(armatures)

    legacy_geometry_hash = digest(geometry_legacy)
    rig_rest_hash = digest(rig)
    rig_hierarchy_hash = digest(rig_hierarchy)
    legacy_skin_hash = digest(skin_legacy)
    geometry_hash = geometry_surface["surface_hash"]
    skin_hash = skin_spatial["spatial_binding_hash"]
    structural_hash = digest({"geometry_topology": geometry_hash, "rig_rest": rig_rest_hash, "skin_weights": skin_hash})
    legacy_structural_hash = digest({"geometry_topology": legacy_geometry_hash, "rig_rest": rig_rest_hash, "skin_weights": legacy_skin_hash})

    payload = {
        "schema": SOURCE_STRUCTURE_SCHEMA,
        "extractor": {
            "name": "BLENDER_CANONICAL_SOURCE_STRUCTURE_V2",
            "blender_version": bpy.app.version_string,
            "canonical_space": "IMPORT_NORMALIZED_GLOBAL_WELDED_SURFACE_V2",
            "cross_format_dedup_ready": True,
            "ignored_import_helper_count": len(ignored_helpers),
        },
        "source": {
            "path": str(source), "name": source.name, "extension": source.suffix.lower(),
            "sha256": file_hash(source), "size_bytes": source.stat().st_size,
        },
        "metrics": {
            "mesh_count": len(mesh_objects),
            "vertex_count": sum(len(obj.data.vertices) for obj in mesh_objects),
            "welded_vertex_count": geometry_surface["welded_vertex_count"],
            "edge_count": sum(len(obj.data.edges) for obj in mesh_objects),
            "polygon_count": sum(len(obj.data.polygons) for obj in mesh_objects),
            "triangle_count": geometry_surface["triangle_count"],
            "armature_count": len(armatures),
            "joint_count": len(rig),
            "skinned_vertex_count": skinned_vertices,
            "animation_count": len(animations),
            "material_count": len(appearance["materials"]),
            "image_count": len(appearance["images"]),
        },
        "capabilities": {
            "geometry": bool(geometry_legacy),
            "appearance": bool(appearance["materials"] or appearance["images"] or appearance["has_vertex_color"]),
            "rig": bool(rig), "skin": skinned_vertices > 0, "animation": bool(animations),
        },
        "fingerprints": {
            "geometry_topology": geometry_hash,
            "geometry_topology_legacy_v1": legacy_geometry_hash,
            "rig_rest": rig_rest_hash,
            "rig_hierarchy": rig_hierarchy_hash,
            "skin_weights": skin_hash,
            "skin_weights_legacy_v1": legacy_skin_hash,
            "structural_subject": structural_hash,
            "structural_subject_legacy_v1": legacy_structural_hash,
            "appearance": digest(appearance),
            "animation_inventory": digest(animations),
            "object_inventory": digest(objects),
        },
        "inventory": {
            "objects": objects,
            "geometry": geometry_legacy,
            "canonical_surface": geometry_surface,
            "rig": rig,
            "rig_hierarchy": rig_hierarchy,
            "skin": skin_legacy,
            "spatial_skin": skin_spatial,
            "appearance": appearance,
            "animations": animations,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("POST139_SOURCE_STRUCTURE_V2_PASS", structural_hash, rig_hierarchy_hash)


if __name__ == "__main__":
    main()
