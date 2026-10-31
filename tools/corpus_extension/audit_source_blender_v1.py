import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict

import bpy

SCHEMA = "realsas.post139_source_blender_audit.v1"
SUPPORTED = {".fbx", ".gltf", ".glb", ".blend"}


def sha_parts(parts):
    h = hashlib.sha256()
    for part in parts:
        if not isinstance(part, (bytes, bytearray)):
            part = str(part).encode("utf-8")
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def q(x):
    return round(float(x), 8)


def matrix_payload(m):
    return [[q(v) for v in row] for row in m]


def reset_and_import(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = Path(path).suffix.lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), use_anim=True)
    elif ext in {".gltf", ".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=str(path))
    else:
        raise RuntimeError(f"unsupported extension: {ext}")
    bpy.context.view_layer.update()


def mesh_measure(obj):
    mesh = obj.data
    verts = [[q(c) for c in v.co] for v in mesh.vertices]
    faces = [[int(i) for i in p.vertices] for p in mesh.polygons]
    nondeg = sum(1 for p in mesh.polygons if float(p.area) > 1e-12)
    uv_parts = []
    for layer in mesh.uv_layers:
        uv_parts.append((layer.name, [[q(d.uv.x), q(d.uv.y)] for d in layer.data]))
    color_parts = []
    if hasattr(mesh, "color_attributes"):
        for layer in mesh.color_attributes:
            values = []
            for d in layer.data:
                c = getattr(d, "color", None)
                if c is not None:
                    values.append([q(x) for x in c])
            color_parts.append((layer.name, str(layer.domain), str(layer.data_type), values))
    materials = []
    for slot in obj.material_slots:
        m = slot.material
        if not m:
            continue
        rec = {"name": m.name, "diffuse": [q(x) for x in m.diffuse_color], "nodes": [], "images": []}
        if m.use_nodes and m.node_tree:
            for node in m.node_tree.nodes:
                rec["nodes"].append(node.bl_idname)
                image = getattr(node, "image", None)
                if image:
                    rec["images"].append(image.name)
        rec["nodes"].sort()
        rec["images"] = sorted(set(rec["images"]))
        materials.append(rec)
    return {
        "object": obj.name,
        "mesh": mesh.name,
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.polygons),
        "nondegenerate_fraction": (nondeg / len(mesh.polygons)) if mesh.polygons else 0.0,
        "geometry_hash": sha_parts([json.dumps(verts, separators=(",", ":"))]),
        "topology_hash": sha_parts([json.dumps(faces, separators=(",", ":"))]),
        "appearance_hash": sha_parts([json.dumps(materials, sort_keys=True), json.dumps(uv_parts), json.dumps(color_parts)]),
        "material_count": len(materials),
        "uv_layer_count": len(uv_parts),
        "color_attribute_count": len(color_parts),
        "materials": materials,
    }


def armature_measure(obj):
    bones = []
    for b in obj.data.bones:
        bones.append({
            "name": b.name,
            "parent": b.parent.name if b.parent else None,
            "use_deform": bool(b.use_deform),
            "matrix_local": matrix_payload(b.matrix_local),
        })
    return {
        "object": obj.name,
        "bone_count": len(bones),
        "bones": bones,
        "hierarchy_hash": sha_parts([json.dumps([(b["name"], b["parent"]) for b in bones], separators=(",", ":"))]),
        "rest_hash": sha_parts([json.dumps([(b["name"], b["matrix_local"]) for b in bones], separators=(",", ":"))]),
    }


def skin_measure(obj, bone_names):
    groups = {vg.index: vg.name for vg in obj.vertex_groups}
    rows = []
    zero = 0
    max_influences = 0
    errors = []
    for v in obj.data.vertices:
        weights = []
        for g in v.groups:
            name = groups.get(g.group)
            if name in bone_names and g.weight > 0:
                weights.append((name, q(g.weight)))
        weights.sort()
        rows.append((int(v.index), weights))
        if not weights:
            zero += 1
        else:
            errors.append(abs(sum(w for _, w in weights) - 1.0))
        max_influences = max(max_influences, len(weights))
    errors.sort()
    p99 = errors[min(len(errors) - 1, max(0, int(math.ceil(0.99 * len(errors))) - 1))] if errors else None
    count = len(obj.data.vertices)
    return {
        "object": obj.name,
        "vertex_count": count,
        "weighted_vertex_count": count - zero,
        "zero_weight_fraction": (zero / count) if count else 1.0,
        "max_influences": max_influences,
        "row_sum_abs_error_p99": p99,
        "skin_hash": sha_parts([json.dumps(rows, separators=(",", ":"))]),
    }


def action_measure(action):
    curves = []
    for fc in action.fcurves:
        keys = [[q(k.co.x), q(k.co.y)] for k in fc.keyframe_points]
        curves.append((fc.data_path, int(fc.array_index), keys))
    curves.sort(key=lambda x: (x[0], x[1]))
    return {
        "name": action.name,
        "frame_range": [q(action.frame_range[0]), q(action.frame_range[1])],
        "fcurve_count": len(action.fcurves),
        "hash": sha_parts([json.dumps(curves, separators=(",", ":"))]),
    }


def image_measure(image):
    rec = {"name": image.name, "size": [int(image.size[0]), int(image.size[1])], "filepath": None}
    if image.filepath:
        rec["filepath"] = bpy.path.abspath(image.filepath)
    if image.packed_file:
        try:
            raw = bytes(image.packed_file.data)
            rec["sha256"] = hashlib.sha256(raw).hexdigest()
            rec["bytes"] = len(raw)
        except Exception:
            pass
    elif rec["filepath"] and os.path.isfile(rec["filepath"]):
        rec["sha256"] = sha_file(rec["filepath"])
        rec["bytes"] = os.path.getsize(rec["filepath"])
    return rec


def audit(path):
    path = Path(path).resolve()
    reset_and_import(path)
    mesh_objects = sorted([o for o in bpy.context.scene.objects if o.type == "MESH"], key=lambda o: o.name)
    armature_objects = sorted([o for o in bpy.context.scene.objects if o.type == "ARMATURE"], key=lambda o: o.name)
    meshes = [mesh_measure(o) for o in mesh_objects]
    armatures = [armature_measure(o) for o in armature_objects]
    bone_names = set()
    for a in armature_objects:
        bone_names.update(b.name for b in a.data.bones)
    skins = [skin_measure(o, bone_names) for o in mesh_objects if o.vertex_groups and bone_names]
    actions = sorted([action_measure(a) for a in bpy.data.actions], key=lambda x: x["name"])
    images = [image_measure(i) for i in bpy.data.images if i.name not in {"Render Result", "Viewer Node"}]

    geometry_pass = bool(meshes) and sum(m["face_count"] for m in meshes) > 0
    appearance_pass = any(m["material_count"] or m["color_attribute_count"] for m in meshes)
    rig_pass = any(a["bone_count"] > 0 for a in armatures)
    weighted = sum(s["weighted_vertex_count"] for s in skins)
    skin_pass = rig_pass and bool(skins) and weighted > 0
    animation_pass = bool(actions)
    subject_class = "FULL_TRUTH_ACTOR" if all([geometry_pass, appearance_pass, rig_pass, skin_pass, animation_pass]) else "PARTIAL_TRUTH_ACTOR" if geometry_pass and rig_pass and skin_pass else "STATIC_OR_RIGID_AUX" if geometry_pass else "INVALID"

    return {
        "schema_version": SCHEMA,
        "source": {"path": str(path), "name": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)},
        "capabilities": {
            "geometry": {"pass": geometry_pass, "mesh_count": len(meshes), "vertex_count_sum": sum(m["vertex_count"] for m in meshes), "face_count_sum": sum(m["face_count"] for m in meshes), "geometry_signature": sha_parts([m["geometry_hash"] for m in meshes]) if meshes else None, "topology_signature": sha_parts([m["topology_hash"] for m in meshes]) if meshes else None},
            "appearance": {"pass": appearance_pass, "material_count": sum(m["material_count"] for m in meshes), "image_count": len(images), "appearance_signature": sha_parts([m["appearance_hash"] for m in meshes] + [json.dumps(images, sort_keys=True)]) if meshes else None},
            "rig": {"pass": rig_pass, "armature_count": len(armatures), "bone_count_sum": sum(a["bone_count"] for a in armatures), "skeleton_hierarchy_signature": sha_parts([a["hierarchy_hash"] for a in armatures]) if armatures else None, "rest_bind_signature": sha_parts([a["rest_hash"] for a in armatures]) if armatures else None},
            "skin": {"pass": skin_pass, "skinned_mesh_count": len(skins), "weighted_vertex_count": weighted, "max_influences": max((s["max_influences"] for s in skins), default=0), "zero_weight_fraction_max": max((s["zero_weight_fraction"] for s in skins), default=None), "row_sum_abs_error_p99_max": max((s["row_sum_abs_error_p99"] for s in skins if s["row_sum_abs_error_p99"] is not None), default=None), "skin_signature": sha_parts([s["skin_hash"] for s in skins]) if skins else None},
            "animation": {"pass": animation_pass, "action_count": len(actions), "action_names": [a["name"] for a in actions], "animation_signature": sha_parts([a["hash"] for a in actions]) if actions else None},
        },
        "subject_class_candidate": subject_class,
        "source_structure": {"meshes": meshes, "armatures": armatures, "skins": skins, "images": images, "actions": actions},
        "authority": {"state": "PRE_ADMISSION_ONLY", "final_gate": "existing Truth139 courts -> realsas.source_admission.v1", "no_threshold_relaxation": True},
    }


def iter_sources(root):
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    inp = Path(args.input).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    sources = iter_sources(inp)
    for i, src in enumerate(sources, 1):
        rel = str(src.relative_to(inp)) if inp.is_dir() else src.name
        key = hashlib.sha256(rel.encode()).hexdigest()[:12]
        dst = out / f"{key}_{src.stem}.json"
        try:
            value = audit(src)
            value["source"]["relative_path"] = rel
            dst.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append({"source": rel, "audit": dst.name, "status": "PASS", "subject_class_candidate": value["subject_class_candidate"]})
            print(f"[{i}/{len(sources)}] PASS {rel} {value['subject_class_candidate']}")
        except Exception as exc:
            dst.write_text(json.dumps({"schema_version": SCHEMA, "source": rel, "status": "ERROR", "error": repr(exc)}, indent=2), encoding="utf-8")
            results.append({"source": rel, "audit": dst.name, "status": "ERROR", "error": repr(exc)})
            print(f"[{i}/{len(sources)}] ERROR {rel}: {exc}")
    (out / "MANIFEST.json").write_text(json.dumps({"schema_version": "realsas.post139_source_blender_audit_manifest.v1", "input": str(inp), "source_count": len(sources), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
