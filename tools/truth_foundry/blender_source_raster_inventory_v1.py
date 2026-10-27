#!/usr/bin/env python3
from __future__ import annotations

"""Read-only inventory of one packed source evaluator scene.

This probe runs *inside* the exact packed ``source_scene.blend`` produced by the
Truth Foundry teacher scene. It does not render, rewrite nodes, infer a sampler,
or mint qualification. The only goal is to expose the real material/render
surface that a later measured raster adapter must support or explicitly reject.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

SCHEMA = "RealSaS.SourceRasterRendererInventory.v1"


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--subject-id", required=True)
    p.add_argument("--teacher-scene", required=True, type=Path)
    p.add_argument("--appearance", required=True, type=Path)
    p.add_argument("--expected-blend-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args(raw)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _socket_value(socket):
    value = getattr(socket, "default_value", None)
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return [float(v) for v in value]
    except Exception:
        return str(value)


def _socket_row(socket) -> dict:
    links = []
    if socket is not None:
        for link in socket.links:
            links.append({
                "from_node": link.from_node.name,
                "from_type": link.from_node.bl_idname,
                "from_socket": link.from_socket.identifier,
            })
    return {
        "present": socket is not None,
        "linked": bool(links),
        "default": _socket_value(socket) if socket is not None else None,
        "links": links,
    }


def _material_row(material) -> dict:
    row = {
        "name": material.name,
        "use_nodes": bool(material.use_nodes),
        "diffuse_rgba": [float(v) for v in material.diffuse_color],
        "surface_render_method": str(getattr(material, "surface_render_method", "UNKNOWN")),
        "use_transparency_overlap": bool(getattr(material, "use_transparency_overlap", False)),
        "show_transparent_back": bool(getattr(material, "show_transparent_back", False)),
        "node_types": [],
        "surface_output_count": 0,
        "surface_output_links": [],
        "principled": [],
        "image_textures": [],
    }
    if not material.use_nodes or material.node_tree is None:
        return row
    nt = material.node_tree
    for node in sorted(nt.nodes, key=lambda n: n.name):
        row["node_types"].append(str(node.bl_idname))
        if node.type == "OUTPUT_MATERIAL" and bool(getattr(node, "is_active_output", True)):
            row["surface_output_count"] += 1
            surface = node.inputs.get("Surface")
            if surface is not None:
                for link in surface.links:
                    row["surface_output_links"].append({
                        "from_node": link.from_node.name,
                        "from_type": link.from_node.bl_idname,
                        "from_socket": link.from_socket.identifier,
                    })
        if node.type == "BSDF_PRINCIPLED":
            row["principled"].append({
                "node": node.name,
                "base_color": _socket_row(node.inputs.get("Base Color")),
                "alpha": _socket_row(node.inputs.get("Alpha")),
                "normal": _socket_row(node.inputs.get("Normal")),
            })
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            image = node.image
            row["image_textures"].append({
                "node": node.name,
                "image_name": image.name,
                "packed": bool(image.packed_file),
                "source": str(image.source),
                "size": [int(v) for v in image.size],
                "channels": int(image.channels),
                "is_float": bool(image.is_float),
                "colour_space": str(image.colorspace_settings.name),
                "alpha_mode": str(image.alpha_mode),
                "interpolation": str(getattr(node, "interpolation", "UNKNOWN")),
                "extension": str(getattr(node, "extension", "UNKNOWN")),
                "projection": str(getattr(node, "projection", "UNKNOWN")),
            })
    row["node_types"] = sorted(set(row["node_types"]))
    return row


def main() -> None:
    a = _args()
    teacher = _json(a.teacher_scene)
    appearance = _json(a.appearance)
    if teacher.get("schema") != "RealSaS.TeacherSceneIR.v1" or teacher.get("subject_key") != a.subject_id:
        raise RuntimeError("SOURCE_RASTER_INVENTORY_TEACHER_BINDING")
    if teacher.get("qualification_minted") is not False or teacher.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS":
        raise RuntimeError("SOURCE_RASTER_INVENTORY_TEACHER_AUTHORITY")
    if appearance.get("schema") != "RealSaS.SourceAppearanceSamplerInputs.v1":
        raise RuntimeError("SOURCE_RASTER_INVENTORY_APPEARANCE_SCHEMA")
    if appearance.get("qualified_sampler") is not False or appearance.get("linear_rgb_and_effective_alpha_measurement") != "NOT_RUN":
        raise RuntimeError("SOURCE_RASTER_INVENTORY_INPUT_ALREADY_QUALIFIED")

    current_blend = Path(bpy.data.filepath).resolve()
    if not current_blend.is_file() or _sha(current_blend) != a.expected_blend_sha256:
        raise RuntimeError("SOURCE_RASTER_INVENTORY_BLEND_HASH_DRIFT")

    scene = bpy.context.scene
    materials = [_material_row(m) for m in sorted((m for m in bpy.data.materials if m), key=lambda m: m.name)]
    node_types = sorted({t for row in materials for t in row["node_types"]})
    render_methods = sorted({row["surface_render_method"] for row in materials})
    images = sorted({
        (img["image_name"], img["colour_space"], img["alpha_mode"], tuple(img["size"]))
        for row in materials for img in row["image_textures"]
    })
    mesh_objects = sorted(o.name for o in scene.objects if o.type == "MESH" and not o.hide_render)

    payload = {
        "schema": SCHEMA,
        "subject_id": a.subject_id,
        "teacher_scene_sha256": _sha(a.teacher_scene),
        "appearance_sha256": _sha(a.appearance),
        "source_scene_blend_sha256": a.expected_blend_sha256,
        "blender_version": str(bpy.app.version_string),
        "render_engine_saved": str(scene.render.engine),
        "film_transparent_saved": bool(scene.render.film_transparent),
        "view_transform_saved": str(scene.view_settings.view_transform),
        "look_saved": str(scene.view_settings.look),
        "exposure_saved": float(scene.view_settings.exposure),
        "gamma_saved": float(scene.view_settings.gamma),
        "world_present": scene.world is not None,
        "renderable_mesh_count": len(mesh_objects),
        "renderable_mesh_names": mesh_objects,
        "material_count": len(materials),
        "material_node_types": node_types,
        "surface_render_methods": render_methods,
        "image_count": len(images),
        "images": [
            {"image_name": n, "colour_space": c, "alpha_mode": am, "size": list(size)}
            for n, c, am, size in images
        ],
        "materials": materials,
        "render_executed": False,
        "material_graph_mutated": False,
        "qualification_minted": False,
        "student_target_allowed": False,
        "training_allowed": False,
        "authority": "RENDERER_CAPABILITY_INVENTORY_ONLY",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in {"materials", "images", "renderable_mesh_names"}}, sort_keys=True))


if __name__ == "__main__":
    main()
