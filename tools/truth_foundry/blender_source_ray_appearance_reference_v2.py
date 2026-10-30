#!/usr/bin/env python3
"""Attribute-complete V2 wrapper for the independent Stage48 source reference.

The sealed V1 reference intentionally renders exact teacher triangles with a fresh
source-observation material evaluator.  V1 copied teacher UVs but did not carry
source mesh color attributes into those isolated triangles.  Materials that read
``ShaderNodeVertexColor`` (for example Image Texture × Color Attribute) therefore
rendered black in the independent reference even when the packed evaluator was
correct.

V2 leaves the V1 scientific court and tolerances unchanged.  It only transports
the evaluated source mesh attributes required by the material evaluator onto the
isolated exact-teacher triangle, after fail-closed owner/triangle/material/world-
geometry/active-UV rebinding.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.truth_foundry import blender_source_ray_appearance_reference_v1 as base


_ORIGINAL_FULL_SUBJECT = base.orbit.full_subject
_FACE_TRANSPORT: dict[int, dict] | None = None
_FACE_RE = re.compile(r"^RealSaS_SourceReference_F(\d+)_H\d+$")


def _argument_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
        raw = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ARGUMENT_MISSING:" + flag) from exc
    return Path(raw).expanduser().resolve()


def _color_tuple(item, data_type: str) -> tuple[float, float, float, float]:
    property_name = "color_srgb" if data_type == "BYTE_COLOR" else "color"
    value = getattr(item, property_name, None)
    if value is None:
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_PROPERTY:" + data_type)
    row = tuple(map(float, value))
    if len(row) != 4 or not np.isfinite(row).all():
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_NONFINITE")
    return row


def _build_face_transport(meshes) -> dict[int, dict]:
    geometry = _argument_path("--geometry-npz")
    vertices, faces, face_owner, face_slot, triangle_uv, uv_valid = base._load_geometry(geometry)
    if not meshes or np.any(face_owner < 0) or np.any(face_owner >= len(meshes)):
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_OWNER_CARDINALITY_DRIFT")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    transport: dict[int, dict] = {}
    for owner_index, source_obj in enumerate(meshes):
        global_faces = np.flatnonzero(face_owner == owner_index)
        evaluated = source_obj.evaluated_get(depsgraph)
        source_mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            source_mesh.calc_loop_triangles()
            triangles = list(source_mesh.loop_triangles)
            if len(triangles) != len(global_faces):
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_OWNER_TRIANGLE_CARDINALITY_DRIFT")

            uv_layers = list(source_mesh.uv_layers)
            active_uv_index = int(source_mesh.uv_layers.active_index) if uv_layers else -1
            if active_uv_index >= len(uv_layers):
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ACTIVE_UV_INDEX_DRIFT")

            color_group = source_mesh.color_attributes
            color_attributes = list(color_group)
            active_color_index = int(getattr(color_group, "active_color_index", -1)) if color_attributes else -1
            render_color_index = int(getattr(color_group, "render_color_index", -1)) if color_attributes else -1
            for selector, value in (("active", active_color_index), ("render", render_color_index)):
                if value >= len(color_attributes):
                    raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_SELECTOR_DRIFT:" + selector)

            for local_index, face_index_raw in enumerate(global_faces):
                face_index = int(face_index_raw)
                tri = triangles[local_index]
                if int(tri.material_index) != int(face_slot[face_index]):
                    raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_MATERIAL_REBIND_DRIFT")

                observed_world = np.asarray(
                    [tuple(float(x) for x in (evaluated.matrix_world @ source_mesh.vertices[int(v)].co)) for v in tri.vertices],
                    dtype=np.float64,
                )
                expected_world = np.asarray(vertices[faces[face_index]], dtype=np.float64)
                if observed_world.shape != (3, 3) or not np.allclose(
                    observed_world, expected_world, atol=2.0e-6, rtol=0.0
                ):
                    raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_WORLD_REBIND_DRIFT")

                uv_rows = []
                for layer in uv_layers:
                    values = [
                        tuple(map(float, layer.data[int(loop_index)].uv))
                        for loop_index in tri.loops
                    ]
                    if np.asarray(values, dtype=np.float64).shape != (3, 2):
                        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_UV_LAYOUT")
                    uv_rows.append(
                        {
                            "name": str(layer.name),
                            "values": values,
                            "active_render": bool(getattr(layer, "active_render", False)),
                        }
                    )

                if bool(uv_valid[face_index]):
                    if active_uv_index < 0:
                        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ACTIVE_UV_MISSING")
                    observed_uv = np.asarray(uv_rows[active_uv_index]["values"], dtype=np.float64)
                    expected_uv = np.asarray(triangle_uv[face_index], dtype=np.float64)
                    if not np.allclose(observed_uv, expected_uv, atol=2.0e-6, rtol=0.0):
                        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ACTIVE_UV_REBIND_DRIFT")
                elif active_uv_index >= 0:
                    observed_uv = np.asarray(uv_rows[active_uv_index]["values"], dtype=np.float64)
                    if np.isfinite(observed_uv).all():
                        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ACTIVE_UV_VALIDITY_DRIFT")

                color_rows = []
                for attribute in color_attributes:
                    domain = str(attribute.domain)
                    data_type = str(attribute.data_type)
                    if domain == "CORNER":
                        values = [_color_tuple(attribute.data[int(i)], data_type) for i in tri.loops]
                    elif domain == "POINT":
                        values = [_color_tuple(attribute.data[int(i)], data_type) for i in tri.vertices]
                    elif domain == "FACE":
                        values = [_color_tuple(attribute.data[int(tri.polygon_index)], data_type)]
                    else:
                        raise RuntimeError(
                            "SOURCE_RAY_REFERENCE_V2_COLOR_ATTRIBUTE_DOMAIN_UNSUPPORTED:"
                            + str(attribute.name)
                            + ":"
                            + domain
                        )
                    color_rows.append(
                        {
                            "name": str(attribute.name),
                            "domain": domain,
                            "data_type": data_type,
                            "values": values,
                        }
                    )

                transport[face_index] = {
                    "face": np.asarray(faces[face_index], dtype=np.int64).tolist(),
                    "uv_layers": uv_rows,
                    "active_uv_index": active_uv_index,
                    "color_attributes": color_rows,
                    "active_color_index": active_color_index,
                    "render_color_index": render_color_index,
                }
        finally:
            evaluated.to_mesh_clear()

    if set(transport) != set(range(len(faces))):
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_COVERAGE_DRIFT")
    return transport


def _capture_full_subject():
    global _FACE_TRANSPORT
    result = _ORIGINAL_FULL_SUBJECT()
    _FACE_TRANSPORT = _build_face_transport(tuple(result[0]))
    return result


def _audit_material_attribute_consumers(material: bpy.types.Material, transport: dict) -> None:
    if not material.use_nodes or material.node_tree is None:
        return
    uv_names = {row["name"] for row in transport["uv_layers"]}
    color_names = {row["name"] for row in transport["color_attributes"]}
    for node in material.node_tree.nodes:
        if node.type == "VERTEX_COLOR":
            layer_name = str(getattr(node, "layer_name", "") or "")
            if layer_name:
                if layer_name not in color_names:
                    raise RuntimeError("SOURCE_RAY_REFERENCE_V2_VERTEX_COLOR_LAYER_MISSING:" + layer_name)
            elif int(transport["active_color_index"]) < 0:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_ACTIVE_COLOR_REQUIRED")
        elif node.type == "UVMAP":
            uv_map = str(getattr(node, "uv_map", "") or "")
            if uv_map and uv_map not in uv_names:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_UV_MAP_MISSING:" + uv_map)
        elif node.type == "ATTRIBUTE":
            name = str(getattr(node, "attribute_name", "") or "")
            if name and name not in color_names and name not in uv_names:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_GENERIC_ATTRIBUTE_UNSUPPORTED:" + name)


def _make_triangle_with_source_attributes(
    vertices: np.ndarray,
    face: np.ndarray,
    triangle_uv: np.ndarray,
    uv_valid: bool,
    material: bpy.types.Material,
    name: str,
) -> bpy.types.Object:
    if _FACE_TRANSPORT is None:
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_TRANSPORT_NOT_CAPTURED")
    match = _FACE_RE.match(name)
    if match is None:
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_NAME")
    face_index = int(match.group(1))
    transport = _FACE_TRANSPORT.get(face_index)
    if transport is None:
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_NOT_BOUND")
    if list(map(int, face)) != list(map(int, transport["face"])):
        raise RuntimeError("SOURCE_RAY_REFERENCE_V2_FACE_ARGUMENT_DRIFT")

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices[face].tolist(), [], [(0, 1, 2)])
    mesh.update()
    mesh.materials.append(material)
    polygon = mesh.polygons[0]

    for row in transport["uv_layers"]:
        layer = mesh.uv_layers.new(name=row["name"])
        values = row["values"]
        for corner, loop_index in enumerate(polygon.loop_indices):
            layer.data[loop_index].uv = tuple(map(float, values[corner]))
        if row["active_render"]:
            try:
                layer.active_render = True
            except Exception as exc:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_UV_ACTIVE_RENDER_TRANSPORT") from exc
    active_uv_index = int(transport["active_uv_index"])
    if active_uv_index >= 0:
        try:
            mesh.uv_layers.active_index = active_uv_index
        except Exception as exc:
            raise RuntimeError("SOURCE_RAY_REFERENCE_V2_UV_ACTIVE_INDEX_TRANSPORT") from exc

    for row in transport["color_attributes"]:
        attribute = mesh.color_attributes.new(
            name=row["name"], type=row["data_type"], domain=row["domain"]
        )
        property_name = "color_srgb" if row["data_type"] == "BYTE_COLOR" else "color"
        values = row["values"]
        if row["domain"] in {"CORNER", "POINT"}:
            if len(values) != 3:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_TRIANGLE_LAYOUT")
            indices = list(polygon.loop_indices) if row["domain"] == "CORNER" else [0, 1, 2]
            for target_index, value in zip(indices, values):
                setattr(attribute.data[int(target_index)], property_name, tuple(map(float, value)))
        elif row["domain"] == "FACE":
            if len(values) != 1:
                raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_FACE_LAYOUT")
            setattr(attribute.data[0], property_name, tuple(map(float, values[0])))
        else:
            raise RuntimeError("SOURCE_RAY_REFERENCE_V2_COLOR_DOMAIN_INTERNAL")

    if transport["color_attributes"]:
        for property_name in ("active_color_index", "render_color_index"):
            value = int(transport[property_name])
            if value >= 0:
                try:
                    setattr(mesh.color_attributes, property_name, value)
                except Exception as exc:
                    raise RuntimeError(
                        "SOURCE_RAY_REFERENCE_V2_COLOR_SELECTOR_TRANSPORT:" + property_name
                    ) from exc

    _audit_material_attribute_consumers(material, transport)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _output_path() -> Path:
    return _argument_path("--output")


def main() -> None:
    base.orbit.full_subject = _capture_full_subject
    base._make_triangle = _make_triangle_with_source_attributes
    base.main()

    output = _output_path()
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["transport_revision"] = 2
    payload["source_attribute_transport"] = {
        "owner_triangle_material_world_rebind": "FAIL_CLOSED",
        "uv_layers": "ALL_EVALUATED_SOURCE_UV_LAYERS__ACTIVE_AND_RENDER_SELECTOR_PRESERVED",
        "color_attributes": "ALL_EVALUATED_SOURCE_COLOR_ATTRIBUTES__CORNER_POINT_FACE__ACTIVE_AND_RENDER_SELECTOR_PRESERVED",
        "sealed_v1_reference_logic_changed": False,
        "tolerance_relaxed": False,
    }
    output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS_SOURCE_RAY_REFERENCE_V2_ATTRIBUTE_TRANSPORT",
                "subject_id": payload.get("subject_id"),
                "hit_count": payload.get("hit_count"),
                "transport_revision": 2,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
