#!/usr/bin/env python3
"""Run the proven source-ray renderer intervention court on TeacherSceneIR.v2.

The historical measurement implementation is intentionally left byte-for-byte
unchanged because its V1 FIT8 court is already sealed. This adapter changes only
the accepted V2 envelope and reconstructs the intervention triangle soup with the
source evaluator's exact mesh attributes (UV layers and color attributes). That
attribute transport is required for source-equivalence when imported materials use
vertex/corner color in addition to texture UVs.

``_sha`` remains untouched, so the measurement receipt binds the real V2 teacher
bytes rather than a rewritten compatibility document.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy
import numpy as np

# Blender executes --python entrypoints with its own embedded Python path. The
# released repository root is not implicitly importable there, so this adapter
# must establish the same standalone bootstrap used by the other Blender truth
# entrypoints before importing `tools.*`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.iris.truth.presentation_pass_domain_v1 import admit_presentation_pass_domain_v1
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as measured


_original_load = measured._load


def _load_with_v2_teacher(path):
    value = _original_load(path)
    if isinstance(value, dict) and value.get("schema") == "RealSaS.TeacherSceneIR.v2":
        if (
            value.get("authority") != "UNQUALIFIED_DERIVED_MEASUREMENTS"
            or value.get("qualification_minted") is not False
            or value.get("rest_contract") != "EXACT_ADMITTED_OBSERVATION_IMPORT_GEOMETRY"
            or (value.get("observation_rest_binding") or {}).get("status")
            != "PASS_EXACT_ADMITTED_IMPORT_GEOMETRY_REPLAY"
        ):
            raise RuntimeError("PRESENTATION_RAY_V2_TEACHER_NOT_EXACT_OBSERVATION_REST")
        compatible = dict(value)
        compatible["schema"] = "RealSaS.TeacherSceneIR.v1"
        return compatible
    return value


def _copy_uv_layers(
    source_mesh,
    target_mesh,
    source_triangles,
    global_faces,
    triangle_uv,
    triangle_uv_valid,
) -> None:
    source_layers = list(source_mesh.uv_layers)
    active = source_mesh.uv_layers.active
    if active is None:
        if any(bool(triangle_uv_valid[int(face_index)]) for face_index in global_faces):
            raise RuntimeError("PRESENTATION_RAY_V2_ACTIVE_UV_LAYER_MISSING")
        return

    # The teacher arrays were exported from this same evaluated source mesh.
    # Rebind that fact before transporting any additional UV layers.
    for local_index, face_index_raw in enumerate(global_faces):
        face_index = int(face_index_raw)
        tri = source_triangles[local_index]
        observed = np.asarray(
            [tuple(active.data[int(loop_index)].uv) for loop_index in tri.loops],
            dtype=np.float64,
        )
        if bool(triangle_uv_valid[face_index]):
            expected = np.asarray(triangle_uv[face_index], dtype=np.float64)
            if observed.shape != (3, 2) or not np.allclose(observed, expected, atol=2.0e-6, rtol=0):
                raise RuntimeError("PRESENTATION_RAY_V2_ACTIVE_UV_REBIND_DRIFT")
        elif np.isfinite(observed).all():
            raise RuntimeError("PRESENTATION_RAY_V2_ACTIVE_UV_VALIDITY_DRIFT")

    for source_layer in source_layers:
        target_layer = target_mesh.uv_layers.new(name=source_layer.name)
        for local_index, tri in enumerate(source_triangles):
            polygon = target_mesh.polygons[local_index]
            for corner, source_loop_index in enumerate(tri.loops):
                target_loop_index = polygon.loop_indices[corner]
                target_layer.data[target_loop_index].uv = tuple(
                    source_layer.data[int(source_loop_index)].uv
                )
        if bool(getattr(source_layer, "active_render", False)):
            try:
                target_layer.active_render = True
            except Exception as exc:
                raise RuntimeError("PRESENTATION_RAY_V2_UV_ACTIVE_RENDER_TRANSPORT") from exc

    try:
        target_mesh.uv_layers.active_index = int(source_mesh.uv_layers.active_index)
    except Exception as exc:
        raise RuntimeError("PRESENTATION_RAY_V2_UV_ACTIVE_INDEX_TRANSPORT") from exc


def _copy_source_color_attributes(source_mesh, target_mesh, source_triangles) -> None:
    source_attributes = list(source_mesh.color_attributes)
    for source_attribute in source_attributes:
        domain = str(source_attribute.domain)
        data_type = str(source_attribute.data_type)
        target_attribute = target_mesh.color_attributes.new(
            name=source_attribute.name,
            type=data_type,
            domain=domain,
        )
        color_property = "color_srgb" if data_type == "BYTE_COLOR" else "color"

        if domain == "CORNER":
            for local_index, tri in enumerate(source_triangles):
                polygon = target_mesh.polygons[local_index]
                for corner, source_loop_index in enumerate(tri.loops):
                    target_loop_index = polygon.loop_indices[corner]
                    source_item = source_attribute.data[int(source_loop_index)]
                    target_item = target_attribute.data[target_loop_index]
                    value = getattr(source_item, color_property, source_item.color)
                    setattr(target_item, color_property, tuple(value))
        elif domain == "POINT":
            for local_index, tri in enumerate(source_triangles):
                for corner, source_vertex_index in enumerate(tri.vertices):
                    target_vertex_index = 3 * local_index + corner
                    source_item = source_attribute.data[int(source_vertex_index)]
                    target_item = target_attribute.data[target_vertex_index]
                    value = getattr(source_item, color_property, source_item.color)
                    setattr(target_item, color_property, tuple(value))
        elif domain == "FACE":
            for local_index, tri in enumerate(source_triangles):
                source_item = source_attribute.data[int(tri.polygon_index)]
                target_item = target_attribute.data[local_index]
                value = getattr(source_item, color_property, source_item.color)
                setattr(target_item, color_property, tuple(value))
        else:
            raise RuntimeError(
                "PRESENTATION_RAY_V2_COLOR_ATTRIBUTE_DOMAIN_UNSUPPORTED:"
                + source_attribute.name
                + ":"
                + domain
            )

    # ShaderNodeVertexColor with an empty layer_name resolves through the mesh's
    # active/render color attribute. Preserve both selectors exactly when present.
    source_group = source_mesh.color_attributes
    target_group = target_mesh.color_attributes
    for property_name in ("active_color_index", "render_color_index"):
        if hasattr(source_group, property_name):
            try:
                setattr(target_group, property_name, int(getattr(source_group, property_name)))
            except Exception as exc:
                raise RuntimeError(
                    "PRESENTATION_RAY_V2_COLOR_SELECTOR_TRANSPORT:" + property_name
                ) from exc


def _build_triangle_soup_with_source_attributes(arrays, owners, transparent):
    faces = arrays["faces"]
    vertices = arrays["rest_vertices_source_world"]
    face_owner = arrays["face_owner_privileged"]
    face_material = arrays["face_material_privileged"]
    triangle_uv = arrays["triangle_uv"]
    triangle_uv_valid = arrays["triangle_uv_valid"]
    if len(owners) == 0 or np.any(face_owner < 0) or np.any(face_owner >= len(owners)):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_OWNER_BINDING")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    global_map = {}
    soup_objects = []
    for owner_index, source_obj in enumerate(owners):
        global_faces = np.flatnonzero(face_owner == owner_index)
        if not len(global_faces):
            continue

        evaluated = source_obj.evaluated_get(depsgraph)
        source_mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            source_mesh.calc_loop_triangles()
            source_triangles = list(source_mesh.loop_triangles)
            if len(source_triangles) != len(global_faces):
                raise RuntimeError("PRESENTATION_RAY_V2_OWNER_TRIANGLE_CARDINALITY_DRIFT")

            local_vertices = []
            local_faces = []
            for local_index, face_index_raw in enumerate(global_faces):
                face_index = int(face_index_raw)
                source_triangle = source_triangles[local_index]
                if int(source_triangle.material_index) != int(face_material[face_index]):
                    raise RuntimeError("PRESENTATION_RAY_V2_FACE_MATERIAL_REBIND_DRIFT")
                local_vertices.extend(vertices[faces[face_index]].tolist())
                local_faces.append((3 * local_index, 3 * local_index + 1, 3 * local_index + 2))

            mesh = bpy.data.meshes.new("RealSaS_RaySoupMesh_" + source_obj.name)
            mesh.from_pydata(local_vertices, [], local_faces)
            mesh.update()
            for slot in source_obj.material_slots:
                if slot.material is None:
                    raise RuntimeError("SOURCE_RAY_APPEARANCE_NULL_MATERIAL_SLOT:" + source_obj.name)
                mesh.materials.append(slot.material)
            transparent_index = len(mesh.materials)
            mesh.materials.append(transparent)

            _copy_uv_layers(
                source_mesh,
                mesh,
                source_triangles,
                global_faces,
                triangle_uv,
                triangle_uv_valid,
            )
            _copy_source_color_attributes(source_mesh, mesh, source_triangles)

            for local_index, face_index_raw in enumerate(global_faces):
                face_index = int(face_index_raw)
                material_index = int(face_material[face_index])
                if not 0 <= material_index < transparent_index:
                    raise RuntimeError("SOURCE_RAY_APPEARANCE_FACE_MATERIAL_BINDING")
                polygon = mesh.polygons[local_index]
                polygon.material_index = material_index
                global_map[face_index] = (
                    mesh,
                    local_index,
                    material_index,
                    transparent_index,
                )

            mesh.update()
            obj = bpy.data.objects.new("RealSaS_RaySoup_" + source_obj.name, mesh)
            bpy.context.scene.collection.objects.link(obj)
            soup_objects.append(obj)
        finally:
            evaluated.to_mesh_clear()

    if len(global_map) != len(faces):
        raise RuntimeError("SOURCE_RAY_APPEARANCE_SOUP_FACE_COVERAGE")
    bpy.context.view_layer.update()
    return tuple(soup_objects), global_map


def _argument_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
        value = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("PRESENTATION_RAY_ARGUMENT_MISSING:" + flag) from exc
    return Path(value).resolve()


def _admit_preregistered_pass_domain_after_raw_failures() -> dict:
    output = _argument_path("--output")
    # Platform layout is <run_root>/artifacts/48_PRESENTATION_RESPONSE_TRUTH/PACKED_X.json.
    # Fail closed outside that exact Attempt-bound layout rather than consulting
    # a mutable repository-global policy file.
    try:
        run_root = output.parents[2]
    except IndexError as exc:
        raise RuntimeError("PRESENTATION_RAY_RUN_ROOT_UNRESOLVED") from exc
    manifest_path = run_root / "run_manifest.json"
    if not output.is_file() or not manifest_path.is_file():
        raise RuntimeError("PRESENTATION_RAY_PASS_DOMAIN_EVIDENCE_MISSING")
    candidate = json.loads(output.read_text())
    run_manifest = json.loads(manifest_path.read_text())
    try:
        receipt = admit_presentation_pass_domain_v1(candidate, run_manifest)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main() -> None:
    measured._load = _load_with_v2_teacher
    measured._build_triangle_soup = _build_triangle_soup_with_source_attributes
    try:
        measured.main()
    except RuntimeError as exc:
        # The sealed historical court writes the complete raw candidate before
        # raising when any individual ray fails scalar factorization. Stage48
        # preregistered a subject-level minimum PASS-ray domain, so preserve the
        # failed rays as evidence and return success only when the exact Attempt
        # manifest's minimum is independently satisfied.
        if str(exc) != "SOURCE_RAY_APPEARANCE_INTERVENTION_REPLAY_FAILED":
            raise
        _admit_preregistered_pass_domain_after_raw_failures()


if __name__ == "__main__":
    main()
