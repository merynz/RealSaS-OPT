"""Common FBX/GLTF rest materializer; emits candidates, never source admission."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.observation import blender_controlled_8view_v1 as orbit  # noqa: E402
from tools.observation.source_scene_contract_v2 import (  # noqa: E402
    prepare_bind_rest_gltf, sha256, validate_imported_scene_snapshot,
)


def arguments():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--frame-scale", type=float, default=1.12)
    parser.add_argument("--minimum-border-margin-px", type=int, default=8)
    parser.add_argument("--material-mode", choices=("source", "unlit_preserve_alpha"),
                        default="unlit_preserve_alpha")
    return parser.parse_args(raw)


def matrix_rows(value):
    return [[float(x) for x in row] for row in value]


def animation_counts(obj):
    blocks = [obj, obj.data, getattr(obj.data, "shape_keys", None)]
    animation = [block.animation_data for block in blocks
                 if block is not None and getattr(block, "animation_data", None)]
    return {"active_action_count": sum(int(bool(row.action)) for row in animation),
            "active_nla_track_count": sum(len(row.nla_tracks) for row in animation),
            "driver_count": sum(len(row.drivers) for row in animation)}


def import_rest(source: Path, output: Path) -> dict:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if source.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source), automatic_bone_orientation=False,
                                 use_anim=False)
        receipt = {"source_sha256": sha256(source), "operation": "FBX_IMPORT_WITHOUT_ANIMATION",
                   "importer_options": {"automatic_bone_orientation": False, "use_anim": False},
                   "native_animation_inventory_status": "PENDING_SEPARATE_ANIMATED_IMPORT"}
    elif source.suffix.lower() in (".gltf", ".glb"):
        derivative = output / "rest_import_source" / source.name
        receipt = prepare_bind_rest_gltf(source, derivative)
        bpy.ops.import_scene.gltf(filepath=str(derivative), guess_original_bind_pose=True)
        receipt["importer_options"] = {"filepath_role": "EXPLICIT_BIND_REST_DERIVATIVE",
                                       "guess_original_bind_pose": True}
    else:
        raise RuntimeError("SOURCE_REST_IMPORT_UNSUPPORTED_FORMAT")
    # No frame0/action evaluation is permitted in this default/rest import.
    for obj in bpy.context.scene.objects:
        if any(animation_counts(obj).values()):
            raise RuntimeError("SOURCE_REST_IMPORT_CREATED_ANIMATION_BINDING")
        if obj.type == "ARMATURE":
            obj.data.pose_position = "REST"
    bpy.context.view_layer.update()
    helpers = {bone.custom_shape for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
               for bone in obj.pose.bones if bone.custom_shape is not None}
    for helper in helpers:
        helper.hide_render = True
        helper["realsas_import_helper_role"] = "ARMATURE_CUSTOM_SHAPE"
    receipt["excluded_import_controls"] = sorted(obj.name for obj in helpers)
    receipt["raw_source_mutated"] = False
    receipt["pose_policy"] = "SOURCE_BIND_REST__NO_ANIMATION"
    receipt["source_bind_parity_claimed"] = False
    return receipt


def scene_snapshot(source_sha: str, normalization: Matrix) -> dict:
    objects = sorted(bpy.context.scene.objects, key=lambda obj: obj.name)
    ids = {obj.name: f"{source_sha}:node:{index}" for index, obj in enumerate(objects)}
    nodes = []
    for obj in objects:
        world = obj.matrix_world.copy()
        local = obj.parent.matrix_world.inverted() @ world if obj.parent else world.copy()
        nodes.append({
            "node_id": ids[obj.name], "source_import_name": obj.name, "type": obj.type,
            "parent_node_id": ids[obj.parent.name] if obj.parent else None,
            "parent_type": obj.parent_type, "parent_bone": obj.parent_bone or None,
            "node_to_source_world": matrix_rows(world), "effective_node_to_parent": matrix_rows(local),
            "authored_import_matrix_basis": matrix_rows(obj.matrix_basis),
            "matrix_parent_inverse": matrix_rows(obj.matrix_parent_inverse),
            "node_to_canonical": matrix_rows(normalization @ world),
            "hide_render": bool(obj.hide_render),
            "import_helper_role": obj.get("realsas_import_helper_role"),
            "animation_capture_scope": "OBJECT_DATA_AND_SHAPE_KEYS",
            **animation_counts(obj),
        })
    bones = []
    for obj in objects:
        if obj.type != "ARMATURE":
            continue
        for bone in obj.data.bones:
            world = obj.matrix_world @ bone.matrix_local
            bones.append({
                "armature_node_id": ids[obj.name], "source_bone_name": bone.name,
                "source_parent_bone_name": bone.parent.name if bone.parent else None,
                "use_deform": bool(bone.use_deform),
                "rest_joint_to_source_world": matrix_rows(world),
                "rest_joint_to_canonical": matrix_rows(normalization @ world),
                "head_source_world": list(obj.matrix_world @ bone.head_local),
                "tail_source_world": list(obj.matrix_world @ bone.tail_local),
                "head_canonical": list(normalization @ obj.matrix_world @ bone.head_local),
                "tail_canonical": list(normalization @ obj.matrix_world @ bone.tail_local),
            })
    return {
        "schema": "RealSaS.ImportedSourceSceneSnapshot.v2", "source_sha256": source_sha,
        "coordinate_owner": "BLENDER_FORMAT_IMPORTER__RECORDED_WORLD_Z_UP",
        "source_world_to_canonical": matrix_rows(normalization),
        "canonical_to_source_world": matrix_rows(normalization.inverted()),
        "nodes": nodes, "bones": bones,
        "parent_capture_scope": "EFFECTIVE_IMPORTED_OBJECT_HIERARCHY__BONE_PARENT_METADATA_PRESERVED",
        "independent_source_bind_parity_claimed": False,
        "canonical_product_joint_ids_minted": False,
    }


def preserve_alpha_unlit(meshes) -> list[dict]:
    """Replace lighting response at the BSDF, retaining upstream/downstream graphs."""
    materials = {slot.material.name: slot.material for obj in meshes for slot in obj.material_slots
                 if slot.material is not None}
    receipts = []
    for name, material in sorted(materials.items()):
        material.use_nodes = True
        tree = material.node_tree
        principled = [node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"]
        emissions = [node for node in tree.nodes if node.type == "EMISSION"]
        if not principled and emissions:
            receipts.append({"material": name, "operation": "NATIVE_EMISSION_GRAPH_PRESERVED"})
            continue
        if len(principled) != 1:
            raise RuntimeError(f"SOURCE_BASE_COLOR_SHADER_BINDING_AMBIGUOUS:{name}")
        bsdf = principled[0]
        color = bsdf.inputs.get("Base Color")
        alpha = bsdf.inputs.get("Alpha")
        if color is None or alpha is None:
            raise RuntimeError(f"SOURCE_BASE_COLOR_ALPHA_SOCKET_MISSING:{name}")
        consumers = [link.to_socket for link in list(bsdf.outputs["BSDF"].links)]
        if not consumers:
            raise RuntimeError(f"SOURCE_BASE_COLOR_SHADER_UNCONSUMED:{name}")
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 1.0
        if color.is_linked:
            tree.links.new(color.links[0].from_socket, emission.inputs["Color"])
        else:
            emission.inputs["Color"].default_value = tuple(color.default_value)
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        mix = tree.nodes.new("ShaderNodeMixShader")
        blend_mode = getattr(material, "blend_method", None)
        # Legacy Eevee OPAQUE ignores the dormant BSDF alpha socket; Cycles
        # does not. Transport that explicit imported mode rather than silently
        # converting opaque surfaces to transparent ones in the CPU producer.
        # Independent source/native material parity is still a separate gate.
        if blend_mode == "OPAQUE":
            mix.inputs[0].default_value = 1.0
            alpha_policy = "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE"
        elif alpha.is_linked:
            tree.links.new(alpha.links[0].from_socket, mix.inputs[0])
            alpha_policy = "IMPORTED_SHADER_ALPHA_LINK_RETAINED"
        else:
            mix.inputs[0].default_value = float(alpha.default_value)
            alpha_policy = "IMPORTED_SHADER_ALPHA_CONSTANT_RETAINED"
        tree.links.new(transparent.outputs[0], mix.inputs[1])
        tree.links.new(emission.outputs[0], mix.inputs[2])
        for consumer in consumers:
            tree.links.new(mix.outputs[0], consumer)
        receipts.append({"material": name, "operation": "BSDF_LIGHT_RESPONSE_TO_EMISSION__ALPHA_GRAPH_RETAINED",
                         "source_color_linked": bool(color.is_linked),
                         "source_alpha_linked": bool(alpha.is_linked),
                         "source_alpha_default": float(alpha.default_value),
                         "imported_blend_method": blend_mode, "alpha_policy": alpha_policy,
                         "downstream_shader_consumer_count": len(consumers),
                         "pixel_alpha_parity_status": "PENDING_INDEPENDENT_REFERENCE"})
    return receipts


def configure(resolution):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.seed = 0
    scene.cycles.use_denoising = False
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 1.0
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)


def render_all(output, center, half, resolution, minimum_margin):
    """One fixed orbit; explicit clipping scales with the whole source subject."""
    data = bpy.data.cameras.new("RealSaS_Controlled8View_Camera_v2")
    data.type = "ORTHO"
    data.ortho_scale = 2.0 * half
    distance = max(4.0 * half, 1.0)
    data.clip_start = max(1e-6, distance - 2.0 * half)
    data.clip_end = distance + 2.0 * half
    cam = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cameras, views = [], []
    for index, (label, yaw) in enumerate(zip(orbit.VIEW_ORDER, orbit.VIEW_YAW_DEG)):
        started = time.perf_counter()
        radial, forward, right, up = orbit.basis(yaw)
        origin = center + radial * distance
        cam.location = origin
        cam.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        png = output / f"V{index}.png"
        bpy.context.scene.render.filepath = str(png)
        bpy.ops.render.render(write_still=True)
        rendered = bpy.data.images.load(str(png), check_existing=False)
        try:
            if tuple(rendered.size) != (resolution, resolution):
                raise RuntimeError(f"SOURCE_REST_RESOLUTION_DRIFT:V{index}")
            pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
            rendered.pixels.foreach_get(pixels)
            alpha = pixels.reshape(resolution, resolution, 4)[::-1, :, 3]
            mask = np.asarray(alpha > 1e-6, dtype=np.uint8)
        finally:
            bpy.data.images.remove(rendered)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            raise RuntimeError(f"SOURCE_REST_EMPTY_FOREGROUND:V{index}")
        margin = int(min(xs.min(), ys.min(), resolution - 1 - xs.max(), resolution - 1 - ys.max()))
        if margin < minimum_margin:
            raise RuntimeError(f"SOURCE_REST_BORDER_MARGIN_FAIL:V{index}:{margin}<{minimum_margin}")
        mask_path = output / f"V{index}.mask.bin"
        mask_path.write_bytes(mask.tobytes())
        cameras.append({
            "schema_version": "RealSaS.FullSurfaceCameraProjection.v3",
            "view_id": f"V{index}", "view_index": index, "origin": list(origin),
            "right": list(right), "screen_up": list(up), "forward": list(forward),
            "half_extent": half, "resolution": resolution,
            "clip_start": data.clip_start, "clip_end": data.clip_end,
        })
        views.append({
            "view_index": index, "view_id": f"V{index}", "direction_label": label, "yaw_deg": yaw,
            "raster": {"path": str(png), "sha256": sha256(png)},
            "foreground_mask": {"path": str(mask_path), "sha256": sha256(mask_path)},
            "alpha_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "foreground_pixel_count": int(mask.sum()), "border_margin_px": margin,
        })
        print(f"[RENDER] V{index} {label} foreground={mask.sum()} margin_px={margin} "
              f"seconds={time.perf_counter() - started:.1f}", flush=True)
    return cameras, views


def main():
    args = arguments()
    source = args.source.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    if not source.is_file() or sha256(source) != args.expected_source_sha256:
        raise RuntimeError("SOURCE_REST_MATERIALIZER_EXACT_SOURCE_HASH_REQUIRED")
    if output == source.parent or output.is_relative_to(source.parent):
        raise RuntimeError("SOURCE_REST_OUTPUT_MUST_BE_OUTSIDE_RAW_DIRECTORY")
    if args.resolution < 1 or args.minimum_border_margin_px < 1:
        raise RuntimeError("SOURCE_REST_RASTER_POLICY_INVALID")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("SOURCE_REST_OUTPUT_MUST_BE_FRESH__NO_STALE_RECEIPT_REUSE")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    print(f"[SOURCE] {args.subject_id} format={source.suffix} sha256={sha256(source)}", flush=True)
    import_receipt = import_rest(source, output)
    meshes, points, by_object = orbit.full_subject()
    center, half, frame = orbit.solve_frame(points, args.frame_scale)
    normalization = Matrix.Scale(1. / half, 4) @ Matrix.Translation(-center)
    snapshot = scene_snapshot(sha256(source), normalization)
    internal = validate_imported_scene_snapshot(snapshot)
    snapshot_path = output / "SOURCE_SCENE_SNAPSHOT.json"
    orbit.dump(snapshot_path, snapshot)
    source_materials = orbit.material_audit()
    material_transform = preserve_alpha_unlit(meshes) if args.material_mode == "unlit_preserve_alpha" else []
    configure(args.resolution)
    cameras, views = render_all(output, center, half, args.resolution, args.minimum_border_margin_px)
    camera_path = output / "camera_bundle.json"
    orbit.dump(camera_path, {
        "schema": "RealSaS.CameraProjectionBundle.v1", "view_order": list(orbit.VIEW_ORDER),
        "common_center_xyz": list(center), "common_half_extent": half, "cameras": cameras,
        "per_view_recenter": False, "per_view_pose_change": False,
    })
    receipt = {
        "schema": "RealSaS.Controlled8ViewSourceRestCandidate.v2",
        "status": "ENGINEERING_MATERIALIZED__SOURCE_ADMISSION_PENDING",
        "subject_id": args.subject_id, "source_sha256": sha256(source),
        "source_bytes": source.stat().st_size, "import": import_receipt,
        "blender_version": bpy.app.version_string,
        "blender_binary_sha256": sha256(Path(bpy.app.binary_path)),
        "render_policy": {"engine": "CYCLES", "device": "CPU", "samples": 16, "seed": 0,
                          "denoising": False, "view_transform": "Standard", "look": "None",
                          "exposure": 0.0, "gamma": 1.0, "color_depth": 8},
        "source_lit_appearance_parity_status": "PENDING_INDEPENDENT_REFERENCE",
        "materializer_sha256": sha256(Path(__file__)),
        "scene_contract_sha256": sha256(Path(__file__).with_name("source_scene_contract_v2.py")),
        "shared_orbit_materializer_sha256": sha256(Path(orbit.__file__)),
        "pose_address": "SOURCE_BIND_REST", "native_frame0_is_rest_assumed": False,
        "source_scene_snapshot": {"sha256": sha256(snapshot_path), "file": snapshot_path.name},
        "internal_consistency": internal, "frame_report": frame, "resolution": args.resolution,
        "mesh_objects": orbit.mesh_rows(meshes, by_object),
        "source_materials": source_materials, "material_mode": args.material_mode,
        "material_transform": material_transform, "views": views,
        "camera_bundle": {"sha256": sha256(camera_path), "file": camera_path.name},
        "source_import_pose_parity_status": "PENDING_INDEPENDENT_REFERENCE",
        "native_motion_admission_status": "PENDING",
        "training_truth_export_status": "PENDING",
        "training_admission_claimed": False, "product_pass_claimed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    orbit.dump(output / "SOURCE_REST_RENDER_CANDIDATE.json", receipt)
    print(f"[REST] {args.subject_id} nodes={len(snapshot['nodes'])} bones={len(snapshot['bones'])} "
          f"meshes={len(meshes)} views={len(views)} margin_px={min(v['border_margin_px'] for v in views)} "
          f"seconds={receipt['wall_seconds']:.1f} admission=PENDING", flush=True)


if __name__ == "__main__":
    main()
