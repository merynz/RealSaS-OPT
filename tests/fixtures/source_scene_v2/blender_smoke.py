"""Blender engineering smoke: animated roots, nested attachments and shader alpha.

Run with Blender --background --factory-startup --python-exit-code 1 --python.
Synthetic evidence only; this does not admit any Fit8 subject.
"""
from pathlib import Path
import json
import sys
import tempfile

import bpy
from mathutils import Matrix, Vector
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.observation import blender_controlled_8view_v2 as producer
from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot


def cube(name, parent, location):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.name = name
    obj.parent = parent
    obj.location = location
    return obj


def scene_import(tmp):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    root = bpy.data.objects.new("AuthoredRoot", None)
    bpy.context.scene.collection.objects.link(root)
    root.location = (3, 4, 5)
    root.scale = (2, 3, 4)
    root.rotation_euler.z = .3
    body = cube("Body", root, (1, 2, 3))
    attachment = cube("RigidAttachment", body, (.4, -.2, .7))
    bpy.context.view_layer.update()
    expected = {obj.name: np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
                for obj in (body, attachment)}
    raw = tmp / "raw"; raw.mkdir()
    source = raw / "scene.gltf"
    bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLTF_SEPARATE", export_animations=False)
    doc = json.loads(source.read_text())
    index = next(i for i, node in enumerate(doc["nodes"]) if node.get("name") == "AuthoredRoot")
    # Deliberately invalid animation data is safe only if absent BEFORE import.
    doc["animations"] = [{"name": "RootActionMustNotBeEvaluated",
                          "samplers": [{"input": 9999, "output": 9999}],
                          "channels": [{"sampler": 0, "target": {"node": index, "path": "translation"}}]}]
    source.write_text(json.dumps(doc))
    original = source.read_bytes()
    imported = producer.import_rest(source, tmp / "derived")
    assert source.read_bytes() == original
    assert imported["removed_animation_count"] == 1
    assert imported["source_bind_parity_claimed"] is False
    normalization = Matrix.Scale(.2, 4) @ Matrix.Translation(Vector((-3, -4, -5)))
    snap = producer.scene_snapshot(producer.sha256(source), normalization)
    assert validate_imported_scene_snapshot(snap)["animation_free_rest"]
    for name, wanted in expected.items():
        obj = bpy.data.objects[name]
        actual = np.array([tuple(obj.matrix_world @ vertex.co) for vertex in obj.data.vertices])
        # Compare point sets; source exporter/importer may reorder vertices.
        distance = np.linalg.norm(actual[:, None] - wanted[None, :], axis=2)
        assert distance.min(axis=1).max() < 2e-4, (name, distance.min(axis=1).max())
        assert distance.min(axis=0).max() < 2e-4
    assert bpy.data.objects["RigidAttachment"].parent.name == "Body"
    print("SOURCE_SCENE_V2_SYNTHETIC_ROOT_ATTACHMENT_IMPORT_OK", flush=True)


def alpha(tmp):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=2, rotation=(np.pi / 2, 0, 0))
    plane = bpy.context.object
    material = bpy.data.materials.new("LinkedAlphaAndDownstreamGraph")
    material.use_nodes = True
    if hasattr(material, "blend_method"):
        material.blend_method = "BLEND"
    plane.data.materials.append(material)
    tree = material.node_tree
    bsdf = next(node for node in tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (.2, .4, .6, 1)
    value = tree.nodes.new("ShaderNodeValue")
    value.outputs[0].default_value = .25
    tree.links.new(value.outputs[0], bsdf.inputs["Alpha"])
    transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
    downstream = tree.nodes.new("ShaderNodeMixShader")
    downstream.inputs[0].default_value = .5
    tree.links.new(transparent.outputs[0], downstream.inputs[1])
    tree.links.new(bsdf.outputs[0], downstream.inputs[2])
    out = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL")
    tree.links.new(downstream.outputs[0], out.inputs["Surface"])
    receipt = producer.preserve_alpha_unlit([plane])
    assert receipt[0]["source_alpha_linked"]
    assert out.inputs["Surface"].links[0].from_node == downstream
    retained = downstream.inputs[2].links[0].from_node
    assert retained.type == "MIX_SHADER"
    assert retained.inputs[0].links[0].from_node == value
    producer.configure(64)
    bpy.context.scene.cycles.samples = 128
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"; camera_data.ortho_scale = 3
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = (0, -4, 0)
    camera.rotation_euler = Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    png = tmp / "alpha.png"
    bpy.context.scene.render.filepath = str(png)
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(str(png), check_existing=False)
    try:
        center = image.pixels[(32 * 64 + 32) * 4 + 3]
        assert abs(center - .125) <= .02, center
    finally:
        bpy.data.images.remove(image)
    print(f"SOURCE_SCENE_V2_SYNTHETIC_ALPHA_OK observed={center:.5f} expected=0.125", flush=True)
    # Explicit legacy OPAQUE mode must not become transparent under Cycles.
    if hasattr(material, "blend_method"):
        opaque = bpy.data.materials.new("OpaqueWithDormantZeroAlpha")
        opaque.use_nodes = True
        opaque.blend_method = "OPAQUE"
        principal = next(node for node in opaque.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        principal.inputs["Alpha"].default_value = 0
        plane.data.materials.clear(); plane.data.materials.append(opaque)
        result = producer.preserve_alpha_unlit([plane])
        assert result[0]["alpha_policy"] == "IMPORTED_OPAQUE_MODE__BSDF_ALPHA_INACTIVE"
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(png), check_existing=False)
        try:
            center = image.pixels[(32 * 64 + 32) * 4 + 3]
            assert center > .99, center
        finally:
            bpy.data.images.remove(image)
        print("SOURCE_SCENE_V2_SYNTHETIC_OPAQUE_BACKEND_SEMANTICS_OK", flush=True)


with tempfile.TemporaryDirectory() as directory:
    scene_import(Path(directory))
    alpha(Path(directory))
print("SOURCE_SCENE_V2_BLENDER_ENGINEERING_SMOKE_OK__NO_TRAINING_ADMISSION", flush=True)
