"""Synthetic GLB/FBX parity smoke for the Post139 canonical structural extractor.

Engineering evidence only. It validates that one authored Blender scene exported
through two source formats receives the same structural-subject fingerprint.
It does not admit any real corpus subject.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parents[3]
EXTRACTOR = ROOT / "tools/corpus/blender_post139_source_structure_v1.py"


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def build_scene() -> None:
    reset_scene()

    mesh = bpy.data.meshes.new("SyntheticMesh")
    mesh.from_pydata(
        [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 1.0), (-0.5, 0.0, 1.0)],
        [],
        [(0, 1, 2), (0, 2, 3)],
    )
    mesh.update()
    mesh_obj = bpy.data.objects.new("SyntheticMesh", mesh)
    bpy.context.collection.objects.link(mesh_obj)

    material = bpy.data.materials.new("SyntheticMaterial")
    material.diffuse_color = (0.2, 0.6, 0.9, 1.0)
    mesh_obj.data.materials.append(material)

    armature = bpy.data.armatures.new("SyntheticRig")
    rig_obj = bpy.data.objects.new("SyntheticRig", armature)
    bpy.context.collection.objects.link(rig_obj)
    bpy.context.view_layer.objects.active = rig_obj
    rig_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = armature.edit_bones.new("Root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.0, 0.5)
    tip = armature.edit_bones.new("Tip")
    tip.head = root.tail
    tip.tail = (0.0, 0.0, 1.0)
    tip.parent = root
    tip.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")

    mesh_obj.parent = rig_obj
    modifier = mesh_obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = rig_obj
    root_group = mesh_obj.vertex_groups.new(name="Root")
    tip_group = mesh_obj.vertex_groups.new(name="Tip")
    root_group.add([0, 1], 1.0, "REPLACE")
    tip_group.add([2, 3], 1.0, "REPLACE")

    bpy.context.view_layer.objects.active = rig_obj
    pose_root = rig_obj.pose.bones["Root"]
    pose_root.rotation_mode = "XYZ"
    pose_root.rotation_euler[2] = 0.0
    pose_root.keyframe_insert(data_path="rotation_euler", frame=1)
    pose_root.rotation_euler[2] = 0.25
    pose_root.keyframe_insert(data_path="rotation_euler", frame=12)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 12
    bpy.context.scene.frame_set(1)


def export_sources(directory: Path) -> tuple[Path, Path]:
    glb = directory / "synthetic.glb"
    fbx = directory / "synthetic.fbx"
    bpy.ops.export_scene.gltf(
        filepath=str(glb),
        export_format="GLB",
        export_animations=True,
    )
    bpy.ops.export_scene.fbx(
        filepath=str(fbx),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
    )
    assert glb.is_file() and fbx.is_file()
    return glb, fbx


def extract(source: Path, output: Path) -> dict:
    command = [
        bpy.app.binary_path,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(EXTRACTOR),
        "--",
        "--source",
        str(source),
        "--out",
        str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)
    return json.loads(output.read_text())


def diagnostic(payload: dict) -> dict:
    return {
        "metrics": payload["metrics"],
        "fingerprints": payload["fingerprints"],
        "geometry": payload["inventory"]["geometry"],
        "rig": payload["inventory"]["rig"],
        "skin": payload["inventory"]["skin"],
    }


def main() -> None:
    build_scene()
    with tempfile.TemporaryDirectory(prefix="realsas-post139-") as raw:
        directory = Path(raw)
        glb, fbx = export_sources(directory)
        glb_result = extract(glb, directory / "glb.json")
        fbx_result = extract(fbx, directory / "fbx.json")

        required = {"geometry": True, "appearance": True, "rig": True, "skin": True}
        for label, payload in (("GLB", glb_result), ("FBX", fbx_result)):
            for key, expected in required.items():
                assert payload["capabilities"][key] is expected, (label, key, payload["capabilities"])
            assert payload["extractor"]["cross_format_dedup_ready"] is True

        glb_hash = glb_result["fingerprints"]["structural_subject"]
        fbx_hash = fbx_result["fingerprints"]["structural_subject"]
        if glb_hash != fbx_hash:
            print("POST139_GLTF_DIAGNOSTIC", json.dumps(diagnostic(glb_result), sort_keys=True))
            print("POST139_FBX_DIAGNOSTIC", json.dumps(diagnostic(fbx_result), sort_keys=True))
        assert glb_hash == fbx_hash, {
            "glb": glb_hash,
            "fbx": fbx_hash,
            "glb_fingerprints": glb_result["fingerprints"],
            "fbx_fingerprints": fbx_result["fingerprints"],
        }
        print("POST139_BLENDER_CROSS_FORMAT_SMOKE_PASS", glb_hash)


if __name__ == "__main__":
    main()
