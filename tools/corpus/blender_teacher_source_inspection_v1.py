"""Read-only source measurements through the canonical animation-free importer."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest, scene_snapshot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect(row, out):
    out = Path(out).resolve()
    source = Path(row["source"]["path"]).resolve()
    if sha(source) != row["source"]["sha256"]:
        raise ValueError("TEACHER_INSPECTION_SOURCE_DRIFT")
    out.mkdir(parents=True, exist_ok=True)
    receipt = import_rest(source, out)
    deps = bpy.context.evaluated_depsgraph_get()
    vertices, faces, owners, meshes, controls = [], [], [], [], []
    joint_index = {}
    for obj in sorted(bpy.context.scene.objects, key=lambda x: x.name):
        if obj.type != "ARMATURE":
            continue
        for bone in obj.data.bones:
            joint_index[(obj.name, bone.name)] = len(controls)
            controls.append({"armature": obj.name, "name": bone.name,
                "parent": bone.parent.name if bone.parent else None,
                "rest_matrix_world": [list(x) for x in obj.matrix_world @ bone.matrix_local],
                "use_deform": bool(bone.use_deform)})
    weight_vertex, weight_joint, weight_value = [], [], []
    for oid, obj in enumerate(sorted(bpy.context.scene.objects, key=lambda x: x.name)):
        if obj.type != "MESH" or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        offset = len(vertices)
        vertices.extend([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
        mesh.calc_loop_triangles()
        faces.extend([[offset + int(i) for i in t.vertices] for t in mesh.loop_triangles])
        owners.extend([oid] * len(mesh.loop_triangles))
        arms = [m.object for m in obj.modifiers if m.type == "ARMATURE" and m.object is not None]
        if len(arms) > 1:
            raise ValueError("MULTIPLE_ARMATURE_MODIFIERS_REQUIRE_ADAPTER")
        if arms and len(mesh.vertices) != len(obj.data.vertices):
            raise ValueError("EVALUATED_VERTEX_WEIGHT_ADDRESSING_CHANGED")
        if arms:
            for v in obj.data.vertices:
                for g in v.groups:
                    joint = joint_index.get((arms[0].name, obj.vertex_groups[g.group].name))
                    if joint is not None and g.weight > 0:
                        weight_vertex.append(offset + v.index)
                        weight_joint.append(joint)
                        weight_value.append(g.weight)
        meshes.append({"owner_id": oid, "name": obj.name, "vertex_count": len(mesh.vertices),
            "triangle_count": len(mesh.loop_triangles), "parent": obj.parent.name if obj.parent else None,
            "parent_type": obj.parent_type, "parent_bone": obj.parent_bone or None,
            "armature_targets": [x.name for x in arms], "material_count": len(obj.material_slots)})
        evaluated.to_mesh_clear()
    if not faces:
        raise ValueError("EMPTY_ACTIVE_REST_GEOMETRY")
    arrays = out / "IMPORTED_SOURCE_REST_INSPECTION.npz"
    np.savez_compressed(arrays, rest_vertices_source_world=np.asarray(vertices, np.float32),
        faces=np.asarray(faces, np.int32), face_owner_privileged=np.asarray(owners, np.int32),
        weight_vertex=np.asarray(weight_vertex, np.int32), weight_joint=np.asarray(weight_joint, np.int32),
        weight_value=np.asarray(weight_value, np.float32))
    snapshot = out / "IMPORTED_SOURCE_REST_SNAPSHOT.json"
    snapshot.write_text(json.dumps(scene_snapshot(row["source"]["sha256"], Matrix.Identity(4)), sort_keys=True, indent=2) + "\n")
    report = {"schema": "RealSaS.ImportedTeacherSourceInspection.v1", "subject_id": row["subject_id"],
        "status": "IMPORTED_REST_MEASURED__RAW_PARITY_AND_SOLID_QUALIFICATION_SEPARATE", "source": row["source"],
        "runtime": {"blender": bpy.app.version_string, "libc": list(platform.libc_ver()), "python": platform.python_version()},
        "producer_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [
            Path(__file__).resolve(), ROOT / "tools/observation/blender_controlled_8view_v2.py",
            ROOT / "tools/observation/blender_controlled_8view_v1.py", ROOT / "tools/observation/source_scene_contract_v2.py"]},
        "import": receipt, "meshes": meshes, "controls": controls,
        "counts": {"mesh_owner_count": len(meshes), "vertex_count": len(vertices), "triangle_count": len(faces),
                   "joint_count": len(controls), "weighted_vertex_count": len(set(weight_vertex))},
        "outputs": {"rest_geometry": {"path": str(arrays), "sha256": sha(arrays)},
                    "snapshot": {"path": str(snapshot), "sha256": sha(snapshot)}},
        "teacher_truth_minted": False, "training_authorized": False}
    (out / "IMPORTED_TEACHER_SOURCE_INSPECTION_V1.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    rows, root = json.loads(Path(args[0]).read_text()), Path(args[1]).resolve()
    failed = []
    for row in rows:
        try:
            result = inspect(row, root / row["subject_id"])
            print(json.dumps({"subject_id": row["subject_id"], "status": result["status"], "counts": result["counts"]}), flush=True)
        except Exception as exc:
            failed.append({"subject_id": row["subject_id"], "error_type": type(exc).__name__, "reason": str(exc)})
            print(json.dumps(failed[-1]), flush=True)
    (root / (Path(args[0]).stem + "_failures.json")).write_text(json.dumps(failed, indent=2) + "\n")
    if failed:
        raise SystemExit(1)
