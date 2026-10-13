"""Export evaluated imported bind geometry for an independent raw-source court."""
import json
import sys
from pathlib import Path
import bpy
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.observation.blender_controlled_8view_v2 import import_rest, matrix_rows

source, output = map(Path, sys.argv[sys.argv.index("--") + 1:])
output.parent.mkdir(parents=True, exist_ok=True)
receipt = import_rest(source.resolve(), output.parent)
deps = bpy.context.evaluated_depsgraph_get()
vertices, faces, objects = [], [], []
for obj in bpy.context.scene.objects:
    if obj.type != "MESH" or obj.hide_render:
        continue
    evaluated = obj.evaluated_get(deps)
    mesh = evaluated.to_mesh()
    offset = len(vertices)
    vertices.extend([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
    mesh.calc_loop_triangles()
    faces.extend([[offset + i for i in tri.vertices] for tri in mesh.loop_triangles])
    objects.append({"name": obj.name, "vertex_count": len(mesh.vertices),
                    "parent": obj.parent.name if obj.parent else None,
                    "parent_bone": obj.parent_bone,
                    "world": matrix_rows(obj.matrix_world),
                    "weights": [[{"group": obj.vertex_groups[g.group].name, "weight": g.weight}
                                 for g in v.groups] for v in obj.data.vertices]})
    evaluated.to_mesh_clear()
bones = sum(len(obj.data.bones) for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
output.write_text(json.dumps({"vertices": vertices, "faces": faces, "bone_count": bones,
                              "objects": objects, "import": receipt}))
print("[PARITY_EXPORT]", len(vertices), bones, flush=True)
