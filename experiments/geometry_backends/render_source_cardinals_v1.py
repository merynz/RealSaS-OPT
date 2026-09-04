from __future__ import annotations

"""Blender-only neutral cardinal renderer for pinned external 3D source assets.

This is an input apparatus for solved-literature reconstruction baselines. It does
not create teacher truth and does not alter the source pose through a rig. All
armatures are put in REST position, the imported geometry is left unchanged, and
four orthographic RGB views are rendered around the source's own world-space bbox.
"""

from pathlib import Path
import sys

import bpy
from mathutils import Vector


argv = sys.argv[sys.argv.index("--") + 1 :]
if len(argv) != 3:
    raise RuntimeError("usage: blender -b --python render_source_cardinals_v1.py -- SOURCE OUTPUT_DIR RESOLUTION")
source = Path(argv[0]).resolve()
out_dir = Path(argv[1]).resolve()
resolution = int(argv[2])
if resolution < 128:
    raise RuntimeError("resolution must be >=128")
if not source.is_file():
    raise FileNotFoundError(source)
out_dir.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = source.suffix.lower()
if ext in {".glb", ".gltf"}:
    bpy.ops.import_scene.gltf(filepath=str(source))
elif ext == ".fbx":
    try:
        bpy.ops.wm.fbx_import(filepath=str(source))
    except Exception:
        bpy.ops.import_scene.fbx(filepath=str(source))
else:
    raise RuntimeError(f"unsupported baseline source format: {ext}")

for obj in bpy.context.scene.objects:
    if obj.type == "ARMATURE":
        obj.data.pose_position = "REST"
bpy.context.view_layer.update()

meshes = [
    obj for obj in bpy.context.scene.objects
    if obj.type == "MESH" and len(obj.data.vertices) > 0 and not obj.hide_render
]
if not meshes:
    raise RuntimeError("no renderable mesh in baseline source")

corners = []
for obj in meshes:
    mw = obj.matrix_world
    corners.extend(mw @ Vector(corner) for corner in obj.bound_box)
lo = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
hi = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
center = (lo + hi) * 0.5
extent = hi - lo
span = max(float(extent.x), float(extent.y), float(extent.z), 1e-3)
radius = span * 3.0
ortho_scale = span * 1.22

scene = bpy.context.scene
# Blender 5.2.0 LTS exposes the Eevee engine as BLENDER_EEVEE.
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.film_transparent = False
scene.render.use_file_extension = True

# --factory-startup + use_empty=True can leave scene.world unset.
if scene.world is None:
    scene.world = bpy.data.worlds.new("RealSaSCardinalWorld")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg is None:
    raise RuntimeError("Blender world Background node missing")
bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
bg.inputs["Strength"].default_value = 0.8

try:
    scene.view_settings.look = "Medium High Contrast"
except Exception:
    pass
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0

# Broad lights preserve the source material/texture while avoiding a directional
# silhouette bias that would make one cardinal systematically darker.
def add_area(name: str, location, energy: float, size: float):
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name=name, object_data=data)
    scene.collection.objects.link(obj)
    obj.location = center + Vector(location)
    direction = center - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return obj

add_area("KeyFront", (0.0, -radius * 0.8, radius * 0.7), 850.0, span * 2.5)
add_area("KeyBack", (0.0, radius * 0.8, radius * 0.7), 650.0, span * 2.5)
add_area("FillRight", (radius * 0.8, 0.0, radius * 0.3), 500.0, span * 2.0)
add_area("FillLeft", (-radius * 0.8, 0.0, radius * 0.3), 500.0, span * 2.0)

cam_data = bpy.data.cameras.new("CanonicalCardinalCamera")
cam_data.type = "ORTHO"
cam_data.ortho_scale = ortho_scale
cam = bpy.data.objects.new("CanonicalCardinalCamera", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

# RealSaS naming: S=front, N=back, E=right, W=left.
views = {
    "S": Vector((0.0, -1.0, 0.0)),
    "N": Vector((0.0, 1.0, 0.0)),
    "E": Vector((1.0, 0.0, 0.0)),
    "W": Vector((-1.0, 0.0, 0.0)),
}
for label, direction in views.items():
    cam.location = center + direction * radius
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_dir / f"{label}.png")
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)

for label in views:
    p = out_dir / f"{label}.png"
    if not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError(f"cardinal render missing: {p}")

print(
    "REALSAS_SOURCE_CARDINAL_RENDER=PASS "
    f"source={source} resolution={resolution} center={tuple(round(float(x),6) for x in center)} "
    f"span={span:.6f} ortho_scale={ortho_scale:.6f}"
)
