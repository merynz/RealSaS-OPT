import bpy, sys, json
from pathlib import Path
import numpy as np

argv=sys.argv[sys.argv.index("--")+1:]
src=Path(argv[0]); out_npz=Path(argv[1]); out_json=Path(argv[2])

ext=src.suffix.lower()
if ext==".blend":
    bpy.ops.wm.open_mainfile(filepath=str(src))
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ext in {".glb",".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif ext==".fbx":
        ok=False
        for op in [lambda:bpy.ops.wm.fbx_import(filepath=str(src)),
                   lambda:bpy.ops.import_scene.fbx(filepath=str(src))]:
            try: op(); ok=True; break
            except Exception: pass
        if not ok: raise RuntimeError("FBX importer unavailable")
    elif ext in {".usd",".usda",".usdc",".usdz"}:
        bpy.ops.wm.usd_import(filepath=str(src))
    elif ext==".dae":
        ok=False
        for op in [lambda:bpy.ops.wm.collada_import(filepath=str(src)),
                   lambda:bpy.ops.import_scene.collada(filepath=str(src))]:
            try: op(); ok=True; break
            except Exception: pass
        if not ok: raise RuntimeError("Collada importer unavailable")
    elif ext==".obj":
        ok=False
        for op in [lambda:bpy.ops.wm.obj_import(filepath=str(src)),
                   lambda:bpy.ops.import_scene.obj(filepath=str(src))]:
            try: op(); ok=True; break
            except Exception: pass
        if not ok: raise RuntimeError("OBJ importer unavailable")
    else:
        raise RuntimeError("Unsupported source extension: "+ext)

for a in bpy.data.objects:
    if a.type=="ARMATURE": a.data.pose_position="REST"
bpy.context.view_layer.update()

meshes=[o for o in bpy.context.scene.objects
        if o.type=="MESH" and len(o.data.vertices)>0 and not getattr(o,"hide_render",False)]
if not meshes: raise RuntimeError("No mesh objects found")

V=[]; F=[]; UV=[]; UV_VALID=[]; FACE_MAT=[]; object_ranges=[]; offset=0
materials=[]

def image_from_base_color(mat):
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    bs=None
    for n in mat.node_tree.nodes:
        if n.type=="BSDF_PRINCIPLED":
            bs=n; break
    if bs is None: return None
    sock=bs.inputs.get("Base Color")
    if sock is not None and sock.is_linked:
        node=sock.links[0].from_node
        if node.type=="TEX_IMAGE" and getattr(node,"image",None) is not None:
            return node.image
    return None

def base_color(mat):
    if mat is None: return [0.8,0.8,0.8,1.0]
    if mat.use_nodes and mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            if n.type=="BSDF_PRINCIPLED":
                s=n.inputs.get("Base Color")
                if s is not None and not s.is_linked:
                    return [float(x) for x in s.default_value]
    try: return [float(x) for x in mat.diffuse_color]
    except Exception: return [0.8,0.8,0.8,1.0]

for o in meshes:
    M=o.matrix_world
    verts=np.array([tuple(M @ v.co) for v in o.data.vertices],dtype=np.float32)
    V.append(verts)
    face_start=len(F)

    local_to_global=[]
    slot_count=max(1,len(o.material_slots))
    for si in range(slot_count):
        mat=o.material_slots[si].material if si<len(o.material_slots) else None
        gi=len(materials)
        local_to_global.append(gi)
        img=image_from_base_color(mat)
        img_path=None; img_name=None
        if img is not None:
            img_name=img.name
            try: img_path=bpy.path.abspath(img.filepath)
            except Exception: img_path=None
        materials.append({
            "object":o.name,"slot":si,"name":mat.name if mat else None,
            "base_color":base_color(mat),"image_name":img_name,"image_path":img_path,
            "image_exists":bool(img_path and Path(img_path).exists())
        })

    uv_layer=o.data.uv_layers.active
    for p in o.data.polygons:
        ids=list(p.vertices)
        if len(ids)<3: continue
        loops=[p.loop_start+k for k in range(p.loop_total)]
        if len(loops)!=len(ids): raise RuntimeError("polygon vertex/loop count mismatch")
        mi=int(p.material_index)
        if mi<0 or mi>=len(local_to_global): mi=0
        gmi=local_to_global[mi]
        for j in range(1,len(ids)-1):
            F.append((ids[0]+offset,ids[j]+offset,ids[j+1]+offset))
            FACE_MAT.append(gmi)
            corners=[0,j,j+1]
            if uv_layer is not None:
                uv=np.array([tuple(uv_layer.data[loops[k]].uv) for k in corners],dtype=np.float32)
                UV.append(uv); UV_VALID.append(True)
            else:
                UV.append(np.zeros((3,2),dtype=np.float32)); UV_VALID.append(False)

    object_ranges.append({
        "name":o.name,"vertex_start":offset,"vertex_count":len(verts),
        "face_start":face_start,"face_count":len(F)-face_start
    })
    offset+=len(verts)

V=np.concatenate(V,axis=0)
F=np.asarray(F,dtype=np.int32)
UV=np.asarray(UV,dtype=np.float32)
UV_VALID=np.asarray(UV_VALID,dtype=np.uint8)
FACE_MAT=np.asarray(FACE_MAT,dtype=np.int32)
if len(F)==0: raise RuntimeError("No triangulated faces found")

out_npz.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out_npz,vertices_source=V,faces=F,
                    face_uv=UV,face_uv_valid=UV_VALID,face_material=FACE_MAT)
out_json.write_text(json.dumps({
    "source":str(src),"mesh_objects":object_ranges,"materials":materials
},indent=2),encoding="utf-8")
