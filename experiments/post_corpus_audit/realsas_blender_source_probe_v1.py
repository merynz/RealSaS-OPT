import bpy, sys, json, math, os
from pathlib import Path
import numpy as np
from collections import Counter

argv=sys.argv[sys.argv.index("--")+1:]
src=Path(argv[0]); out=Path(argv[1])

def import_source(src):
    ext=src.suffix.lower()
    if ext==".blend":
        bpy.ops.wm.open_mainfile(filepath=str(src))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if ext in {".glb",".gltf"}:
            bpy.ops.import_scene.gltf(filepath=str(src))
        elif ext==".fbx":
            try: bpy.ops.wm.fbx_import(filepath=str(src))
            except Exception: bpy.ops.import_scene.fbx(filepath=str(src))
        elif ext in {".usd",".usda",".usdc",".usdz"}:
            bpy.ops.wm.usd_import(filepath=str(src))
        elif ext==".dae":
            bpy.ops.wm.collada_import(filepath=str(src))
        elif ext==".obj":
            try: bpy.ops.wm.obj_import(filepath=str(src))
            except Exception: bpy.ops.import_scene.obj(filepath=str(src))
        else:
            raise RuntimeError("unsupported extension "+ext)

def world_vertices(obj, me):
    M=np.asarray(obj.matrix_world,dtype=np.float64)
    V=np.array([list(v.co)+[1.0] for v in me.vertices],dtype=np.float64)
    return (V@M.T)[:,:3]

def fan_faces(me):
    fs=[]; ngon=0; maxn=0
    for p in me.polygons:
        vs=list(p.vertices); maxn=max(maxn,len(vs))
        if len(vs)>4: ngon+=1
        for k in range(1,len(vs)-1): fs.append((vs[0],vs[k],vs[k+1]))
    return np.asarray(fs,dtype=np.int64),ngon,maxn

def looptri_faces(me):
    try: me.calc_loop_triangles()
    except Exception: pass
    return np.asarray([list(t.vertices) for t in me.loop_triangles],dtype=np.int64)

def tri_counter(F): return Counter(tuple(sorted(map(int,t))) for t in np.asarray(F))

def builder_normals(V,F):
    N=np.zeros_like(V,dtype=np.float64)
    if len(F)==0: return N
    tri=V[F]; fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    for k in range(3): np.add.at(N,F[:,k],fn)
    nn=np.linalg.norm(N,axis=1,keepdims=True)
    return N/np.maximum(nn,1e-12)

def corner_normal_angles(obj,me,V,F):
    BN=builder_normals(V,F)
    try: nm=obj.matrix_world.to_3x3().inverted().transposed()
    except Exception: nm=obj.matrix_world.to_3x3()
    ang=[]
    for loop in me.loops:
        try: n=nm@loop.normal
        except Exception: continue
        nv=np.asarray(n,dtype=np.float64); d=np.linalg.norm(nv)
        if d<=1e-12: continue
        nv/=d; bv=BN[int(loop.vertex_index)]; bd=np.linalg.norm(bv)
        if bd<=1e-12: continue
        ang.append(math.degrees(math.acos(float(np.clip(np.dot(nv,bv),-1,1)))))
    if not ang: return None
    a=np.asarray(ang)
    return {"count":int(len(a)),"median_deg":float(np.quantile(a,.5)),"p90_deg":float(np.quantile(a,.9)),"p95_deg":float(np.quantile(a,.95)),"gt30_fraction":float(np.mean(a>30)),"gt60_fraction":float(np.mean(a>60))}

def geom_object(obj,deps):
    me=obj.data; V=world_vertices(obj,me); Ffan,ngon,maxn=fan_faces(me); Floop=looptri_faces(me); tri_mismatch=tri_counter(Ffan)!=tri_counter(Floop)
    flat=sum(not p.use_smooth for p in me.polygons); sharp=sum(bool(getattr(e,"use_edge_sharp",False)) for e in me.edges); custom=getattr(me,"has_custom_normals",None); corner=corner_normal_angles(obj,me,V,Ffan)
    mods=[m.type for m in obj.modifiers]; shapekeys=[]
    if getattr(me,"shape_keys",None) is not None:
        for kb in me.shape_keys.key_blocks: shapekeys.append({"name":kb.name,"value":float(kb.value)})
    ev=obj.evaluated_get(deps); eme=ev.to_mesh()
    try:
        EV=world_vertices(ev,eme); EF=looptri_faces(eme); same_counts=(len(V)==len(EV) and len(Floop)==len(EF)); delta=float(np.max(np.linalg.norm(V-EV,axis=1))) if len(V)==len(EV) and len(V) else (0.0 if len(V)==len(EV) else None); eval_diff=(not same_counts) or (delta is not None and delta>1e-7)
        eval_summary={"vertices":int(len(EV)),"triangles":int(len(EF)),"same_counts":bool(same_counts),"indexwise_world_vertex_max_delta":delta,"diff_from_base":bool(eval_diff)}
    finally: ev.to_mesh_clear()
    return {"name":obj.name,"vertices":int(len(V)),"fan_triangles":int(len(Ffan)),"blender_loop_triangles":int(len(Floop)),"ngon_gt4_count":int(ngon),"max_polygon_vertices":int(maxn),"fan_vs_blender_triangulation_mismatch":bool(tri_mismatch),"flat_polygon_fraction":float(flat/max(1,len(me.polygons))),"sharp_edge_count":int(sharp),"has_custom_normals":custom,"builder_vs_corner_normal_angle":corner,"modifiers":mods,"shape_keys":shapekeys,"evaluated":eval_summary}

def appearance_summary():
    mats=[]; image_refs=[]
    for mat in bpy.data.materials:
        rec={"name":mat.name,"use_nodes":bool(mat.use_nodes),"image_nodes":[]}
        if mat.use_nodes and mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type=="TEX_IMAGE" and getattr(n,"image",None):
                    im=n.image; fp=str(getattr(im,"filepath","") or ""); packed=bool(getattr(im,"packed_file",None)); exists=None
                    if fp and not packed:
                        try: exists=os.path.exists(bpy.path.abspath(fp,library=im.library))
                        except Exception: pass
                    ir={"image":im.name,"filepath":fp,"packed":packed,"source":str(getattr(im,"source","")),"external_exists":exists}; rec["image_nodes"].append(ir); image_refs.append(ir)
        try: rec["diffuse_color"]=list(map(float,mat.diffuse_color))
        except Exception: pass
        mats.append(rec)
    unique_images={}
    for im in bpy.data.images:
        if im.name in {"Render Result","Viewer Node"}: continue
        fp=str(getattr(im,"filepath","") or ""); packed=bool(getattr(im,"packed_file",None)); exists=None
        if fp and not packed:
            try: exists=os.path.exists(bpy.path.abspath(fp,library=im.library))
            except Exception: pass
        unique_images[im.name]={"filepath":fp,"packed":packed,"source":str(getattr(im,"source","")),"external_exists":exists}
    vals=list(unique_images.values())
    return {"material_count":len(mats),"materials":mats,"image_datablock_count":len(vals),"image_node_reference_count":len(image_refs),"packed_image_count":sum(bool(x["packed"]) for x in vals),"external_image_present_count":sum(x["external_exists"] is True for x in vals),"external_image_missing_count":sum(x["external_exists"] is False for x in vals),"images":unique_images}

res={"source":str(src),"extension":src.suffix.lower(),"errors":[]}
try:
    import_source(src)
    for o in bpy.data.objects:
        if o.type=="ARMATURE":
            try: o.data.pose_position="REST"
            except Exception: pass
    bpy.context.view_layer.update(); deps=bpy.context.evaluated_depsgraph_get(); objs=[]
    for o in bpy.context.scene.objects:
        if o.type=="MESH" and len(o.data.vertices)>0:
            try: objs.append(geom_object(o,deps))
            except Exception as e: res["errors"].append(f"object {o.name}: {type(e).__name__}: {e}")
    res["mesh_objects"]=objs; res["mesh_object_count"]=len(objs); res["evaluated_diff_object_count"]=sum(bool(x["evaluated"]["diff_from_base"]) for x in objs); res["triangulation_mismatch_object_count"]=sum(bool(x["fan_vs_blender_triangulation_mismatch"]) for x in objs); res["modifier_object_count"]=sum(bool(x["modifiers"]) for x in objs); res["shape_key_object_count"]=sum(bool(x["shape_keys"]) for x in objs)
    normals=[x["builder_vs_corner_normal_angle"] for x in objs if x["builder_vs_corner_normal_angle"]]
    if normals: res["normal_authority_summary"]={"object_count":len(normals),"median_of_object_medians_deg":float(np.median([x["median_deg"] for x in normals])),"max_object_p95_deg":float(max(x["p95_deg"] for x in normals))}
    res["appearance"]=appearance_summary()
except Exception as e: res["errors"].append(f"{type(e).__name__}: {e}")
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2),encoding="utf-8")
