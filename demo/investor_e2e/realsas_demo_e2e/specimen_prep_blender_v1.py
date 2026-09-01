from __future__ import annotations

"""Generic Blender-side producer for the preregistered demo fit envelope.

Input is one rigged source file. No specimen names, bone coordinates, topology or
weight arrays are embedded in code. The output separates fit truth from the
image-only final-inference payload.
"""

from pathlib import Path
import argparse
import json
import math
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

YAW_DEG=[0,45,90,135,180,225,270,315]
AUTH_RES=1024
TEACHER_RES=256
TARGET_RADIUS=0.55
MARGIN=1.12


def log(x):print("REALSAS_DEMO_PREP|"+str(x),flush=True)


def parse_args():
    a=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True)
    return p.parse_args(a)


def clean():
    bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)


def import_asset(path:str):
    e=Path(path).suffix.lower()
    if e==".blend":bpy.ops.wm.open_mainfile(filepath=path);return
    clean()
    if e==".fbx":bpy.ops.import_scene.fbx(filepath=path,use_anim=True)
    elif e in {".glb",".gltf"}:bpy.ops.import_scene.gltf(filepath=path)
    else:raise RuntimeError("DEMO_PREP_UNSUPPORTED_SOURCE:"+e)


def bone_group_indices(obj,arm):
    names={b.name for b in arm.data.bones};return {g.index for g in obj.vertex_groups if g.name in names}


def weighted_count(obj,arm):
    valid=bone_group_indices(obj,arm)
    return sum(1 for v in obj.data.vertices if any(g.group in valid and g.weight>1e-8 for g in v.groups))


def relation(obj,arm):
    r=[]
    if any(m.type=="ARMATURE" and m.object==arm for m in obj.modifiers):r.append("ARMATURE_MODIFIER")
    if obj.parent==arm:r.append("BONE_PARENT" if obj.parent_type=="BONE" else "ARMATURE_PARENT")
    return r


def resolve_primary(meshes,arms):
    rows=[]
    for arm in arms:
        w=sum(weighted_count(o,arm) for o in meshes);rel=sum(bool(relation(o,arm)) for o in meshes)
        rows.append((w+100*rel+.01*len(arm.data.bones),arm,w))
    rows.sort(key=lambda x:x[0],reverse=True)
    return rows[0][1] if rows and rows[0][2]>0 else None


def subject_gate(meshes,arm):
    admitted=[];excluded=[]
    for o in meshes:
        w=weighted_count(o,arm);r=relation(o,arm)
        if w>=max(4,int(.002*max(1,len(o.data.vertices)))) and ("ARMATURE_MODIFIER" in r or "ARMATURE_PARENT" in r):admitted.append(o)
        elif "BONE_PARENT" in r:admitted.append(o)
        else:excluded.append(o)
    return admitted,excluded


def bbox_points(objs):
    dg=bpy.context.evaluated_depsgraph_get();pts=[]
    for o in objs:
        eo=o.evaluated_get(dg)
        for c in eo.bound_box:pts.append(eo.matrix_world@Vector(c))
    return pts


def infer_up(arm):
    bs=list(arm.data.bones);heads=[b for b in bs if re.search("head|neck",b.name,re.I)];hips=[b for b in bs if re.search("hip|pelvis|root",b.name,re.I)]
    if heads and hips:
        h=sum((arm.matrix_world@b.head_local for b in heads),Vector())/len(heads)
        p=sum((arm.matrix_world@b.head_local for b in hips),Vector())/len(hips);v=h-p
        if v.length>1e-8:return v.normalized(),"NAMED_HIP_TO_HEAD"
    return Vector((0,0,1)),"SOURCE_Z_UP"


def normalize_subject(admitted,arm):
    root=bpy.data.objects.new("REALSAS_DEMO_CANONICAL_ROOT",None);bpy.context.collection.objects.link(root)
    ss=set(admitted+[arm])
    for o in list(ss):
        if o.parent not in ss:
            mw=o.matrix_world.copy();o.parent=root;o.matrix_world=mw
    up,method=infer_up(arm);root.rotation_mode="QUATERNION";root.rotation_quaternion=up.rotation_difference(Vector((0,0,1)));bpy.context.view_layer.update()
    pts=bbox_points(admitted);center=sum(pts,Vector())/len(pts);root.location-=center;bpy.context.view_layer.update()
    pts=bbox_points(admitted);radius=max(p.length for p in pts)
    if radius<=1e-9:raise RuntimeError("DEMO_PREP_DEGENERATE_SUBJECT")
    s=TARGET_RADIUS/radius;root.scale=(s,s,s);bpy.context.view_layer.update()
    pts=bbox_points(admitted);radius=max(p.length for p in pts)
    return root,radius,method


def setup_render():
    s=bpy.context.scene
    for e in ("BLENDER_EEVEE","BLENDER_EEVEE_NEXT"):
        try:s.render.engine=e;break
        except Exception:continue
    s.render.film_transparent=True;s.render.resolution_x=AUTH_RES;s.render.resolution_y=AUTH_RES;s.render.resolution_percentage=100
    s.render.image_settings.file_format="PNG";s.render.image_settings.color_mode="RGBA";s.render.image_settings.color_depth="8";s.render.image_settings.compression=15
    if hasattr(s,"eevee") and hasattr(s.eevee,"taa_render_samples"):s.eevee.taa_render_samples=8
    w=s.world or bpy.data.worlds.new("World");s.world=w;w.use_nodes=True;bg=w.node_tree.nodes.get("Background")
    if bg:bg.inputs["Color"].default_value=(.18,.18,.18,1);bg.inputs["Strength"].default_value=.65
    for o in [x for x in bpy.data.objects if x.type=="LIGHT"]:bpy.data.objects.remove(o,do_unlink=True)
    for name,v,power,size in [("Key",(3,-4,5),900,4),("Fill",(-4,-2,2),450,3),("Rim",(0,4,5),550,3)]:
        d=bpy.data.lights.new(name,"AREA");d.energy=power;d.shape="DISK";d.size=size
        o=bpy.data.objects.new(name,d);bpy.context.collection.objects.link(o);o.location=Vector(v);o.rotation_euler=((Vector((0,0,0))-o.location).to_track_quat("-Z","Y")).to_euler()
    cd=bpy.data.cameras.new("REALSAS_DEMO_CAMERA");cam=bpy.data.objects.new("REALSAS_DEMO_CAMERA",cd);bpy.context.collection.objects.link(cam);cam.data.type="ORTHO";s.camera=cam
    return s,cam


def camera_basis(yaw):
    z=Vector((0,0,1));f0=Vector((0,-1,0));f=(Matrix.Rotation(math.radians(yaw),4,z)@f0).normalized();up=Vector((0,0,1));right=f.cross(up).normalized();return f,right,up


def common_half_extent(admitted,dist):
    pts=bbox_points(admitted);best=0.
    for yaw in YAW_DEG:
        f,r,u=camera_basis(yaw);xs=[p.dot(r) for p in pts];ys=[p.dot(u) for p in pts]
        best=max(best,max(max(xs)-min(xs),max(ys)-min(ys))*.5)
    return max(1e-4,best*MARGIN)


def build_world_bvh(admitted):
    dg=bpy.context.evaluated_depsgraph_get();verts=[];polys=[];offset=0
    for o in admitted:
        eo=o.evaluated_get(dg);me=eo.to_mesh()
        verts.extend([tuple(eo.matrix_world@v.co) for v in me.vertices])
        polys.extend([tuple(offset+i for i in p.vertices) for p in me.polygons if len(p.vertices)>=3])
        offset+=len(me.vertices);eo.to_mesh_clear()
    if not verts or not polys:raise RuntimeError("DEMO_PREP_EMPTY_BVH")
    return BVHTree.FromPolygons(verts,polys,all_triangles=False)


def raycast_teacher(bvh,forward,right,up,half):
    d=np.zeros((TEACHER_RES,TEACHER_RES),np.float32);s=np.zeros_like(d);start=2.0
    for py in range(TEACHER_RES):
        gy=((py+.5)/TEACHER_RES)*2-1
        for px in range(TEACHER_RES):
            gx=((px+.5)/TEACHER_RES)*2-1
            plane=gx*half*right-gy*half*up;origin=plane-forward*start
            hit,_,_,_=bvh.ray_cast(origin,forward,start*2)
            if hit is not None:
                d[py,px]=float(hit.dot(forward));s[py,px]=1.0
    return d,s


def weighted_bone_truth(admitted,arm):
    weighted=set()
    for o in admitted:
        if o.type!="MESH":continue
        name_by_idx={g.index:g.name for g in o.vertex_groups};bone_names={b.name for b in arm.data.bones}
        for v in o.data.vertices:
            for g in v.groups:
                if g.weight>1e-8 and name_by_idx.get(g.group) in bone_names:weighted.add(name_by_idx[g.group])
    include=set(weighted)
    for name in list(weighted):
        b=arm.data.bones.get(name)
        while b and b.parent:
            b=b.parent;include.add(b.name)
    roots=[b for b in arm.data.bones if b.name in include and (b.parent is None or b.parent.name not in include)]
    if len(roots)!=1:raise RuntimeError(f"DEMO_PREP_REQUIRES_SINGLE_ROOT:roots={[b.name for b in roots]}")
    root=roots[0];order=[];queue=[root]
    while queue:
        b=queue.pop(0)
        if b.name not in include:continue
        order.append(b);queue.extend(sorted([c for c in b.children if c.name in include],key=lambda x:x.name))
    idx={b.name:i for i,b in enumerate(order)}
    joints=np.asarray([tuple(arm.matrix_world@b.head_local) for b in order],np.float32)
    parent=np.asarray([-1 if b.parent is None or b.parent.name not in idx else idx[b.parent.name] for b in order],np.int64)
    source_p=[];source_w=[]
    for o in admitted:
        if o.type!="MESH":continue
        vg_by_idx={g.index:g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            row=np.zeros(len(order),np.float32)
            for g in v.groups:
                name=vg_by_idx.get(g.group)
                if name in idx and g.weight>0:row[idx[name]]+=float(g.weight)
            total=float(row.sum())
            if total<=1e-8:continue
            row/=total;source_p.append(tuple(o.matrix_world@v.co));source_w.append(row)
    if len(source_p)<32:raise RuntimeError("DEMO_PREP_TOO_FEW_WEIGHTED_SOURCE_VERTICES")
    return joints,parent,0,np.asarray(source_p,np.float32),np.asarray(source_w,np.float32),[b.name for b in order]


def main():
    a=parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);views=out/"views_1024";views.mkdir(exist_ok=True)
    import_asset(a.input);meshes=[o for o in bpy.context.scene.objects if o.type=="MESH" and not o.hide_render];arms=[o for o in bpy.context.scene.objects if o.type=="ARMATURE"]
    arm=resolve_primary(meshes,arms)
    if arm is None:raise RuntimeError("DEMO_PREP_NO_PRIMARY_ARMATURE_WITH_SKIN")
    try:arm.data.pose_position="REST"
    except Exception:pass
    admitted,excluded=subject_gate(meshes,arm)
    if not admitted:raise RuntimeError("DEMO_PREP_NO_ADMITTED_SUBJECT")
    for o in excluded:o.hide_render=True;o.hide_viewport=True
    _,radius,up_method=normalize_subject(admitted,arm);scene,cam=setup_render();dist=max(2.0,radius*6);half=common_half_extent(admitted,dist);cam.data.ortho_scale=2*half
    bvh=build_world_bvh(admitted);rgba=[];depth=[];support=[];cameras=[]
    for vi,yaw in enumerate(YAW_DEG):
        f,r,u=camera_basis(yaw);cam.location=-f*dist;cam.rotation_euler=((Vector((0,0,0))-cam.location).to_track_quat("-Z","Y")).to_euler();bpy.context.view_layer.update()
        fn=views/f"V{vi}.png";scene.render.filepath=str(fn);log(f"RENDER {vi}/8 yaw={yaw}");bpy.ops.render.render(write_still=True)
        im=bpy.data.images.load(str(fn),check_existing=False);arr=np.asarray(im.pixels[:],np.float32).reshape(im.size[1],im.size[0],4);rgba.append(arr.copy());bpy.data.images.remove(im)
        d,s=raycast_teacher(bvh,f,r,u,half);depth.append(d);support.append(s)
        cameras.append({"contract":"realsas.level_orthographic_z_orbit.v1","yaw_deg":yaw,"forward":list(map(float,f)),"right":list(map(float,r)),"screen_up":list(map(float,u)),"half_extent":float(half)})
    rgba=np.asarray(rgba,np.float32);depth=np.asarray(depth,np.float32);support=np.asarray(support,np.float32)
    joints,parent,root_index,sv,sw,bone_names=weighted_bone_truth(admitted,arm)
    np.savez_compressed(out/"image_only_inputs.npz",rgba=rgba,yaw_deg=np.asarray(YAW_DEG,np.float32))
    np.savez_compressed(out/"observations.npz",rgba=rgba,yaw_deg=np.asarray(YAW_DEG,np.float32),teacher_depth=depth,teacher_support=support)
    np.savez_compressed(out/"mechanical_truth.npz",joint_positions=joints,parent_index=parent,root_index=np.asarray([root_index],np.int64),source_vertices=sv,source_vertex_weights=sw)
    (out/"cameras.json").write_text(json.dumps(cameras,indent=2),encoding="utf-8")
    report={"schema":"RealSaS.DemoSpecimenPrep.v1","status":"PASS","input":str(Path(a.input).resolve()),"admitted_objects":[o.name for o in admitted],"excluded_objects":[o.name for o in excluded],"primary_armature":arm.name,"up_method":up_method,"radius":radius,"half_extent":half,"joint_count":len(joints),"weighted_source_vertex_count":len(sv),"bone_names_for_audit_only":bone_names,"image_only_payload_contains_truth":False,"generalization_claim":False}
    (out/"PREP_REPORT_V1.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");log("PASS")

if __name__=="__main__":
    try:main()
    except Exception as e:
        import traceback;log("FAIL "+type(e).__name__+":"+str(e));traceback.print_exc();raise
