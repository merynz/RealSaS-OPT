from __future__ import annotations

"""Deterministic full-subject eight-view Blender observation materializer.

This is subject-agnostic. It deliberately forbids the Mage-era failure mode:
no body-only framing, no per-view bbox/centering/scale, no per-view pose edits.
All renderable imported meshes are framed once, then viewed by one frozen
level +Z-up orthographic orbit.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

VIEW_ORDER=("S","SE","E","NE","N","NW","W","SW")
VIEW_YAW_DEG=(0,45,90,135,180,225,270,315)
UP=Vector((0.0,0.0,1.0))


def args():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--source-fbx",required=True)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--subject-id",default="SUBJECT")
    p.add_argument("--resolution",type=int,default=1024)
    p.add_argument("--frame-scale",type=float,default=1.12)
    p.add_argument("--minimum-border-margin-px",type=int,default=8)
    return p.parse_args(raw)


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def dump(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def box(points):
    mn=[min(float(p[i]) for p in points) for i in range(3)]
    mx=[max(float(p[i]) for p in points) for i in range(3)]
    return {"min_xyz":mn,"max_xyz":mx,"size_xyz":[mx[i]-mn[i] for i in range(3)]}


def basis(deg):
    t=math.radians(float(deg))
    radial=Vector((math.sin(t),-math.cos(t),0.0)).normalized()
    forward=(-radial).normalized()
    right=forward.cross(UP).normalized()
    if abs(forward.dot(UP))>1e-9 or abs(right.dot(UP))>1e-9:
        raise RuntimeError("OBSERVATION_CAMERA_BASIS_DRIFT")
    return radial,forward,right,UP.copy()


def reset_and_import(path:Path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(path),automatic_bone_orientation=False)
    bpy.context.scene.frame_set(0)
    for o in bpy.context.scene.objects:
        if o.type=="ARMATURE":
            o.data.pose_position="REST"
    bpy.context.view_layer.update()


def full_subject():
    meshes=sorted(
        (o for o in bpy.context.scene.objects if o.type=="MESH" and not o.hide_render),
        key=lambda o:o.name,
    )
    if not meshes: raise RuntimeError("OBSERVATION_NO_RENDERABLE_MESH")
    dg=bpy.context.evaluated_depsgraph_get()
    points=[]; by_object={}
    for o in meshes:
        eo=o.evaluated_get(dg)
        m=eo.to_mesh(preserve_all_data_layers=True,depsgraph=dg)
        try: pts=[eo.matrix_world@v.co for v in m.vertices]
        finally: eo.to_mesh_clear()
        if pts:
            by_object[o.name]=pts
            points.extend(pts)
    if not points: raise RuntimeError("OBSERVATION_EMPTY_FULL_SUBJECT")
    return meshes,points,by_object


def solve_frame(points,scale):
    if not math.isfinite(scale) or scale<=1.0:
        raise RuntimeError("OBSERVATION_FRAME_SCALE_INVALID")
    b=box(points)
    center=Vector(tuple((b["min_xyz"][i]+b["max_xyz"][i])*0.5 for i in range(3)))
    screen_max=0.0; per=[]
    for name,yaw in zip(VIEW_ORDER,VIEW_YAW_DEG):
        _,_,right,up=basis(yaw)
        extent=max(
            max(abs((p-center).dot(right)) for p in points),
            max(abs((p-center).dot(up)) for p in points),
        )
        screen_max=max(screen_max,float(extent))
        per.append({"view":name,"yaw_deg":yaw,"required_half_extent":float(extent)})
    xyz_max=max(abs(float(p[i]-center[i])) for p in points for i in range(3))
    half=max(screen_max,xyz_max)*float(scale)
    return center,half,{
        "full_subject_world_bbox":b,
        "ground_z":float(b["min_xyz"][2]),
        "height":float(b["size_xyz"][2]),
        "max_screen_abs_before_margin":screen_max,
        "max_xyz_abs_before_margin":xyz_max,
        "frame_scale":float(scale),
        "per_view_required_half_extent":per,
    }


def material_audit():
    rows=[]
    for m in sorted((x for x in bpy.data.materials if x),key=lambda x:x.name):
        images=[]
        if m.use_nodes and m.node_tree:
            for n in m.node_tree.nodes:
                if n.type=="TEX_IMAGE" and getattr(n,"image",None):
                    im=n.image
                    images.append({
                        "image_name":im.name,
                        "filepath":bpy.path.abspath(im.filepath) if im.filepath else "",
                        "colorspace":im.colorspace_settings.name,
                    })
        rows.append({"name":m.name,"use_nodes":bool(m.use_nodes),"image_textures":images})
    return rows


def force_unlit():
    # Preserve source base-color signal, remove lighting/material response.
    for mat in [m for m in bpy.data.materials if m]:
        mat.use_nodes=True
        nt=mat.node_tree
        if not nt: continue
        out=next((n for n in nt.nodes if n.type=="OUTPUT_MATERIAL"),None)
        if out is None: out=nt.nodes.new("ShaderNodeOutputMaterial")
        p=next((n for n in nt.nodes if n.type=="BSDF_PRINCIPLED"),None)
        e=nt.nodes.new("ShaderNodeEmission")
        e.inputs["Strength"].default_value=1.0
        if p is not None and p.inputs.get("Base Color") is not None:
            src=p.inputs["Base Color"]
            if src.is_linked: nt.links.new(src.links[0].from_socket,e.inputs["Color"])
            else: e.inputs["Color"].default_value=tuple(src.default_value)
        else:
            e.inputs["Color"].default_value=tuple(mat.diffuse_color)
        for link in list(out.inputs["Surface"].links): nt.links.remove(link)
        nt.links.new(e.outputs[0],out.inputs["Surface"])


def armature_rows():
    rows=[]
    for o in sorted((x for x in bpy.context.scene.objects if x.type=="ARMATURE"),key=lambda x:x.name):
        rows.append({
            "object_name":o.name,
            "pose_position":o.data.pose_position,
            "bone_count":len(o.data.bones),
            "bones":[{
                "name":b.name,
                "parent":b.parent.name if b.parent else None,
                "head_local":list(map(float,b.head_local)),
                "tail_local":list(map(float,b.tail_local)),
                "use_deform":bool(b.use_deform),
            } for b in o.data.bones],
        })
    return rows


def mesh_rows(meshes,by_object):
    rows=[]
    for o in meshes:
        group_stats={int(g.index):{"name":g.name,"weighted_vertex_count":0,"max_weight":0.0} for g in o.vertex_groups}
        for v in o.data.vertices:
            for membership in v.groups:
                row=group_stats.get(int(membership.group))
                if row is None:
                    continue
                w=float(membership.weight)
                if w>0.0:
                    row["weighted_vertex_count"]+=1
                    row["max_weight"]=max(float(row["max_weight"]),w)
        group_counts=[
            group_stats[idx] for idx in sorted(group_stats)
            if int(group_stats[idx]["weighted_vertex_count"])>0
        ]
        rows.append({
            "object_name":o.name,
            "parent":o.parent.name if o.parent else None,
            "vertex_count":len(o.data.vertices),
            "polygon_count":len(o.data.polygons),
            "material_slots":[s.material.name if s.material else None for s in o.material_slots],
            "modifiers":[{"name":m.name,"type":m.type,"object":getattr(getattr(m,"object",None),"name",None)} for m in o.modifiers],
            "vertex_groups":group_counts,
            "evaluated_world_bbox":box(by_object[o.name]) if o.name in by_object else None,
        })
    return rows


def configure(res):
    s=bpy.context.scene
    s.render.engine="BLENDER_EEVEE_NEXT"
    s.render.resolution_x=res; s.render.resolution_y=res; s.render.resolution_percentage=100
    s.render.image_settings.file_format="PNG"; s.render.image_settings.color_mode="RGBA"; s.render.image_settings.color_depth="8"
    s.render.film_transparent=True
    s.view_settings.view_transform="Standard"; s.view_settings.exposure=0.0; s.view_settings.gamma=1.0
    for o in list(s.objects):
        if o.type=="LIGHT": bpy.data.objects.remove(o,do_unlink=True)


def render_all(out_dir,center,half,res,min_margin):
    data=bpy.data.cameras.new("RealSaS_Controlled8View_Camera")
    data.type="ORTHO"; data.ortho_scale=2.0*half
    cam=bpy.data.objects.new("RealSaS_Controlled8View_Camera",data)
    bpy.context.scene.collection.objects.link(cam); bpy.context.scene.camera=cam
    distance=max(4.0*half,1.0)
    cams=[]; views=[]
    for vi,(name,yaw) in enumerate(zip(VIEW_ORDER,VIEW_YAW_DEG)):
        radial,forward,right,up=basis(yaw)
        origin=center+radial*distance
        cam.location=origin
        cam.rotation_euler=forward.to_track_quat("-Z","Y").to_euler()
        bpy.context.view_layer.update()
        png=out_dir/f"V{vi}.png"
        bpy.context.scene.render.filepath=str(png)
        bpy.ops.render.render(write_still=True)
        rr=bpy.data.images.get("Render Result")
        if rr is None: raise RuntimeError("OBSERVATION_RENDER_RESULT_MISSING")
        w,h=map(int,rr.size)
        if (w,h)!=(res,res): raise RuntimeError("OBSERVATION_RESOLUTION_DRIFT")
        pix=list(rr.pixels[:]); mask=bytearray(w*h); xs=[]; ys=[]
        # Blender render pixels are bottom-up; observation mask bytes are top-down.
        for yb in range(h):
            y=h-1-yb
            for x in range(w):
                fg=1 if float(pix[(yb*w+x)*4+3])>1e-6 else 0
                mask[y*w+x]=fg
                if fg: xs.append(x); ys.append(y)
        if not xs: raise RuntimeError(f"OBSERVATION_EMPTY_FOREGROUND:V{vi}")
        mp=out_dir/f"V{vi}.mask.bin"; mp.write_bytes(bytes(mask))
        margin=int(min(min(xs),min(ys),w-1-max(xs),h-1-max(ys)))
        if margin<min_margin: raise RuntimeError(f"OBSERVATION_BORDER_MARGIN_FAIL:V{vi}:{margin}<{min_margin}")
        cams.append({
            "schema_version":"RealSaS.FullSurfaceCameraProjection.v3",
            "view_id":f"V{vi}","view_index":vi,"origin":list(map(float,origin)),
            "right":list(map(float,right)),"screen_up":list(map(float,up)),
            "forward":list(map(float,forward)),"half_extent":float(half),"resolution":res,
        })
        views.append({
            "view_index":vi,"view_id":f"V{vi}","direction_label":name,"yaw_deg":yaw,
            "raster":{"path":str(png.resolve()),"sha256":sha(png)},
            "foreground_mask":{"path":str(mp.resolve()),"sha256":sha(mp)},
            "alpha_bbox_xyxy":[min(xs),min(ys),max(xs),max(ys)],
            "foreground_pixel_count":int(sum(mask)),"border_margin_px":margin,
        })
    return cams,views


def main():
    a=args(); source=Path(a.source_fbx).expanduser().resolve(); out=Path(a.out_dir).expanduser().resolve()
    if not source.is_file(): raise RuntimeError("OBSERVATION_SOURCE_FBX_MISSING")
    out.mkdir(parents=True,exist_ok=True)
    reset_and_import(source)
    meshes,points,by_object=full_subject()
    center,half,frame=solve_frame(points,a.frame_scale)
    audit={
        "schema":"RealSaS.SourceMechanicalSceneAudit.v1","subject_id":a.subject_id,
        "source_fbx":{"path":str(source),"sha256":sha(source),"bytes":source.stat().st_size},
        "pose_policy":"SOURCE_BIND_REST_V1",
        "full_admitted_subject_policy":"ALL_RENDERABLE_IMPORTED_MESH_OBJECTS_V1",
        "world_up":[0.0,0.0,1.0],"per_view_recenter_forbidden":True,
        "per_view_pose_change_forbidden":True,"uniform_scale_only":True,
        "full_subject_world_bbox":frame["full_subject_world_bbox"],
        "canonical_center_xyz":list(map(float,center)),"canonical_ground_z":frame["ground_z"],
        "canonical_height":frame["height"],"armatures":armature_rows(),
        "mesh_objects":mesh_rows(meshes,by_object),"materials":material_audit(),
    }
    ap=out/"SUBJECT_SOURCE_AUDIT.json"; dump(ap,audit)
    force_unlit(); configure(a.resolution)
    cams,views=render_all(out,center,half,a.resolution,a.minimum_border_margin_px)
    cp=out/"camera_bundle.json"
    dump(cp,{
        "schema":"RealSaS.CameraProjectionBundle.v1",
        "view_contract":"LEVEL_ORTHOGRAPHIC_ORBIT_ABOUT_GLOBAL_Z_V1",
        "view_order":list(VIEW_ORDER),"view_yaw_deg":list(VIEW_YAW_DEG),
        "common_center_xyz":list(map(float,center)),"common_half_extent":float(half),
        "cameras":cams,
    })
    norm=[[float((p[i]-center[i])/half) for i in range(3)] for p in points]
    maxabs=[max(abs(r[i]) for r in norm) for i in range(3)]
    outside=sum(1 for r in norm if max(map(abs,r))>1.0+1e-9)
    np=out/"normalization.json"
    dump(np,{
        "schema":"RealSaS.CanonicalSubjectNormalization.v1","subject_id":a.subject_id,
        "coordinate_frame":"REALSAS_OBJECT_FRAME","center_xyz":list(map(float,center)),
        "half_extent":float(half),"normalized_bbox":{
            "min_xyz":[min(r[i] for r in norm) for i in range(3)],
            "max_xyz":[max(r[i] for r in norm) for i in range(3)]},
        "max_abs_normalized_coordinate_by_axis":maxabs,
        "outside_domain_vertex_count":outside,
        "margin_to_domain_boundary":1.0-max(maxabs),
        "full_admitted_vertex_count":len(points),
    })
    if outside!=0 or 1.0-max(maxabs)<=0.0: raise RuntimeError("OBSERVATION_NORMALIZATION_DOMAIN_FAIL")
    rp=out/"RENDER_RECEIPT.json"
    dump(rp,{
        "schema":"RealSaS.Controlled8ViewObservationMaterialization.v1","status":"PASS",
        "subject_id":a.subject_id,"source_fbx_sha256":sha(source),"resolution":a.resolution,
        "camera_contract":{"orbit_axis":"+Z","camera_elevation_deg":0.0,"screen_up":"+Z",
            "S_camera_radial":"-Y","S_camera_forward":"+Y","view_order":list(VIEW_ORDER),
            "view_yaw_deg":list(VIEW_YAW_DEG),"common_center_xyz":list(map(float,center)),
            "common_half_extent":float(half),"per_view_recenter":False,"per_view_scale":False},
        "appearance_contract":{"source_base_color_only":True,"lighting":"NONE__EMISSION_ONLY",
            "texture_redesign":False,"image_space_warp":False,"silhouette_cleanup":False,"ai_correction":False},
        "frame_report":frame,"minimum_required_border_margin_px":a.minimum_border_margin_px,
        "minimum_observed_border_margin_px":min(v["border_margin_px"] for v in views),
        "all_views_meet_margin":all(v["border_margin_px"]>=a.minimum_border_margin_px for v in views),
        "views":views,"source_audit":{"path":str(ap.resolve()),"sha256":sha(ap)},
        "camera_bundle":{"path":str(cp.resolve()),"sha256":sha(cp)},
        "normalization":{"path":str(np.resolve()),"sha256":sha(np)},
    })
    print("CONTROLLED_8VIEW_PASS")
    print(json.dumps({"half_extent":half,"min_margin":min(v["border_margin_px"] for v in views)},sort_keys=True))


if __name__=="__main__":
    main()
