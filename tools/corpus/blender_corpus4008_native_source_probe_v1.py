#!/usr/bin/env python3
"""Pinned Blender imported-source scene and mechanics preflight, not truth.

Run from Blender --background --python ... -- --source ... --expected-sha ...
Keeps observed importer geometry, transforms, bone-parented meshes, rig and
material inventory distinct from SOURCE TRUTH, semantic regions, mechanical
attachment behavior, and anatomical camera authority. No training admitted.

Native FBX importer option use_anim=False preserves source rest for geometry;
a separate fresh import with use_anim=True inventories authored actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "RealSaS.Corpus4008BlenderSourceScenePreflight.v1"


def require(cond,reason):
    if not cond:
        raise ValueError("CORPUS4008_BLENDER_IMPORT_"+reason)


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8<<20),b""):
            h.update(part)
    return h.hexdigest()


def topology_of_polygons(n_vertices, polygons):
    """Pure indexed incidence only; no automatic watertight or semantic claim."""
    require(0<=n_vertices<=15_000_000,"VERTEX_BUDGET")
    parent=list(range(n_vertices))
    active=set()
    edges=Counter()

    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i

    for face in polygons:
        require(len(face)>=3,"SHORT_POLYGON")
        for i in face:
            require(type(i) is int and 0<=i<n_vertices,"POLYGON_VERTEX_OOB")
            active.add(i)
        for a,b in zip(face,face[1:]+face[:1]):
            if a==b:
                continue
            parent[root(a)]=root(b)
            edges[(min(a,b),max(a,b))]+=1
    roots={root(i) for i in active}
    return {
        "source_vertex_count":n_vertices,
        "polygon_used_vertex_count":len(active),
        "loose_vertex_count":n_vertices-len(active),
        "triangle_fan_equivalent_count":sum(max(0,len(f)-2) for f in polygons),
        "source_index_component_count":len(roots),
        "boundary_edge_count":sum(n==1 for n in edges.values()),
        "nonmanifold_edge_count":sum(n>2 for n in edges.values()),
        "all_used_edges_two_sided":bool(edges) and all(n==2 for n in edges.values()),
        "topology_index_measurement_only":True,
        "watertight_solid_sdf_authorized":False,
        "drawable_count_minted":False,
    }


def finite_matrix(obj):
    flat=[float(obj[r][c]) for r in range(4) for c in range(4)]
    require(all(math.isfinite(x) for x in flat),"NONFINITE_IMPORT_MATRIX")
    return flat


def import_source(bpy,path,*,load_anim):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext=path.suffix.lower()
    if ext==".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path),
            automatic_bone_orientation=False,use_anim=load_anim)
    elif ext in {".gltf",".glb"}:
        bpy.ops.import_scene.gltf(filepath=str(path),guess_original_bind_pose=True)
    elif ext==".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise ValueError("UNSUPPORTED_EXTENSION_FOR_PINNED_BLENDER")
    for o in bpy.context.scene.objects:
        if o.type=="ARMATURE":
            o.data.pose_position="REST"
    bpy.context.view_layer.update()


def material_inventory(bpy):
    imgs=[]
    for im in bpy.data.images:
        if im.name in {"Render Result","Viewer Node"}: continue
        path=str(getattr(im,"filepath","") or "")
        packed=bool(getattr(im,"packed_file",None))
        external_present=None
        if path and not packed:
            try:external_present=Path(bpy.path.abspath(path,library=im.library)).exists()
            except Exception:external_present=False
        imgs.append({
            "image_name_teacher_only":im.name,
            "external_source_path_teacher_only":path,
            "packed":packed,
            "external_source_found":external_present,
            "image_source":str(getattr(im,"source","")),
        })
    return {
        "material_count":len(bpy.data.materials),
        "image_count":len(imgs),
        "images_teacher_only":imgs,
        "images_missing_external_count":sum(x["external_source_found"] is False for x in imgs),
        "source_original_material_parity_courted":False,
        "source_native_rgba_render_courted":False,
    }


def scene_import_report(bpy):
    objs=sorted(bpy.context.scene.objects,key=lambda x:x.name)
    aid={x.name:i for i,x in enumerate(objs)}
    nodes=[]
    mesh_rows=[]
    rig_rows=[]
    bone_parent=[]
    for o in objs:
        parent=aid.get(o.parent.name) if o.parent else None
        rec={
            "imported_scene_node_index":aid[o.name],
            "name_teacher_only":o.name,
            "object_type":o.type,
            "parent_node_index":parent,
            "parent_type":o.parent_type,
            "parent_bone_teacher_only":o.parent_bone or None,
            "matrix_world_imported":finite_matrix(o.matrix_world),
            "matrix_local_imported":finite_matrix(o.matrix_local),
            "matrix_parent_inverse_imported":finite_matrix(o.matrix_parent_inverse),
            "hide_render":bool(o.hide_render),
            "constraints":[x.type for x in o.constraints],
            "modifiers":[x.type for x in o.modifiers],
        }
        nodes.append(rec)
        if o.type=="MESH":
            me=o.data
            polygons=[list(map(int,p.vertices)) for p in me.polygons]
            topo=topology_of_polygons(len(me.vertices),polygons)
            groups={g.index:g.name for g in o.vertex_groups}
            sampled=range(len(me.vertices)) if len(me.vertices)<=64000 else range(0,len(me.vertices),max(1,len(me.vertices)//64000))
            sample_rows=0
            abs_bad=0
            max_err=0.0
            zero_rows=0
            for idx in sampled:
                v=me.vertices[idx]
                ws=[float(w.weight) for w in v.groups if w.group in groups]
                if not ws:zero_rows+=1
                if ws:
                    err=abs(sum(ws)-1)
                    max_err=max(max_err,err)
                    if err>1e-3:abs_bad+=1
                sample_rows+=1
            mesh_rows.append({
                "scene_node_index":aid[o.name],
                "vertex_count":len(me.vertices),
                "polygon_count":len(polygons),
                "index_topology":topo,
                "material_slots":len(o.material_slots),
                "vertex_groups":len(groups),
                "skin_rows_sampled":sample_rows,
                "skin_rows_without_any_groups":zero_rows,
                "skin_rows_nonunity_above_1e-3":abs_bad,
                "sampled_skin_row_max_abs_error":max_err,
                "actual_skin_binding_court":False,
                "source_group_is_semantic_component":False,
            })
            if o.parent and o.parent_type=="BONE":
                bone_parent.append({
                    "mesh_node_index":aid[o.name],
                    "armature_node_index":aid.get(o.parent.name),
                    "parent_bone_teacher_only":o.parent_bone,
                    "candidate_kind":"IMPORTED_BONE_PARENTED_MESH",
                    "attachment_motion_consequence_courted":False,
                })
        elif o.type=="ARMATURE":
            bones=[]
            for bone in o.data.bones:
                bones.append({
                    "name_teacher_only":bone.name,
                    "parent_name_teacher_only":bone.parent.name if bone.parent else None,
                    "use_deform":bool(bone.use_deform),
                    "matrix_local_imported":finite_matrix(bone.matrix_local),
                    "head_local_imported":list(map(float,bone.head_local)),
                    "tail_local_imported":list(map(float,bone.tail_local)),
                })
            rig_rows.append({
                "scene_node_index":aid[o.name],
                "bone_count":len(bones),
                "deform_bone_count":sum(x["use_deform"] for x in bones),
                "bones_teacher_only":bones,
                "rigprogram_or_controller_courted":False,
            })
    return {
        "imported_nodes":nodes,
        "meshes":mesh_rows,
        "armatures":rig_rows,
        "bone_parented_unskinned_mesh_candidates":bone_parent,
        "bone_parented_mesh_candidate_count":len(bone_parent),
        "source_rest_import_observed":True,
        "source_rest_parity_independently_courted":False,
        "source_anatomical_up":"UNKNOWN",
        "source_anatomical_front":"UNKNOWN",
        "source_original_pose_class":"UNKNOWN",
        "character_vs_prop":"UNKNOWN",
        "morphology":"UNKNOWN",
        "native_camera_courted":False,
        "eight_view_camera_courted":False,
        "semantic_teacher_courted":False,
        "graphics_source_rgba_courted":False,
        "training_authorized":False,
    }


def run(bpy,*,source,expected_sha,asset_id,candidate_id):
    require(source.is_file(),"SOURCE_NOT_FOUND")
    actual=sha(source)
    require(actual==expected_sha,"EXACT_ORIGINAL_SHA")
    require(candidate_id.endswith(expected_sha) or candidate_id.endswith(expected_sha[:24]),
            "CANDIDATE_SOURCE_SHA_DRIFT")
    import_source(bpy,source,load_anim=False)
    scene=scene_import_report(bpy)
    materials=material_inventory(bpy)
    # Fresh animated import is a separate measurement: do not interpret frame 0
    # as rest, and do not let action binding influence rest pose snapshot above.
    source_action_names=[]
    if source.suffix.lower() in {".fbx",".glb",".gltf"}:
        import_source(bpy,source,load_anim=True)
        source_action_names=sorted(a.name for a in bpy.data.actions)
    return {
        "schema":SCHEMA,
        "status":"PINNED_BLENDER_IMPORTED_SOURCE_SCENE__NO_INDEPENDENT_COURTS",
        "asset_id":asset_id,
        "candidate_id":candidate_id,
        "original_sha256":actual,
        "source_format":source.suffix.lower(),
        "blender_importer_version":bpy.app.version_string,
        "importer_coordinate_owner":"BLENDER_IMPORT_CONVENTION__NOT_AUTHORED_ANATOMY",
        "imported_source_scene":scene,
        "material_image_inventory":materials,
        "native_action_names_teacher_only":source_action_names,
        "native_action_inventory_courted":False,
        "source_original_bytes_preserved":True,
        "animation_ready_mesh_authorized":False,
        "training_authorized":False,
    }


def main():
    import bpy
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--expected-sha256",required=True)
    ap.add_argument("--asset-id",required=True)
    ap.add_argument("--candidate-id",required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    o=run(bpy,source=a.source.resolve(),expected_sha=a.expected_sha256,
          asset_id=a.asset_id,candidate_id=a.candidate_id)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":o["status"],
        "source_format":o["source_format"],
        "mesh_count":len(o["imported_source_scene"]["meshes"]),
        "armature_count":len(o["imported_source_scene"]["armatures"]),
        "bone_parent_candidates":o["imported_source_scene"]["bone_parented_mesh_candidate_count"],
        "native_action_inventory_courted":False,
        "training_authorized":False,
    },sort_keys=True))


if __name__=="__main__":
    import sys
    argv=sys.argv[sys.argv.index("--")+1:]
    old_argv=sys.argv
    try:
        sys.argv=[sys.argv[0]]+argv
        main()
    finally:sys.argv=old_argv
