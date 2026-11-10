#!/usr/bin/env python3
"""Real isolated imported-bone commands, never semantic drawable ownership.

Runs under byte-verified Blender. Rig/skin and exact imported vertex groups are
teacher-only control identities. Does not infer labels from names, components,
colors or co-motion. Absent suitable parent/child controls are MASKED, not FAIL.
The independent numeric court must run separately on the resulting NPZ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4

SCHEMA="RealSaS.Corpus370NativeBoneCounterfactual.v1"
STATES=("ZERO_CONTROL","ZERO_RESET","A_POS","A_NEG",
        "B_POS","B_NEG","AB_POS","COMMON_ROOT_SHAM")
ANGLE=math.radians(6.0)


def require(ok,code):
    if not ok:raise RuntimeError(code)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def _select_pair(mesh_objects):
    # Source vertex group weights establish a candidate rig command set.
    # The bone / group names are addresses only and are never student targets.
    options=[]
    armatures=sorted((obj for obj in bpy.context.scene.objects
                      if obj.type=="ARMATURE"),key=lambda a:a.name)
    for arm in armatures:
        masses={}
        for obj in mesh_objects:
            if not any(mod.type=="ARMATURE" and mod.object is arm
                       and mod.show_viewport for mod in obj.modifiers):
                continue
            groups={g.index:g.name for g in obj.vertex_groups}
            for vertex in obj.data.vertices:
                for g in vertex.groups:
                    name=groups.get(g.group)
                    if name in arm.pose.bones and g.weight>0:
                        masses[name]=masses.get(name,0.)+float(g.weight)
        bones={bone.name:bone for bone in arm.pose.bones}
        for child in arm.pose.bones:
            parent=child.parent
            if parent is None: continue
            a=masses.get(parent.name,0.);b=masses.get(child.name,0.)
            if a<=1e-8 or b<=1e-8: continue
            # Mass is selection before intervention outcomes, not semantics.
            options.append((-min(a,b),-max(a,b),arm.name,parent.name,
                            child.name,arm,parent,child))
    if not options:return None
    options.sort(key=lambda r:r[:5])
    _,_,_,_,_,arm,parent,child=options[0]
    root=parent.parent
    return arm,parent,child,root


def _restore(arms):
    for arm in arms:
        arm.data.pose_position="POSE"
        for bone in arm.pose.bones:
            bone.matrix_basis=Matrix.Identity(4)
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()


def _snapshot(mesh_objects,signature,faces,owners,materials):
    snap=v4._extract_snapshot(mesh_objects,freeze_triangulation=False)
    require(snap["polygon_signature"]==signature
            and snap["vertices"].ndim==2
            and np.isfinite(snap["vertices"]).all()
            and len(snap["owners"])==len(owners),
            "NATIVE_BONE_INTERVENTION_SOURCE_TOPOLOGY_DRIFT")
    return snap["vertices"]


def main():
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--subject-id",required=True)
    p.add_argument("--admission",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(raw)
    source=a.source.resolve()
    admission=a.admission.resolve()
    out=a.output.resolve()
    require(source.is_file() and sha(source)==a.source_sha256,
            "NATIVE_BONE_INTERVENTION_SOURCE_SHA")
    adm=json.loads(admission.read_text())
    require(adm.get("subject_id")==a.subject_id
            and adm.get("source_sha256")==a.source_sha256
            and adm.get("native_motion_inventory"),
            "NATIVE_BONE_INTERVENTION_ADMISSION_SHA_OR_MOTION")
    base._import_animated(source)
    mesh_objects=base._mesh_objects()
    base._neutralize_animation()
    rest=v4._extract_snapshot(mesh_objects,freeze_triangulation=True)
    require(len(rest["vertices"])>0 and len(rest["faces"])>0,
            "NATIVE_BONE_INTERVENTION_NO_REST_SOURCE")
    path=out.with_suffix(out.suffix+".json")
    data={
        "schema":SCHEMA,"subject_id":a.subject_id,
        "status":"MASKED_NO_TWO_WEIGHTED_PARENT_CHILD_COMMANDS",
        "source_sha256":a.source_sha256,
        "observation_admission_sha256":sha(admission),
        "source_domain":"SOURCE_WORLD__NOT_CANONICAL",
        "control_policy":"RIG_WEIGHTED_SOURCE_PARENT_CHILD_WITH_6DEG_LOCAL_Z_ISOLATED_COMMANDS",
        "control_selection_uses_render_or_motion_outcome":False,
        "labels_from_source_names":False,
        "teacher_only":True,"independent_replay_courted":False,
        "same_drawable":"UNKNOWN","different_drawable":"UNKNOWN",
        "mechanical_owner":"UNKNOWN","causal_response_authorized":False,
        "training_authorized":False
    }
    pair=_select_pair(mesh_objects)
    if pair is None:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
        print("NATIVE_BONE_COUNTERFACTUAL_MASKED",a.subject_id,flush=True)
        return
    arm,bone_a,bone_b,root=pair
    arms=[obj for obj in bpy.context.scene.objects if obj.type=="ARMATURE"]
    poses=[]
    for name in STATES:
        _restore(arms)
        if name in ("A_POS","A_NEG","B_POS","B_NEG","AB_POS","COMMON_ROOT_SHAM"):
            def turn(bone,value):
                bone.matrix_basis=Matrix.Rotation(float(value),4,"Z")
            if name in ("A_POS","A_NEG"):
                turn(bone_a,ANGLE if name=="A_POS" else -ANGLE)
            elif name in ("B_POS","B_NEG"):
                turn(bone_b,ANGLE if name=="B_POS" else -ANGLE)
            elif name=="AB_POS":
                turn(bone_a,ANGLE);turn(bone_b,ANGLE)
            elif root is not None:
                turn(root,ANGLE)
        bpy.context.view_layer.update()
        poses.append(_snapshot(mesh_objects,rest["polygon_signature"],
                               rest["faces"],rest["face_owner"],
                               rest["face_material"]))
    _restore(arms)
    values=np.stack(poses,axis=0)
    require(values.shape==(len(STATES),*rest["vertices"].shape),
            "NATIVE_BONE_INTERVENTION_POSE_SHAPE")
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,
        rest_vertices_source_world=rest["vertices"],
        faces=rest["faces"],face_owner=rest["face_owner"],
        face_material=rest["face_material"],
        intervention_positions_source_world=values)
    data.update({
        "status":"MEASURED_ISOLATED_RIG_CONTROL_CANDIDATE__DRAWABLE_AND_CAUSAL_COURTS_SEPARATE",
        "output":{"path":str(out),"sha256":sha(out),"size_bytes":out.stat().st_size},
        "control_states":list(STATES),
        "amplitude_radians":ANGLE,
        "control_parent_child_bones_teacher_only":{
            "armature_name":arm.name,
            "parent_bone_name":bone_a.name,
            "child_bone_name":bone_b.name,
            "common_ancestor":root.name if root is not None else None,
        },
        "root_sham_available":root is not None,
        "rest_vertex_count":len(rest["vertices"]),
        "face_count":len(rest["faces"]),
        "pose_count":len(STATES),
        "rest_face_triangulation_frozen":True,
        "source_polygon_connectivity_stable":True,
    })
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    print("NATIVE_BONE_COUNTERFACTUAL_MEASURED",a.subject_id,
          "pose_count=",len(STATES),"sha=",sha(out),flush=True)


if __name__=="__main__":
    main()
