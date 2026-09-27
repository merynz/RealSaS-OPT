from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector

REPO=Path.cwd().resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from tools.motion.blender_extract_motion_source_v2 import (
    action_by_requested_take,
    canonical_basis,
    canonical_transform,
    import_fbx,
    selected_bones,
    selected_parent,
)


def _assign_exact_action(armature, action):
    if armature.animation_data is None:
        armature.animation_data_create()
    ad=armature.animation_data
    if hasattr(ad,"use_nla"):
        ad.use_nla=False
    for track in tuple(ad.nla_tracks):
        track.mute=True
    ad.action=action
    slot_identifier="LEGACY_ACTION_NO_SLOT_API"
    if hasattr(ad,"action_suitable_slots"):
        suitable=tuple(ad.action_suitable_slots)
        if not suitable:
            raise RuntimeError(
                "MOTION_ACTION_DIAGNOSTIC_NO_SUITABLE_SLOT:"+action.name
            )
        ad.action_slot=suitable[0]
        slot_identifier=str(suitable[0].identifier)
    if hasattr(ad,"action_blend_type"):
        ad.action_blend_type="REPLACE"
    if hasattr(ad,"action_influence"):
        ad.action_influence=1.0
    return slot_identifier


def _hash_rows(rows):
    payload=json.dumps(
        rows,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def diagnose(source_fbx:Path,spec_path:Path):
    spec=json.loads(spec_path.read_text())
    armature=import_fbx(source_fbx)
    C=canonical_basis(armature)
    bones=selected_bones(armature)
    names={b.name for b in bones}
    parent={b.name:selected_parent(b,names) for b in bones}
    rest_global={b.name:canonical_transform(b.matrix_local,C) for b in bones}
    root_name=next(name for name,p in parent.items() if p is None)
    root_pos=rest_global[root_name].to_translation()
    body_scale=max(
        (
            (rest_global[b.name].to_translation()-root_pos).length
            for b in bones
        ),
        default=1.0,
    )
    body_scale=max(float(body_scale),1e-8)
    scene=bpy.context.scene
    fps=float(scene.render.fps)/float(scene.render.fps_base or 1.0)

    reports=[]
    stream_hashes={}
    for clip in spec["clips"]:
        requested=str(clip["source_take"])
        action=action_by_requested_take(requested)
        slot=_assign_exact_action(armature,action)
        start,end=(float(action.frame_range[0]),float(action.frame_range[1]))
        first=int(math.floor(start+1e-6))
        last=int(math.ceil(end-1e-6))
        frames=list(range(first,last+1))
        rows=[]
        translations=[]
        top=[]
        per_bone={}
        per_bone_rotation={}
        for frame in frames:
            scene.frame_set(frame)
            pose_global={}
            for bone in bones:
                pb=armature.pose.bones.get(bone.name)
                if pb is None:
                    raise RuntimeError(
                        "MOTION_ACTION_DIAGNOSTIC_POSE_BONE_MISSING:"+bone.name
                    )
                pose_global[bone.name]=canonical_transform(pb.matrix,C)
            for bone in bones:
                jid=bone.name
                p=parent[jid]
                if p is None:
                    L_rest=rest_global[jid]
                    L_pose=pose_global[jid]
                else:
                    L_rest=rest_global[p].inverted_safe() @ rest_global[jid]
                    L_pose=pose_global[p].inverted_safe() @ pose_global[jid]
                delta=L_rest.inverted_safe() @ L_pose
                q=delta.to_quaternion()
                q.normalize()
                t=delta.to_translation()
                normalized=float(t.length)/body_scale
                rotation_angle=2.0*math.acos(
                    min(1.0,max(-1.0,abs(float(q.w))))
                )
                rows.append([
                    int(frame),jid,
                    float(q.x),float(q.y),float(q.z),float(q.w),
                    float(t.x)/body_scale,
                    float(t.y)/body_scale,
                    float(t.z)/body_scale,
                ])
                per_bone_rotation[jid]=max(
                    per_bone_rotation.get(jid,0.0),
                    rotation_angle,
                )
                if jid!=root_name:
                    translations.append(normalized)
                    per_bone[jid]=max(per_bone.get(jid,0.0),normalized)
                    if normalized>0.0:
                        top.append({
                            "frame":int(frame),
                            "source_joint_id":jid,
                            "normalized_translation":normalized,
                        })
        values=np.asarray(translations,dtype=np.float64)
        top.sort(
            key=lambda row:(
                -row["normalized_translation"],
                row["frame"],
                row["source_joint_id"],
            )
        )
        stream_hash=_hash_rows(rows)
        stream_hashes[str(clip["clip_id"])]=stream_hash
        reports.append({
            "clip_id":str(clip["clip_id"]),
            "clip_kind":str(clip["clip_kind"]),
            "requested_take":requested,
            "resolved_action_name":str(action.name),
            "action_slot_identifier":slot,
            "frame_first":first,
            "frame_last":last,
            "sample_frame_count":len(frames),
            "source_fps":fps,
            "pose_stream_sha256":stream_hash,
            "nonroot_translation":{
                "sample_count":int(len(values)),
                "maximum":float(np.max(values)) if len(values) else 0.0,
                "p95":float(np.quantile(values,0.95)) if len(values) else 0.0,
                "p99":float(np.quantile(values,0.99)) if len(values) else 0.0,
                "fraction_gt_1e_5":float(np.mean(values>1e-5)) if len(values) else 0.0,
                "fraction_gt_1e_4":float(np.mean(values>1e-4)) if len(values) else 0.0,
                "fraction_gt_1e_3":float(np.mean(values>1e-3)) if len(values) else 0.0,
                "top_samples":top[:24],
                "top_bones":[
                    {"source_joint_id":jid,"maximum":float(value)}
                    for jid,value in sorted(
                        per_bone.items(),
                        key=lambda item:(-item[1],item[0]),
                    )[:16]
                ],
            },
            "rotation_amplitude":{
                "top_bones":[
                    {
                        "source_joint_id":jid,
                        "maximum_angle_radians":float(value),
                        "maximum_angle_degrees":float(math.degrees(value)),
                    }
                    for jid,value in sorted(
                        per_bone_rotation.items(),
                        key=lambda item:(-item[1],item[0]),
                    )
                ],
            },
        })
    return {
        "schema":"RealSaS.MotionSourceActionDiagnostic.v1",
        "status":"MEASURED",
        "source_fbx":str(source_fbx),
        "body_scale":body_scale,
        "clip_reports":reports,
        "distinct_pose_stream_hash_count":len(set(stream_hashes.values())),
        "clip_count":len(stream_hashes),
        "all_requested_takes_pose_distinct":(
            len(set(stream_hashes.values()))==len(stream_hashes)
        ),
        "thresholds_changed":False,
        "diagnostic_only":True,
    }


def main():
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-fbx",required=True)
    ap.add_argument("--spec",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args(argv)
    report=diagnose(
        Path(args.source_fbx).resolve(),
        Path(args.spec).resolve(),
    )
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("MOTION_SOURCE_ACTION_DIAGNOSTIC",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
