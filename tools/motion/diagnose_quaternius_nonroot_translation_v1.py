from __future__ import annotations
import json, math, sys
from pathlib import Path
import bpy
from mathutils import Vector

REPO=Path.cwd().resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from tools.motion.blender_extract_motion_source_v2 import (
    import_fbx, canonical_basis, selected_bones, selected_parent,
    canonical_transform, action_by_requested_take,
)

def action_slot_for(armature, action):
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.use_nla=False
    armature.animation_data.action=action
    suitable=list(armature.animation_data.action_suitable_slots)
    by_user=[slot for slot in suitable if armature in tuple(slot.users())]
    by_name=[slot for slot in suitable if str(slot.name_display)==str(armature.name)]
    if len(by_user)==1:
        slot=by_user[0]
    elif len(by_name)==1:
        slot=by_name[0]
    elif len(suitable)==1:
        slot=suitable[0]
    else:
        raise RuntimeError("DIAG_SLOT_AMBIGUOUS:"+action.name)
    armature.animation_data.action_slot=slot
    return slot

def main():
    source=Path(sys.argv[sys.argv.index("--")+1]).resolve()
    spec_path=Path(sys.argv[sys.argv.index("--")+2]).resolve()
    spec=json.loads(spec_path.read_text(encoding="utf-8"))
    armature=import_fbx(source)
    C=canonical_basis(armature)
    bones=selected_bones(armature)
    names={b.name for b in bones}
    parent={b.name:selected_parent(b,names) for b in bones}
    rest_global={b.name:canonical_transform(b.matrix_local,C) for b in bones}
    root_name=next(name for name,p in parent.items() if p is None)
    root_pos=rest_global[root_name].to_translation()
    body_scale=max(((rest_global[b.name].to_translation()-root_pos).length for b in bones),default=1.0)
    body_scale=max(float(body_scale),1e-8)
    report={"source":str(source),"body_scale":body_scale,"takes":[]}
    for row in spec["clips"]:
        action=action_by_requested_take(str(row["source_take"]))
        slot=action_slot_for(armature,action)
        start,end=(float(action.frame_range[0]),float(action.frame_range[1]))
        first=int(math.floor(start+1e-6)); last=int(math.ceil(end-1e-6))
        samples=list(range(first,last+1))
        per={b.name:[] for b in bones if b.name!=root_name}
        for frame in samples:
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()
            pose_global={}
            for bone in bones:
                pb=armature.pose.bones.get(bone.name)
                pose_global[bone.name]=canonical_transform(pb.matrix,C)
            for bone in bones:
                jid=bone.name
                if jid==root_name: continue
                p=parent[jid]
                L_rest=rest_global[p].inverted_safe() @ rest_global[jid]
                L_pose=pose_global[p].inverted_safe() @ pose_global[jid]
                delta_local=L_rest.inverted_safe() @ L_pose
                v=delta_local.to_translation()
                per[jid].append((frame,float(v.x),float(v.y),float(v.z),float(v.length/body_scale)))
        ranked=[]
        for jid,rows in per.items():
            mags=[r[4] for r in rows]
            max_row=max(rows,key=lambda r:r[4])
            minv=min(mags); maxv=max(mags)
            vec_ranges=[
                max(r[k] for r in rows)-min(r[k] for r in rows)
                for k in (1,2,3)
            ]
            ranked.append({
                "bone":jid,
                "max_norm":maxv,
                "min_norm":minv,
                "max_frame":max_row[0],
                "max_vector":list(max_row[1:4]),
                "first_vector":list(rows[0][1:4]),
                "last_vector":list(rows[-1][1:4]),
                "component_range":vec_ranges,
                "over_1e5":sum(1 for x in mags if x>1e-5),
                "over_1e4":sum(1 for x in mags if x>1e-4),
                "sample_count":len(rows),
            })
        ranked.sort(key=lambda x:x["max_norm"],reverse=True)
        take={
            "requested_take":row["source_take"],"resolved_action":action.name,
            "slot_identifier":str(slot.identifier),"frames":[first,last],
            "top_nonroot":ranked[:20],
            "count_over_1e5":sum(1 for x in ranked if x["max_norm"]>1e-5),
            "count_over_1e4":sum(1 for x in ranked if x["max_norm"]>1e-4),
        }
        report["takes"].append(take)
    print("REALSAS_NONROOT_TRANSLATION_DIAGNOSTIC_BEGIN")
    print(json.dumps(report,indent=2,sort_keys=True))
    print("REALSAS_NONROOT_TRANSLATION_DIAGNOSTIC_END")

if __name__=="__main__":
    main()
