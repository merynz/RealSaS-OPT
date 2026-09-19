from __future__ import annotations

"""Deterministic Blender extractor for RealSaS MotionSourceClip.v2.

Run:
  blender --background --python tools/motion/blender_extract_motion_source_v2.py -- \
    --source-fbx KnightCharacter.fbx --spec canonical/MOTION_PRESET_SOURCE_QUATERNIUS_KNIGHT_V1.json \
    --out-dir assets/motion/quaternius_knight_v1

Only source skeleton rest geometry and animation transforms are exported.
Source mesh, skin, materials and textures are never product authority.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector

# When Blender is launched from repository root, make core frame derivation importable.
REPO=Path.cwd().resolve()
if str(REPO) not in sys.path:
    sys.path.insert(0,str(REPO))

from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_rows


EXTRACTOR_SCHEMA="RealSaS.BlenderMotionExtractor.v2"
FRAME_SEMANTICS="REALSAS_DERIVED_JOINT_FRAME_V1"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value)->bytes:
    return (json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode("utf-8")


def vec3(v):
    return [float(v[0]),float(v[1]),float(v[2])]


def matrix3_rows(M):
    return [[float(M[r][c]) for c in range(3)] for r in range(3)]


def np_like_dot(a,b):
    return float(sum(float(x)*float(y) for x,y in zip(a,b)))


def normalize(v,label):
    x=Vector(v)
    if x.length<=1e-10:
        raise RuntimeError(label)
    x.normalize()
    return x


def find_named_bone(bones,*names):
    for name in names:
        if name in bones:
            return bones[name]
    return None


def canonical_basis(armature):
    bones=armature.data.bones
    hips=find_named_bone(bones,"Hips","hips","Root","root")
    head=find_named_bone(bones,"Head","head")
    if hips is None or head is None:
        raise RuntimeError("MOTION_EXTRACTOR_REQUIRES_HIPS_AND_HEAD_FOR_OBJECT_FRAME")
    up=normalize(head.head_local-hips.head_local,"MOTION_EXTRACTOR_UP_DEGENERATE")

    lateral=[]
    for base in ("Shoulder","UpperArm","Palm","UpperLeg","Foot"):
        left=find_named_bone(bones,base+".L",base+"_L",base+"Left")
        right=find_named_bone(bones,base+".R",base+"_R",base+"Right")
        if left is not None and right is not None:
            lateral.append(right.head_local-left.head_local)
    if not lateral:
        raise RuntimeError("MOTION_EXTRACTOR_REQUIRES_LEFT_RIGHT_PAIR")
    right=Vector((0.0,0.0,0.0))
    for row in lateral:
        right+=row
    right=right-up*right.dot(up)
    right=normalize(right,"MOTION_EXTRACTOR_RIGHT_DEGENERATE")
    forward=normalize(up.cross(right),"MOTION_EXTRACTOR_FORWARD_DEGENERATE")
    # Re-orthogonalize so X x Y = Z.
    right=normalize(forward.cross(up),"MOTION_EXTRACTOR_RIGHT_ORTHO_FAIL")
    B=Matrix((
        (right.x,forward.x,up.x),
        (right.y,forward.y,up.y),
        (right.z,forward.z,up.z),
    ))
    # Columns of B are canonical axes expressed in source armature coordinates.
    # canonical vector = B^T * source vector.
    C=B.transposed()
    if C.determinant()<0.999:
        raise RuntimeError("MOTION_EXTRACTOR_OBJECT_FRAME_NOT_RIGHT_HANDED")
    return C


def canonical_transform(M4,C3):
    R=C3 @ M4.to_3x3() @ C3.transposed()
    t=C3 @ M4.to_translation()
    out=R.to_4x4()
    out.translation=t
    return out


def selected_bones(armature):
    # Keep the complete authored transform tree except terminal helper "_end" bones.
    rows=[b for b in armature.data.bones if not b.name.lower().endswith("_end")]
    if not rows:
        raise RuntimeError("MOTION_EXTRACTOR_BONE_SET_EMPTY")
    names={b.name for b in rows}
    roots=[b for b in rows if b.parent is None or b.parent.name not in names]
    if len(roots)!=1:
        raise RuntimeError("MOTION_EXTRACTOR_REQUIRES_SINGLE_SELECTED_ROOT")
    return tuple(rows)


def selected_parent(bone,names):
    p=bone.parent
    while p is not None and p.name not in names:
        p=p.parent
    return None if p is None else p.name


def quaternion_xyzw(R):
    q=R.to_quaternion()
    q.normalize()
    return [float(q.x),float(q.y),float(q.z),float(q.w)]


def angular_scale(M):
    _,_,scale=M.decompose()
    return [float(scale.x),float(scale.y),float(scale.z)]


def extract_clip(*,armature,action,source_path,source_sha,license_sha,spec,C):
    bones=selected_bones(armature)
    names={b.name for b in bones}
    parent={b.name:selected_parent(b,names) for b in bones}

    rest_global={b.name:canonical_transform(b.matrix_local,C) for b in bones}
    rest_rows=[
        {
            "source_joint_id":b.name,
            "parent_source_joint_id":parent[b.name],
            "rest_position":vec3(rest_global[b.name].to_translation()),
        }
        for b in bones
    ]
    derived=derive_joint_frames_from_rows(
        rest_rows,
        joint_id_key="source_joint_id",
        parent_id_key="parent_source_joint_id",
        position_key="rest_position",
    )
    body_scale=max(
        (
            (Vector(row["rest_position"])-Vector(next(x["rest_position"] for x in rest_rows if x["parent_source_joint_id"] is None))).length
            for row in rest_rows
        ),
        default=1.0,
    )
    body_scale=max(float(body_scale),1e-8)

    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action=action
    start,end=(float(action.frame_range[0]),float(action.frame_range[1]))
    scene=bpy.context.scene
    fps=float(scene.render.fps)/float(scene.render.fps_base or 1.0)
    first=int(math.floor(start+1e-6))
    last=int(math.ceil(end-1e-6))
    sample_frames=list(range(first,last+1))
    if not sample_frames:
        raise RuntimeError("MOTION_EXTRACTOR_ACTION_EMPTY")

    tracks={b.name:[] for b in bones}
    root_name=next(name for name,p in parent.items() if p is None)
    root_rest_translation=rest_global[root_name].to_translation()

    for frame in sample_frames:
        scene.frame_set(frame)
        pose_global={}
        for bone in bones:
            pb=armature.pose.bones.get(bone.name)
            if pb is None:
                raise RuntimeError("MOTION_EXTRACTOR_POSE_BONE_MISSING:"+bone.name)
            pose_global[bone.name]=canonical_transform(pb.matrix,C)

        time_seconds=(float(frame)-float(first))/fps
        for bone in bones:
            jid=bone.name
            p=parent[jid]
            if p is None:
                L_rest=rest_global[jid]
                L_pose=pose_global[jid]
            else:
                L_rest=rest_global[p].inverted_safe() @ rest_global[jid]
                L_pose=pose_global[p].inverted_safe() @ pose_global[jid]
            delta_local=L_rest.inverted_safe() @ L_pose
            scale=angular_scale(delta_local)
            if max(abs(x-1.0) for x in scale)>1e-4:
                raise RuntimeError(f"MOTION_EXTRACTOR_SCALE_ANIMATION_UNSUPPORTED:{jid}:{frame}:{scale}")

            # Convert authored bone-local delta to the geometry-derived RealSaS joint
            # frame so source and target do not need matching FBX local-axis conventions.
            R_child_rest=rest_global[jid].to_3x3()
            R_world_delta=R_child_rest @ delta_local.to_3x3() @ R_child_rest.transposed()
            F=Matrix(derived[jid].rotation_matrix)
            R_derived=F.transposed() @ R_world_delta @ F

            translation=(0.0,0.0,0.0)
            if jid==root_name:
                world_delta=pose_global[jid].to_translation()-root_rest_translation
                translation=tuple(float(x)/body_scale for x in world_delta)
            else:
                local_translation=delta_local.to_translation()
                if local_translation.length/body_scale>1e-5:
                    raise RuntimeError(
                        f"MOTION_EXTRACTOR_NONROOT_TRANSLATION_UNSUPPORTED:{jid}:{frame}:{local_translation.length/body_scale}"
                    )
            tracks[jid].append({
                "time_seconds":float(time_seconds),
                "local_rotation_quat_xyzw":quaternion_xyzw(R_derived),
                "local_translation_xyz":[float(x) for x in translation],
                "local_scale_xyz":[1.0,1.0,1.0],
            })

    duration=float(sample_frames[-1]-sample_frames[0])/fps
    if duration<=0:
        duration=1.0/fps
    payload={
        "schema":"RealSaS.MotionSourceClip.v2",
        "clip_id":str(spec["clip_id"]),
        "clip_kind":str(spec["clip_kind"]),
        "source_space":"SOURCE_RIG_TRACKS_V2",
        "coordinate_frame":"REALSAS_OBJECT_FRAME_V1",
        "joint_frame_semantics":FRAME_SEMANTICS,
        "duration_seconds":duration,
        "loop":bool(spec["loop"]),
        "channel_contract":[
            "LOCAL_ROTATION_QUAT_XYZW",
            "LOCAL_TRANSLATION_XYZ",
            "LOCAL_SCALE_XYZ",
        ],
        "source_skeleton":rest_rows,
        "tracks":[
            {
                "source_joint_id":jid,
                "keyframes":tracks[jid],
                "metadata":{
                    "source_joint_frame_hash":derived[jid].frame_hash,
                    "bone_local_axis_convention_discarded":True,
                },
            }
            for jid in sorted(tracks)
        ],
        "metadata":{
            "extractor_schema":EXTRACTOR_SCHEMA,
            "source_fbx_sha256":source_sha,
            "license_evidence_sha256":license_sha,
            "source_take":action.name,
            "source_fps":fps,
            "sample_frame_first":sample_frames[0],
            "sample_frame_last":sample_frames[-1],
            "source_body_scale":body_scale,
            "source_mesh_used_as_product_authority":False,
            "source_skin_used_as_product_authority":False,
            "source_material_used_as_product_authority":False,
            "source_texture_used_as_product_authority":False,
            "source_skeleton_used_as_final_target_authority":False,
            "motion_semantics_only":True,
        },
    }
    return payload


def import_fbx(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=str(path),use_anim=True)
    except TypeError:
        bpy.ops.import_scene.fbx(filepath=str(path))
    arms=[obj for obj in bpy.context.scene.objects if obj.type=="ARMATURE"]
    if len(arms)!=1:
        raise RuntimeError(f"MOTION_EXTRACTOR_REQUIRES_ONE_ARMATURE:{len(arms)}")
    return arms[0]


def action_by_requested_take(requested):
    exact=[a for a in bpy.data.actions if a.name==requested]
    if len(exact)==1:
        return exact[0]
    # FBX importers may strip the "HumanArmature|" prefix.
    short=requested.split("|")[-1]
    matches=[a for a in bpy.data.actions if a.name==short or a.name.endswith("|"+short)]
    if len(matches)!=1:
        raise RuntimeError(
            "MOTION_EXTRACTOR_TAKE_NOT_UNIQUE:"
            +requested+":available="+",".join(sorted(a.name for a in bpy.data.actions))
        )
    return matches[0]


def main():
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-fbx",required=True)
    ap.add_argument("--spec",required=True)
    ap.add_argument("--out-dir",required=True)
    args=ap.parse_args(argv)

    source=Path(args.source_fbx).resolve()
    spec_path=Path(args.spec).resolve()
    out_dir=Path(args.out_dir).resolve()
    spec=json.loads(spec_path.read_text(encoding="utf-8"))
    source_sha=sha256(source)
    expected=str(spec["source"]["fbx_sha256"])
    if source_sha!=expected:
        raise RuntimeError("MOTION_EXTRACTOR_SOURCE_SHA_DRIFT")
    license_sha=str(spec["source"]["license_evidence_sha256"])

    armature=import_fbx(source)
    C=canonical_basis(armature)
    out_dir.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for row in spec["clips"]:
        action=action_by_requested_take(str(row["source_take"]))
        payload=extract_clip(
            armature=armature,action=action,source_path=source,
            source_sha=source_sha,license_sha=license_sha,spec=row,C=C,
        )
        target=out_dir/str(row["output_filename"])
        target.write_bytes(canonical_json(payload))
        outputs.append({
            "clip_id":payload["clip_id"],
            "clip_kind":payload["clip_kind"],
            "source_take":action.name,
            "path":str(target),
            "sha256":sha256(target),
            "payload_sha256":__import__("compiler.realsas_compiler_core.hashing",fromlist=["content_sha256"]).content_sha256(payload),
        })
    receipt={
        "schema":"RealSaS.MotionPresetExtractionReceipt.v1",
        "status":"PASS",
        "source_fbx_sha256":source_sha,
        "spec_sha256":sha256(spec_path),
        "extractor_schema":EXTRACTOR_SCHEMA,
        "outputs":outputs,
        "source_mesh_skin_appearance_product_authority":False,
    }
    receipt_path=out_dir/"EXTRACTION_RECEIPT.json"
    receipt_path.write_bytes(canonical_json(receipt))
    print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
