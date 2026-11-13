#!/usr/bin/env python3
from __future__ import annotations

"""Audit authored Blender attachment inventory against one sealed IRIS teacher.

Run under the SAME pinned Blender importer as the original native teacher:

 blender -b --python tools/truth_foundry/blender_native_attachment_inventory_v1.py -- \\
   --source CHAR.glb --expected-source-sha256 SHA --subject-id ID \\
   --observation-admission ADMISSION.json \\
   --teacher-receipt ORIGINAL_TEACHER.npz.json \\
   --expected-teacher-receipt-sha256 SHA \\
   --teacher-npz ORIGINAL_TEACHER.npz --output-directory OUTPUT

The original native teacher owns vertex and triangle ordering. All world/root
transform series here are **source-only diagnostic candidates**, sampled on
the EXACT teacher action/frame identities and compared against the original
rest/posed source mesh bytes. This exporter never mints authored sockets,
held/holder face IDs, NO_FACE labels, trainer admission or product authority.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.iris_vnext import blender_authored_teacher_export_v1 as native
from tools.truth_foundry.native_blender_attachment_inventory_v1 import (
    compile_blender_native_attachment_inventory_candidate_v1,
)

SCHEMA="RealSaS.TruthFoundry.BlenderNativeAttachmentObservation.v1"


def _fail(reason:str)->None:
    raise RuntimeError("RSS_VNEXT_BLENDER_NATIVE_ATTACHMENT_EXPORT_"+reason)


def _sha(path:Path)->str:
    h=sha256()
    with path.open("rb") as f:
        for blob in iter(lambda:f.read(8*1024*1024),b""):
            h.update(blob)
    return h.hexdigest()


def _args()->argparse.Namespace:
    raw=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--expected-source-sha256",required=True)
    p.add_argument("--subject-id",required=True)
    p.add_argument("--observation-admission",type=Path,required=True)
    p.add_argument("--teacher-receipt",type=Path,required=True)
    p.add_argument("--expected-teacher-receipt-sha256",required=True)
    p.add_argument("--teacher-npz",type=Path,required=True)
    p.add_argument("--output-directory",type=Path,required=True)
    return p.parse_args(raw)


def _constraint_row(c)->dict:
    owner=getattr(c,"target",None)
    row={
        "name":str(c.name),
        "type":str(c.type),
        "mute":getattr(c,"mute",None),
        "influence":getattr(c,"influence",None),
        "target":None if owner is None else str(owner.name),
        "target_type":None if owner is None else str(owner.type),
        "subtarget":str(getattr(c,"subtarget","") or ""),
    }
    if str(c.type)=="CHILD_OF":
        for name in (
            "use_location_x","use_location_y","use_location_z",
            "use_rotation_x","use_rotation_y","use_rotation_z",
            "use_scale_x","use_scale_y","use_scale_z",
        ):
            row[name]=getattr(c,name,None)
        # The inverse/rest offset is preserved but is NOT an exact
        # authored held/holder triangle or an AXIS anchor.
        inverse=getattr(c,"inverse_matrix",None)
        row["inverse_matrix"]=(
            None if inverse is None
            else [[float(v) for v in r] for r in inverse]
        )
    return row


def _source_row(obj,owner_index:int)->dict:
    parent=obj.parent
    return {
        "owner_index":owner_index,
        "name":obj.name,
        "parent_type":str(obj.parent_type),
        "parent_name":None if parent is None else str(parent.name),
        "parent_object_type":None if parent is None else str(parent.type),
        "parent_bone":str(obj.parent_bone or ""),
        "constraints":[_constraint_row(c) for c in obj.constraints],
        "modifiers":[{
            "name":str(mod.name),"type":str(mod.type),
            "target":(
                None if getattr(mod,"object",None) is None
                else str(mod.object.name)
            ),
        } for mod in obj.modifiers],
    }


def _frames_for_owner(candidate)->tuple[str,str]|None:
    if candidate.parent_armature_name and candidate.parent_bone_name:
        return candidate.parent_armature_name,candidate.parent_bone_name
    return None


def _source_bone_frames(armature_name:str,bone_name:str)->tuple[np.ndarray,np.ndarray]:
    arm=bpy.context.scene.objects.get(armature_name)
    if arm is None or arm.type!="ARMATURE":
        _fail("SOURCE_ARMATURE_MISSING:"+armature_name)
    eval_arm=arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bone=eval_arm.pose.bones.get(bone_name)
    if bone is None:
        _fail("SOURCE_POSE_BONE_MISSING:"+armature_name+":"+bone_name)
    parent=np.asarray(eval_arm.matrix_world,dtype=np.float64)
    bone_world=np.asarray(eval_arm.matrix_world@bone.matrix,dtype=np.float64)
    if (
        parent.shape!=(4,4) or bone_world.shape!=(4,4)
        or not np.isfinite(parent).all()
        or not np.isfinite(bone_world).all()
    ):
        _fail("SOURCE_PARENT_ROOT_FRAME_NONFINITE")
    return bone_world,parent


def _assert_same_mesh(source_mesh:dict,original:dict,*,label:str)->None:
    if (
        not np.array_equal(source_mesh["faces"],original["faces"])
        or not np.array_equal(source_mesh["face_owner"],original["face_owner"])
        or source_mesh["vertices"].shape!=original["vertices"].shape
        or not np.isfinite(source_mesh["vertices"]).all()
        or not np.isfinite(original["vertices"]).all()
    ):
        _fail("NATIVE_TEACHER_MESH_IDENTITY_DRIFT:"+label)
    source=source_mesh["vertices"]
    target=original["vertices"]
    scale=float(np.linalg.norm(
        target.max(axis=0)-target.min(axis=0)
    ))
    threshold=1.e-8*max(scale,1.)
    difference=float(np.max(np.abs(source-target)))
    if difference>threshold:
        _fail("NATIVE_TEACHER_MESH_POSITION_DRIFT:"+label)


def run()->None:
    a=_args()
    paths=(a.source,a.observation_admission,a.teacher_receipt,a.teacher_npz)
    if any(not p.is_file() for p in paths):
        _fail("SOURCE_ADMISSION_OR_TEACHER_FILE_MISSING")
    source_sha=_sha(a.source)
    teacher_receipt_sha=_sha(a.teacher_receipt)
    teacher_npz_sha=_sha(a.teacher_npz)
    if (
        source_sha!=a.expected_source_sha256
        or teacher_receipt_sha!=a.expected_teacher_receipt_sha256
    ):
        _fail("SOURCE_OR_TEACHER_RECEIPT_SHA256_DRIFT")
    admission=json.loads(a.observation_admission.read_text("utf8"))
    receipt=json.loads(a.teacher_receipt.read_text("utf8"))
    if (
        admission.get("subject_id")!=a.subject_id
        or admission.get("source_sha256")!=source_sha
        or receipt.get("schema")!="RealSaS.IRISVNextAuthoredTeacherExport.v1"
        or receipt.get("subject_id")!=a.subject_id
        or receipt.get("source",{}).get("sha256")!=source_sha
        or receipt.get("observation_admission_sha256")!=_sha(a.observation_admission)
        or receipt.get("output",{}).get("sha256")!=teacher_npz_sha
    ):
        _fail("NATIVE_TEACHER_ADMISSION_IDENTITY_DRIFT")
    with np.load(a.teacher_npz,allow_pickle=False) as data:
        original={name:np.asarray(data[name]) for name in (
            "rest_vertices_source_world","faces","face_owner",
            "posed_vertices_source_world",
        )}
    samples=receipt.get("sampled_frames")
    original_frames=original["posed_vertices_source_world"]
    if (
        not isinstance(samples,list)
        or not samples
        or len(samples)!=len(original_frames)
        or original_frames.ndim!=3
        or original_frames.shape[1:]!=original["rest_vertices_source_world"].shape
    ):
        _fail("NATIVE_TEACHER_FRAME_RECEIPT_DOMAIN")

    imported=native._import_animated(a.source)
    if imported!=receipt.get("import"):
        _fail("BLENDER_IMPORTER_POLICY_DRIFT")
    objects=native._mesh_objects()
    old_owner=receipt.get("owners")
    if (
        len(objects)!=len(old_owner)
        or any(obj.name!=row["source_import_name"]
               or row["anonymous_owner_index"]!=i
               for i,(obj,row) in enumerate(zip(objects,old_owner)))
    ):
        _fail("NATIVE_TEACHER_RENDERABLE_MESH_OWNER_DRIFT")
    rows=[_source_row(obj,index) for index,obj in enumerate(objects)]
    native._neutralize_animation()
    exact_rest=native._extract_world_mesh(objects)
    _assert_same_mesh(exact_rest,dict(
        faces=original["faces"],face_owner=original["face_owner"],
        vertices=original["rest_vertices_source_world"],
    ),label="REST")
    # Preserve original candidate/unknown classifications without converting
    # skin-deformation ARMATURE modifiers into rigid object attachments.
    inventory=compile_blender_native_attachment_inventory_candidate_v1(
        owner_rows=rows,source_face_owner=original["face_owner"],
        source_asset_sha256=source_sha,
        native_teacher_receipt_sha256=teacher_receipt_sha,
        native_teacher_npz_sha256=teacher_npz_sha,
        observation_admission_sha256=_sha(a.observation_admission),
    )
    nowners=len(objects);nframes=len(samples)
    parent=np.full((nowners,nframes,4,4),np.nan,dtype=np.float64)
    root=np.full_like(parent,np.nan)
    source_rest_parent=np.full((nowners,4,4),np.nan,dtype=np.float64)
    source_rest_root=np.full_like(source_rest_parent,np.nan)
    present=np.zeros((nowners,nframes),dtype=np.bool_)
    # REST frames sampled before activating any source animation action.
    for candidate in inventory.owners:
        locator=_frames_for_owner(candidate)
        if locator is None:
            continue
        source_rest_parent[candidate.owner_index],source_rest_root[
            candidate.owner_index
        ]=_source_bone_frames(*locator)
    action_by_name={action.name:action for action in bpy.data.actions}
    expected_actions=set(receipt.get("native_motion_inventory_exported") or ())
    if not expected_actions or not expected_actions.issubset(action_by_name):
        _fail("NATIVE_TEACHER_ACTION_MISSING")
    last_action=None;active_armature=None
    for frame_index,sample in enumerate(samples):
        if (
            not isinstance(sample,dict)
            or type(sample.get("frame")) is not int
            or not isinstance(sample.get("action"),str)
            or sample["action"] not in expected_actions
            or not isinstance(sample.get("armature"),str)
        ):
            _fail("NATIVE_TEACHER_ACTION_FRAME_ROW_INVALID")
        if last_action!=sample["action"]:
            active_armature=native._activate_action(action_by_name[sample["action"]])
            last_action=sample["action"]
        if active_armature!=sample["armature"]:
            _fail("NATIVE_TEACHER_ACTION_ARMATURE_DRIFT")
        bpy.context.scene.frame_set(sample["frame"])
        bpy.context.view_layer.update()
        current=native._extract_world_mesh(objects)
        _assert_same_mesh(current,dict(
            faces=original["faces"],face_owner=original["face_owner"],
            vertices=original_frames[frame_index],
        ),label=str(frame_index))
        for candidate in inventory.owners:
            locator=_frames_for_owner(candidate)
            # Another armature may not be actuated by this source action.
            # Keep missing, do not synthesize motion or a negative label.
            if locator is None or locator[0]!=active_armature:
                continue
            at=candidate.owner_index
            parent[at,frame_index],root[at,frame_index]=_source_bone_frames(*locator)
            present[at,frame_index]=True
    native._neutralize_animation()

    target=a.output_directory.resolve()
    if target.exists() and any(target.iterdir()):
        _fail("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    target.mkdir(parents=True,exist_ok=True)
    arrays_path=target/"native_attachment_parent_root_frames.npz"
    np.savez_compressed(
        arrays_path,
        parent_rest_world=source_rest_parent,
        root_rest_world=source_rest_root,
        parent_sampled_world=parent,
        root_sampled_world=root,
        sample_available=present,
    )
    receipt_payload={
        "schema":SCHEMA,
        "subject_id":a.subject_id,
        "source_asset_sha256":source_sha,
        "teacher_receipt_sha256":teacher_receipt_sha,
        "teacher_npz_sha256":teacher_npz_sha,
        "observation_admission_sha256":_sha(a.observation_admission),
        "importer_policy":imported,
        "original_teacher_frame_rows":samples,
        "source_owner_inventory":rows,
        "source_link_candidates":asdict(inventory),
        "parent_root_npz_sha256":_sha(arrays_path),
        "parent_root_npz_filename":arrays_path.name,
        "source_coordinate_domain":"SOURCE_WORLD_NOT_REALSAS_CANONICAL",
        "attachment_face_authority":False,
        "exhaustive_no_face_authority":False,
        "trainer_admission":False,
        "product_revision_authority":False,
    }
    (target/"native_attachment_inventory.json").write_text(
        json.dumps(receipt_payload,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf8",
    )
    print(json.dumps({
        "schema":SCHEMA,"subject_id":a.subject_id,
        "source_owner_count":len(objects),
        "candidate_bone_link_count":sum(
            c.state=="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"
            for c in inventory.owners
        ),
        "heldout_source_motion_qualified":False,
        "source_native_training_admitted":False,
        "parent_frame_available_samples":int(present.sum()),
        "parent_root_npz_sha256":_sha(arrays_path),
    },sort_keys=True),flush=True)


if __name__=="__main__":
    run()
