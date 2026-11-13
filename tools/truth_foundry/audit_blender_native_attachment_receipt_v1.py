#!/usr/bin/env python3
from __future__ import annotations

"""Portable readback court for source-native Blender attachment inventory.

Re-verify exported native owner/constraint records and parent/root world
transform samples against the exact original teacher receipt/NPZ. Does NOT
reinterpret authored links as socket/held/holder-face ground truth. Does NOT
mint source training eligibility, attach/detach truth, or NO_FACE negatives.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from tools.truth_foundry.native_blender_attachment_inventory_v1 import (
    compile_blender_native_attachment_inventory_candidate_v1,
)

SCHEMA="RealSaS.TruthFoundry.BlenderNativeAttachmentReadback.v1"
SOURCE_SCHEMA="RealSaS.TruthFoundry.BlenderNativeAttachmentObservation.v1"
_CANDIDATE="UNQUALIFIED_AUTHORED_BONE_LINK_CANDIDATE"


def _fail(reason:str)->None:
    raise ValueError("RSS_VNEXT_BLENDER_ATTACHMENT_READBACK_"+reason)


def _digest(path:Path)->str:
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_world_affine(frames:np.ndarray)->np.ndarray:
    if frames.ndim!=3 or frames.shape[1:]!=(4,4):
        _fail("WORLD_TRANSFORM_SHAPE")
    if not np.isfinite(frames).all():
        _fail("AVAILABLE_FRAME_NOT_FINITE")
    if not np.allclose(frames[:,3,:],[0,0,0,1],atol=1.e-8,rtol=0):
        _fail("AVAILABLE_FRAME_NOT_HOMOGENEOUS")
    matrix=frames[:,:3,:3]
    det=np.linalg.det(matrix)
    if np.any(np.abs(det)<1.e-10):
        _fail("AVAILABLE_FRAME_SINGULAR")
    # Source transforms may have authored non-uniform scale; do not
    # promote those to the SE3 rest-frame claim used by the motion oracle.
    rigid=(
        np.max(np.abs(np.swapaxes(matrix,1,2)@matrix-
                      np.eye(3)[None]),axis=(1,2))<=1.e-6
    )&(np.abs(det-1.)<=1.e-6)
    return rigid


def audit_native_blender_attachment_receipt_v1(
    *,receipt:dict,teacher_receipt:dict,
    teacher_npz:dict[str,np.ndarray],
    parent_root_npz:dict[str,np.ndarray],
    source_asset_sha256:str,
    teacher_receipt_sha256:str,
    teacher_npz_sha256:str,
    observation_admission_sha256:str,
    parent_root_npz_sha256:str,
)->dict:
    if not isinstance(receipt,dict) or not isinstance(teacher_receipt,dict):
        _fail("RECEIPT_SCHEMA")
    if (
        receipt.get("schema")!=SOURCE_SCHEMA
        or teacher_receipt.get("schema")!="RealSaS.IRISVNextAuthoredTeacherExport.v1"
        or receipt.get("subject_id")!=teacher_receipt.get("subject_id")
        or receipt.get("original_teacher_frame_rows")!=teacher_receipt.get("sampled_frames")
        or receipt.get("source_asset_sha256")!=source_asset_sha256
        or receipt.get("teacher_receipt_sha256")!=teacher_receipt_sha256
        or receipt.get("teacher_npz_sha256")!=teacher_npz_sha256
        or receipt.get("observation_admission_sha256")!=observation_admission_sha256
        or receipt.get("parent_root_npz_sha256")!=parent_root_npz_sha256
        or receipt.get("parent_root_npz_filename")
           !="native_attachment_parent_root_frames.npz"
        or any(receipt.get(name) is not False for name in (
            "attachment_face_authority",
            "exhaustive_no_face_authority",
            "trainer_admission",
            "product_revision_authority",
        ))
        or receipt.get("source_coordinate_domain")
            !="SOURCE_WORLD_NOT_REALSAS_CANONICAL"
    ):
        _fail("PARENT_OR_AUTHORITY_DRIFT")
    if (
        teacher_receipt.get("source",{}).get("sha256")!=source_asset_sha256
        or teacher_receipt.get("output",{}).get("sha256")!=teacher_npz_sha256
        or teacher_receipt.get("observation_admission_sha256")
            !=observation_admission_sha256
        or receipt.get("importer_policy")!=teacher_receipt.get("import")
    ):
        _fail("TEACHER_SOURCE_AND_IMPORT_IDENTITY_DRIFT")
    try:
        faces=np.asarray(teacher_npz["faces"])
        face_owner=np.asarray(teacher_npz["face_owner"])
        rest=np.asarray(teacher_npz["rest_vertices_source_world"])
        posed=np.asarray(teacher_npz["posed_vertices_source_world"])
    except (KeyError,TypeError) as exc:
        _fail("TEACHER_NPZ_REQUIRED_FIELDS")
    if (
        faces.ndim!=2 or faces.shape[1:]!=(3,)
        or rest.ndim!=2 or rest.shape[1:]!=(3,)
        or posed.ndim!=3 or posed.shape[1:]!=rest.shape
        or len(posed)!=len(teacher_receipt["sampled_frames"])
        or not np.isfinite(rest).all() or not np.isfinite(posed).all()
        or len(faces)!=len(face_owner)
        or len(faces)!=teacher_receipt.get("face_count")
        or not isinstance(receipt.get("source_owner_inventory"),list)
    ):
        _fail("TEACHER_MESH_FRAME_DOMAIN")
    inventory=compile_blender_native_attachment_inventory_candidate_v1(
        owner_rows=receipt["source_owner_inventory"],
        source_face_owner=face_owner,
        source_asset_sha256=source_asset_sha256,
        native_teacher_receipt_sha256=teacher_receipt_sha256,
        native_teacher_npz_sha256=teacher_npz_sha256,
        observation_admission_sha256=observation_admission_sha256,
    )
    if (
        json.loads(json.dumps(asdict(inventory),allow_nan=False))
        !=receipt.get("source_link_candidates")
        or len(inventory.owners)!=teacher_receipt.get("anonymous_owner_count")
        or any(candidate.source_object_name!=row.get("source_import_name")
               for candidate,row in zip(inventory.owners,teacher_receipt.get("owners",[])))
    ):
        _fail("AUTHORING_LEDGER_OR_ORIGINAL_TEACHER_OWNERS_DRIFT")
    nown=len(inventory.owners)
    nframe=len(posed)
    try:
        rest_parent=np.asarray(parent_root_npz["parent_rest_world"])
        rest_root=np.asarray(parent_root_npz["root_rest_world"])
        sample_parent=np.asarray(parent_root_npz["parent_sampled_world"])
        sample_root=np.asarray(parent_root_npz["root_sampled_world"])
        available=np.asarray(parent_root_npz["sample_available"])
    except (KeyError,TypeError) as exc:
        _fail("PARENT_ROOT_NPZ_REQUIRED_FIELDS")
    if (
        rest_parent.shape!=(nown,4,4)
        or rest_root.shape!=(nown,4,4)
        or sample_parent.shape!=(nown,nframe,4,4)
        or sample_root.shape!=(nown,nframe,4,4)
        or available.shape!=(nown,nframe)
        or available.dtype!=np.bool_
        or any(v.dtype!=np.float64 for v in (
            rest_parent,rest_root,sample_parent,sample_root
        ))
    ):
        _fail("PARENT_ROOT_SAMPLED_SHAPE_OR_DTYPE")
    invalid_rigid=0
    for owner in inventory.owners:
        ix=owner.owner_index
        rows=teacher_receipt["sampled_frames"]
        expected=np.array([
            owner.state==_CANDIDATE
            and frame["armature"]==owner.parent_armature_name
            for frame in rows
        ],dtype=np.bool_)
        if not np.array_equal(available[ix],expected):
            _fail("SOURCE_PARENT_ACTION_FRAME_MASK_DRIFT")
        if owner.state!=_CANDIDATE:
            if not (
                np.isnan(rest_parent[ix]).all()
                and np.isnan(rest_root[ix]).all()
            ):
                _fail("UNQUALIFIED_OWNER_HAS_PARENT_REST_FRAME")
        else:
            rigid_parent=_valid_world_affine(rest_parent[ix][None])
            rigid_root=_valid_world_affine(rest_root[ix][None])
            invalid_rigid+=int((~rigid_parent).sum()+(~rigid_root).sum())
        if np.any(available[ix]):
            rigid_parent=_valid_world_affine(sample_parent[ix][available[ix]])
            rigid_root=_valid_world_affine(sample_root[ix][available[ix]])
            invalid_rigid+=int((~rigid_parent).sum()+(~rigid_root).sum())
        if (
            not np.isnan(sample_parent[ix][~available[ix]]).all()
            or not np.isnan(sample_root[ix][~available[ix]]).all()
        ):
            _fail("MISSING_ACTION_FRAME_WAS_SYNTHESIZED")
    return {
        "schema":SCHEMA,
        "source_asset_sha256":source_asset_sha256,
        "inventory_sha256":inventory.inventory_sha256,
        "owner_count":nown,
        "original_teacher_frame_count":nframe,
        "bone_link_candidate_count":sum(o.state==_CANDIDATE for o in inventory.owners),
        "parent_root_available_samples":int(available.sum()),
        "non_se3_source_frames":invalid_rigid,
        "source_world_affine_frames_observed":True,
        "source_attachment_training_admitted":False,
        "source_attachment_contact_qualified":False,
        "exhaustive_no_face_verified":False,
        "heldout_intervention_qualified":False,
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--teacher-receipt",type=Path,required=True)
    p.add_argument("--teacher-npz",type=Path,required=True)
    p.add_argument("--observation-admission",type=Path,required=True)
    p.add_argument("--inventory-directory",type=Path,required=True)
    a=p.parse_args()
    directory=a.inventory_directory
    receipt=json.loads((directory/"native_attachment_inventory.json").read_text("utf8"))
    teacher=json.loads(a.teacher_receipt.read_text("utf8"))
    original=json.loads(a.observation_admission.read_text("utf8"))
    source_sha=_digest(a.source)
    if original.get("source_sha256")!=source_sha:
        _fail("SOURCE_OBSERVATION_HASH")
    path=directory/"native_attachment_parent_root_frames.npz"
    with np.load(a.teacher_npz,allow_pickle=False) as z:
        data={k:np.asarray(z[k]) for k in (
            "faces","face_owner","rest_vertices_source_world",
            "posed_vertices_source_world"
        )}
    with np.load(path,allow_pickle=False) as z:
        frames={k:np.asarray(z[k]) for k in (
            "parent_rest_world","root_rest_world",
            "parent_sampled_world","root_sampled_world","sample_available"
        )}
    answer=audit_native_blender_attachment_receipt_v1(
        receipt=receipt,teacher_receipt=teacher,
        teacher_npz=data,parent_root_npz=frames,
        source_asset_sha256=source_sha,
        teacher_receipt_sha256=_digest(a.teacher_receipt),
        teacher_npz_sha256=_digest(a.teacher_npz),
        observation_admission_sha256=_digest(a.observation_admission),
        parent_root_npz_sha256=_digest(path),
    )
    print(json.dumps(answer,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
