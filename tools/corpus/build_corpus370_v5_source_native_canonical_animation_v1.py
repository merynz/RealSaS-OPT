#!/usr/bin/env python3
"""Project V5 native-action teacher poses into sealed canonical coordinates.

Strict original SourceRest geometry + exact authored Action/owner teacher arrays
are both required. This producer DOES NOT replay native controls independently,
therefore its output is a canonical animation CANDIDATE, not a family court.
Separate independent consumer/replay courts must qualify it before ANIMATION
KINEMATICS can be admitted. Action names and mesh IDs remain privileged.

Animation frame displacement is kinematics per authored frame, NEVER physical
stiffness, isolated joint intervention or causal Response ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA="RealSaS.Corpus370V5SourceNativeCanonicalAnimationCandidate.v1"
PASS="PASS_NATIVE_SOURCE_REST_BOUND_CANONICAL_POSE_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
MASKED="MASKED_STATIC_NATIVE_ACTION__NO_POSITIVE_MOTION_LABEL"
MATRIX_TOLERANCE=1e-7
REST_REL_TOLERANCE=1e-6


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as reader:
        return hashlib.file_digest(reader,"sha256").hexdigest()


def bound(ref,code):
    require(isinstance(ref,dict),"NATIVE_CANONICAL_REF:"+code)
    path=Path(str(ref.get("path") or "")).resolve()
    require(path.is_file() and sha(path)==ref.get("sha256"),
            "NATIVE_CANONICAL_PARENT_BYTES:"+code)
    return path


def _surface(v,f,owner,code):
    v=np.asarray(v,np.float64)
    f=np.asarray(f,np.int64)
    owner=np.asarray(owner,np.int64)
    require(v.ndim==2 and v.shape[1]==3 and len(v)>0 and np.isfinite(v).all(),
            code+"_VERTICES")
    require(f.ndim==2 and f.shape[1]==3 and len(f)>0
            and f.min()>=0 and f.max()<len(v),
            code+"_FACES")
    require(owner.shape==(len(f),) and np.all(owner>=0),
            code+"_OWNER")
    return v,f,owner


def _transform(points,matrix):
    shape=points.shape
    homo=np.concatenate([points.reshape(-1,3),np.ones((np.prod(shape[:-1],dtype=int),1))],axis=1)
    transformed=homo @ matrix.T
    require(np.isfinite(transformed).all() and
            np.max(np.abs(transformed[:,3]-1.))<=1e-7,
            "NATIVE_CANONICAL_HOMOGENEOUS_GAUGE")
    return transformed[:,:3].reshape(shape)


def frame_schedule(receipt,n_frames):
    rows=receipt.get("sampled_frames")
    native=receipt.get("native_motion_inventory_exported")
    require(isinstance(rows,list) and len(rows)==n_frames
            and isinstance(native,list) and native
            and len(native)==len(set(native))
            and all(isinstance(x,str) and x for x in native),
            "NATIVE_CANONICAL_AUTHORED_SCHEDULE")
    action_to_index={action:i for i,action in enumerate(native)}
    ordinal=[];frame=[];last={};encounter=set()
    for row in rows:
        require(isinstance(row,dict)
                and row.get("action") in action_to_index
                and type(row.get("frame")) is int,
                "NATIVE_CANONICAL_FRAME_PROVENANCE")
        act=row["action"];value=row["frame"]
        require(act not in last or value>last[act],
                "NATIVE_CANONICAL_MONOTONIC_ACTION_FRAMES")
        require(act not in encounter or ordinal[-1]==action_to_index[act],
                "NATIVE_CANONICAL_INTERLEAVED_ACTION_FRAMES")
        last[act]=value
        encounter.add(act)
        ordinal.append(action_to_index[act])
        frame.append(value)
    require(encounter==set(native),"NATIVE_CANONICAL_MISSING_ACTION")
    return np.asarray(ordinal,np.int32),np.asarray(frame,np.int64)


def build_numeric(rest_geometry,native_teacher,snapshot,receipt):
    rest,faces,owners=_surface(
        rest_geometry["rest_vertices_source_world"],
        rest_geometry["faces"],
        rest_geometry["face_owner_privileged"],
        "NATIVE_CANONICAL_ADMITTED_SOURCE")
    native_rest,native_faces,native_owners=_surface(
        native_teacher["rest_vertices_source_world"],
        native_teacher["faces"],
        native_teacher["face_owner"],
        "NATIVE_CANONICAL_NATIVE_IMPORT")
    require(np.array_equal(faces,native_faces)
            and np.array_equal(owners,native_owners)
            and native_rest.shape==rest.shape,
            "NATIVE_CANONICAL_REST_TOPOLOGY_DRIFT")
    extent=float(np.linalg.norm(np.ptp(rest,axis=0)))
    require(np.isfinite(extent) and extent>0.,"NATIVE_CANONICAL_ZERO_GEOMETRY_EXTENT")
    mismatch=float(np.max(np.abs(rest-native_rest)))
    tolerance=max(1.,extent)*REST_REL_TOLERANCE
    require(mismatch<=tolerance,"NATIVE_CANONICAL_NATIVE_REST_NOT_OBSERVATION_EQUIVALENT")
    posed=np.asarray(native_teacher["posed_vertices_source_world"],np.float64)
    require(posed.ndim==3 and posed.shape[1:]==rest.shape
            and len(posed)>0 and np.isfinite(posed).all(),
            "NATIVE_CANONICAL_POSED_VERTEX_LAYOUT")
    ordinal,frames=frame_schedule(receipt,len(posed))
    m=np.asarray(snapshot.get("source_world_to_canonical"),np.float64)
    inv=np.asarray(snapshot.get("canonical_to_source_world"),np.float64)
    require(m.shape==inv.shape==(4,4) and
            np.isfinite(m).all() and np.isfinite(inv).all()
            and np.max(np.abs(m[3]-[0,0,0,1]))<=MATRIX_TOLERANCE
            and np.max(np.abs(inv[3]-[0,0,0,1]))<=MATRIX_TOLERANCE
            and np.max(np.abs(m@inv-np.eye(4)))<=MATRIX_TOLERANCE
            and np.max(np.abs(inv@m-np.eye(4)))<=MATRIX_TOLERANCE,
            "NATIVE_CANONICAL_SOURCE_FRAME_INVERSE")
    canonical_rest=_transform(rest,m)
    canonical_posed=_transform(posed,m)
    source_back=_transform(canonical_posed,inv)
    inverse_error=float(np.max(np.abs(source_back-posed)))
    require(inverse_error<=max(1.,extent)*MATRIX_TOLERANCE,
            "NATIVE_CANONICAL_POSE_FRAME_RECONSTRUCTION")
    rest_motion=np.linalg.norm(canonical_posed-canonical_rest[None],axis=2)
    max_motion=float(rest_motion.max())
    require(np.isfinite(max_motion),"NATIVE_CANONICAL_POSE_DISPLACEMENT")
    edges=[];edge_displacements=[];edge_dframes=[]
    for i in range(1,len(ordinal)):
        if ordinal[i]!=ordinal[i-1]:
            continue
        dt=int(frames[i]-frames[i-1])
        require(dt>0,"NATIVE_CANONICAL_NONPOSITIVE_FRAME_DELTA")
        # source-world/canonical per authored frame is a kinematic delta,
        # NOT velocity per physical second (timebase independently unknown).
        delta=float(np.linalg.norm(canonical_posed[i]-canonical_posed[i-1],axis=1).max())
        edges.append(i)
        edge_dframes.append(dt)
        edge_displacements.append(delta)
    positive=max_motion>1e-8
    return {
        "arrays":{
            "rest_vertices_canonical":canonical_rest,
            "posed_vertices_canonical":canonical_posed,
            "faces":faces,
            "source_triangle_owner_teacher_only":owners.astype(np.int32),
            "frame_action_ordinal_teacher_only":ordinal,
            "frame_index_native":frames,
            "valid_same_action_edge_ending_frame":np.asarray(edges,np.int64),
            "source_action_edge_frame_delta":np.asarray(edge_dframes,np.int64),
            "source_action_edge_max_displacement_canonical":np.asarray(edge_displacements,np.float64),
        },
        "measures":{
            "max_native_rest_replay_error_source_world":mismatch,
            "native_rest_equivalence_tolerance_source_world":tolerance,
            "max_canonical_roundtrip_error":inverse_error,
            "frame_count":int(len(posed)),
            "native_authored_action_count":len(receipt["native_motion_inventory_exported"]),
            "valid_within_action_delta_count":len(edges),
            "measured_max_rest_relative_displacement_canonical":max_motion,
            "action_per_native_frame_units_only":True,
            "native_frame_to_seconds_authority":False,
            "source_motion_is_causal_intervention":False,
            "model_training_authorized":False,
        },
        "status":PASS if positive else MASKED,
    }


def process(*,subject_id,source_sha,rest_manifest,teacher_court,snapshot_ref,output_dir):
    output_dir=Path(output_dir).resolve();output_dir.mkdir(parents=True,exist_ok=True)
    source=bound(rest_manifest,"SOURCE_REST_MANIFEST")
    rest_doc=json.loads(source.read_text())
    require(rest_doc.get("subject_id")==subject_id and
            rest_doc.get("source_sha256")==source_sha,
            "NATIVE_CANONICAL_REST_SUBJECT_ID")
    rest_npz=bound(rest_doc["output"],"SOURCE_REST_NPZ")
    teacher_doc_path=bound(teacher_court,"NATIVE_TEACHER_COURT")
    teacher_doc=json.loads(teacher_doc_path.read_text())
    require(teacher_doc.get("subject_id")==subject_id and
            teacher_doc.get("source_sha256")==source_sha and
            teacher_doc.get("status")==
            "PASS_SOURCE_AUTHORED_TEMPORAL_GEOMETRY_CANDIDATE__CAUSAL_RESPONSE_SEPARATE"
            and teacher_doc.get("training_authorized") is False
            and teacher_doc.get("family_authority_minted") is False,
            "NATIVE_CANONICAL_TEACHER_COURT_NOT_POSITIVE")
    native_npz=bound(teacher_doc["teacher_arrays"],"NATIVE_TEACHER_NPZ")
    native_receipt_path=bound(teacher_doc["teacher_manifest"],"NATIVE_TEACHER_MANIFEST")
    native_receipt=json.loads(native_receipt_path.read_text())
    require(native_receipt.get("transport_revision")==4
            and native_receipt.get("subject_id")==subject_id
            and native_receipt.get("source",{}).get("sha256")==source_sha
            and native_receipt.get("teacher_only") is True
            and native_receipt.get("teacher_rig_skin_motion_allowed_as_inference_input") is False,
            "NATIVE_CANONICAL_RECEIPT_PROVENANCE")
    snap_path=bound(snapshot_ref,"ADMITTED_SOURCE_SNAPSHOT")
    snap=json.loads(snap_path.read_text())
    require(snap.get("source_sha256")==source_sha,"NATIVE_CANONICAL_SNAPSHOT_SOURCE")
    with np.load(rest_npz,allow_pickle=False) as a,np.load(native_npz,allow_pickle=False) as b:
        measured=build_numeric(a,b,snap,native_receipt)
    out_npz=output_dir/"CANONICAL_NATIVE_ANIMATION_CANDIDATE_V1.npz"
    temp=output_dir/"CANONICAL_NATIVE_ANIMATION_CANDIDATE_V1.partial.npz"
    np.savez_compressed(temp,**measured["arrays"])
    temp.replace(out_npz)
    manifest={
        "schema":SCHEMA,"status":measured["status"],
        "subject_id":subject_id,"source_sha256":source_sha,
        "parents":{
            "source_rest_manifest":rest_manifest,
            "source_rest_arrays":rest_doc["output"],
            "native_teacher_court":teacher_court,
            "native_teacher_arrays":teacher_doc["teacher_arrays"],
            "native_teacher_manifest":teacher_doc["teacher_manifest"],
            "observation_source_scene_snapshot":snapshot_ref,
        },
        "canonical_arrays":{"path":str(out_npz),"sha256":sha(out_npz)},
        "measures":measured["measures"],
        "authority":{
            "candidate_only":True,
            "independent_native_action_replay_court_passed":False,
            "student_privileged_action_names_allowed":False,
            "causal_response_minted":False,
            "animation_kinematics_family_minted":False,
            "tessa_mechanical_teacher_minted":False,
            "training_authorized":False,
        },
    }
    out_manifest=output_dir/"CANONICAL_NATIVE_ANIMATION_CANDIDATE_V1.json"
    out_manifest.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--subject-id",required=True)
    p.add_argument("--source-sha256",required=True)
    p.add_argument("--source-rest-manifest",type=Path,required=True)
    p.add_argument("--native-teacher-court",type=Path,required=True)
    p.add_argument("--observation-snapshot",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    result=process(subject_id=a.subject_id,source_sha=a.source_sha256,
                   rest_manifest={"path":str(a.source_rest_manifest.resolve()),"sha256":sha(a.source_rest_manifest)},
                   teacher_court={"path":str(a.native_teacher_court.resolve()),"sha256":sha(a.native_teacher_court)},
                   snapshot_ref={"path":str(a.observation_snapshot.resolve()),"sha256":sha(a.observation_snapshot)},
                   output_dir=a.output_dir)
    print(json.dumps({"status":result["status"],"subject_id":result["subject_id"],
                      "measures":result["measures"],"training_authorized":False},sort_keys=True))


if __name__=="__main__":main()
