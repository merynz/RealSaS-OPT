#!/usr/bin/env python3
"""V13 anonymous native source actuator ± perturbation on original authored rig.

The source rig is PRIVILEGED instrumentation only, NEVER a compiler label or a
semantic body-part taxonomy. This measures actual Blender evaluated source mesh
effects of local rig control interventions versus restored no-op. No V13 pixels
or human drawable identity are inferred.
"""
from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
import numpy as np
import bpy
from mathutils import Matrix

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v2 as transport
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4
from tools.iris_vnext import blender_authored_teacher_export_v4_actions as actions
from tools.truth_foundry.blender_corpus370_multi_partition_edit_v7 import (
    sha,verify,require
)

SCHEMA="RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b"
JOB="RealSaS.Corpus370NativeCarrierActuatorInterventionJob.v13b"
AXES=(("X",(1.,0.,0.)),("Y",(0.,1.,0.)),("Z",(0.,0.,1.)))

def _read(path):
    return json.loads(Path(path).read_text())

def run(job_path, output_path):
    job=_read(job_path)
    require(job.get("schema")==JOB
        and job.get("teacher_only") is True
        and job.get("graphics_owner_authorized") is False
        and job.get("training_authorized") is False
        and job.get("source_bone_name_as_label") is False
        and job.get("pose_indices") is not None
        and len(job["pose_indices"])==3
        and job.get("native_actuation_degrees")==5.0
        and job.get("axes")==["X","Y","Z"]
        and job.get("allow_source_rig_teacher_only") is True
        and job.get("source_control_budget")==128,
        "V13B_NATIVE_RIG_SOURCE_TEACHER_ONLY_NO_LABEL")
    source=verify(job["source"],"V13B_ORIGINAL_SOURCE_SHA")
    rest_path=verify(job["rest_geometry"],"V13B_SOURCE_REST_SHA")
    teacher_path=verify(job["native_teacher"],"V13B_SOURCE_AUTHORED_NATIVE_TEACHER_SHA")
    receipt_path=verify(job["native_teacher_receipt"],"V13B_NATIVE_AUTHOR_SAMPLE_RECEIPT_SHA")
    verify(job["native_teacher_court"],"V13B_NATIVE_INDEPENDENT_COURT_SHA")
    proposal_path=verify(job["group_proposal"],"V13B_ANONYMOUS_GROUP_PROPOSAL_SHA")
    v11_path=verify(job["v11_private_plan"],"V13B_PRECOMMITTED_HELDOUT_QUERY_PLAN_SHA")
    receipt=_read(receipt_path)
    group=_read(proposal_path)
    plan=_read(v11_path)
    require(job["source"]["sha256"]==receipt["source"]["sha256"]==
            group["source_sha256"]==plan["source_sha256"],
            "V13B_SOURCE_SHA_AND_GEOMETRY_PARENT")
    require(plan["new_source_face_intervention_query_count"]==12
            and plan["graphics_owner_authorized"] is False
            and plan["training_authorized"] is False
            and plan["subject_id"]==job["subject_id"]
            and job["group_proposal"]["sha256"]==
                 plan["parents"]["original_geometry_proposal_sha256"]
            and job["native_teacher"]["sha256"]==
                 receipt["output"]["sha256"],
            "V13B_NEW_SOURCE_GROUP_ORIGINAL_AUTHOR_TEACHER")
    with np.load(rest_path,allow_pickle=False) as r:
        rv=np.asarray(r["rest_vertices_source_world"],np.float64)
        faces=np.asarray(r["faces"],np.int64)
        owners=np.asarray(r["face_owner_privileged"],np.int32)
    with np.load(teacher_path,allow_pickle=False) as n:
        nv=np.asarray(n["posed_vertices_source_world"],np.float64)
        rf=np.asarray(n["faces"],np.int64)
        no=np.asarray(n["face_owner"],np.int32)
        original_rest=np.asarray(n["rest_vertices_source_world"],np.float64)
    extent=float(np.linalg.norm(np.ptp(rv,axis=0)))
    tol=1e-6*max(1.,extent)
    require(np.array_equal(faces,rf) and np.array_equal(owners,no)
            and rv.shape==original_rest.shape
            and np.max(np.abs(rv-original_rest))<=tol
            and len(nv)==len(receipt["sampled_frames"])
            and len(nv)>=max(job["pose_indices"])+1,
            "V13B_PREEXISTING_REST_TO_AUTHORED_POSED_TRIANGLE_IDENTITY")
    base._import_animated(source)
    names=receipt["native_motion_inventory_expected"]
    resolved={a:transport._action_by_requested_take(a) for a in names}
    bindings=actions.capture_imported_action_bindings(resolved)
    meshes=base._mesh_objects()
    base._neutralize_animation()
    snapshot=v4._extract_snapshot(meshes,freeze_triangulation=True)
    require(np.array_equal(np.asarray(snapshot["faces"]),faces)
            and np.array_equal(np.asarray(snapshot["face_owner"]),owners)
            and np.max(np.abs(np.asarray(snapshot["vertices"],np.float64)-rv))<=tol,
            "V13B_IMPORT_ORIGINAL_SOURCE_REST_DRIFT")
    # All anonymous control axes are explored, without topology, bone-name,
    # morphology category, or anatomical source-specific filtering.
    # Ontology-free native mechanism inventory: an authored source can bind
    # animation directly to imported OBJECT/EMPTY/MESH nodes (Cat), not just
    # armature pose bones (Triceratops/Cowfish). Object controls are selected
    # ONLY if importer-native action bindings occur in the three original
    # authored action samples. This is NOT a model-visible actor type prior.
    bone_carriers=[("POSE_BONE",obj,pb) for obj in bpy.context.scene.objects
                   if obj.type=="ARMATURE" for pb in obj.pose.bones]
    original_actions={receipt["sampled_frames"][p]["action"]
                      for p in job["pose_indices"]}
    object_owners={}
    unsupported_native_owner_types=set()
    for action_name in original_actions:
        for row in bindings[action_name]:
            owner=row["owner"]
            if isinstance(owner,bpy.types.Object) and owner.type in ("EMPTY","MESH","ARMATURE"):
                object_owners[id(owner)]=owner
            elif not isinstance(owner,bpy.types.Object):
                unsupported_native_owner_types.add(str(owner.bl_rna.identifier))
    node_carriers=[("IMPORTED_ANIMATED_OBJECT",obj,None)
                   for obj in object_owners.values()]
    controls=sorted(bone_carriers+node_carriers,
                    key=lambda x:(x[0],x[1].name,
                                  x[2].name if x[2] is not None else ""))
    count_total=len(controls)
    controls=controls[:job["source_control_budget"]]
    selected=len(controls)
    counts={
        "POSE_BONE":sum(row[0]=="POSE_BONE" for row in controls),
        "IMPORTED_ANIMATED_OBJECT":sum(row[0]=="IMPORTED_ANIMATED_OBJECT" for row in controls)
    }
    baseline=[]
    tensors=[]
    noop_errors=[]
    pose_receipts=[]
    for pidx in job["pose_indices"]:
        frame=receipt["sampled_frames"][pidx]
        require(frame["action"] in resolved and type(frame["frame"]) is int,
                "V13B_AUTHOR_ORIGINAL_NATIVE_ACTION_POSE_BINDING")
        carrier=actions.activate_action(resolved[frame["action"]],bindings)
        bpy.context.scene.frame_set(frame["frame"])
        bpy.context.view_layer.update()
        evaluated=v4._extract_snapshot(meshes,freeze_triangulation=False)
        native=np.asarray(evaluated["vertices"],np.float64)
        require(native.shape==rv.shape and
                evaluated["polygon_signature"]==snapshot["polygon_signature"]
                and np.max(np.abs(native-nv[pidx]))<=tol,
                "V13B_SOURCE_NATIVE_ACTION_POSE_NOT_EQUAL_ORIGINAL_TEACHER")
        baseline.append(native.astype(np.float32))
        perturb=np.empty((selected,3,2,len(rv),3),np.float32)
        for ci,(kind,obj,bone) in enumerate(controls):
            carrier=bone if kind=="POSE_BONE" else obj
            before=carrier.matrix_basis.copy()
            for ai,(aname,axis) in enumerate(AXES):
                for si,sign in enumerate((1.,-1.)):
                    # Source-local physical rig perturbation. Only source rig
                    # authoring is used as an ORACLE measurement mechanism.
                    rotation=Matrix.Rotation(float(sign*job["native_actuation_degrees"]*
                                         np.pi/180.),4,axis)
                    carrier.matrix_basis=before @ rotation
                    bpy.context.view_layer.update()
                    variant=v4._extract_snapshot(meshes,freeze_triangulation=False)
                    position=np.asarray(variant["vertices"],np.float64)
                    require(variant["polygon_signature"]==snapshot["polygon_signature"]
                        and position.shape==native.shape
                        and np.isfinite(position).all(),
                        "V13B_ORIGINAL_SOURCE_INTERVENTION_TOPOLOGY_CHANGED")
                    perturb[ci,ai,si]=position.astype(np.float32)
                    carrier.matrix_basis=before
                    bpy.context.view_layer.update()
            # Every control must be exactly reversible when returned to its
            # authored action matrix (no hidden incremental drift).
            restored=np.asarray(
                v4._extract_snapshot(meshes,freeze_triangulation=False)["vertices"],
                np.float64)
            err=float(np.max(np.abs(restored-native)))
            noop_errors.append(err)
            require(err<=tol,
                    "V13B_SOURCE_ACTUATION_NOT_INVERTIBLE_TO_AUTHOR_POSE")
        tensors.append(perturb)
        pose_receipts.append({
            "teacher_pose_index":pidx,
            "original_action_ordinal_teacher_only":names.index(frame["action"]),
            "original_frame_teacher_only":frame["frame"],
            "exact_original_native_pose_replayed":True,
            "control_inventory_count":count_total,
            "controls_measured":selected,
            "action_carrier_kind_teacher_only":carrier["carrier_kind"]
        })
        base._neutralize_animation()
    out=Path(output_path).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    npz=out.with_suffix(".npz")
    np.savez_compressed(npz,baseline_vertices=np.stack(baseline),
                       native_control_perturbed_vertices=np.stack(tensors))
    digest=hashlib.sha256("\\n".join(k+":"+obj.name+":"+(bone.name if bone is not None else "")
                                       for k,obj,bone in controls).encode()).hexdigest()
    result={
        "schema":SCHEMA,
        "status":"V13B_REAL_ORIGINAL_AUTHORED_SOURCE_ANONYMOUS_RIG_INTERVENTION__INDEPENDENT_COURT_REQUIRED",
        "subject_id":job["subject_id"],"source_sha256":job["source"]["sha256"],
        "original_source_teacher_pose_count":3,
        "source_control_inventory_count":count_total,
        "source_controls_measured_count":selected,
        "native_control_carrier_types_teacher_only":counts,
        "source_animated_object_owner_count":len(object_owners),
        "native_unsupported_source_action_owner_types_teacher_only":
            sorted(unsupported_native_owner_types),
        "source_object_controls_preselected_from_original_imported_action_bindings":True,
        "source_control_inventory_truncated":count_total>selected,
        "native_rig_control_names_private_inventory_sha256":digest,
        "anonymous_control_count_inference_target":False,
        "source_control_carrier_type_as_model_prior":False,
        "native_source_actions_as_student_input":False,
        "original_selected_pose_indices_teacher_only":job["pose_indices"],
        "axes":["X","Y","Z"],"rotation_degrees":5.0,
        "source_rest_vertex_count":len(rv),"source_face_count":len(faces),
        "source_pose_receipts_teacher_only":pose_receipts,
        "rest_source_sha256":sha(rest_path),
        "native_original_pose_teacher_npz_sha256":sha(teacher_path),
        "original_heldout_group_plan_sha256":sha(v11_path),
        "group_proposal_sha256":sha(proposal_path),
        "all_control_restoration_noop_max_error":max(noop_errors,default=0.),
        "source_pose_articulation_array_path":str(npz),
        "source_pose_articulation_array_sha256":sha(npz),
        "no_anon_control_intervention_result_as_drawable_truth":True,
        "artist_same_drawable_positive_pairs":0,
        "artist_different_drawable_negative_pairs":0,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],"subject_id":job["subject_id"],
        "control_count":selected,"pose_count":3,
        "perturbations":3*selected*3*2,
        "source_names_as_targets":False,
        "training_authorized":False
    }),flush=True)
    return result

if __name__=="__main__":
    p=__import__("argparse").ArgumentParser()
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    run(a.job,a.output)
