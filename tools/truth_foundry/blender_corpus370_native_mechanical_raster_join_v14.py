#!/usr/bin/env python3
"""V14: SAME original native control perturbation, independent 3D and 2D pixels.

Real Blender: original authored FBX/GLTF action + exact source carrier,
per-precommitted control/axis ±5°; compare original UV/material image, identical
posed source-face soup, and active causal face isolation on same camera. The
source rig is privileged experiment instrumentation, NOT an ontology or target.
No alpha/draw order/appearance CAA or artist drawable ownership is minted.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v2 as transport
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4
from tools.iris_vnext import blender_authored_teacher_export_v4_actions as actions
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as probe
from tools.truth_foundry.blender_teacher_scene_v1 import _appearance_and_uv
from tools.truth_foundry.blender_corpus370_multi_partition_edit_v7 import (
    sha,verify,require,_camera_full,_frame,_group_mode
)
from tools.truth_foundry.blender_corpus370_heldout_native_edit_render_v12 import _set_animated_soup
from tools.corpus.corpus370_native_action_binding_v13b import bound_rows_for_source_action
from tools.corpus.corpus370_native_raster_join_plan_v14 import (
    SCHEMA as PLAN_SCHEMA,arms_for_plan,select,verify_locked_source_selection
)

SCHEMA="RealSaS.Corpus370SourceNativeMechanicalRasterJointRender.v14"
JOB="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
AXES=((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))

def load(path):
    return json.loads(Path(path).read_text())

def _inventory(original_actions,resolved,bindings):
    bones=[("POSE_BONE",obj,pb) for obj in bpy.context.scene.objects
           if obj.type=="ARMATURE" for pb in obj.pose.bones]
    owners={}
    unsupported=set()
    for name in original_actions:
        for row in bound_rows_for_source_action(name,resolved,bindings):
            owner=row["owner"]
            if isinstance(owner,bpy.types.Object) and owner.type in ("EMPTY","MESH","ARMATURE"):
                owners[id(owner)]=owner
            elif not isinstance(owner,bpy.types.Object):
                unsupported.add(str(owner.bl_rna.identifier))
    objects=[("IMPORTED_ANIMATED_OBJECT",obj,None) for obj in owners.values()]
    controls=sorted(bones+objects,
                    key=lambda x:(x[0],x[1].name,x[2].name if x[2] else ""))
    return controls,unsupported

def _inventory_digest(rows):
    return hashlib.sha256("\n".join(
        kind+":"+obj.name+":"+(bone.name if bone is not None else "")
        for kind,obj,bone in rows).encode()).hexdigest()

def render(job_path,output_path):
    job=load(job_path)
    require(job.get("schema")==JOB
            and job.get("teacher_only") is True
            and job.get("training_authorized") is False
            and job.get("graphics_owner_authorized") is False
            and job.get("artist_drawable_identity_inferred") is False
            and job.get("source_bone_names_as_targets") is False
            and job.get("view_indices")==[0,2,4,6]
            and job.get("raster_resolution")==192
            and job.get("cycles_samples")==32
            and job.get("native_actuation_degrees")==5.0
            and job.get("root_sham_source_world_fraction")==.025,
            "V14_RENDER_ONLY_SHA_LOCKED_ORIGINAL_SOURCE_BOUNDED_NO_OWNER")
    source=verify(job["source"],"V14_ORIGINAL_SOURCE_SHA")
    for key in ("admission","native_teacher_court","v11_private_plan"):
        verify(job[key],"V14_"+key.upper()+"_SHA")
    candidate=load(verify(job["candidate"],"V14_ORIGINAL_CANDIDATE_SHA"))
    cameras=load(verify(job["camera"],"V14_SEALED_SOURCE_CAMERA_SHA"))
    restpath=verify(job["rest_geometry"],"V14_REST_SOURCE_FACE_SHA")
    teacherpath=verify(job["native_teacher"],"V14_ORIGINAL_POSED_SOURCE_SHA")
    receipt=load(verify(job["native_teacher_receipt"],"V14_ORIGINAL_NATIVE_RECEIPT_SHA"))
    prior=load(verify(job["v13b_manifest"],"V14_V13B_ORIGINAL_INTERVENTION_MANIFEST_SHA"))
    original_npz=verify(job["v13b_array"],"V14_ORIGINAL_V13B_SOURCE_PERTURBATION_VERTEX_SHA")
    selection=load(verify(job["selected_source_controls"],"V14_PRECOMMITTED_SOURCE_ONLY_CONTROL_SELECTION_SHA"))
    require(selection.get("schema")==PLAN_SCHEMA
            and selection.get("status")=="ORIGINAL_NATIVE_GEOMETRY_ONLY_PRECOMMITTED_BEFORE_V14_RGBA"
            and selection["native_pose_indices_teacher_only"]==job["pose_indices"]
            and prior["original_selected_pose_indices_teacher_only"]==job["pose_indices"]
            and prior["source_sha256"]==job["source"]["sha256"]==receipt["source"]["sha256"]
            and prior["source_pose_articulation_array_sha256"]==job["v13b_array"]["sha256"]
            and selection["source_control_inventory_count_teacher_only"]==
                prior["source_controls_measured_count"]
            and prior["training_authorized"] is False
            and prior["graphics_owner_authorized"] is False
            and prior["no_anon_control_intervention_result_as_drawable_truth"] is True
            and cameras["schema"]=="RealSaS.CameraProjectionBundle.v1",
            "V14_FROZEN_REAL_V13B_SOURCE_RASTER_ONLY_CONTROL_PRECOMMIT")
    arms,pairs=arms_for_plan(selection)
    with np.load(restpath,allow_pickle=False) as r:
        rv=np.asarray(r["rest_vertices_source_world"],np.float64)
        faces=np.asarray(r["faces"],np.int64)
        owner=np.asarray(r["face_owner_privileged"],np.int32)
    with np.load(teacherpath,allow_pickle=False) as n:
        native_rest=np.asarray(n["rest_vertices_source_world"],np.float64)
        native_faces=np.asarray(n["faces"],np.int64)
        native_owner=np.asarray(n["face_owner"],np.int32)
        native_posed=np.asarray(n["posed_vertices_source_world"],np.float64)
    with np.load(original_npz,allow_pickle=False) as d:
        baselines=np.asarray(d["baseline_vertices"],np.float64)
        interventions=np.asarray(d["native_control_perturbed_vertices"],np.float64)
    extent=max(1e-12,float(np.linalg.norm(np.ptp(rv,axis=0))))
    tol=1e-6*max(1.,extent)
    require(np.array_equal(faces,native_faces)
            and np.array_equal(owner,native_owner)
            and np.max(np.abs(rv-native_rest))<=tol
            and native_posed.shape[1:]==rv.shape
            and baselines.shape==(3,len(rv),3)
            and interventions.shape==(3,prior["source_controls_measured_count"],3,2,len(rv),3)
            and len(receipt["sampled_frames"])==len(native_posed)
            and len(job["pose_indices"])==3
            and all(0<=p<len(native_posed) for p in job["pose_indices"])
            and np.max(np.abs(baselines-native_posed[job["pose_indices"]]))<=tol,
            "V14_ORIGINAL_SOURCE_V13B_NATIVE_VERTEX_POSE_TOPOLOGY_IDENTITY")
    expected=select(rv,baselines,interventions,job["pose_indices"],
                    prior["source_controls_measured_count"],job["pose_indices"])
    require(verify_locked_source_selection(expected,selection),
            "V14_PRECOMMIT_SOURCE_CONTROL_NOT_RECONSTRUCTIBLE_FROM_ORIGINAL_V13B_VERTICES")
    camera={int(x["view_index"]):x for x in cameras["cameras"]}
    require(set(camera)==set(range(8)),
            "V14_ORIGINAL_NATIVE_EIGHT_CAMERA_BUNDLE")
    images=[]
    pose_evidence=[]
    with tempfile.TemporaryDirectory(prefix="corpus370_native_mech_raster_v14_") as td:
        tmp=Path(td)
        base._import_animated(source)
        names=receipt["native_motion_inventory_expected"]
        resolved={name:transport._action_by_requested_take(name) for name in names}
        bindings=actions.capture_imported_action_bindings(resolved)
        meshes=base._mesh_objects()
        base._neutralize_animation()
        snap=v4._extract_snapshot(meshes,freeze_triangulation=True)
        require(np.array_equal(np.asarray(snap["faces"]),faces)
                and np.array_equal(np.asarray(snap["face_owner"]),owner)
                and np.max(np.abs(np.asarray(snap["vertices"],np.float64)-rv))<=tol,
                "V14_REIMPORTED_ORIGINAL_SOURCE_REST_FAILED")
        mat=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(mat,list(candidate.get("material_transform") or []))
        uv,uv_valid,_,_=_appearance_and_uv(meshes,snap)
        original={
            "rest_vertices_source_world":rv,
            "faces":faces,
            "face_owner_privileged":owner,
            "face_material_privileged":np.asarray(snap["face_material"],np.int32),
            "triangle_uv":uv,
            "triangle_uv_valid":uv_valid,
        }
        soup,face_map=probe._build_triangle_soup(
            original,meshes,probe._transparent_material())
        require(len(face_map)==len(faces)
                and all(np.max(np.abs(np.asarray(o.matrix_world)-np.eye(4)))<=1e-8
                        for o in soup),"V14_MATERIAL_UV_SOURCE_FACE_SOUP_WORLD_ADDRESS")
        all_face_labels=np.arange(len(faces),dtype=np.int32)
        probe._configure_measurement(32)
        cam=probe._make_camera()
        selected_native_actions={receipt["sampled_frames"][p]["action"]
                                 for p in job["pose_indices"]}
        controls,unsupported=_inventory(selected_native_actions,resolved,bindings)
        controls=controls[:128]
        require(_inventory_digest(controls)==
                prior["native_rig_control_names_private_inventory_sha256"]
                and len(controls)==prior["source_controls_measured_count"]
                and sum(k=="POSE_BONE" for k,_,_ in controls)==
                    prior["native_control_carrier_types_teacher_only"]["POSE_BONE"]
                and sum(k=="IMPORTED_ANIMATED_OBJECT" for k,_,_ in controls)==
                    prior["native_control_carrier_types_teacher_only"]["IMPORTED_ANIMATED_OBJECT"],
                "V14_SOURCE_CONTROL_NAMES_PRIVATE_SHA_CANNOT_CHANGE")
        active_by_pose=[]
        for pose_ordinal,pidx in enumerate(job["pose_indices"]):
            sample=receipt["sampled_frames"][pidx]
            require(sample["action"] in resolved and type(sample["frame"]) is int,
                    "V14_REAL_ORIGINAL_NATIVE_POSE_AUTHORED_SOURCE_ONLY")
            replayed=actions.activate_action(resolved[sample["action"]],bindings)
            bpy.context.scene.frame_set(sample["frame"])
            bpy.context.view_layer.update()
            current=v4._extract_snapshot(meshes,freeze_triangulation=False)
            actual=np.asarray(current["vertices"],np.float64)
            require(current["polygon_signature"]==snap["polygon_signature"]
                    and actual.shape==rv.shape
                    and np.max(np.abs(actual-native_posed[pidx]))<=tol
                    and np.max(np.abs(actual-baselines[pose_ordinal]))<=tol,
                    "V14_REPLAYED_ORIGINAL_NATIVE_POSE_NO_SOURCE_COPY_SHORTCUT")
            _set_animated_soup(face_map,faces,actual)
            supports=[]
            for q in selection["controls"]:
                c=q["control_index_teacher_only"];a=q["axis_index_teacher_only"]
                displacement=np.linalg.norm(interventions[pose_ordinal,c,a,0]-actual,axis=-1)
                face_moved=np.mean(displacement[faces],axis=1)>2e-4*extent
                supports.append(face_moved)
            require(all(bool(np.any(mask)) for mask in supports),
                    "V14_PRESELECTED_CAUSAL_CONTROL_NO_SOURCE_ACTIVE_FACES")
            active_by_pose.append([int(np.count_nonzero(z)) for z in supports])
            view_images=[]
            for view in job["view_indices"]:
                _camera_full(cam,camera[view],192)
                def draw(name,mode,face_mask=None):
                    source_render=mode=="SOURCE"
                    for o in meshes:o.hide_render=not source_render
                    for o in soup:o.hide_render=source_render
                    if not source_render:
                        excluded={int(i) for i in range(len(faces))
                                  if face_mask is not None and not bool(face_mask[i])}
                        _group_mode(face_map,all_face_labels,excluded=excluded)
                    bpy.context.view_layer.update()
                    return _frame(tmp/(f"p{pidx}_v{view}_{name}.exr"),192)
                rendered={
                    "ORIGINAL_BASE":draw("ORIGINAL_BASE","SOURCE"),
                    "SOUP_BASE":draw("SOUP_BASE","SOUP"),
                    "SOUP_NOOP":draw("SOUP_NOOP","SOUP")
                }
                # An independent full-scene uniform translation control:
                # translating the CAMERA by -delta is the same projection as
                # rigidly translating the entire source scene by +delta.
                # It is NOT an authored root/attachment perturbation.
                cam.location=cam.location-Vector((.025*extent,0.,0.))
                bpy.context.view_layer.update()
                rendered["ROOT_SHAM_ORIGINAL"]=draw("ROOT_SHAM_ORIGINAL","SOURCE")
                rendered["ROOT_SHAM_SOUP"]=draw("ROOT_SHAM_SOUP","SOUP")
                _camera_full(cam,camera[view],192)
                for ci,support in enumerate(supports):
                    rendered[f"ACTIVE_C{ci:02d}_BASE"]=draw(
                        f"ACTIVE_C{ci:02d}_BASE","SOUP",support)
                for ci,query in enumerate(selection["controls"]):
                    c=query["control_index_teacher_only"]
                    a=query["axis_index_teacher_only"]
                    kind,obj,bone=controls[c]
                    carrier=bone if kind=="POSE_BONE" else obj
                    original_basis=carrier.matrix_basis.copy()
                    for sj,sign in enumerate((1.,-1.)):
                        sign_name="PLUS" if sign>0 else "MINUS"
                        rot=Matrix.Rotation(sign*np.pi*5./180.,4,AXES[a])
                        carrier.matrix_basis=original_basis @ rot
                        bpy.context.view_layer.update()
                        observed=v4._extract_snapshot(meshes,freeze_triangulation=False)
                        pos=np.asarray(observed["vertices"],np.float64)
                        expected_pos=interventions[pose_ordinal,c,a,sj]
                        require(observed["polygon_signature"]==snap["polygon_signature"]
                                and pos.shape==rv.shape
                                and np.max(np.abs(pos-expected_pos))<=tol,
                                "V14_ACTUAL_SOURCE_NATIVE_CONTROL_NOT_ORIGINAL_V13B_INTERVENTION")
                        _set_animated_soup(face_map,faces,expected_pos)
                        rendered[f"SOURCE_C{ci:02d}_{sign_name}"]=draw(
                            f"SOURCE_C{ci:02d}_{sign_name}","SOURCE")
                        rendered[f"SOUP_C{ci:02d}_{sign_name}"]=draw(
                            f"SOUP_C{ci:02d}_{sign_name}","SOUP")
                        rendered[f"ACTIVE_C{ci:02d}_{sign_name}"]=draw(
                            f"ACTIVE_C{ci:02d}_{sign_name}","SOUP",supports[ci])
                        carrier.matrix_basis=original_basis
                        bpy.context.view_layer.update()
                        _set_animated_soup(face_map,faces,actual)
                    require(np.max(np.abs(np.asarray(
                        v4._extract_snapshot(meshes,freeze_triangulation=False)["vertices"],
                        np.float64)-actual))<=tol,
                        "V14_SOURCE_NATIVE_CONTROL_NOOP_RESTORE_FAILURE")
                require(set(rendered)==set(arms),"V14_REAL_SOURCE_RGBA_EXPECTED_ARMS")
                view_images.append(np.stack([rendered[name] for name in arms],axis=0))
            images.append(np.stack(view_images,axis=0))
            pose_evidence.append({
                "pose_index_teacher_only":pidx,
                "source_action_ordinal_teacher_only":names.index(sample["action"]),
                "source_original_frame_teacher_only":sample["frame"],
                "source_motion_original_binding_kind_teacher_only":replayed["carrier_kind"],
                "cause_active_source_face_count_per_selected_control":active_by_pose[-1],
                "source_bone_or_object_identity_as_model_label":False,
            })
            base._neutralize_animation()
    pixels=np.stack(images,axis=0).astype(np.float16)
    require(pixels.shape==(3,4,len(arms),192,192,4)
            and np.isfinite(pixels).all(),"V14_RAW_ORIGINAL_NATIVE_RGBA_IMAGE_ARRAY_INCOMPLETE")
    out=Path(output_path).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    arrays=out.with_suffix(".npz")
    np.savez_compressed(arrays,original_native_actuator_rgba=pixels)
    manifest={
        "schema":SCHEMA,
        "status":"V14_REAL_SAME_CAUSAL_SOURCE_NATIVE_CONTROL_VERTEX_AND_RGBA__INDEPENDENT_COURT_REQUIRED",
        "subject_id":job["subject_id"],
        "source_sha256":job["source"]["sha256"],
        "native_poses_exact_source_author_actions":True,
        "original_source_rest_and_animated_face_identity":True,
        "source_pose_matches_original_v13b_vertex_oracle":True,
        "perturbed_source_pose_matches_original_v13b_native_control":True,
        "original_native_source_carrier_inventory_sha256":
            prior["native_rig_control_names_private_inventory_sha256"],
        "source_private_control_selection_sha256":sha(job["selected_source_controls"]["path"]),
        "source_private_v13b_intervention_manifest_sha256":sha(job["v13b_manifest"]["path"]),
        "source_private_v13b_intervention_array_sha256":sha(original_npz),
        "source_camera_bundle_sha256":sha(job["camera"]["path"]),
        "source_rest_geometry_sha256":sha(restpath),
        "source_native_teacher_sha256":sha(teacherpath),
        "selected_source_controls":selection,
        "source_pose_evidence_teacher_only":pose_evidence,
        "original_source_control_carrier_kinds_teacher_only":
            prior["native_control_carrier_types_teacher_only"],
        "original_source_action_name_as_model_feature":False,
        "source_bone_mesh_material_names_as_drawable_labels":False,
        "source_active_faces_are_not_artist_drawables":True,
        "rendered_arms":arms,
        "rendered_pose_indices":job["pose_indices"],
        "rendered_source_camera_indices":job["view_indices"],
        "raster_resolution":192,
        "cycles_samples":32,
        "root_sham":"CAMERA_NEGATIVE_TRANSLATION_EQUIVALENT_TO_POSITIVE_GLOBAL_SOURCE_TRANSLATION_NOT_AUTHORED_ROOT",
        "rgba_and_visibility_only_not_actual_draw_order_proof":True,
        "original_perturbed_pose_rgba_npz_path":str(arrays),
        "original_perturbed_pose_rgba_npz_sha256":sha(arrays),
        "private_job_sha256":sha(job_path),
        "artist_same_drawable_pairs":0,
        "artist_different_drawable_pairs":0,
        "native_full_resolution_caa_authorized":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }
    out.write_text(json.dumps(manifest,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":manifest["status"],"subject_id":job["subject_id"],
                      "real_render_arms":3*4*len(arms),"arms_per_view":len(arms),
                      "source_control_count":len(selection["controls"]),
                      "source_semantic_layer_authority":False}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--job",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    render(a.job,a.output)
