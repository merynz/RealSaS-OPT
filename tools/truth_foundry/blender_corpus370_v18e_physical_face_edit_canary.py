#!/usr/bin/env python3
"""Real original Blender V18b heldout-pose edit replay for V18a stain groups.

There is NO simulated edit mask. Original source UV/texture/material preserving
triangles are physically hidden or isolated in bpy/Cycles at original pose 2.
V11 pre-registered heldout source-face targets provide blind edit loci. A
single-face remove/isolate is a deliberately limited *proxy reference*, not
the original artist's desired layer or graphics ownership label.

All original source pixels/rig names/face labels remain PRIVATE.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile

import bpy
import bmesh
import numpy as np

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
    _camera_full,_frame,_group_mode,sha,verify,require
)
from tools.truth_foundry.blender_corpus370_heldout_native_edit_render_v12 import (
    _set_animated_soup
)
from tools.corpus.compile_corpus370_v18a_reactive_staining import THRESHOLDS as SOURCE_THRESHOLDS

SCHEMA="RealSaS.Corpus370OriginalPhysicalFaceEditCanaryRender.v18e"
STAIN="RealSaS.Corpus370NativeReactiveStainCandidate.v18a"
STAIN_COURT="RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
V14_JOB="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
POSE=2
VIEWS=(0,2,4,6)
TARGET_INDICES=(0,)
THRESHOLDS=(.1,)
PHYSICAL_VISIBILITY="ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"


def ops():
    rows=["SOURCE_ORIGINAL","SOUP_FULL","SOUP_NOOP","SOUP_EMPTY"]
    for i in range(len(TARGET_INDICES)):
        rows += [f"SINGLE_FACE_REMOVE_T{i:02d}",
                 f"SINGLE_FACE_ISOLATE_T{i:02d}"]
        for t in THRESHOLDS:
            key=str(t).replace(".","p")
            rows += [f"STAIN_REMOVE_{key}_T{i:02d}",
                     f"STAIN_ISOLATE_{key}_T{i:02d}"]
    return rows


def physical_group_mode(soup, face_map, base_meshes, labels, excluded):
    """Physically delete excluded Blender polygons; never fake deletion with alpha.

    Original posed UV/material soup mesh datablocks remain immutable, and every
    intervention starts from these originals. No retained deletion across arms.
    """
    require(isinstance(excluded,set) and all(type(i) is int and i>=0
             for i in excluded),"V18E_PHYSICAL_GROUP_EXCLUSION_TYPE")
    require(len(labels)==len(face_map),"V18E_PHYSICAL_FACE_LABEL_CENSUS")
    omitted_by_mesh={mesh:[] for mesh in base_meshes.values()}
    visible_count=0
    for face,(mesh,local,_,_) in face_map.items():
        require(mesh in omitted_by_mesh, "V18E_PHYSICAL_ORIGINAL_MESH_BINDING")
        if int(labels[face]) in excluded:
            omitted_by_mesh[mesh].append(int(local))
        else:
            visible_count+=1
    for obj in soup:
        original=base_meshes[obj.name]
        removed=omitted_by_mesh[original]
        old=obj.data
        if len(removed)==len(original.polygons):
            obj.data=original
            obj.hide_render=True
        elif not removed:
            obj.data=original
            obj.hide_render=False
        else:
            new_mesh=original.copy()
            bm=bmesh.new()
            try:
                bm.from_mesh(new_mesh)
                bm.faces.ensure_lookup_table()
                require(len(bm.faces)==len(original.polygons),
                        "V18E_SOURCE_BMESH_FACE_ORDER_NOT_PRESERVED")
                bmesh.ops.delete(bm,
                    geom=[bm.faces[i] for i in sorted(removed)],
                    context='FACES')
                bm.to_mesh(new_mesh)
            finally:
                bm.free()
            new_mesh.update()
            require(len(new_mesh.polygons)==len(original.polygons)-len(removed),
                    "V18E_PHYSICAL_SOURCE_FACE_DELETE_COUNT")
            obj.data=new_mesh
            obj.hide_render=False
        if old is not original and old.users==0:
            bpy.data.meshes.remove(old)
    bpy.context.view_layer.update()
    require(sum(len(obj.data.polygons) for obj in soup if not obj.hide_render)
            ==visible_count,"V18E_PHYSICAL_VISIBLE_TRIANGLE_CENSUS")
    return visible_count


def run(job_path,stain_report_path,stain_npz_path,stain_court_path,
        v14_original_pixels,output_path):
    job=json.loads(Path(job_path).read_text())
    stain=json.loads(Path(stain_report_path).read_text())
    independent=json.loads(Path(stain_court_path).read_text())
    require(job.get("schema")==V14_JOB
        and job.get("teacher_only") is True
        and job.get("training_authorized") is False
        and job.get("graphics_owner_authorized") is False
        and job.get("pose_indices") is not None
        and len(job["pose_indices"])==3
        and job["view_indices"]==list(VIEWS)
        and job["raster_resolution"]==192
        and job["cycles_samples"]==32
        and stain.get("schema")==STAIN
        and independent.get("schema")==STAIN_COURT
        and independent.get("status")==
            "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
        and independent.get("private_stain_report_sha256")==sha(stain_report_path)
        and independent.get("private_stain_arrays_sha256")==sha(stain_npz_path)
        and independent.get("subject_id")==stain.get("subject_id")==job["subject_id"]
        and independent.get("original_source_sha256")==job["source"]["sha256"]
        and stain.get("status")==
            "ORIGINAL_SOURCE_REACTIVE_STAINING_MECHANICAL_CANDIDATES__HELDOUT_EDITOR_COURT_PENDING"
        and stain.get("fit_source_authored_poses")==[0,1]
        and stain.get("heldout_source_authored_pose")==POSE
        and stain.get("training_authorized") is False
        and stain.get("source_graphics_owner_authorized") is False
        and independent.get("training_authorized") is False
        and independent.get("actual_editor_action_independence_proven") is False,
        "V18B_ORIGINAL_SOURCE_BLIND_STAIN_SHA_AND_AUTHORITY")
    source=verify(job["source"],"V18B_ORIGINAL_SOURCE_IMMUTABLE_SHA")
    restpath=verify(job["rest_geometry"],"V18B_ORIGINAL_REST_WORLD_SHA")
    teacherpath=verify(job["native_teacher"],"V18B_ORIGINAL_NATIVE_POSES_SHA")
    receiptpath=verify(job["native_teacher_receipt"],"V18B_ORIGINAL_NATIVE_RECEIPT_SHA")
    camerapath=verify(job["camera"],"V18B_ORIGINAL_CALIBRATED_CAMERAS_SHA")
    candidatepath=verify(job["candidate"],"V18B_ORIGINAL_SOURCE_APPEARANCE_SHA")
    v11path=verify(job["v11_private_plan"],"V18B_ORIGINAL_HELDOUT_FACE_SELECTION_SHA")
    precommitted=json.loads(v11path.read_text())
    receipt=json.loads(receiptpath.read_text())
    candidate=json.loads(candidatepath.read_text())
    cameras=json.loads(camerapath.read_text())
    require(precommitted["subject_id"]==job["subject_id"]
        and precommitted["source_sha256"]==job["source"]["sha256"]
        and precommitted["new_source_face_intervention_query_count"]==12
        and precommitted["real_rgba_consequences_measured_for_heldout_queries"] is False
        and all(row["same_drawable"]=="UNKNOWN"
                and row["different_drawable"]=="UNKNOWN"
                for row in precommitted["query_targets_source_teacher_only"])
        and cameras["schema"]=="RealSaS.CameraProjectionBundle.v1"
        and {int(v["view_index"]) for v in cameras["cameras"]}==set(range(8)),
        "V18B_ORIGINAL_V11_EDITOR_TARGETS_AND_CAMERAS")
    camera={int(c["view_index"]):c for c in cameras["cameras"]}

    with np.load(restpath,allow_pickle=False) as d:
        faces=np.asarray(d["faces"],np.int64)
        rest=np.asarray(d["rest_vertices_source_world"],np.float64)
        owner=np.asarray(d["face_owner_privileged"],np.int32)
    with np.load(teacherpath,allow_pickle=False) as d:
        expected_faces=np.asarray(d["faces"],np.int64)
        posed=np.asarray(d["posed_vertices_source_world"],np.float64)
    with np.load(stain_npz_path,allow_pickle=False) as d:
        require(set(d.files)=={
            "faces","stain_channels","graph_edges",
            "selected_source_control_axes_teacher_only"
        }|{"labels_"+str(t).replace(".","p") for t in SOURCE_THRESHOLDS},
            "V18B_STAIN_ORIGINAL_FACE_ARRAY_MEMBERS")
        require(np.array_equal(d["faces"],faces),
            "V18B_ORIGINAL_GRAPHICS_FACE_ADDRESS_CHANGED")
        labels_by_t={
            t:np.asarray(d["labels_"+str(t).replace(".","p")],np.int32)
            for t in SOURCE_THRESHOLDS}
    with np.load(v14_original_pixels,allow_pickle=False) as d:
        require(d.files==["original_native_actuator_rgba"],
                "V18B_REAL_ORIGINAL_V14_RGBA_MEMBERS")
        v14=np.asarray(d["original_native_actuator_rgba"],np.float32)
    from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan
    selection=json.loads(Path(job["selected_source_controls"]["path"]).read_text())
    v14_arms,_=arms_for_plan(selection)
    origindex=v14_arms.index("ORIGINAL_BASE")
    require(v14.shape==(3,4,len(v14_arms),192,192,4)
        and np.array_equal(faces,expected_faces)
        and posed.shape[1:]==rest.shape
        and all(len(x)==len(faces) for x in labels_by_t.values())
        and all(len(set(labels_by_t[t]))==stain["candidate_policies"][i][
            "predicted_connected_source_graphics_layer_candidate_count"]
            for i,t in enumerate(SOURCE_THRESHOLDS)),
        "V18B_EXACT_NATIVE_SOURCE_POSE_AND_UNSEEN_STAIN_EDIT_FACE_COUNTS")
    originals=receipt["native_motion_inventory_expected"]
    sampled=receipt["sampled_frames"][job["pose_indices"][POSE]]
    targets=[
        precommitted["query_targets_source_teacher_only"][i]["source_face_teacher_only"]
        for i in TARGET_INDICES
    ]
    require(len(set(targets))==len(targets)
        and all(type(i) is int and 0<=i<len(faces) for i in targets),
        "V18B_REAL_V11_HELDOUT_SOURCE_FACE_TARGETS_NONUNIQUE")
    names=ops()
    with tempfile.TemporaryDirectory(prefix="c370_v18b_stain_real_") as td:
        scratch=Path(td)
        base._import_animated(source)
        resolved={k:transport._action_by_requested_take(k) for k in originals}
        bindings=actions.capture_imported_action_bindings(resolved)
        meshes=base._mesh_objects()
        base._neutralize_animation()
        snapshot=v4._extract_snapshot(meshes,freeze_triangulation=True)
        extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
        tol=1e-6*max(1.,extent)
        require(np.array_equal(np.asarray(snapshot["faces"]),faces)
            and np.array_equal(np.asarray(snapshot["face_owner"]),owner)
            and np.max(np.abs(np.asarray(snapshot["vertices"],np.float64)-rest))<=tol,
            "V18B_ORIGINAL_SOURCE_REST_MESH_TOPOLOGY_DRIFT")
        mat=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(mat,list(candidate.get("material_transform") or []))
        uv,uv_valid,_,_=_appearance_and_uv(meshes,snapshot)
        carrier=dict(rest_vertices_source_world=rest,faces=faces,
            face_owner_privileged=owner,
            face_material_privileged=np.asarray(snapshot["face_material"],np.int32),
            triangle_uv=uv,triangle_uv_valid=uv_valid)
        soup,face_map=probe._build_triangle_soup(
            carrier,meshes,probe._transparent_material())
        require(len(face_map)==len(faces)
            and all(np.max(np.abs(np.asarray(obj.matrix_world)-np.eye(4)))<=1e-8
                    for obj in soup),
            "V18B_SOURCE_TRIANGLE_SOUP_SHARED_WORLD_IDENTITY")
        probe._configure_measurement(32)
        cam=probe._make_camera()
        actions.activate_action(resolved[sampled["action"]],bindings)
        bpy.context.scene.frame_set(int(sampled["frame"]))
        bpy.context.view_layer.update()
        posed_actual=v4._extract_snapshot(meshes,freeze_triangulation=False)
        original_vertices=np.asarray(posed_actual["vertices"],np.float64)
        require(posed_actual["polygon_signature"]==snapshot["polygon_signature"]
            and np.max(np.abs(original_vertices-posed[job["pose_indices"][POSE]]))<=tol,
            "V18B_REAL_ORIGINAL_AUTHORED_HELDOUT_POSE_NOT_REPLAYED")
        _set_animated_soup(face_map,faces,original_vertices)

        base_meshes={obj.name:obj.data for obj in soup}
        require(len(base_meshes)==len(soup),"V18E_SOURCE_SOUP_MESH_NAMES_UNIQUE")
        face_ids=np.arange(len(faces),dtype=np.int32)
        all_faces=set(range(len(faces)))
        render_cases=[]
        face_count_checks=[]
        for vi,view in enumerate(VIEWS):
            _camera_full(cam,camera[view],192)
            # True source with original rig/skin, never synthetic geometry.
            for ob in meshes:ob.hide_render=False
            for ob in soup:ob.hide_render=True
            bpy.context.view_layer.update()
            source_frame=_frame(scratch/f"source_v{view}.exr",192)
            parent_frame=v14[POSE,vi,origindex]
            p99=float(np.percentile(
                np.max(np.abs(source_frame-parent_frame),axis=-1),99))
            require(p99<=.025,
                "V18B_V14_AND_NEW_ORIGINAL_SOURCE_RGBA_NOT_EQUAL")
            for ob in meshes:ob.hide_render=True
            for ob in soup:ob.hide_render=False
            current={"SOURCE_ORIGINAL":source_frame}
            counts={"SOURCE_ORIGINAL":len(faces)}
            for arm in names[1:]:
                if arm in ("SOUP_FULL","SOUP_NOOP"):
                    labels=face_ids;excluded=set()
                elif arm=="SOUP_EMPTY":
                    labels=face_ids;excluded=all_faces
                elif arm.startswith("SINGLE_FACE_"):
                    ix=int(arm.split("_T")[1]);fidx=targets[ix]
                    labels=face_ids
                    excluded=({fidx} if "REMOVE" in arm else all_faces-{fidx})
                else:
                    ix=int(arm.split("_T")[1])
                    tnext=arm.split("_T")[0].split("_")[2]
                    t=float(tnext.replace("p","."))
                    require(t in labels_by_t,"V18B_GROUP_THRESHOLD_NOT_PRECOMMITTED")
                    labels=labels_by_t[t]
                    target_group=int(labels[targets[ix]])
                    excluded=({target_group} if "REMOVE" in arm else
                              set(range(int(labels.max(initial=-1)+1)))-{target_group})
                counts[arm]=physical_group_mode(
                    soup,face_map,base_meshes,labels,excluded)
                current[arm]=_frame(scratch/f"{arm}_v{view}.exr",192)
            require(set(current)==set(names),
                "V18B_STAIN_ORIGINAL_SOURCE_ALL_EDIT_ARMS_MISSING")
            render_cases.append(np.stack([current[n] for n in names],axis=0))
            face_count_checks.append([counts[n] for n in names])
        out=np.stack(render_cases,axis=0).astype(np.float16)
        require(out.shape==(4,len(names),192,192,4) and np.isfinite(out).all()
            and np.min(out[...,3])>=-1e-5 and np.max(out[...,3])<=1.00001,
            "V18B_SOURCE_EDITOR_RGBA_ARRAY_FINITE")
    result_path=Path(output_path).resolve()
    result_path.parent.mkdir(parents=True,exist_ok=True)
    npz=result_path.with_suffix(".npz")
    np.savez_compressed(npz,original_stain_editor_rgba=out)
    result={
        "schema":SCHEMA,
        "status":"V18E_REAL_ORIGINAL_PHYSICAL_FACE_EDITS_RENDERED__INDEPENDENT_COURT_REQUIRED",
        "subject_id":job["subject_id"],
        "original_source_sha256":job["source"]["sha256"],
        "source_original_v14_job_sha256":sha(job_path),
        "original_v14_RGBA_sha256":sha(v14_original_pixels),
        "source_reactive_stain_report_sha256":sha(stain_report_path),
        "source_reactive_stain_npz_sha256":sha(stain_npz_path),
        "independent_source_stain_court_sha256":sha(stain_court_path),
        "original_v11_heldout_target_plan_sha256":sha(v11path),
        "original_source_teacher_manifest_sha256":sha(receiptpath),
        "original_source_teacher_vertices_sha256":sha(teacherpath),
        "original_source_camera_sha256":sha(camerapath),
        "heldout_original_authored_pose_ordinal":POSE,
        "heldout_source_camera_views":list(VIEWS),
        "source_face_probe_target_ordinals_teacher_only":targets,
        "source_edit_arms":names,
        "physical_face_visibility_operator":PHYSICAL_VISIBILITY,
        "actual_render_arms_per_camera":len(names),
        "actual_visible_original_source_triangles_per_arm_view":face_count_checks,
        "precommitted_v11_target_query_indices":list(TARGET_INDICES),
        "frozen_tested_stain_similarity_thresholds":list(THRESHOLDS),
        "original_full_stain_similarity_thresholds":list(SOURCE_THRESHOLDS),
        "original_source_heldout_pose_vertices_verified":True,
        "source_rest_triangle_UV_material_authority_reused":True,
        "source_vs_original_v14_RGBA_reference_source_parity_verified":True,
        "new_source_actual_edited_rgba_npz_sha256":sha(npz),
        "artist_original_source_part_names_as_labels":False,
        "source_face_single_triangle_is_artist_gold":False,
        "source_editor_edit_independence_proven":False,
        "full_original_caa_draw_order_qualified":False,
        "physical_removal_not_author_drawable_layer":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }
    result_path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],"subject_id":result["subject_id"],
        "source_edit_arms_per_view":len(names),
        "original_heldout_camera_views":len(VIEWS),
        "private_RGBA_sha256":result["new_source_actual_edited_rgba_npz_sha256"],
        "training_authorized":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("job","stain_report","stain_npz","stain_court",
                "v14_original_pixels","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
    a=p.parse_args(argv)
    run(a.job,a.stain_report,a.stain_npz,a.stain_court,
        a.v14_original_pixels,a.output)
