#!/usr/bin/env python3
"""V9: real original authored-motion source vs independently posed face-soup RGBA.

The soup is built ONLY from source-rest faces, UV and material slots. Per pose
vertices are indexed into the exact SHA-sealed teacher array, never inferred
from source mesh, bone or semantic names. All images stay private. Both
original scene reimport and actual evaluated vertices MUST equal the teacher.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import bpy

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v2 as transport
from tools.iris_vnext import blender_authored_teacher_export_v4 as v4
from tools.iris_vnext import blender_authored_teacher_export_v4_actions as actions
from tools.truth_foundry import blender_source_appearance_policy_replay_v1 as replay
from tools.truth_foundry import blender_source_ray_appearance_measure_v1 as image_probe
from tools.truth_foundry.blender_teacher_scene_v1 import _appearance_and_uv
from tools.truth_foundry.blender_corpus370_multi_partition_edit_v7 import (
    _camera_full,_frame,_group_mode,sha,verify,require
)
from tools.corpus.corpus370_partition_edit_plan_v7 import build_plan
from tools.corpus.corpus370_adaptive_edit_probe_plan_v11 import SCHEMA as V11_SCHEMA

SCHEMA="RealSaS.Corpus370HeldoutNativeAnimatedSourceRender.v12"
JOB_SCHEMA="RealSaS.Corpus370HeldoutNativeAnimatedSourceJob.v12"


def _set_animated_soup(face_map,faces,vertices):
    """Source world vertices map into duplicated, identity-transform soup."""
    meshes=set()
    for face,(mesh,local,_,_) in face_map.items():
        ids=faces[face]
        for k,idx in enumerate(ids):
            mesh.vertices[3*local+k].co=tuple(float(x) for x in vertices[idx])
        meshes.add(mesh)
    for mesh in meshes:
        mesh.update()
    bpy.context.view_layer.update()


def run(job_path,output_path):
    job=json.loads(Path(job_path).read_text())
    require(job.get("schema")==JOB_SCHEMA
            and job.get("teacher_only") is True
            and job.get("training_authorized") is False
            and job.get("graphics_owner_authorized") is False
            and job.get("native_full_caa_authorized") is False
            and job.get("source_bone_names_as_targets") is False
            and job.get("view_indices")==[0,2,4,6]
            and job.get("raster_resolution")==192
            and job.get("cycles_samples")==32
            and job.get("reference_policy")=="DIRECT_PAIRWISE_HELDOUT_SOURCE_GEOMETRIC_EDIT_NO_ARTIST_TRUTH"
            and isinstance(job.get("pose_indices"),list)
            and len(job["pose_indices"])==3
            and len(set(job["pose_indices"]))==3
            and all(type(x) is int and x>=0 for x in job["pose_indices"]),
            "V12_ANIMATED_RENDER_BOUNDED_AUTHORITY_AND_FRAME_POLICY")
    source=verify(job["source"],"V12_ORIGINAL_SOURCE_SHA")
    verify(job["admission"],"V12_ORIGINAL_V5_ADMISSION_SHA")
    candidate_path=verify(job["candidate"],"V12_ORIGINAL_V5_CANDIDATE_SHA")
    camera_path=verify(job["camera"],"V12_ORIGINAL_V5_CAMERAS_SHA")
    rest_path=verify(job["rest_geometry"],"V12_ORIGINAL_SOURCE_REST_SHA")
    group_path=verify(job["layer_proposal"],"V12_ORIGINAL_LAYER_PROPOSAL_SHA")
    pair_path=verify(job["source_graphics_pairs"],"V12_ORIGINAL_SOURCE_PAIR_SHA")
    teacher_path=verify(job["native_teacher"],"V12_NATIVE_TEACHER_ARRAY_SHA")
    teacher_receipt_path=verify(job["native_teacher_receipt"],"V12_NATIVE_TEACHER_MANIFEST_SHA")
    teacher_court_path=verify(job["native_teacher_court"],"V12_NATIVE_TEACHER_INDEPENDENT_COURT_SHA")
    candidate=json.loads(candidate_path.read_text())
    camera_bundle=json.loads(camera_path.read_text())
    layer=json.loads(group_path.read_text())
    pairs=json.loads(pair_path.read_text())
    receipt=json.loads(teacher_receipt_path.read_text())
    court=json.loads(teacher_court_path.read_text())
    original_plan=build_plan(layer,pairs)
    v11_path=verify(job["v11_private_plan"],"V12_ORIGINAL_HELDOUT_PLAN_SHA")
    heldout=json.loads(v11_path.read_text())
    require(heldout.get("schema")==V11_SCHEMA
            and heldout.get("subject_id")==job["subject_id"]
            and heldout.get("source_sha256")==layer["source_sha256"]
            and heldout.get("source_rest_geometry_sha256")==job["rest_geometry"]["sha256"]
            and heldout.get("new_source_face_intervention_query_count")==12
            and len(heldout.get("query_targets_source_teacher_only",[]))==12
            and heldout.get("real_rgba_consequences_measured_for_heldout_queries") is False
            and heldout.get("graphics_owner_authorized") is False
            and heldout.get("training_authorized") is False
            and heldout.get("parents",{}).get("original_geometry_proposal_sha256")==job["layer_proposal"]["sha256"]
            and heldout.get("parents",{}).get("original_source_hit_pair_sha256")==job["source_graphics_pairs"]["sha256"]
            and heldout.get("parents",{}).get("original_native_v9_manifest_sha256")==job["prior_v9_manifest_sha256"]
            and heldout.get("parents",{}).get("independent_native_v10_court_sha256")==job["prior_v10_court_sha256"]
            and heldout.get("parents",{}).get("original_rest_npz_sha256")==job["rest_geometry"]["sha256"],
            "V12_IMMUTABLE_ORIGINAL_GEOMETRIC_QUERY_PRECOMMIT_BINDINGS")
    require(job["subject_id"]==receipt["subject_id"]==court["subject_id"]
            and job["source"]["sha256"]==receipt["source"]["sha256"]==
                court["source_sha256"]==layer["source_sha256"]==
                pairs["parents"]["source_sha256"]==candidate["source_sha256"]
            and receipt["output"]["sha256"]==job["native_teacher"]["sha256"]==
                court["teacher_arrays"]["sha256"]
            and court["teacher_manifest"]["sha256"]==job["native_teacher_receipt"]["sha256"]
            and court["family_authority_minted"] is False
            and court["training_authorized"] is False
            and receipt["teacher_rig_skin_motion_allowed_as_inference_input"] is False
            and job["expected_partition_sizes"]==original_plan["source_geometry_candidate_sizes"]
            and job["expected_proxy_task_count"]==original_plan["task_count"]
            and camera_bundle["schema"]=="RealSaS.CameraProjectionBundle.v1",
            "V12_SEALED_SOURCE_NATIVE_GROUP_TEACHER_BINDINGS")
    with np.load(rest_path,allow_pickle=False) as rest:
        rv=np.asarray(rest["rest_vertices_source_world"],np.float64)
        faces=np.asarray(rest["faces"],np.int64)
        face_owner=np.asarray(rest["face_owner_privileged"],np.int32)
    with np.load(teacher_path,allow_pickle=False) as teacher:
        native_rest=np.asarray(teacher["rest_vertices_source_world"],np.float64)
        native_faces=np.asarray(teacher["faces"],np.int64)
        native_owner=np.asarray(teacher["face_owner"],np.int32)
        native_posed=np.asarray(teacher["posed_vertices_source_world"],np.float64)
    extent=float(np.linalg.norm(np.ptp(rv,axis=0)))
    tol=1e-6*max(1.,extent)
    require(np.array_equal(faces,native_faces)
            and np.array_equal(face_owner,native_owner)
            and rv.shape==native_rest.shape
            and np.max(np.abs(rv-native_rest))<=tol
            and native_posed.shape[1:]==rv.shape
            and len(receipt["sampled_frames"])==len(native_posed)
            and len(native_posed)>=max(job["pose_indices"])+1,
            "V12_INDEPENDENT_REAL_NATIVE_REST_TO_POSE_FACE_CORRESPONDENCE")
    # The compiler hypotheses are frozen before native pose/RGBA consequences.
    group_labels={
        c["slot_count"]:np.asarray(c["face_slot_teacher_only"],np.int32)
        for c in layer["candidates"]
        if c["slot_count"] in original_plan["source_geometry_candidate_sizes"]
    }
    sizes=original_plan["source_geometry_candidate_sizes"]
    tasks=heldout["query_targets_source_teacher_only"]
    old_finest=group_labels[sizes[-1]]
    prior_leaves={int(old_finest[t["source_face_teacher_only"]])
                  for t in original_plan["tasks"]}
    new_leaves=set()
    operations={
        "SOUP_FULL":(group_labels[8],set()),
        "SOUP_NOOP":(group_labels[8],set()),
        "REMOVE_ALL_GROUPS":(group_labels[8],set(range(8))),
    }
    arms=["SOURCE_ORIGINAL"]+list(operations)
    for index,task in enumerate(tasks):
        face=task["source_face_teacher_only"]
        leaf=int(old_finest[face])
        require(type(index) is int and task["test_index"]==index
                and type(face) is int and 0<=face<len(faces)
                and task["finest_group_id_teacher_only"]==leaf
                and leaf not in prior_leaves and leaf not in new_leaves
                and task["same_drawable"]=="UNKNOWN"
                and task["different_drawable"]=="UNKNOWN"
                and task["training_authorized"] is False,
                "V12_HELDOUT_QUERY_LEAF_INVALID_OR_PREVIOUSLY_EXPOSED")
        new_leaves.add(leaf)
        for size in sizes:
            slot=int(group_labels[size][face])
            require(task["source_groups_per_candidate_size_teacher_only"][str(size)]==slot,
                    "V12_UNSEALED_SOURCE_FACE_TO_GROUP_SLOT")
            rem=f"GROUP_REMOVE_K{size:03d}_T{index:02d}"
            iso=f"GROUP_ISOLATE_K{size:03d}_T{index:02d}"
            operations[rem]=(group_labels[size],{slot})
            operations[iso]=(group_labels[size],set(range(size))-{slot})
            arms.extend((rem,iso))
    require(len(arms)==4+2*len(sizes)*len(tasks)
            and len(arms)==len(set(arms)),
            "V12_ALL_12_PRIVATE_SOURCE_HIERARCHY_EDIT_ARMS")
    cameras={int(x["view_index"]):x for x in camera_bundle["cameras"]}
    require(set(cameras)==set(range(8)),"V12_ORIGINAL_EIGHT_CAMERA_CONTRACT")
    rendered=[]
    validated_pose_details=[]
    out=Path(output_path).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c370_v12_heldout_") as t:
        scratch=Path(t)
        base._import_animated(source)
        action_names=receipt["native_motion_inventory_expected"]
        resolved={
            name:transport._action_by_requested_take(name)
            for name in action_names
        }
        original_bindings=actions.capture_imported_action_bindings(resolved)
        meshes=base._mesh_objects()
        base._neutralize_animation()
        snapshot=v4._extract_snapshot(meshes,freeze_triangulation=True)
        actual=np.asarray(snapshot["vertices"],np.float64)
        require(np.array_equal(np.asarray(snapshot["faces"]),faces)
                and np.array_equal(np.asarray(snapshot["face_owner"]),face_owner)
                and actual.shape==rv.shape
                and np.max(np.abs(actual-rv))<=tol,
                "V12_ORIGINAL_SOURCE_NATIVE_ANIMATED_IMPORT_REST_DRIFT")
        material=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(material,list(candidate.get("material_transform") or []))
        uv,uv_valid,_,_=_appearance_and_uv(meshes,snapshot)
        arrays={
            "rest_vertices_source_world":rv,
            "faces":faces,
            "face_owner_privileged":face_owner,
            "face_material_privileged":np.asarray(snapshot["face_material"],np.int32),
            "triangle_uv":uv,
            "triangle_uv_valid":uv_valid,
        }
        clear=image_probe._transparent_material()
        soup,face_map=image_probe._build_triangle_soup(arrays,meshes,clear)
        require(len(face_map)==len(faces),"V12_ANIMATED_SOURCE_FACE_MATERIAL_MAP")
        for obj in soup:
            require(np.max(np.abs(np.asarray(obj.matrix_world)-np.eye(4)))<1e-8,
                    "V12_POSED_SOURCE_WORLD_TRIANGLE_SOUP_LOCAL_WORLD_MISMATCH")
        image_probe._configure_measurement(job["cycles_samples"])
        cam=image_probe._make_camera()
        for pidx in job["pose_indices"]:
            frame=receipt["sampled_frames"][pidx]
            require(frame["action"] in resolved and type(frame["frame"]) is int,
                    "V12_SELECTED_AUTHORED_ACTION_FRAME_BINDING")
            # Same importer-owner and action-slot replay as the original
            # v4 source teacher exporter; no fabricated bone trajectories.
            control=actions.activate_action(
                resolved[frame["action"]],original_bindings)
            bpy.context.scene.frame_set(int(frame["frame"]))
            bpy.context.view_layer.update()
            evaluated=v4._extract_snapshot(meshes,freeze_triangulation=False)
            current=np.asarray(evaluated["vertices"],np.float64)
            max_pose_error=float(np.max(np.abs(current-native_posed[pidx])))
            # The V4 authored-teacher exporter deliberately freezes triangle
            # addresses only in the rest snapshot. Subsequent evaluated poses
            # are sampled WITHOUT triangulation, and prove source topology by
            # the exact per-owner polygon cycles/material signature. Requiring
            # a non-existent animated triangle list is a false failure.
            require(evaluated["polygon_signature"]==snapshot["polygon_signature"]
                    and current.shape==rv.shape
                    and max_pose_error<=tol,
                    "V12_REPLAYED_ORIGINAL_SOURCE_POSE_NOT_TEACHER_EQUIVALENT:"
                    +str({"pose_index":pidx,
                          "polygon_signature_equal":evaluated["polygon_signature"]==snapshot["polygon_signature"],
                          "vertex_shape_equal":current.shape==rv.shape,
                          "max_pose_error":max_pose_error,"tolerance":tol}))
            _set_animated_soup(face_map,faces,native_posed[pidx])
            pose_views=[]
            for view in job["view_indices"]:
                _camera_full(cam,cameras[view],job["raster_resolution"])
                for obj in meshes:obj.hide_render=False
                for obj in soup:obj.hide_render=True
                bpy.context.view_layer.update()
                raster=[_frame(scratch/f"original_p{pidx}_v{view}.exr",192)]
                for obj in meshes:obj.hide_render=True
                for obj in soup:obj.hide_render=False
                for arm in arms[1:]:
                    labels,exclude=operations[arm]
                    _group_mode(face_map,labels,excluded=exclude)
                    raster.append(_frame(scratch/f"{arm}_p{pidx}_v{view}.exr",192))
                pose_views.append(np.stack(raster,axis=0))
            rendered.append(np.stack(pose_views,axis=0))
            validated_pose_details.append({
                "pose_index":pidx,
                "source_action_ordinal_teacher_only":action_names.index(frame["action"]),
                "source_frame_native_teacher_only":int(frame["frame"]),
                "max_original_evaluated_vs_sealed_teacher_coordinate_error":max_pose_error,
                "action_carrier_kind_teacher_only":control["carrier_kind"],
                "source_native_exact_face_order":True,
            })
            base._neutralize_animation()
        pixels=np.stack(rendered,axis=0)
        require(pixels.shape==(len(job["pose_indices"]),4,len(arms),192,192,4)
                and np.isfinite(pixels).all(),"V12_RENDER_ARRAY_SHAPE")
        npz=out.with_suffix(".npz")
        np.savez_compressed(npz,animated_original_and_group_edits_rgba=pixels)
        result={
            "schema":SCHEMA,
            "status":"REAL_V11_HELDOUT_SOURCE_NATIVE_POSE_RGBA_RENDER__INDEPENDENT_V12_COURT_REQUIRED",
            "subject_id":job["subject_id"],
            "source_sha256":job["source"]["sha256"],
            "source_rest_exact_geometry_verified":True,
            "source_selected_pose_exact_teacher_replay_verified":True,
            "source_cameras_exact_sealed_rest_bundle":True,
            "native_teacher_arrays_sha256":sha(teacher_path),
            "native_teacher_manifest_sha256":sha(teacher_receipt_path),
            "source_group_proposal_sha256":sha(group_path),
            "source_pair_sha256":sha(pair_path),
            "private_job_sha256":sha(job_path),
            "v11_heldout_plan_sha256":sha(v11_path),
            "prior_v9_manifest_sha256":job["prior_v9_manifest_sha256"],
            "prior_v10_court_sha256":job["prior_v10_court_sha256"],
            "v11_public_seal_sha256":job["v11_public_seal_sha256"],
            "v11_original_heldout_targets_verified":True,
            "pose_indices":job["pose_indices"],
            "selected_pose_details_teacher_only":validated_pose_details,
            "view_indices":job["view_indices"],
            "arms":arms,
            "candidate_sizes":sizes,
            "heldout_query_plan":heldout,
            "previous_tested_source_faces_teacher_only":[t["source_face_teacher_only"] for t in original_plan["tasks"]],
            "raster_resolution":192,
            "cycles_samples":32,
            "reference_policy":"DIRECT_PAIRWISE_HELDOUT_SOURCE_GEOMETRIC_EDIT_NO_ARTIST_TRUTH",
            "array_path":str(npz),
            "array_sha256":sha(npz),
            "teacher_only":True,
            "heldout_edit_responses_measured_but_artist_truth_unknown":True,
            "source_owner_semantics_minted":False,
            "source_bone_names_as_targets":False,
            "animated_render_caa_authorized":False,
            "graphics_owner_authorized":False,
            "materialization_authorized":False,
            "training_authorized":False,
        }
        out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        print(json.dumps({
            "status":result["status"],
            "real_pose_count":len(job["pose_indices"]),
            "views_per_pose":len(job["view_indices"]),
            "arms_per_view":len(arms),
            "training_authorized":False,
        }),flush=True)
        return result


def main():
    parser=__import__("argparse").ArgumentParser()
    parser.add_argument("--job",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    run(args.job,args.output)


if __name__=="__main__":
    main()
