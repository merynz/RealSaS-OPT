#!/usr/bin/env python3
"""V16 real original native *paired* control interventions.

This is an independent original Blender source experiment, not an extrapolation
from V14's single-control images, not a source artist-layer name and not a K.
For each original authored pose and camera, simultaneously actuate the two
precommitted anonymous V14 original source-native controls at +/-5 degrees,
render ORIGINAL + UV/material preserving SOURCE-FACE SOUP + isolated union
source-face support. The real Blender animation action is detached only during
SOURCE raster after exact local source matrices/poses are frozen. No native
semantic names enter the model. The source is independently SHA-pinned.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

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
from tools.truth_foundry import blender_corpus370_native_mechanical_raster_join_v14 as v14
from tools.corpus.corpus370_native_raster_join_plan_v14 import (
    arms_for_plan,select,verify_locked_source_selection
)

SCHEMA="RealSaS.Corpus370OriginalDualNativeControlCausalInteractionRender.v16"
PAIR_SIGN=((1.,1.),(1.,-1.),(-1.,1.),(-1.,-1.))
AXES=v14.AXES


def _json(path):
    return json.loads(Path(path).read_text())


def _difference(a,b):
    return float(np.percentile(
        np.max(np.abs(np.asarray(a,np.float32)-np.asarray(b,np.float32)),axis=-1),99
    ))


def _joint_arms():
    return ["ORIGINAL_BASE","ISOLATED_AB_BASE"]+[
        mode+"_"+a+"_"+b
        for a,b in (("PLUS","PLUS"),("PLUS","MINUS"),("MINUS","PLUS"),("MINUS","MINUS"))
        for mode in ("SOURCE_AB","SOUP_AB","ISOLATED_AB")
    ]


def render(job_path, parent_path, original_npz, output_path):
    job=_json(job_path)
    parent=_json(parent_path)
    require(job.get("schema")==v14.JOB
        and job.get("teacher_only") is True
        and job.get("source_bone_names_as_targets") is False
        and job.get("artist_drawable_identity_inferred") is False
        and job.get("training_authorized") is False
        and job.get("graphics_owner_authorized") is False
        and parent.get("schema")==v14.SCHEMA
        and parent.get("status")==
            "V14_REAL_SAME_CAUSAL_SOURCE_NATIVE_CONTROL_VERTEX_AND_RGBA__INDEPENDENT_COURT_REQUIRED"
        and parent.get("source_sha256")==job["source"]["sha256"]
        and parent.get("private_job_sha256")==sha(job_path)
        and parent.get("original_perturbed_pose_rgba_npz_sha256")==sha(original_npz)
        and parent.get("source_native_control_raster_animated_action_frozen_then_restored") is True
        and parent.get("source_native_perturbed_vertices_verified_before_and_after_render") is True
        and job["pose_indices"]==parent["rendered_pose_indices"]
        and job["view_indices"]==parent["rendered_source_camera_indices"]==[0,2,4,6]
        and parent["raster_resolution"]==job["raster_resolution"]==192
        and parent["cycles_samples"]==job["cycles_samples"]==32
        and job["native_actuation_degrees"]==5.
        and parent["artist_same_drawable_pairs"]==
            parent["artist_different_drawable_pairs"]==0
        and parent["training_authorized"] is False,
        "V16_ORIGINAL_SHA_PINNED_V14_NATIVE_ACTION_JOB_REQUIRED")

    source=verify(job["source"],"V16_ORIGINAL_SOURCE_BYTES_SHA")
    for field in ("admission","native_teacher_court","v11_private_plan"):
        verify(job[field],"V16_PRIVATE_"+field.upper()+"_SHA")
    candidate=_json(verify(job["candidate"],"V16_ORIGINAL_RENDER_CANDIDATE_SHA"))
    cameras=_json(verify(job["camera"],"V16_ORIGINAL_CAMERA_BUNDLE_SHA"))
    restpath=verify(job["rest_geometry"],"V16_ORIGINAL_SOURCE_REST_GEOMETRY_SHA")
    teacherpath=verify(job["native_teacher"],"V16_ORIGINAL_AUTHORED_TEACHER_SHA")
    receipt=_json(verify(job["native_teacher_receipt"],"V16_SOURCE_NATIVE_TEACHER_RECEIPT_SHA"))
    prior=_json(verify(job["v13b_manifest"],"V16_ORIGINAL_PRECOMMITTED_V13B_SHA"))
    intervention_path=verify(job["v13b_array"],"V16_REAL_V13B_SOURCE_VERTICES_SHA")
    selection=_json(verify(job["selected_source_controls"],"V16_V14_BLIND_SELECTED_CONTROLS_SHA"))
    require(selection==parent["selected_source_controls"]
            and len(selection["controls"])==2
            and selection["source_bone_or_object_identity_as_artist_target"] is False
            and selection["raster_pixels_seen_during_control_selection"] is False,
            "V16_NO_POSTHOC_SOURCE_CONTROL_OR_ARTIST_SELECTION")

    with np.load(original_npz,allow_pickle=False) as data:
        original_arms,_=arms_for_plan(selection)
        require(data.files==["original_native_actuator_rgba"]
                and original_arms==parent["rendered_arms"],
                "V16_V14_ORIGINAL_FROZEN_RENDER_ARMS")
        original_images=np.asarray(data["original_native_actuator_rgba"],np.float32)
    require(original_images.shape==(3,4,len(original_arms),192,192,4),
            "V16_PARENT_IMAGE_SHAPE")
    oldix={a:i for i,a in enumerate(original_arms)}

    with np.load(restpath,allow_pickle=False) as data:
        rest=np.asarray(data["rest_vertices_source_world"],np.float64)
        faces=np.asarray(data["faces"],np.int64)
        owner=np.asarray(data["face_owner_privileged"],np.int32)
    with np.load(teacherpath,allow_pickle=False) as data:
        native_faces=np.asarray(data["faces"],np.int64)
        native_rest=np.asarray(data["rest_vertices_source_world"],np.float64)
        native_owner=np.asarray(data["face_owner"],np.int32)
        poses=np.asarray(data["posed_vertices_source_world"],np.float64)
    with np.load(intervention_path,allow_pickle=False) as data:
        native_baselines=np.asarray(data["baseline_vertices"],np.float64)
        native_controls=np.asarray(data["native_control_perturbed_vertices"],np.float64)
    extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
    tol=1e-6*max(1.,extent)
    require(np.array_equal(faces,native_faces)
            and np.array_equal(owner,native_owner)
            and np.max(np.abs(rest-native_rest))<=tol
            and native_baselines.shape==(3,len(rest),3)
            and native_controls.shape==(3,prior["source_controls_measured_count"],3,2,len(rest),3)
            and np.max(np.abs(native_baselines-poses[job["pose_indices"]]))<=tol,
            "V16_ORIGINAL_REAL_V13B_SOURCE_VERTEX_FAMILY")
    expected_selection=select(rest,native_baselines,native_controls,
        job["pose_indices"],prior["source_controls_measured_count"],job["pose_indices"])
    require(verify_locked_source_selection(expected_selection,selection),
        "V16_V14_CONTROL_SELECTION_GEOMETRIC_PRECOMMIT")
    camera={int(x["view_index"]):x for x in cameras["cameras"]}
    require(set(camera)==set(range(8)),
        "V16_ONLY_EXACT_ORIGINAL_EIGHT_CAMERAS")

    with tempfile.TemporaryDirectory(prefix="corpus370_native_causal_pair_v16_") as td:
        temp=Path(td)
        base._import_animated(source)
        names=receipt["native_motion_inventory_expected"]
        resolved={name:transport._action_by_requested_take(name) for name in names}
        bindings=actions.capture_imported_action_bindings(resolved)
        meshes=base._mesh_objects()
        base._neutralize_animation()
        snapshot=v4._extract_snapshot(meshes,freeze_triangulation=True)
        require(np.array_equal(np.asarray(snapshot["faces"]),faces)
                and np.array_equal(np.asarray(snapshot["face_owner"]),owner)
                and np.max(np.abs(np.asarray(snapshot["vertices"],np.float64)-rest))<=tol,
                "V16_REIMPORTED_REAL_SOURCE_REST_TOPOLOGY")
        mat=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(mat,list(candidate.get("material_transform") or []))
        uv,uv_valid,_,_=_appearance_and_uv(meshes,snapshot)
        canonical={
            "rest_vertices_source_world":rest,
            "faces":faces,
            "face_owner_privileged":owner,
            "face_material_privileged":np.asarray(snapshot["face_material"],np.int32),
            "triangle_uv":uv,
            "triangle_uv_valid":uv_valid,
        }
        soup,face_map=probe._build_triangle_soup(
            canonical,meshes,probe._transparent_material()
        )
        require(len(face_map)==len(faces)
            and all(np.max(np.abs(np.asarray(o.matrix_world)-np.eye(4)))<=1e-8
                for o in soup),
            "V16_ORIGINAL_UV_MATERIAL_AND_FACE_SOUP_WORLD_IDENTITY")
        probe._configure_measurement(32)
        cam=probe._make_camera()
        all_labels=np.arange(len(faces),dtype=np.int32)
        selected_actions={receipt["sampled_frames"][p]["action"]
                          for p in job["pose_indices"]}
        controls,unsupported=v14._inventory(selected_actions,resolved,bindings)
        controls=controls[:128]
        require(len(controls)==prior["source_controls_measured_count"]
            and v14._inventory_digest(controls)==
                prior["native_rig_control_names_private_inventory_sha256"],
            "V16_ORIGINAL_SOURCE_ANIMATED_CARRIER_INVENTORY_SHA")
        arrays=[]
        witness=[]
        control_specs=[]
        for selected in selection["controls"]:
            idx=selected["control_index_teacher_only"]
            axis=selected["axis_index_teacher_only"]
            require(type(idx) is int and type(axis) is int
                    and 0<=idx<len(controls) and 0<=axis<3,
                "V16_LOCKED_ANONYMOUS_CONTROL_INDEX")
            kind,obj,bone=controls[idx]
            carrier=bone if kind=="POSE_BONE" else obj
            control_specs.append((idx,axis,kind,obj,carrier))
        # Two handles are required; no ontology that source controls equal author layers.
        require(control_specs[0][0]!=control_specs[1][0],
            "V16_NO_DUPLICATED_NATIVE_CONTROL_PAIR")
        for pi,pidx in enumerate(job["pose_indices"]):
            sample=receipt["sampled_frames"][pidx]
            replayed=actions.activate_action(resolved[sample["action"]],bindings)
            bpy.context.scene.frame_set(sample["frame"])
            bpy.context.view_layer.update()
            at_rest=v4._extract_snapshot(meshes,freeze_triangulation=False)
            xyz=np.asarray(at_rest["vertices"],np.float64)
            require(at_rest["polygon_signature"]==snapshot["polygon_signature"]
                and np.max(np.abs(xyz-native_baselines[pi]))<=tol,
                "V16_ORIGINAL_AUTHORED_POSE_V13B_BASELINE")
            _set_animated_soup(face_map,faces,xyz)
            support=[]
            for ci,(idx,axis,kind,obj,carrier) in enumerate(control_specs):
                moved=np.linalg.norm(native_controls[pi,idx,axis,0]-xyz,axis=-1)
                face_moved=np.mean(moved[faces],axis=1)>2e-4*extent
                require(np.any(face_moved),"V16_BLIND_SELECTED_NATIVE_CONTROL_NO_ACTIVE_FACES")
                support.append(face_moved)
            joint_support=support[0] | support[1]
            base_basis=[spec[-1].matrix_basis.copy() for spec in control_specs]
            camera_images=[]
            for vi,view in enumerate(job["view_indices"]):
                _camera_full(cam,camera[view],192)
                def draw(label, mode, visible_faces=None):
                    source_mode=mode=="SOURCE"
                    for obj in meshes:obj.hide_render=not source_mode
                    for obj in soup:obj.hide_render=source_mode
                    if not source_mode:
                        excluded={int(i) for i in range(len(faces))
                            if visible_faces is not None and not visible_faces[i]}
                        _group_mode(face_map,all_labels,excluded=excluded)
                    bpy.context.view_layer.update()
                    return _frame(temp/f"p{pidx}_v{view}_{label}.exr",192)
                now_base=draw("ORIGINAL_BASE","SOURCE")
                old_base=original_images[pi,vi,oldix["ORIGINAL_BASE"]]
                require(_difference(now_base,old_base)<=.025,
                    "V16_V14_EXACT_ORIGINAL_SOURCE_RGBA_BASELINE_DRIFT")
                current={
                    "ORIGINAL_BASE":now_base,
                    "ISOLATED_AB_BASE":draw("ISOLATED_AB_BASE","SOUP",joint_support)
                }
                pair_witness=[]
                for sign_a,sign_b in PAIR_SIGN:
                    key=("PLUS" if sign_a>0 else "MINUS")+"_"+(
                        "PLUS" if sign_b>0 else "MINUS")
                    for spec,basis,sign in zip(control_specs,base_basis,(sign_a,sign_b)):
                        idx,axis,kind,obj,carrier=spec
                        carrier.matrix_basis=basis @ Matrix.Rotation(
                            sign*np.pi*5./180.,4,AXES[axis])
                    bpy.context.view_layer.update()
                    edited=v4._extract_snapshot(meshes,freeze_triangulation=False)
                    joint=np.asarray(edited["vertices"],np.float64)
                    require(edited["polygon_signature"]==snapshot["polygon_signature"]
                        and joint.shape==xyz.shape and np.isfinite(joint).all(),
                        "V16_ORIGINAL_SIMULTANEOUS_NATIVE_VERTEX_FINITE_AND_TOPOLOGY")
                    a=control_specs[0];b=control_specs[1]
                    sa=0 if sign_a>0 else 1;sb=0 if sign_b>0 else 1
                    pos_a=native_controls[pi,a[0],a[1],sa]
                    pos_b=native_controls[pi,b[0],b[1],sb]
                    additive=pos_a+pos_b-xyz
                    residual=np.linalg.norm(joint-additive,axis=-1)
                    require(float(np.linalg.norm(joint-xyz))>2e-4*extent,
                        "V16_REAL_JOINT_NATIVE_CONTROL_NO_MECHANICAL_EFFECT")
                    _set_animated_soup(face_map,faces,joint)
                    unique_objects=[]
                    for s in control_specs:
                        if all(s[3] is not z for z in unique_objects):
                            unique_objects.append(s[3])
                    with contextlib.ExitStack() as stack:
                        for obj in unique_objects:
                            stack.enter_context(
                                v14._freeze_original_action_for_native_source_render(
                                    obj,meshes,joint,faces,
                                    snapshot["polygon_signature"],tol
                                ))
                        src=draw("SOURCE_AB_"+key,"SOURCE")
                    sou=draw("SOUP_AB_"+key,"SOUP")
                    iso=draw("ISOLATED_AB_"+key,"SOUP",joint_support)
                    require(_difference(src,sou)<=.025,
                        "V16_JOINT_NATIVE_SOURCE_VS_SOURCE_FACE_SOUP_RGBA_PARITY")
                    current["SOURCE_AB_"+key]=src
                    current["SOUP_AB_"+key]=sou
                    current["ISOLATED_AB_"+key]=iso
                    pair_witness.append({
                        "source_pose_ordinal":pi,
                        "source_camera_view":view,
                        "sign_a":int(sign_a),
                        "sign_b":int(sign_b),
                        "source_to_soup_p99":_difference(src,sou),
                        "joint_native_max_displacement":
                            float(np.max(np.linalg.norm(joint-xyz,axis=-1))),
                        "joint_native_vs_single_control_additive_max_residual":
                            float(residual.max(initial=0)),
                        "joint_native_vs_single_control_additive_mean_residual":
                            float(residual.mean()),
                        "joint_active_face_union_count":int(joint_support.sum()),
                        "author_drawable_equality":"UNKNOWN",
                    })
                    for s,basis in zip(control_specs,base_basis):
                        s[-1].matrix_basis=basis.copy()
                    bpy.context.view_layer.update()
                    restore=v4._extract_snapshot(meshes,freeze_triangulation=False)
                    require(restore["polygon_signature"]==snapshot["polygon_signature"]
                        and np.max(np.abs(
                            np.asarray(restore["vertices"],np.float64)-xyz))<=tol,
                        "V16_ORIGINAL_ACTION_NATIVE_JOINT_NOOP_RESTORE_FAILED")
                    _set_animated_soup(face_map,faces,xyz)
                arms=_joint_arms()
                require(set(current)==set(arms),
                    "V16_ORIGINAL_TWO_NATIVE_CONTROLS_FOUR_SIGN_PAIRS_PER_VIEW")
                camera_images.append(np.stack([current[name] for name in arms],axis=0))
                witness.extend(pair_witness)
            arrays.append(np.stack(camera_images,axis=0))
            base._neutralize_animation()
    data=np.stack(arrays,axis=0).astype(np.float16)
    arms=_joint_arms()
    require(data.shape==(3,4,len(arms),192,192,4)
        and len(witness)==48 and np.isfinite(data).all(),
        "V16_FULL_REAL_POSE_VIEW_PAIRED_RGBA_ARRAYS")
    out=Path(output_path).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    npz=out.with_suffix(".npz")
    np.savez_compressed(npz,original_native_dual_control_rgba=data)
    result={
        "schema":SCHEMA,
        "status":"V16_REAL_ORIGINAL_TWO_NATIVE_CONTROLS_SIMULTANEOUS_RGBA__INDEPENDENT_COURT_REQUIRED",
        "subject_id":job["subject_id"],
        "source_sha256":job["source"]["sha256"],
        "parent_v14_job_sha256":sha(job_path),
        "parent_v14_render_manifest_sha256":sha(parent_path),
        "parent_v14_rgba_npz_sha256":sha(original_npz),
        "original_v13b_control_vertex_array_sha256":sha(intervention_path),
        "source_rest_geometry_sha256":sha(restpath),
        "original_native_source_inventory_sha256":
            prior["native_rig_control_names_private_inventory_sha256"],
        "precommitted_blind_v14_control_plan_sha256":
            sha(job["selected_source_controls"]["path"]),
        "native_joint_perturbation_degrees":5.,
        "anonymous_source_controls_tested":2,
        "source_pose_count":3,
        "source_camera_views":[0,2,4,6],
        "signed_control_pairs":["PLUS_PLUS","PLUS_MINUS","MINUS_PLUS","MINUS_MINUS"],
        "rendered_arms":arms,
        "blender_render_arms":int(np.prod(data.shape[:3])),
        "joint_original_source_to_soup_parity_rechecked":True,
        "original_source_action_frozen_and_exact_vertices_before_after_render":True,
        "joint_visuals_from_real_original_native_blender_source":True,
        "no_expected_joint_ground_truth_fabricated_from_single_v13b":True,
        "native_source_joint_mechanics_vs_independent_single_effect_witness":witness,
        "original_dual_control_rgba_npz_sha256":sha(npz),
        "original_dual_control_rgba_npz_path":str(npz),
        "mechanical_controller_is_a_drawable_label":False,
        "source_rig_bone_material_names_as_model_features":False,
        "source_artist_drawable_names_or_part_count_used":False,
        "author_same_drawable_pairs":0,
        "author_different_drawable_pairs":0,
        "source_presentation_draw_order_proven":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
        "materialization_authorized":False
    }
    out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in (
        "schema","status","subject_id","blender_render_arms",
        "original_dual_control_rgba_npz_sha256",
        "no_expected_joint_ground_truth_fabricated_from_single_v13b"
    )},sort_keys=True),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for arg in ("job","v14_parent_manifest","v14_parent_rgba","output"):
        ap.add_argument("--"+arg.replace("_","-"),type=Path,required=True)
    a=ap.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None)
    render(a.job,a.v14_parent_manifest,a.v14_parent_rgba,a.output)
