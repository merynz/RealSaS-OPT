#!/usr/bin/env python3
"""Minimal original-source V18b EMPTY-soup transport witness.

Original 192px / 32 Cycles samples / pose2 / views 0,2,4,6. Renders only
(1) original Transparent-BSDF-empty soup, (2) 256 transparency bounces,
(3) physically hidden soup. No artist labeling, training or edited layers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import bpy
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

SCHEMA="RealSaS.Corpus370V18bSourceEmptyTransparencyBounceWitness.v1"
STATUS="REAL_SOURCE_TRANSPARENT_BOUNCE_VS_HIDDEN_EMPTY_RENDER_MEASURED__NO_AUTHORITY"
VIEWS=(0,2,4,6)
TOL=.025
SAMPLES=32
HIGH_BOUNCES=256


def alpha_metrics(frame):
    x=np.asarray(frame,np.float32)
    require(x.shape==(192,192,4) and np.isfinite(x).all(),
            "V18B_EMPTY_WITNESS_MALFORMED_FRAME")
    alpha=x[...,3]
    return {
        "max_alpha":float(alpha.max()),
        "alpha_over_original_threshold_pixel_count":int((alpha>TOL).sum()),
        "alpha_over_original_threshold":bool((alpha>TOL).any()),
        "alpha_p99":float(np.percentile(alpha,99)),
    }


def run(job_path, archived_diagnostic, output_path):
    job=json.loads(Path(job_path).read_text())
    parent=json.loads(Path(archived_diagnostic).read_text())
    require(job.get("schema")=="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
        and job.get("training_authorized") is False
        and job.get("graphics_owner_authorized") is False
        and parent.get("schema")=="RealSaS.Corpus370V18bArchivedRenderParityDiagnosis.v1"
        and parent.get("status")==
            "ARCHIVED_RENDER_PARITY_AND_PROBE_VISIBILITY_DIAGNOSED__NO_AUTHORITY"
        and parent.get("source_subject_id_private")==job["subject_id"]
        and parent.get("original_source_sha256")==job["source"]["sha256"]
        and parent.get("aggregate",{}).get("source_soup_parity_pass_views")==4
        and parent.get("aggregate",{}).get("original_v14_parity_pass_views")==4
        and parent.get("aggregate",{}).get("soup_noop_parity_pass_views")==4
        and parent.get("aggregate",{}).get("empty_soup_alpha_pass_views")<4
        and parent.get("source_editability_qualified") is False
        and parent.get("training_authorized") is False
        and job["view_indices"]==list(VIEWS)
        and len(job["pose_indices"])==3
        and job["raster_resolution"]==192
        and job["cycles_samples"]==SAMPLES,
        "V18B_EMPTY_WITNESS_PARENT_NOT_EXACT_FAILED_EMPTY_TRANSPORT")
    source=verify(job["source"],"V18B_EMPTY_WITNESS_SOURCE_SHA")
    restpath=verify(job["rest_geometry"],"V18B_EMPTY_WITNESS_REST_SHA")
    teacherpath=verify(job["native_teacher"],"V18B_EMPTY_WITNESS_TEACHER_SHA")
    receiptpath=verify(job["native_teacher_receipt"],"V18B_EMPTY_WITNESS_RECEIPT_SHA")
    camerapath=verify(job["camera"],"V18B_EMPTY_WITNESS_CAMERA_SHA")
    candidatepath=verify(job["candidate"],"V18B_EMPTY_WITNESS_CANDIDATE_SHA")
    receipt=json.loads(receiptpath.read_text())
    camera={int(x["view_index"]):x for x in json.loads(camerapath.read_text())["cameras"]}
    candidate=json.loads(candidatepath.read_text())
    with np.load(restpath,allow_pickle=False) as d:
        rest=np.asarray(d["rest_vertices_source_world"],np.float64)
        faces=np.asarray(d["faces"],np.int64)
        owner=np.asarray(d["face_owner_privileged"],np.int32)
    with np.load(teacherpath,allow_pickle=False) as d:
        truth_vertices=np.asarray(d["posed_vertices_source_world"],np.float64)
        truth_faces=np.asarray(d["faces"],np.int64)
    require(np.array_equal(faces,truth_faces),
            "V18B_EMPTY_WITNESS_SOURCE_FACE_IDENTITY")
    sampled=receipt["sampled_frames"][job["pose_indices"][2]]
    with tempfile.TemporaryDirectory(prefix="c370_empty_bounce_witness_") as td:
        scratch=Path(td)
        base._import_animated(source)
        original_actions=receipt["native_motion_inventory_expected"]
        resolved={k:transport._action_by_requested_take(k) for k in original_actions}
        bindings=actions.capture_imported_action_bindings(resolved)
        meshes=base._mesh_objects()
        base._neutralize_animation()
        snapshot=v4._extract_snapshot(meshes,freeze_triangulation=True)
        extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
        tolerance=1e-6*max(1.,extent)
        require(np.array_equal(np.asarray(snapshot["faces"]),faces)
            and np.array_equal(np.asarray(snapshot["face_owner"]),owner)
            and np.max(np.abs(np.asarray(snapshot["vertices"])-rest))<=tolerance,
            "V18B_EMPTY_WITNESS_REST_TRANSPORT_DRIFT")
        materials=replay._materials_from_render_meshes(meshes)
        replay._replay_transform(materials,list(candidate.get("material_transform") or []))
        uv,uv_valid,_,_=_appearance_and_uv(meshes,snapshot)
        carrier=dict(
            rest_vertices_source_world=rest,faces=faces,
            face_owner_privileged=owner,
            face_material_privileged=np.asarray(snapshot["face_material"],np.int32),
            triangle_uv=uv,triangle_uv_valid=uv_valid,
        )
        soup,face_map=probe._build_triangle_soup(
            carrier,meshes,probe._transparent_material())
        require(len(face_map)==len(faces)
            and all(np.max(np.abs(np.asarray(obj.matrix_world)-np.eye(4)))<=1e-8
                    for obj in soup),
            "V18B_EMPTY_WITNESS_SOURCE_WORLD_SOUP_DRIFT")
        probe._configure_measurement(SAMPLES)
        cam=probe._make_camera()
        actions.activate_action(resolved[sampled["action"]],bindings)
        bpy.context.scene.frame_set(int(sampled["frame"]))
        bpy.context.view_layer.update()
        posed=v4._extract_snapshot(meshes,freeze_triangulation=False)
        actual_vertices=np.asarray(posed["vertices"],np.float64)
        require(posed["polygon_signature"]==snapshot["polygon_signature"]
            and np.max(np.abs(actual_vertices-truth_vertices[
                job["pose_indices"][2]]))<=tolerance,
            "V18B_EMPTY_WITNESS_HELDOUT_NATIVE_POSE_DRIFT")
        _set_animated_soup(face_map,faces,actual_vertices)
        scene=bpy.context.scene
        baseline_bounces=int(scene.cycles.transparent_max_bounces)
        require(0<=baseline_bounces<HIGH_BOUNCES,
                "V18B_EMPTY_WITNESS_BOUNCE_CONTRAST_NOT_VALID")
        for mesh in meshes:mesh.hide_render=True
        for obj in soup:obj.hide_render=False
        # Explicitly force every known source triangle to Transparent BSDF.
        face_index=np.arange(len(faces),dtype=np.int32)
        _group_mode(face_map,face_index,excluded=set(range(len(faces))))
        result=[]
        observed_frames=[]
        for view in VIEWS:
            _camera_full(cam,camera[view],192)
            scene.cycles.transparent_max_bounces=baseline_bounces
            bpy.context.view_layer.update()
            low=_frame(scratch/f"empty_low_v{view}.exr",192)
            scene.cycles.transparent_max_bounces=HIGH_BOUNCES
            bpy.context.view_layer.update()
            high=_frame(scratch/f"empty_high_v{view}.exr",192)
            for obj in soup:obj.hide_render=True
            bpy.context.view_layer.update()
            hidden=_frame(scratch/f"empty_hidden_v{view}.exr",192)
            for obj in soup:obj.hide_render=False
            bpy.context.view_layer.update()
            observed_frames.append(np.stack((low,high,hidden),axis=0))
            low_a=alpha_metrics(low);high_a=alpha_metrics(high)
            hidden_a=alpha_metrics(hidden)
            result.append({
                "original_source_view":int(view),
                "transparent_shader_original_bounce_limit":low_a,
                "transparent_shader_high_bounce_limit":high_a,
                "physically_hidden_soup":hidden_a,
                "stochastic_transparency_limit_hypothesis_supported":
                    low_a["alpha_over_original_threshold"] and
                    not high_a["alpha_over_original_threshold"] and
                    not hidden_a["alpha_over_original_threshold"],
                "non_soup_scene_alpha_suspected":
                    hidden_a["alpha_over_original_threshold"],
            })
    out=Path(output_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    archive=out.with_suffix(".npz")
    np.savez_compressed(archive,empty_variants_rgba=np.stack(observed_frames,axis=0).astype(np.float32))
    payload={
        "schema":SCHEMA,"status":STATUS,
        "original_source_id_private":job["subject_id"],
        "original_source_sha256":job["source"]["sha256"],
        "parent_v14_job_sha256":sha(job_path),
        "parent_v18b_archived_diagnostic_sha256":sha(archived_diagnostic),
        "independently_reopenable_render_rgba_npz_sha256":sha(archive),
        "original_v18b_source_run":"42100552844-1",
        "archived_retrospective_run":"42105598958-1",
        "original_heldout_pose_ordinal":2,
        "camera_views":list(VIEWS),
        "cycles_samples":SAMPLES,
        "original_transparent_max_bounces":baseline_bounces,
        "high_transparent_max_bounces":HIGH_BOUNCES,
        "original_alpha_tolerance":TOL,
        "per_view":result,
        "new_renders_performed":3*len(VIEWS),
        "source_artist_layer_identity":"UNKNOWN",
        "source_editability_qualified":False,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    out.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":STATUS,
        "source_subject":out.parent.name,
        "baseline_transparent_bounces":baseline_bounces,
        "high_transparent_bounces":HIGH_BOUNCES,
        "per_view":result,
        "new_renders":3*len(VIEWS),
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    args=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True,type=Path)
    parser.add_argument("--archived-diagnostic",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args(args)
    run(a.job,a.archived_diagnostic,a.output)
