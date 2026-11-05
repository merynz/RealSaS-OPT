#!/usr/bin/env python3
"""V18m original-source per-index Cycles depth evidence without FULL oracle input.

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

SCHEMA="RealSaS.Corpus370OriginalCyclesSampleDepthEvidence.v18m"
FROZEN_T=.1
SAMPLE_INDICES=(0,4,8,12,16,20,24,28)
ARMS=("GROUP_DEPTH","REMAINDER_DEPTH")
STAIN="RealSaS.Corpus370NativeReactiveStainCandidate.v18a"
STAIN_COURT="RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
V14_JOB="RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
POSE=2
VIEWS=(0,2,4,6)
TARGET_INDICES=(0,)
THRESHOLDS=SOURCE_THRESHOLDS
PHYSICAL_VISIBILITY="ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"


def ops():
    return ["EMISSIVE_EMPTY"] + [
        f"EMISSIVE_{str(t).replace('.', 'p')}_{mode}"
        for t in THRESHOLDS for mode in ("FULL","ISOLATE","REMOVE")
    ]


def emission_material(name, rgb):
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    nodes=mat.node_tree.nodes
    nodes.clear()
    o=nodes.new("ShaderNodeOutputMaterial")
    emit=nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value=(*rgb,1.0)
    emit.inputs["Strength"].default_value=1.0
    mat.node_tree.links.new(emit.outputs["Emission"],o.inputs["Surface"])
    return mat


def source_depth_material(name):
    """Render Cycles camera View Z depth as normalized EMISSION red, not RGB art.

    Material is diagnostic only. Per camera's frozen authored pose depth range
    comes from original 3D vertices and original calibrated camera, not FULL
    outcome RGB. Camera rays still use actual Cycles sample subset indices.
    """
    mat=bpy.data.materials.new(name)
    mat.use_nodes=True
    nodes=mat.node_tree.nodes
    nodes.clear()
    output=nodes.new("ShaderNodeOutputMaterial")
    emit=nodes.new("ShaderNodeEmission")
    camera=nodes.new("ShaderNodeCameraData")
    mapping=nodes.new("ShaderNodeMapRange")
    mapping.clamp=True
    mapping.inputs["To Min"].default_value=.02
    mapping.inputs["To Max"].default_value=.98
    combine=nodes.new("ShaderNodeCombineXYZ")
    combine.inputs["Y"].default_value=0.
    combine.inputs["Z"].default_value=0.
    links=mat.node_tree.links
    links.new(camera.outputs["View Z Depth"],mapping.inputs["Value"])
    links.new(mapping.outputs["Result"],combine.inputs["X"])
    links.new(combine.outputs["Vector"],emit.inputs["Color"])
    links.new(emit.outputs["Emission"],output.inputs["Surface"])
    return mat,mapping


def normalize_source_depth_for_camera(mapping,camrow,posed_vertices):
    """Frozen geometry-only depth scale; no V18l FULL pixel read."""
    origin=np.asarray(camrow["origin"],np.float64)
    forward=np.asarray(camrow["forward"],np.float64)
    n=float(np.linalg.norm(forward))
    require(n>0.99 and n<1.01,"V18M_SOURCE_CAMERA_FORWARD_INVALID")
    forward=forward/n
    depth=(np.asarray(posed_vertices,np.float64)-origin)@forward
    lo,hi=float(np.min(depth)),float(np.max(depth))
    span=max(hi-lo,1e-7)
    lo-=.05*span
    hi+=.05*span
    require(hi>lo and lo>0
        and lo>float(camrow["clip_start"])-1e-3
        and hi<float(camrow["clip_end"])+1e-3,
        "V18M_SOURCE_CYCLES_DEPTH_RANGE_OUTSIDE_CAMERA_CLIP")
    mapping.inputs["From Min"].default_value=lo
    mapping.inputs["From Max"].default_value=hi
    return [lo,hi]


def colorize_soup(face_map, base_meshes, labels, wanted_group,
                  group_material, remainder_material):
    # Re-color the immutable original posed soup *for controlled diagnostic
    # rendering only*. No source artist material, UV or graphics identity
    # is admitted; V18f already independent-courted source fidelity.
    require(len(labels)==len(face_map),"V18H_EMISSION_SOURCE_FACE_COUNT")
    by_mesh={}
    for face,(mesh,local,_orig,_clear) in face_map.items():
        by_mesh.setdefault(mesh,[]).append((int(local),int(labels[face])==wanted_group))
    require(set(by_mesh)==set(base_meshes.values()),
            "V18H_EMISSION_SOUP_BASE_MESH_BINDING")
    for mesh,rows in by_mesh.items():
        mi_g=len(mesh.materials)
        if group_material.name not in [m.name for m in mesh.materials]:
            mesh.materials.append(group_material)
        else:
            mi_g=[m.name for m in mesh.materials].index(group_material.name)
        mi_r=len(mesh.materials)
        if remainder_material.name not in [m.name for m in mesh.materials]:
            mesh.materials.append(remainder_material)
        else:
            mi_r=[m.name for m in mesh.materials].index(remainder_material.name)
        require(len(rows)==len(mesh.polygons),
                "V18H_EMISSION_ORIGINAL_FACE_MAP_CENSUS")
        for local,grouped in rows:
            mesh.polygons[local].material_index=mi_g if grouped else mi_r
        mesh.update()
    bpy.context.view_layer.update()


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
            ==visible_count,"V18F_PHYSICAL_VISIBLE_TRIANGLE_CENSUS")
    return visible_count


def run(job_path,stain_report_path,stain_npz_path,stain_court_path,
        v14_original_pixels,parent_v18f_court_path,parent_v18g_court_path,
        expected_v18f_sha,expected_v18g_sha,v18l_manifest,v18l_rgba,
        v18l_court,expected_v18l_court_sha,output_path):
    job=json.loads(Path(job_path).read_text())
    parent_f=json.loads(Path(parent_v18f_court_path).read_text())
    parent_g=json.loads(Path(parent_v18g_court_path).read_text())
    parent_l=json.loads(Path(v18l_court).read_text())
    parent_m=json.loads(Path(v18l_manifest).read_text())
    require(sha(v18l_court)==expected_v18l_court_sha
        and len(expected_v18l_court_sha)==64
        and parent_l.get("schema")=="RealSaS.Corpus370IndependentCyclesCommonSampleSourceCourt.v18l"
        and parent_l.get("status")==
           "ORIGINAL_CYCLES_SUBSET_COMMON_SAMPLE_LAYER_REMEASURED__SOURCE_ARTIST_UNKNOWN"
        and parent_l.get("matched_sample_invariant_qualified") is True
        and parent_l.get("source_render_manifest_sha256")==sha(v18l_manifest)
        and parent_l.get("source_render_npz_sha256")==sha(v18l_rgba)
        and parent_m.get("schema")==
           "RealSaS.Corpus370OriginalCyclesCommonSampleComplement.v18l"
        and parent_m.get("status")==
           "V18L_ORIGINAL_CYCLES_COMMON_SOURCE_SAMPLE_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and parent_m.get("original_source_cycles_sample_indices")==list(SAMPLE_INDICES)
        and parent_m.get("source_cycles_pixel_filter")=="BOX"
        and parent_m.get("source_cycles_pixel_filter_width")==.01
        and parent_m.get("source_original_v14_job_sha256")==sha(job_path)
        and parent_m.get("source_reactive_stain_npz_sha256")==sha(stain_npz_path)
        and parent_m.get("original_source_camera_sha256")==job["camera"]["sha256"]
        and parent_m.get("original_source_teacher_vertices_sha256")==
            job["native_teacher"]["sha256"]
        and parent_m.get("subject_id")==job["subject_id"]
        and parent_m.get("original_source_sha256")==job["source"]["sha256"]
        and parent_l.get("original_source_sha256")==job["source"]["sha256"]
        and parent_l.get("subject_id_private")==job["subject_id"]
        and all(parent_l.get(z) is False for z in (
            "graphics_owner_authorized","training_authorized")),
        "V18M_FROZEN_V18L_INDEPENDENT_SOURCE_PIXEL_COURT_BINDING")

    require(sha(parent_v18f_court_path)==expected_v18f_sha
        and sha(parent_v18g_court_path)==expected_v18g_sha
        and len(expected_v18f_sha)==64 and len(expected_v18g_sha)==64
        and parent_f.get("schema")=="RealSaS.Corpus370IndependentFullPhysicalFaceEditCourt.v18f"
        and parent_f.get("status")==
             "ORIGINAL_FULL_PHYSICAL_EDITS_SOURCE_CONSEQUENCES_COURTED__ARTIST_DRAWABLE_UNKNOWN"
        and parent_f.get("source_sha256")==job["source"]["sha256"]
        and parent_f.get("original_source_id")==job["subject_id"]
        and parent_f.get("physical_render_face_counts_courted") is True
        and parent_f.get("training_authorized") is False
        and parent_g.get("schema")==
             "RealSaS.Corpus370IndependentOriginalPhysicalCompositingDiagnostics.v18g"
        and parent_g.get("status")==
             "ORIGINAL_SOURCE_PHYSICAL_LAYER_COMPLEMENT_ALPHA_COMPOSING_DIAGNOSED__NO_OWNER_TRUTH"
        and parent_g.get("original_source_sha256")==job["source"]["sha256"]
        and parent_g.get("source_subject_private")==job["subject_id"]
        and parent_g.get("exact_v18f_independent_court_sha256")==expected_v18f_sha
        and parent_g.get("training_authorized") is False
        and parent_g.get("graphics_owner_authorized") is False,
        "V18H_ORIGINAL_V18F_V18G_INDEPENDENT_COURTS_SHA_BINDING")
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
    names=list(ARMS)
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
        # Blender 4.5 Cycles renders one *specific* sample index into each
        # physical arm, preserving the sampler's pixel coordinates / seed.
        # This is NOT a claim that 8 spaced sample indices equal 32spp film.
        scene=bpy.context.scene
        require(scene.render.engine=="CYCLES","V18L_REAL_CYCLES_REQUIRED")
        require(all(hasattr(scene.cycles,k) for k in (
            "use_sample_subset","sample_offset","sample_subset_length",
            "use_adaptive_sampling","samples","seed","pixel_filter_type","filter_width")),
            "V18L_BLENDER_45_SAMPLE_SUBSET_API_MISSING")
        scene.cycles.samples=32
        scene.cycles.seed=0
        scene.cycles.use_adaptive_sampling=False
        scene.cycles.use_sample_subset=True
        scene.cycles.sample_subset_length=1
        scene.cycles.pixel_filter_type="BOX"
        scene.cycles.filter_width=0.01
        require(scene.cycles.pixel_filter_type=="BOX"
            and scene.cycles.use_sample_subset
            and scene.cycles.sample_subset_length==1
            and scene.cycles.samples==32,
            "V18L_SINGLE_COMMON_SOURCE_SAMPLE_POLICY_CHANGED")
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
        emit_group,depth_mapping=source_depth_material("V18M_SOURCE_CYCLES_VIEW_DEPTH")
        emit_rest=emit_group
        for ob in meshes: ob.hide_render=True
        for ob in soup: ob.hide_render=False
        # Geometry-only per-camera depth mapping. Material and heldout source
        # geometry are identical for both isolated G and R, with only original
        # physically excluded triangles differing between render arms.
        names=list(ARMS)
        render_cases=[]
        face_count_checks=[]
        depth_range_checks=[]
        t=FROZEN_T
        labels=labels_by_t[t]
        group=int(labels[targets[0]])
        present=set(int(v) for v in np.unique(labels))
        require(group in present and len(present)>1,
                "V18M_FROZEN_SOURCE_GROUP_VACUOUS")
        count_group=int(np.count_nonzero(labels==group))
        require(0<count_group<len(faces),"V18M_GROUP_IS_WHOLE_SOURCE")
        for view in VIEWS:
            _camera_full(cam,camera[view],192)
            depth_range=normalize_source_depth_for_camera(
                depth_mapping,camera[view],original_vertices)
            depth_range_checks.append(depth_range)
            physical_group_mode(soup,face_map,base_meshes,face_ids,set())
            colorize_soup(face_map,base_meshes,labels,group,emit_group,emit_rest)
            view_samples=[]
            view_counts=[]
            for source_sample_index in SAMPLE_INDICES:
                scene.cycles.sample_offset=int(source_sample_index)
                frames=[]
                counts=[]
                for mode in names:
                    excluded=(present-{group} if mode=="GROUP_DEPTH" else {group})
                    count=physical_group_mode(
                        soup,face_map,base_meshes,labels,excluded)
                    frame=_frame(scratch/f"v{view}_sample{source_sample_index}_{mode}.exr",192)
                    frames.append(frame)
                    counts.append(int(count))
                view_samples.append(np.stack(frames,axis=0))
                view_counts.append(counts)
            require(all(row==view_counts[0] for row in view_counts),
                "V18M_SOURCE_FACE_COUNTS_CHANGED_ACROSS_SAMPLE_INDICES")
            render_cases.append(np.stack(view_samples,axis=0))
            face_count_checks.append(view_counts[0])
        out=np.stack(render_cases,axis=0).astype(np.float32)
        require(out.shape==(len(VIEWS),len(SAMPLE_INDICES),len(ARMS),192,192,4)
            and np.isfinite(out).all()
            and np.min(out[...,3])>=-1e-5 and np.max(out[...,3])<=1.00001,
            "V18M_ORIGINAL_SOURCE_DEPTH_RGBA_SHAPE")
    result_path=Path(output_path).resolve()
    result_path.parent.mkdir(parents=True,exist_ok=True)
    npz=result_path.with_suffix(".npz")
    np.savez_compressed(npz,source_cycles_depth_rgba=out)
    result={
        "schema":SCHEMA,
        "status":"V18M_ORIGINAL_CYCLES_SOURCE_DEPTH_REPLAYED__INDEPENDENT_COURT_REQUIRED",
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
        "original_source_cycles_sample_indices":list(SAMPLE_INDICES),
        "source_cycles_max_samples":32,
        "source_cycle_subset_length":1,
        "source_cycle_sampling_seed":0,
        "source_cycles_adaptive_sampling":False,
        "source_cycles_pixel_filter":"BOX",
        "source_cycles_pixel_filter_width":0.01,
        "source_cycle_sample_pattern_equals_original_v18h_32spp_film":False,
        "source_independent_v18l_court_sha256":sha(v18l_court),
        "source_independent_v18l_render_manifest_sha256":sha(v18l_manifest),
        "source_independent_v18l_raw_rgba_sha256":sha(v18l_rgba),
        "source_depth_per_view_source_geometry_only_normalized_ranges":depth_range_checks,
        "source_depth_normalization_to_min":.02,
        "source_depth_normalization_to_max":.98,
        "source_cycle_depth_material":"CAMERA_DATA_VIEW_Z_DEPTH_TO_LINEAR_EMISSION_R",
        "source_face_depth_evidence_is_teacher_private_not_art_layer_label":True,
        "no_source_artist_material_authority_from_emission":True,
        "source_face_probe_target_ordinals_teacher_only":targets,
        "source_edit_arms":names,
        "physical_face_visibility_operator":PHYSICAL_VISIBILITY,
        "source_material_policy":"SOURCE_ORTHOGRAPHIC_VIEW_Z_DEPTH_EMISSION_R_NO_ART_SHADING",
        "source_material_label_authority":"NONE",
        "actual_render_arms_per_camera":len(names),
        "actual_visible_original_source_triangles_per_arm_view":face_count_checks,
        "original_source_v18l_parent_v18h_run":"42120479356-1",
        "original_source_v18m_parent_v18l_run":"42131579613-1",
        "v18l_source_nontrivial_group_face_count":int(np.count_nonzero(labels==group)),
        "v18l_original_source_face_count":len(faces),
        "precommitted_v11_target_query_indices":list(TARGET_INDICES),
        "frozen_tested_stain_similarity_thresholds":[FROZEN_T],
        "original_full_stain_similarity_thresholds":list(SOURCE_THRESHOLDS),
        "original_source_heldout_pose_vertices_verified":True,
        "source_rest_triangle_UV_material_authority_reused":False,
        "original_UV_material_textures_deliberately_not_tested_under_emission":True,
        "source_vs_original_v14_RGBA_reference_source_parity_verified":False,
        "source_v14_parity_authority_inherited_only_from_sealed_v18f_parent":True,
        "parent_v18f_independent_court_sha256":sha(parent_v18f_court_path),
        "parent_v18g_independent_compositing_sha256":sha(parent_v18g_court_path),
        "new_source_actual_edited_rgba_npz_sha256":sha(npz),
        "diagnostic_emission_color_group_rgb":None,
        "diagnostic_emission_color_complement_rgb":None,
        "no_artist_layer_labels_minted":True,
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
        "single_cycles_source_subsets_per_view":len(SAMPLE_INDICES),
        "original_heldout_camera_views":len(VIEWS),
        "private_RGBA_sha256":result["new_source_actual_edited_rgba_npz_sha256"],
        "training_authorized":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("job","stain_report","stain_npz","stain_court",
                "v14_original_pixels","parent_v18f_court","parent_v18g_court",
                "v18l_manifest","v18l_rgba","v18l_court","output"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    p.add_argument("--expected-v18f-sha",required=True)
    p.add_argument("--expected-v18g-sha",required=True)
    p.add_argument("--expected-v18l-court-sha",required=True)
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else None
    a=p.parse_args(argv)
    run(a.job,a.stain_report,a.stain_npz,a.stain_court,
        a.v14_original_pixels,a.parent_v18f_court,a.parent_v18g_court,
        a.expected_v18f_sha,a.expected_v18g_sha,a.v18l_manifest,a.v18l_rgba,
        a.v18l_court,a.expected_v18l_court_sha,a.output)
