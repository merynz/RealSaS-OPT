#!/usr/bin/env python3
"""V18k: original-source geometry-only-selected joint-sample vs postfiltered OVER.

Fixed two-color opaque emission; exact original V14 posed 3D geometry, frozen
V18a anonymous groups and SHA-sealed V18h source frames. 4x4 geometry samples
are NOT Cycles' 32spp stochastic sample positions. No drawable/model authority.

The pixel candidate set uses ONLY original posed mesh triangles, frozen group
membership and original cameras. V18i/V18j error labels are not selection inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from models.iris.truth.graphics_oracle_v1 import (
    camera_from_mapping,face_centroid_candidate_pixels_v1,
    measure_orthographic_ray_stack_v1,
)
from tools.corpus.measure_corpus370_v17b_original_subpixel_coverage_v1 import (
    X_OFFSETS,Y_OFFSETS,offset_camera,
)
from tools.corpus.court_corpus370_v18h_emissive_compositing_witness import (
    THRESHOLDS,VIEWS,arms,RENDER as V18H_RENDER,SCHEMA as V18H_COURT,
    STATUS as V18H_STATUS,
)
from tools.corpus.court_corpus370_v18j_source_subpixel_order_attribution import (
    SCHEMA as V18J_COURT,PASS as V18J_PASS,
)

SCHEMA="RealSaS.Corpus370OriginalSourceGeometryJointSampleCompositing.v18k"
STATUS="V18K_ORIGINAL_GEOMETRY_PRESELECTED_JOINT_SAMPLE_COMPOSITING_DIAGNOSED__ARTIST_UNKNOWN"
POSE_ORDINAL=2
RESOLUTION=192
TOL=.025
MAX_GLOBAL_CENTROIDS=32
MAX_GROUP_CENTROIDS=16
MAX_COMPLEMENT_CENTROIDS=16
NEIGHBOR_ANCHORS_PER_SIDE=2
SAMPLE_COUNT=len(X_OFFSETS)*len(Y_OFFSETS)
HELDOUT_VIEWS=(4,6)
EXPLORATORY_VIEWS=(0,2)
RED=np.array([.90,.08,.035],dtype=np.float64)
BLUE=np.array([.04,.14,.92],dtype=np.float64)


def require(ok,reason):
    if not ok:raise ValueError(reason)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def proposed_pixels(vertices,faces,camera,labels,group_id):
    """Original geometry only; never read RGBA or any V18i residual here."""
    require(type(group_id) is int
        and len(labels)==len(faces)
        and np.issubdtype(np.asarray(labels).dtype,np.integer),
        "V18K_GEOMETRY_ONLY_GROUP_SELECTION")
    gmask=labels==group_id
    rmask=~gmask
    require(bool(np.any(gmask)),"V18K_TARGET_GROUP_NOT_FOUND")
    full=face_centroid_candidate_pixels_v1(
        vertices,faces,camera,maximum=MAX_GLOBAL_CENTROIDS)
    group=face_centroid_candidate_pixels_v1(
        vertices,faces[gmask],camera,maximum=MAX_GROUP_CENTROIDS)
    remain=(face_centroid_candidate_pixels_v1(
        vertices,faces[rmask],camera,maximum=MAX_COMPLEMENT_CENTROIDS)
            if np.any(rmask) else ())
    chosen=set(full)|set(group)|set(remain)
    for anchors in (group[:NEIGHBOR_ANCHORS_PER_SIDE],
                    remain[:NEIGHBOR_ANCHORS_PER_SIDE]):
        for x,y in anchors:
            for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx,ny=x+dx,y+dy
                if 0<=nx<RESOLUTION and 0<=ny<RESOLUTION:
                    chosen.add((nx,ny))
    return tuple(sorted(chosen))


def sample_geometry(vertices,faces,camera,x,y,tie):
    """A source-ray query independent of any layer proposal or image pixel."""
    require(type(x) is int and type(y) is int
        and 0<=x<RESOLUTION and 0<=y<RESOLUTION,
        "V18K_GEOMETRIC_PIXEL_CONTRACT")
    samples=[]
    for oy in Y_OFFSETS:
        for ox in X_OFFSETS:
            raycam=offset_camera(camera,ox,oy)
            ray=measure_orthographic_ray_stack_v1(
                vertices,faces,raycam,x,y,tie_epsilon=tie)
            samples.append({
                "source_subpixel_offset_x":float(ox),
                "source_subpixel_offset_y":float(oy),
                "complete":bool(ray.complete),
                "depth_tie":bool(ray.has_depth_tie),
                "source_hit_face_ordinals_teacher_only":[int(z.face_index)
                    for z in ray.hits],
                "source_hit_depths":[float(z.depth) for z in ray.hits],
            })
    require(len(samples)==SAMPLE_COUNT,"V18K_SAMPLE_GRID_CHANGED")
    return samples


def over_premul(top,bottom):
    require(np.asarray(top).shape==(4,) and np.asarray(bottom).shape==(4,),
            "V18K_OVER_RGBA_SHAPE")
    return np.asarray(top,dtype=np.float64)+(
        np.asarray(bottom,dtype=np.float64)*
        (1.-float(top[3])))


def render_sampled(samples,source_labels,group_id):
    """Joint sample-first and independently filtered group/rest AFTER filter.

    RGB is fixed opaque emission. A ray with only one group yields same pixel
    for both approaches. Opposing nearest faces within the pixel can cause
    a difference; alpha overlap is correlated, not statistically independent.
    """
    require(len(samples)==SAMPLE_COUNT,"V18K_WRONG_SUBPIXEL_COUNT")
    j=np.zeros((4,),dtype=np.float64)
    g=np.zeros((4,),dtype=np.float64)
    r=np.zeros((4,),dtype=np.float64)
    sides=[]
    alternation=0
    unknown=0
    for z in samples:
        if not z["complete"] or z["depth_tie"]:
            unknown+=1
            continue
        hits=z["source_hit_face_ordinals_teacher_only"]
        depths=z["source_hit_depths"]
        require(len(hits)==len(depths)
            and all(type(i) is int and 0<=i<len(source_labels) for i in hits)
            and all(a<=b for a,b in zip(depths,depths[1:])),
            "V18K_ORIGINAL_RAY_FACE_ORDER_DRIFT")
        classes=[bool(source_labels[f]==group_id) for f in hits]
        alternation+=int(sum(a!=b for a,b in zip(classes,classes[1:]))>=2)
        if hits:
            first_side=classes[0]
            sides.append("G" if first_side else "R")
            j+=np.array([*(RED if first_side else BLUE),1.])
        else:
            sides.append("NONE")
        if any(classes):
            g+=np.array([*RED,1.])
        if any(not x for x in classes):
            r+=np.array([*BLUE,1.])
    if unknown:
        return {
            "source_sample_count":SAMPLE_COUNT,
            "ray_unknown_sample_count":unknown,
            "status":"INCOMPLETE_OR_TIED_RAY__NO_PIXEL_COMPOSITION",
            "geometric_opaque_sample_composition":None,
        }
    j/=SAMPLE_COUNT
    g/=SAMPLE_COUNT
    r/=SAMPLE_COUNT
    over_gr=over_premul(g,r)
    over_rg=over_premul(r,g)
    norm=lambda x:float(np.max(np.abs(x)))
    return {
        "source_sample_count":SAMPLE_COUNT,"ray_unknown_sample_count":0,
        "status":"COMPLETE_UNTIED_OPAQUE_EMISSION_GEOMETRY_SAMPLE",
        "geometric_opaque_sample_composition":{
            "joint_pre_filter_premultiplied_rgba":j.tolist(),
            "postfilter_group_premultiplied_rgba":g.tolist(),
            "postfilter_remainder_premultiplied_rgba":r.tolist(),
            "postfilter_g_over_r_premultiplied_rgba":over_gr.tolist(),
            "postfilter_r_over_g_premultiplied_rgba":over_rg.tolist(),
            "joint_vs_best_postfilter_order_max_rgba_delta":min(
                norm(j-over_gr),norm(j-over_rg)),
            "joint_alpha_minus_postfilter_alpha":float(j[3]-over_gr[3]),
            "mixed_source_nearest_G_R_within_pixel":bool(
                "G" in sides and "R" in sides),
            "subsample_rays_with_multiple_depth_side_alternations":alternation,
            "geometrically_nonempty_subsamples":sum(z!="NONE" for z in sides),
            "sample_level_compositing_before_raster_filter":True,
        },
    }


def premul_source(x):
    p=np.asarray(x,dtype=np.float64)
    require(p.shape==(4,) and np.isfinite(p).all()
            and -1e-5<=p[3]<=1+1e-5,
            "V18K_REAL_SOURCE_RGBA_INVALID")
    alpha=np.clip(p[3],0,1)
    return np.array([*(p[:3]*alpha),alpha],dtype=np.float64)


def compare_to_source(geometry,full,isolated,remaining):
    j=premul_source(full)
    g=premul_source(isolated)
    r=premul_source(remaining)
    actual_gr=over_premul(g,r)
    actual_rg=over_premul(r,g)
    dist=lambda a,b:float(np.max(np.abs(a-b)))
    original_best=min(dist(j,actual_gr),dist(j,actual_rg))
    base={
        "real_blender_v18h_full_source_premultiplied_rgba":j.tolist(),
        "real_blender_v18h_postfilter_best_of_two_source_orders_error":original_best,
        "real_blender_v18h_postfilter_order_invariant_alpha_error":
            float(abs(j[3]-actual_gr[3])),
        "real_blender_v18h_postfilter_both_orders_fail":bool(original_best>TOL),
        "no_artist_layer_identity":True,
    }
    if geometry["geometric_opaque_sample_composition"] is None:
        return dict(base,source_joint_sample_error=None,
                    geometry_postfilter_best_order_error=None,
                    geometric_joint_delta_from_geometric_postfilter=None,
                    sample_reconstruction_improves_actual_blender_postfilter=None)
    x=geometry["geometric_opaque_sample_composition"]
    joint=np.asarray(x["joint_pre_filter_premultiplied_rgba"])
    gr=np.asarray(x["postfilter_g_over_r_premultiplied_rgba"])
    rg=np.asarray(x["postfilter_r_over_g_premultiplied_rgba"])
    sample_error=dist(j,joint)
    return dict(base,
        source_joint_sample_error=sample_error,
        geometry_postfilter_best_order_error=min(dist(j,gr),dist(j,rg)),
        geometric_joint_delta_from_geometric_postfilter=
            x["joint_vs_best_postfilter_order_max_rgba_delta"],
        sample_reconstruction_improves_actual_blender_postfilter=bool(
            sample_error+1e-12<original_best),
        sample_evidence_not_exact_cycles_filter=True)


def measure(v14_job,teacher,camera,rest,stain_npz,
            v18h_manifest,v18h_rgba,v18h_court,v18j_court,
            expected_v18h_court_sha,expected_v18j_court_sha):
    require(all(isinstance(s,str) and len(s)==64 for s in
       (expected_v18h_court_sha,expected_v18j_court_sha))
       and sha(v18h_court)==expected_v18h_court_sha
       and sha(v18j_court)==expected_v18j_court_sha,
       "V18K_PARENT_INDEPENDENT_COURT_SHA_DRIFT")
    job=load(v14_job)
    manifest=load(v18h_manifest)
    hc=load(v18h_court)
    jc=load(v18j_court)
    cb=load(camera)
    require(job.get("schema")==
        "RealSaS.Corpus370SourceNativeMechanicalRasterJointJob.v14"
        and job.get("teacher_only") is True
        and job.get("training_authorized") is False
        and job.get("graphics_owner_authorized") is False
        and job.get("view_indices")==list(VIEWS)
        and job.get("raster_resolution")==RESOLUTION
        and len(job.get("pose_indices",[]))==3
        and all(sha(path)==job[key]["sha256"] for path,key in (
            (teacher,"native_teacher"),(camera,"camera"),
            (rest,"rest_geometry")))
        and manifest.get("schema")==V18H_RENDER
        and manifest.get("status")==
            "V18H_ORIGINAL_EMISSIVE_COMPLEMENT_RENDERED__INDEPENDENT_COURT_REQUIRED"
        and manifest.get("subject_id")==job["subject_id"]
        and manifest.get("original_source_sha256")==job["source"]["sha256"]
        and manifest.get("source_original_v14_job_sha256")==sha(v14_job)
        and manifest.get("original_source_teacher_vertices_sha256")==sha(teacher)
        and manifest.get("original_source_camera_sha256")==sha(camera)
        and manifest.get("source_reactive_stain_npz_sha256")==sha(stain_npz)
        and manifest.get("new_source_actual_edited_rgba_npz_sha256")==sha(v18h_rgba)
        and manifest.get("source_edit_arms")==arms()
        and manifest.get("heldout_original_authored_pose_ordinal")==POSE_ORDINAL
        and manifest.get("heldout_source_camera_views")==list(VIEWS)
        and manifest.get("frozen_tested_stain_similarity_thresholds")==list(THRESHOLDS)
        and manifest.get("source_material_policy")==
            "FIXED_TWO_COLOR_OPAQUE_EMISSION_NO_MATERIAL_SHADING"
        and manifest.get("diagnostic_emission_color_group_rgb")==list(RED)
        and manifest.get("diagnostic_emission_color_complement_rgb")==list(BLUE)
        and manifest.get("source_material_label_authority")=="NONE"
        and manifest.get("physical_face_visibility_operator")==
            "ACTUAL_TRIANGLE_REMOVAL_AND_OBJECT_RENDER_HIDE"
        and hc.get("schema")==V18H_COURT and hc.get("status")==V18H_STATUS
        and hc.get("original_source_sha256")==job["source"]["sha256"]
        and hc.get("source_subject_id_private")==job["subject_id"]
        and hc.get("exact_source_v18h_render_manifest_sha256")==sha(v18h_manifest)
        and hc.get("exact_source_v18h_render_rgba_npz_sha256")==sha(v18h_rgba)
        and jc.get("schema")==V18J_COURT
        and jc.get("status")==V18J_PASS
        and jc.get("original_source_sha256")==job["source"]["sha256"]
        and jc.get("original_source_subject_id_private")==job["subject_id"]
        and jc.get("source_v14_teacher_sha256")==sha(teacher)
        and jc.get("source_v14_camera_sha256")==sha(camera)
        and jc.get("source_v14_rest_sha256")==sha(rest)
        and jc.get("source_v18h_rgba_sha256")==sha(v18h_rgba)
        and jc.get("complete_private_source_ray_reread") is True
        and all(d.get(k) is False for d in (job,manifest,hc,jc) for k in
            ("graphics_owner_authorized","training_authorized")),
       "V18K_EXACT_ORIGINAL_SOURCE_GEOMETRY_RENDER_PARENT_BINDING")
    with np.load(teacher,allow_pickle=False) as z:
        faces=np.asarray(z["faces"],np.int64)
        poses=np.asarray(z["posed_vertices_source_world"],np.float64)
    with np.load(rest,allow_pickle=False) as z:
        rest_faces=np.asarray(z["faces"],np.int64)
        rest_vertices=np.asarray(z["rest_vertices_source_world"],np.float64)
    with np.load(stain_npz,allow_pickle=False) as z:
        require(set(z.files)=={"faces","stain_channels","graph_edges",
            "selected_source_control_axes_teacher_only"}|
            {"labels_"+str(t).replace(".","p") for t in THRESHOLDS},
            "V18K_FROZEN_V18A_ORIGINAL_SOURCE_ARRAYS")
        stain_faces=np.asarray(z["faces"],np.int64)
        labels={t:np.asarray(z["labels_"+str(t).replace(".","p")],np.int32)
                for t in THRESHOLDS}
    with np.load(v18h_rgba,allow_pickle=False) as z:
        require(z.files==["emissive_complement_rgba"],
                "V18K_ORIGINAL_V18H_RGB_MEMBER")
        rgb=np.asarray(z["emissive_complement_rgba"],np.float32)
    require(np.array_equal(faces,rest_faces)
        and np.array_equal(faces,stain_faces)
        and rgb.shape==(4,10,RESOLUTION,RESOLUTION,4)
        and np.isfinite(rgb).all()
        and poses.ndim==3 and poses.shape[1:]==rest_vertices.shape
        and len(poses)>max(job["pose_indices"])
        and all(x.shape==(len(faces),) and np.all(x>=0)
                for x in labels.values())
        and cb.get("schema")=="RealSaS.CameraProjectionBundle.v1"
        and len(cb.get("cameras",[]))==8,
        "V18K_GEOMETRY_SOURCE_VIEW_AND_RGBA_INCOMPATIBLE")
    xyz=poses[job["pose_indices"][POSE_ORDINAL]]
    cams={int(z["view_index"]):camera_from_mapping(dict(z,resolution=192))
          for z in cb["cameras"]}
    require(set(cams)==set(range(8)),"V18K_SOURCE_CAMERA_COUNT")
    face_anchor=int(manifest["source_face_probe_target_ordinals_teacher_only"][0])
    require(0<=face_anchor<len(faces),"V18K_ANCHOR_SOURCE_FACE_INVALID")
    tie=max(1e-9,1e-7*max(1e-12,float(
        np.linalg.norm(np.ptp(xyz,axis=0)))))
    index={name:i for i,name in enumerate(arms())}
    raw_cache={}
    rows=[]
    summaries={}
    for t in THRESHOLDS:
        gid=int(labels[t][face_anchor])
        gsize=int(np.count_nonzero(labels[t]==gid))
        candidate_groups=int(np.unique(labels[t]).size)
        for vi,view in enumerate(VIEWS):
            pixels=proposed_pixels(xyz,faces,cams[view],labels[t],gid)
            area="LOCKED_EVALUATION_VIEW" if view in HELDOUT_VIEWS else "EXPLORATORY_VIEW"
            key=str(t).replace(".","p")
            full=rgb[vi,index[f"EMISSIVE_{key}_FULL"]]
            iso=rgb[vi,index[f"EMISSIVE_{key}_ISOLATE"]]
            rem=rgb[vi,index[f"EMISSIVE_{key}_REMOVE"]]
            for x,top_y in pixels:
                ray_key=(int(view),int(x),int(top_y))
                if ray_key not in raw_cache:
                    raw_cache[ray_key]=sample_geometry(
                        xyz,faces,cams[view],int(x),int(top_y),tie)
                geometry=render_sampled(raw_cache[ray_key],labels[t],gid)
                original_y=RESOLUTION-1-top_y
                comparison=compare_to_source(
                    geometry,full[original_y,x],iso[original_y,x],
                    rem[original_y,x])
                rows.append({
                    "source_camera_view":int(view),
                    "camera_evaluation_partition":area,
                    "source_pose_ordinal":POSE_ORDINAL,
                    "source_pixel_topdown":[int(x),int(top_y)],
                    "source_blender_framebuffer_pixel_bottomup":[int(x),int(original_y)],
                    "geometry_precommitted_selection":True,
                    "rgba_error_never_used_for_pixel_selection":True,
                    "frozen_v18a_threshold":float(t),
                    "source_group_ordinal_teacher_only":gid,
                    "source_group_face_count":gsize,
                    "whole_source_face_count":len(faces),
                    "proposed_group_count":candidate_groups,
                    "vacuous_whole_character_group":bool(gsize==len(faces)),
                    "geometry_subsample_compositor":geometry,
                    "actual_original_blender_source_comparison":comparison,
                    "source_graphic_drawable_owner":"UNKNOWN",
                })
            summaries[f"{view}:{t}"]={
                "geometry_preselected_pixels":len(pixels),
                "qualified_joint_sample_pixels":sum(
                    z["geometry_subsample_compositor"][
                        "geometric_opaque_sample_composition"] is not None
                    for z in rows if z["source_camera_view"]==view and
                    z["frozen_v18a_threshold"]==float(t)),
                "source_group_face_count":gsize,
                "whole_character_vacuous":bool(gsize==len(faces)),
                "evaluated_in":"HELDOUT_VIEWS_4_6_LOCKED" if
                    view in HELDOUT_VIEWS else "EXPLORATORY_VIEWS_0_2",
            }
    require(len(rows)==sum(x["geometry_preselected_pixels"]
                            for x in summaries.values())
        and len(raw_cache)>0,
        "V18K_SOURCE_GEOMETRIC_PIXEL_COUNTS_INCOMPLETE")
    return {
        "schema":SCHEMA,"status":STATUS,
        "source_subject_id_private":job["subject_id"],
        "original_source_sha256":job["source"]["sha256"],
        "source_original_v14_job_sha256":sha(v14_job),
        "source_original_teacher_sha256":sha(teacher),
        "source_original_rest_sha256":sha(rest),
        "source_camera_bundle_sha256":sha(camera),
        "frozen_original_v18a_groups_sha256":sha(stain_npz),
        "source_emissive_v18h_manifest_sha256":sha(v18h_manifest),
        "source_emissive_v18h_rgba_sha256":sha(v18h_rgba),
        "exact_v18h_independent_court_sha256":sha(v18h_court),
        "exact_v18j_independent_court_sha256":sha(v18j_court),
        "selection":"SOURCE_GEOMETRY_CENTROIDS_GROUP_AND_REMAINDER_PLUS_FIXED_NEIGHBORS",
        "no_rgba_error_in_pixel_selection":True,
        "sample_grid":"V17B_GEOMETRY_4X4_NOT_EXACT_CYCLES_32SPP",
        "camera_exploratory_views":list(EXPLORATORY_VIEWS),
        "camera_locked_evaluation_views":list(HELDOUT_VIEWS),
        "original_source_face_count":len(faces),
        "original_source_pose_ordinal":POSE_ORDINAL,
        "geometric_original_view_pixel_rays_unique":len(raw_cache)*SAMPLE_COUNT,
        "per_original_source_view_threshold":summaries,
        "original_source_pixel_rows_private_teacher_only":rows,
        "heldout_camera_policy_frozen_before_original_source_measurements":True,
        "geometrical_sample_filter_is_not_exact_cycles":True,
        "k1_perfect_reconstruction_is_vacuous":True,
        "source_opaque_emission_is_not_artist_painted_alpha":True,
        "artist_drawable_identity":"UNKNOWN",
        "operational_drawable_identity":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    p=argparse.ArgumentParser()
    for name in ("v14_job","teacher","camera","rest","stain_npz",
                 "v18h_manifest","v18h_rgba","v18h_court","v18j_court","output"):
        p.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    for name in ("expected_v18h_court_sha","expected_v18j_court_sha"):
        p.add_argument("--"+name.replace("_","-"),required=True)
    a=p.parse_args()
    result=measure(a.v14_job,a.teacher,a.camera,a.rest,a.stain_npz,
        a.v18h_manifest,a.v18h_rgba,a.v18h_court,a.v18j_court,
        a.expected_v18h_court_sha,a.expected_v18j_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                   allow_nan=False)+"\n")
    print(json.dumps({
        "status":result["status"],
        "source":result["source_subject_id_private"],
        "original_view_pixel_rows":len(result["original_source_pixel_rows_private_teacher_only"]),
        "source_rays_unique":result["geometric_original_view_pixel_rays_unique"],
        "training_authorized":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
