#!/usr/bin/env python3
"""Original Cat face→pixel→UV edit-causality source court producer.

Conservatively certifies an ENTIRE source pixel square (with safety guard)
is covered by ONE original indexed triangle and cannot have ANY closer
original triangle anywhere inside that square. Then it proves the full
square's original face-corner glTF UV address remains inside ONE predeclared
source edit tile. This is sufficient for all unknown jitter samples inside
the original pixel square; NOT an exact reconstruction of Cycles jitter.

Original physical source RGB edit effects are observed from immutable
independently source-courted V2 sample arrays, NOT from a new renderer.
The output is a *sufficient subset*, never negative coverage, artist owner,
drawables or an IRIS training target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.OriginalSourceFacePixelUVInterventionWitness.v1"
CAT="1e73aaf7d497dd7d2946badf92d189d2333f163e66dced91feb446bc9bdee850"
PINS={
    "source.glb":CAT,
    "teacher.npz":"c2792c8efb36c4ccdf9178c6b933351b50364bbd28707e9365a3c115db00d603",
    "rest.npz":"748609aca5ba7c509aa390ed65baeca7612b8879aa00409735da3a96758379cf",
    "face-uv.npz":"7f3afdc4f0f848de3aa06bda2220151427d352368a0f235ca09d5d96f7834787",
    "uv-binding.json":"b966a3793c682cb06ec70341e659aabbded9343ace3d645c08643de40f2799d7",
    "camera.json":"300b935a47379f6c44d4b4356554d1c32a86ca3b732a62405c73da9e0ec6adf0",
    "uv-plan.json":"8be8fccde3d9cc97f7ba452fd50c1a9f4ebea2e34da590a22d30872b2b0f497b",
    "physical.npz":"d422b7d7d082fba5deac5dd1239b8c6aec078f4d635a7bd5b1f10e514100b114",
    "physical-manifest.json":"8a56e47a09f5c1216de2c97e6f13d5e04385f7fe7d2e580d0085b2ab69dc2da1",
    "physical-court.json":"90b85b9c1a963b23f7c3037ca443975bf2d5c5d4123568d8aa8e54557e433c08",
}
VIEWS=(1,3,5,7)
CYCLES_SAMPLES=(0,4,8,12,16,20,24,28)
RES=192
PIXEL_GUARD=0.05
UV_TEXEL_GUARD=1.0/512
DEPTH_MARGIN=1e-5
BARY_MARGIN=1e-7
TOL=.025
STUDENT_FALSE={
    "graphics_owner_authorized":False,
    "iris_training_authorized":False,
    "tessa_training_authorized":False,
    "training_authorized":False,
    "materialization_authorized":False,
    "same_drawable_training_pairs_qualified":0,
    "different_drawable_training_pairs_qualified":0,
}


def require(ok,why):
    if not ok:raise ValueError("ORIGINAL_FACE_PIXEL_EDIT_"+why)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def project_original(vertices,faces,camera,*,resolution=RES):
    """Original source ortho frame → *top-first* image pixel coordinates."""
    v=np.asarray(vertices,np.float64)
    f=np.asarray(faces,np.int64)
    require(v.ndim==2 and v.shape[1]==3 and f.ndim==2 and f.shape[1]==3
            and f.min()>=0 and f.max()<len(v) and np.isfinite(v).all(),
            "SOURCE_ORIGINAL_TRIANGLE_DOMAIN")
    origin=np.asarray(camera["origin"],np.float64)
    right=np.asarray(camera["right"],np.float64)
    up=np.asarray(camera["screen_up"],np.float64)
    forward=np.asarray(camera["forward"],np.float64)
    basis=np.stack((right,up,forward))
    require(basis.shape==(3,3)
            and np.allclose(basis@basis.T,np.eye(3),rtol=0,atol=3e-5)
            and camera["view_index"] in range(8)
            and camera["schema_version"]=="RealSaS.FullSurfaceCameraProjection.v3",
            "SOURCE_CAMERA_ORTHONORMAL_FRAME")
    half=float(camera["half_extent"])
    require(half>0 and camera["clip_start"]>0 and
            camera["clip_end"]>camera["clip_start"],"SOURCE_CAMERA_CLIP_INVALID")
    xyz=v[f]-origin
    plane=2*half
    coords=np.stack((
        (0.5+np.sum(xyz*right,axis=-1)/plane)*resolution,
        (0.5-np.sum(xyz*up,axis=-1)/plane)*resolution,
    ),axis=-1)
    depth=np.sum(xyz*forward,axis=-1)
    return coords,depth


def guarded_corners(x,y,guard=PIXEL_GUARD):
    return np.asarray([
        (x-guard,y-guard),(x+1+guard,y-guard),
        (x-guard,y+1+guard),(x+1+guard,y+1+guard)
    ],np.float64)


def triangle_corner_barycentric(triangle,corners):
    p=np.asarray(triangle,np.float64)
    c=np.asarray(corners,np.float64)
    ab=p[1]-p[0];ac=p[2]-p[0]
    det=ab[0]*ac[1]-ab[1]*ac[0]
    if abs(det)<1e-10:
        return None
    rel=c-p[0]
    b=(rel[...,0]*ac[1]-rel[...,1]*ac[0])/det
    d=(ab[0]*rel[...,1]-ab[1]*rel[...,0])/det
    return np.stack((1-b-d,b,d),axis=-1)


def box(tri,resolution=RES):
    p=np.asarray(tri,np.float64)
    return (
        max(0,int(np.floor(p[:,0].min()))-1),
        min(resolution,int(np.ceil(p[:,0].max()))+1),
        max(0,int(np.floor(p[:,1].min()))-1),
        min(resolution,int(np.ceil(p[:,1].max()))+1),
    )


def depth_envelopes(projected_triangles,depths,*,resolution=RES):
    """Conservative lower bound for ALL possible projected occluders.

    For every pixel, compute minimum source depth over all triangle vertices
    of each projected AABB-overlapping triangle. Triangles not actually
    intersecting pixel can spuriously block qualification; NEVER certify an
    occluded face. Keep top 2 minima and owner index to exclude candidate.
    """
    best=np.full((resolution,resolution),np.inf,np.float64)
    second=np.full_like(best,np.inf)
    owner=np.full((resolution,resolution),-1,np.int64)
    boxes=[]
    for fi,(tri,d) in enumerate(zip(projected_triangles,depths)):
        xmin,xmax,ymin,ymax=box(tri,resolution)
        boxes.append((xmin,xmax,ymin,ymax))
        if xmin>=xmax or ymin>=ymax:
            continue
        new=float(d.min())
        first=best[ymin:ymax,xmin:xmax]
        other=second[ymin:ymax,xmin:xmax]
        assigned=owner[ymin:ymax,xmin:xmax]
        prev=first.copy()
        closer=new<prev
        np.minimum(other,np.where(closer,prev,new),out=other)
        np.minimum(first,new,out=first)
        assigned[closer]=fi
    return best,second,owner,boxes


def certified_face_uv_pixel_rows(vertices,faces,face_uv,camera,uv_edit_rectangles,
                                 *,resolution=RES):
    """Sound sufficient source-face/edit region witnesses, not complete rays."""
    tri,depth=project_original(vertices,faces,camera,resolution=resolution)
    uv=np.asarray(face_uv,np.float64)
    require(uv.shape==(len(faces),3,2) and np.isfinite(uv).all(),
            "GLTF_SOURCE_FACE_CORNER_UV_DOMAIN")
    require(len(uv_edit_rectangles)>=1 and
            all(len(rect)==4 and 0<=rect[0]<rect[2]<=512 and
                0<=rect[1]<rect[3]<=512 for rect in uv_edit_rectangles),
            "SOURCE_EDIT_TEXTURE_RECT_INVALID")
    _,second,owner,boxes=depth_envelopes(tri,depth,resolution=resolution)
    rows=[]
    census={"whole_pixel_triangle_coverage":0,
            "whole_pixel_unique_nearest_source_face":0}
    for fi,(points,z,uvtri,bbox) in enumerate(zip(tri,depth,uv,boxes)):
        xmin,xmax,ymin,ymax=bbox
        if xmin>=xmax or ymin>=ymax:
            continue
        xs,ys=np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
        corners=np.stack([
            np.stack((xs-PIXEL_GUARD,ys-PIXEL_GUARD),axis=-1),
            np.stack((xs+1+PIXEL_GUARD,ys-PIXEL_GUARD),axis=-1),
            np.stack((xs-PIXEL_GUARD,ys+1+PIXEL_GUARD),axis=-1),
            np.stack((xs+1+PIXEL_GUARD,ys+1+PIXEL_GUARD),axis=-1),
        ],axis=-2)
        bary=triangle_corner_barycentric(points,corners)
        if bary is None:
            continue
        full_pixel=np.all(bary>=BARY_MARGIN,axis=(-2,-1))
        census["whole_pixel_triangle_coverage"]+=int(np.count_nonzero(full_pixel))
        if not np.any(full_pixel):
            continue
        z_four=np.sum(bary*z[None,None,None,:],axis=-1)
        maxz=np.max(z_four,axis=-1)
        minz=np.min(z_four,axis=-1)
        first=(full_pixel & (owner[ymin:ymax,xmin:xmax]==fi)
               & (maxz+DEPTH_MARGIN < second[ymin:ymax,xmin:xmax])
               & (minz>float(camera["clip_start"])+DEPTH_MARGIN)
               & (maxz<float(camera["clip_end"])-DEPTH_MARGIN))
        census["whole_pixel_unique_nearest_source_face"]+=int(np.count_nonzero(first))
        if not np.any(first):
            continue
        uv_four=np.sum(bary[...,None]*uvtri[None,None,None,:,:],axis=-2)
        for operation_index,rect in enumerate(uv_edit_rectangles):
            lo=np.asarray(rect[:2],np.float64)/512
            hi=np.asarray(rect[2:],np.float64)/512
            inside=(np.all(uv_four>lo+UV_TEXEL_GUARD,axis=(-2,-1))
                    & np.all(uv_four<hi-UV_TEXEL_GUARD,axis=(-2,-1)))
            qualified=first&inside
            yy,xx=np.where(qualified)
            rows.extend({
                "face_index":int(fi),
                "pixel_y_top_first":int(ymin+y),
                "pixel_x":int(xmin+x),
                "source_uv_edit_operation_index":int(operation_index)
            } for y,x in zip(yy,xx))
    require(len(rows)==len({(row["pixel_y_top_first"],row["pixel_x"])
                            for row in rows}),
            "DUPLICATE_OR_AMBIGUOUS_TOP_FIRST_PIXEL")
    rows.sort(key=lambda r:(r["pixel_y_top_first"],r["pixel_x"]))
    return rows,census


def physically_court_witness(rows,original_samples,*,view_ordinal):
    """Observed same-Cycles sample physical effects for certified face pixels."""
    img=np.asarray(original_samples,np.float32)
    require(img.shape==(8,8,RES,RES,4) and np.isfinite(img).all(),
            "SOURCE_PHYSICAL_RGBA_SHAPE_INVALID")
    if not rows:
        return {"face_pixels":0,"independently_edited_sample_incidents":0,
                "absent_other_edit_effect_obligations":0,
                "alpha_violations":0,"effect_violations":0,"source_face_count":0}
    ys=np.asarray([r["pixel_y_top_first"] for r in rows],np.int64)
    xs=np.asarray([r["pixel_x"] for r in rows],np.int64)
    edit=np.asarray([r["source_uv_edit_operation_index"] for r in rows],np.int64)
    require(np.all((ys>=0)&(ys<RES)&(xs>=0)&(xs<RES))
            and np.all((edit>=0)&(edit<3)),"SOURCE_PIXEL_OR_OPERATION_OUT_OF_RANGE")
    # Blender EXR Image.pixels.foreach_get rows are BOTTOM-FIRST.
    sample=img[:,:,RES-1-ys,xs].astype(np.float64)
    sample[...,:3]*=sample[...,3:4]   # linear premultiplied RGBA.
    base=sample[:,0]                  # [SAMPLE,ROW,RGBA]
    changed=np.max(np.abs(sample[:,1:4]-base[:,None]),axis=-1) # [8,3,ROW]
    mask=np.arange(3)[None,:,None]==edit[None,None,:]
    positives=int(np.count_nonzero((changed>TOL)&mask))
    negatives=int(np.count_nonzero((changed>TOL)&~mask))
    missing=int(np.count_nonzero((changed<=TOL)&mask))
    alpha_invalid=int(np.count_nonzero(np.abs(base[...,3]-1)>TOL))
    rows_per_op=[int(np.count_nonzero(edit==i)) for i in range(3)]
    require(negatives==0 and missing==0 and alpha_invalid==0,
            "PHYSICAL_SOURCE_FACE_PIXEL_INTERVENTION_CONTRADICTION")
    return {
        "face_pixels":len(rows),
        "original_face_count":len({r["face_index"] for r in rows}),
        "source_uv_edit_pixel_counts":rows_per_op,
        "independently_edited_sample_incidents":positives,
        "absent_other_edit_effect_obligations":int(np.count_nonzero(~mask)*8),
        "source_expected_edit_sample_absence_count":missing,
        "source_other_edit_sample_false_positive_count":negatives,
        "source_opaque_base_sample_violations":alpha_invalid,
        "physical_sample_per_pixel":8,
        "source_original_view_ordinal":view_ordinal,
    }


def produce(source_root:Path,output:Path):
    root=Path(source_root)
    for name,digest in PINS.items():
        require(sha(root/name)==digest,"SOURCE_SHA_PIN_DRIFT:"+name)
    j=lambda f:json.loads((root/f).read_text())
    manifest=j("physical-manifest.json")
    court=j("physical-court.json")
    bind=j("uv-binding.json")
    plan=j("uv-plan.json")
    camera=j("camera.json")
    require(manifest.get("schema")==
            "RealSaS.Corpus370OriginalCatSourceTextureSameCyclesEditEvidence.v2"
            and court.get("status")==
            "PASS_ORIGINAL_CAT_LINKED_UV_EDIT_CONSEQUENCE_SUPERPOSITION__OWNER_UNKNOWN"
            and court.get("failed_obligations")==[]
            and bind.get("source_to_teacher_face_uv_binding")==
            "VERIFIED_NATIVE_BLENDER_FACE_ORDER_AND_RAW_GLTF_UV_MULTISET"
            and bind.get("source_uv_comparison",{}).get("matched") is True
            and bind.get("original_glb_sha256")==CAT
            and bind.get("original_teacher_sha256")==PINS["teacher.npz"]
            and bind.get("face_uv_array_sha256")==PINS["face-uv.npz"] or
            False,
            "SOURCE_FAMILY_IDENTITY_UNVERIFIED")
    require(manifest.get("original_source_sha256")==CAT
            and manifest.get("original_camera_bundle_sha256")==PINS["camera.json"]
            and manifest.get("original_source_uv_edit_plan_sha256")==PINS["uv-plan.json"]
            and manifest.get("source_original_sample_npz_sha256")==PINS["physical.npz"]
            and manifest.get("camera_indices")==list(VIEWS)
            and manifest.get("cycles_subset_indices")==list(CYCLES_SAMPLES)
            and manifest.get("original_source_rest_only") is True
            and manifest.get("cycles_seed")==0
            and manifest.get("cycles_subset_length")==1
            and manifest.get("cycles_pixel_filter")=="BOX"
            and manifest.get("cycles_filter_width")==.01
            and manifest.get("cycles_resolution")==RES
            and plan.get("source_original_glb_sha256")==CAT
            and plan.get("source_uv_coordinate_origin")==
              "GLTF_2_IMAGE_TOP_LEFT__UV_V_INCREASES_DOWN"
            and plan.get("source_original_camera_sha256")==PINS["camera.json"]
            and plan.get("original_source_uv_operation_count")==3
            and plan.get("source_texture_source_image_alpha_all_opaque") is True
            and court.get("source_sampler_tolerance_rgba_premultiplied_max")==TOL
            and all(z.get(k) is False for z in (manifest,court,plan,bind)
                    for k in ("graphics_owner_authorized",
                              "iris_training_authorized",
                              "materialization_authorized","training_authorized")),
            "ORIGINAL_SOURCE_PHYSICAL_SCOPE_OR_NONOWNER_INVALID")
    with np.load(root/"teacher.npz",allow_pickle=False) as t, \
         np.load(root/"rest.npz",allow_pickle=False) as s, \
         np.load(root/"face-uv.npz",allow_pickle=False) as u, \
         np.load(root/"physical.npz",allow_pickle=False) as samples:
        xyz=np.asarray(t["rest_vertices_source_world"],np.float64)
        faces=np.asarray(t["faces"],np.int64)
        uv=np.asarray(u["face_uv"],np.float64)
        require(np.array_equal(faces,s["faces"])
                and np.array_equal(xyz,np.asarray(s["rest_vertices_source_world"],np.float64))
                and faces.shape==(684,3) and xyz.shape==(1204,3),
                "EXACT_NATIVE_REST_FACE_ORDER_OR_POSITION_MISMATCH")
        original=np.asarray(samples["source_uv_edit_cycles_sample_rgba"],np.float32)
    require(original.shape==(4,8,8,RES,RES,4),
            "SOURCE_CYCLES_ORIGINAL_SAMPLE_ARRAY_INVALID")
    edits=plan["selected_original_source_uv_edits"]
    require([x["source_uv_edit_operation_index"] for x in edits]==list(range(3)),
            "SOURCE_ORIGINAL_EDIT_ORDER")
    rectangles=[tuple(r["source_rgba_png_pixel_rectangle_xyxy_exclusive"]) for r in edits]
    cameras=camera["cameras"]
    result_rows=[]
    per_view=[]
    for ix,view in enumerate(VIEWS):
        require(cameras[view]["view_index"]==view,
                "ORIGINAL_CAMERA_ORDER")
        rows,geom=certified_face_uv_pixel_rows(xyz,faces,uv,cameras[view],rectangles)
        outcome=physically_court_witness(rows,original[ix],view_ordinal=view)
        per_view.append({**outcome,**geom})
        result_rows.extend([{"original_camera_view":view,**r} for r in rows])
    counts={
        "qualified_original_face_pixel_witness_count":len(result_rows),
        "qualified_original_source_sample_face_edit_effect_count":
            sum(v["independently_edited_sample_incidents"] for v in per_view),
        "qualified_original_source_other_edit_absence_obligations":
            sum(v["absent_other_edit_effect_obligations"] for v in per_view),
        "qualified_original_source_face_count":
            len({r["face_index"] for r in result_rows}),
        "source_pixels_by_edit":[sum(v["source_uv_edit_pixel_counts"][i]
                                      for v in per_view) for i in range(3)],
    }
    require(counts["qualified_original_face_pixel_witness_count"]>0,
            "NO_SOURCE_FACE_EDIT_WITNESSES")
    report={
        "schema":SCHEMA,
        "status":"ORIGINAL_SOURCE_FACE_TO_PIXEL_TO_UV_EDIT_CONSERVATIVE_WITNESSES__INDEPENDENT_COURT_REQUIRED",
        "source_original_glb_sha256":CAT,
        "original_parent_bytes_sha256":PINS,
        "original_face_count":len(faces),
        "original_camera_views":list(VIEWS),
        "source_pixel_sample_offsets":list(CYCLES_SAMPLES),
        "source_pixel_origin":"TOP_FIRST_GEOMETRY_VS_BOTTOM_FIRST_ORIGINAL_BLENDER_EXR",
        "whole_pixel_guard_px":PIXEL_GUARD,
        "original_source_uv_interior_guard_texels":1,
        "all_other_faces_conservative_depth_envelope":"PROJECTED_TRIANGLE_AABB_OVERLAP_AND_GLOBAL_TRIANGLE_MIN_DEPTH",
        "first_visible_original_triangle":"SUFFICIENT_ENTIRE_PIXEL_PROOF_ONLY__NOT_COMPLETE",
        "source_observation_reuses_original_256_frames":True,
        "re_rendered_source_frames":0,
        "original_physical_source_opaque_alpha_observation":True,
        "per_source_camera":per_view,
        "counts":counts,
        "source_original_face_pixel_witness_rows_PRIVATE":result_rows,
        "source_artist_layer_or_drawable_owner":"UNKNOWN",
        "source_unwitnessed_faces_and_uv_edits":"UNKNOWN__NOT_FALSE",
        "source_exact_cycles_random_ray_offsets":"UNKNOWN__INTERIOR_CONSERVATIVE_NO_NEED_TO_INFER",
        "source_face_to_pixel_entire_observed_screen":"NOT_PROVEN",
        **STUDENT_FALSE,
    }
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],"counts":counts,
        "source_owner":"UNKNOWN","training_authorized":False
    },sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    produce(args.source_root,args.output)


if __name__=="__main__":
    main()
