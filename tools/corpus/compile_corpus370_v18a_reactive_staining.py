#!/usr/bin/env python3
"""V18a source-native reactive-staining *candidate* graphics layers.

A virtual "dye" is the anonymous response fingerprint of SOURCE TRIANGLES to
real original Blender-controlled +/- perturbations. It is not a generated art
texture or pretrained semantic segmentation. All source rigs, controls, face
ids and visual previews remain private teacher instruments, never student
input or human-authored drawable targets.

Fit: ORIGINAL authored poses 0 and 1, original V13b native control +/- effects
plus V16 actual nonlinear A+B geometry. Heldout: pose 2 native effects; V17
actual original RGBA/ray evidence is READ ONLY AFTER proposals are sealed.
Precommitted variable-K geometry-adjacent thresholds, no target artist part K.
The result is a testable candidate, never an artist ownership PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.measure_corpus370_v17_posed_source_ray_to_causal_rgba_v1 import (
    SIGNS,VIEWS,validate_and_open,require
)
from tools.corpus.corpus370_native_raster_join_plan_v14 import rigid_rms

SCHEMA="RealSaS.Corpus370NativeReactiveStainCandidate.v18a"
STATUS="ORIGINAL_SOURCE_REACTIVE_STAINING_MECHANICAL_CANDIDATES__HELDOUT_EDITOR_COURT_PENDING"
V17_PUBLIC="RealSaS.Corpus370V17OriginalPosedSourceRayToCausalRGBAIndependentRealResult.v1"
V17_PROPOSAL="RealSaS.Corpus370OriginalPosedSourceRayToCausalRGBAJoinProposal.v17"
V17_COURT="RealSaS.Corpus370IndependentOriginalPosedSourceRayCausalRGBAJoinCourt.v17"
BUDGET=12
MAX_CHANNEL=1.
THRESHOLDS=(.045,.10,.18)
TRAIN_POSES=(0,1)
HELDOUT_POSE=2
SOURCE_RESPONSE_FLOOR=2e-4


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def shape_check(rest,faces,baselines,controls,joint):
    require(rest.ndim==2 and rest.shape[1]==3
        and faces.ndim==2 and faces.shape[1]==3
        and len(faces)>8 and faces.min()>=0 and faces.max()<len(rest)
        and baselines.shape==(3,len(rest),3)
        and controls.ndim==6
        and controls.shape[0]==3 and controls.shape[2:4]==(3,2)
        and controls.shape[4:]==(len(rest),3)
        and joint.shape==(3,4,4,len(rest),3)
        and np.isfinite(rest).all() and np.isfinite(baselines).all()
        and np.isfinite(controls).all() and np.isfinite(joint).all(),
        "V18A_ACTUAL_SOURCE_VERTEX_FACE_AND_INTERVENTION_SHAPES")


def mean_face_magnitude(delta,faces):
    """Magnitude of a signed ORIGINAL source vertex effect, never an owner."""
    v=np.linalg.norm(np.asarray(delta,np.float64),axis=-1)
    return v[np.asarray(faces,np.int64)].mean(axis=1)


def intrinsic_rigid_residual(reference,altered):
    """Remove best-fitting global 3D rigid motion to expose *local* actuation.

    Global nonzero motion remains in a separate channel: a character's
    root motion is valid, just not a reliable local layer discriminator.
    """
    x=np.asarray(reference,np.float64)
    y=np.asarray(altered,np.float64)
    require(x.shape==y.shape and x.ndim==2 and x.shape[1]==3
            and len(x)>=3,"V18A_SOURCE_RIGID_SUBTRACTION_SHAPE")
    xa=x-x.mean(axis=0);ya=y-y.mean(axis=0)
    u,_,vt=np.linalg.svd(xa.T@ya)
    parity=-1. if np.linalg.det(vt.T@u.T)<0 else 1.
    r=vt.T@np.diag([1.,1.,parity])@u.T
    rigid=(xa@r.T)+y.mean(axis=0)
    return np.asarray(y-rigid,np.float64)


def select_blind_control_axes(base,interventions,faces,extent,budget=BUDGET):
    """Fit-pose-0 source geometry only; no names, heldout pose or edit RGBA."""
    require(budget>=2,"V18A_CONTROL_BUDGET_INVALID")
    threshold=SOURCE_RESPONSE_FLOOR*extent
    rows=[]
    for ci in range(interventions.shape[1]):
        for ax in range(3):
            plus=mean_face_magnitude(interventions[0,ci,ax,0]-base[0],faces)
            minus=mean_face_magnitude(interventions[0,ci,ax,1]-base[0],faces)
            score=np.maximum(plus,minus)
            active=score>threshold
            if not np.any(active):
                continue
            # Stable geometric support, no anatomical category.
            rows.append((float(np.quantile(score,.9)),
                         int(active.sum()),ci,ax,active))
    rows.sort(key=lambda r:(-r[0],-r[1],r[2],r[3]))
    picks=[]
    for peak,area,ci,ax,active in rows:
        if len(picks)>=budget:break
        # Do not spend all controls probing essentially the same native
        # support, but still permit identical controls if no diversity exists.
        if any(np.count_nonzero(active & p[2])/
               max(1,np.count_nonzero(active | p[2]))>.98
               for p in picks):
            continue
        picks.append((ci,ax,active))
    if len(picks)<2:
        for peak,area,ci,ax,active in rows:
            if len(picks)>=min(budget,len(rows)):break
            if (ci,ax) not in {(p[0],p[1]) for p in picks}:
                picks.append((ci,ax,active))
    require(len(picks)>=2,"V18A_NATIVE_SOURCE_INDEPENDENT_CONTROL_ABSTAIN")
    return [(int(ci),int(ax)) for ci,ax,_ in picks]


def normalize_channels(features):
    """Per-channel precommitted robust normalization, no labels/outcomes."""
    x=np.asarray(features,np.float64)
    require(x.ndim==2 and len(x)>0 and x.shape[1]>0
            and np.isfinite(x).all(),"V18A_STAIN_CHANNELS_NONFINITE")
    scale=np.quantile(np.abs(x),.95,axis=0)
    scale=np.maximum(scale,1e-7)
    return np.asarray(np.clip(x/scale,-MAX_CHANNEL,MAX_CHANNEL),np.float32)


def face_stain_features(rest,faces,base,controls,joint,selection,fit_poses=TRAIN_POSES):
    """Per-face multiple reactive dyes: signed motion, local strain, AB coupling."""
    extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
    out=[]
    for ci,axis in selection:
        for pose in fit_poses:
            for sign in (0,1):
                moved=controls[pose,ci,axis,sign]
                disp=(moved-base[pose])/extent
                residual=intrinsic_rigid_residual(base[pose],moved)/extent
                # Keep direction as well as magnitude: rigid co-motion alone
                # must not force same graphic layer.
                signed_vec=disp[faces].mean(axis=1)
                norm=mean_face_magnitude(disp,faces)
                local=mean_face_magnitude(residual,faces)
                out.extend([signed_vec[:,i] for i in range(3)])
                out.extend((norm,local))
    # Nonlinear source joint response is *measured* V16 original Blender
    # geometry, not an additive synthetic source prediction.
    for pose in fit_poses:
        for si,sign in enumerate(SIGNS):
            s0,s1=sign.split("_")
            selected0,selected1=joint_selection_indices
            i0=0 if s0=="PLUS" else 1
            i1=0 if s1=="PLUS" else 1
            a=controls[pose,selected0[0],selected0[1],i0]
            b=controls[pose,selected1[0],selected1[1],i1]
            nonlinear=(joint[pose,0,si]-a-b+base[pose])/extent
            out.append(mean_face_magnitude(nonlinear,faces))
    return normalize_channels(np.stack(out,axis=1))


# This is set per verified source by run(), after it reads original V14 blind
# control plan. It is deliberately PRIVATE and never controls artist labels.
joint_selection_indices=((0,0),(0,1))


def face_edges(faces):
    """Undirected shared *edge* only; vertex-touching art stays unmerged."""
    faces=np.asarray(faces,np.int64)
    claims={}
    for f,(a,b,c) in enumerate(faces):
        for i,j in ((a,b),(b,c),(c,a)):
            key=(min(int(i),int(j)),max(int(i),int(j)))
            claims.setdefault(key,[]).append(f)
    pairs=set()
    for hits in claims.values():
        # Nonmanifold 3+ face edge requires abstention from automatic merge.
        if len(hits)==2 and hits[0]!=hits[1]:
            pairs.add((min(hits),max(hits)))
    return np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)


def component_labels(features,edges,threshold):
    """No fixed K and no source-part count: graph-local stain similarity."""
    feats=np.asarray(features,np.float32)
    require(0<threshold<1 and feats.ndim==2 and np.isfinite(feats).all()
            and edges.ndim==2 and edges.shape[1]==2,
            "V18A_COMPONENT_SIMILARITY_PARAMETERS")
    count=len(feats)
    parent=np.arange(count,dtype=np.int64)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    if len(edges):
        delta=feats[edges[:,0]]-feats[edges[:,1]]
        distance=np.sqrt(np.mean(np.square(delta,dtype=np.float64),axis=1))
        for (a,b),dist in zip(edges,distance):
            if dist<=threshold:
                a=root(int(a));b=root(int(b))
                if a!=b: parent[max(a,b)]=min(a,b)
    labels=np.asarray([root(i) for i in range(count)],np.int64)
    _,inv=np.unique(labels,return_inverse=True)
    return inv.astype(np.int32)


def heldout_response_error(faces,base,controls,selection,labels,extent):
    """Unseen pose 2 oracle, only for subsequent *evaluation*.

    Counts how much actual independently measured source motion varies within
    components predicted from original authored poses 0+1. An artificially
    fragmented face-level oracle can score 0; ALWAYS report component count
    and largest/smallest size; no unpenalized 'best model' claim.
    """
    held=[]
    for ci,axis in selection:
        for sign in (0,1):
            disp=(controls[HELDOUT_POSE,ci,axis,sign]-
                  base[HELDOUT_POSE])/extent
            held.append(mean_face_magnitude(disp,faces))
    x=np.stack(held,axis=1).astype(np.float64)
    global_var=np.sum((x-x.mean(axis=0))**2)
    groups=np.unique(labels)
    residual=0.
    for k in groups:
        sub=x[labels==k]
        residual+=float(np.sum((sub-sub.mean(axis=0))**2))
    return float(residual/global_var) if global_var>1e-15 else None


def rasterize_debug_stain(faces,world_vertices,camera_row,labels,original_rgba,path):
    """Private diagnostic ONLY: triangle painter sorting, not source CAA."""
    from PIL import Image,ImageDraw
    from models.iris.truth.graphics_oracle_v1 import camera_from_mapping
    cam=camera_from_mapping(dict(camera_row,resolution=192))
    verts=np.asarray(world_vertices,np.float64)
    a=verts-np.asarray(cam.origin)
    sx=(np.dot(a,np.asarray(cam.right))/cam.half_extent+1)*.5*192
    sy=(1.-np.dot(a,np.asarray(cam.screen_up))/cam.half_extent)*.5*192
    depth=np.dot(a,np.asarray(cam.forward))
    canvas=Image.new("RGB",(192,192),(12,17,28))
    draw=ImageDraw.Draw(canvas)
    order=np.argsort(depth[faces].mean(axis=1))[::-1]
    for fi in order:
        k=int(labels[fi])
        h=hashlib.sha256(str(k).encode()).digest()
        col=(64+h[0]*3//4,65+h[1]*3//4,65+h[2]*3//4)
        poly=[(float(sx[j]),float(sy[j])) for j in faces[fi]]
        draw.polygon(poly,fill=col)
    rgba=np.asarray(original_rgba,np.float32)
    require(rgba.shape==(192,192,4),"V18A_DEBUG_SOURCE_RGBA_DIM")
    base=np.flipud(rgba.copy())
    color=np.clip(base[...,:3]*base[...,3:]+(.045)*(1-base[...,3:]),0,1)
    original=Image.fromarray(np.uint8(np.rint(color*255)),"RGB")
    both=Image.new("RGB",(384,192))
    both.paste(original,(0,0));both.paste(canvas,(192,0))
    both.resize((1152,576),Image.Resampling.NEAREST).save(path)


def execute(a):
    from tools.corpus.corpus370_native_raster_join_plan_v14 import arms_for_plan
    identity,manifest,job,faces,bundle,oldix,newix,singles,rgba,joint,_,parents=(
        validate_and_open(a.v14_public,a.v16_public,a.source,a.v14_dir,
                          a.v14_job,a.v16_manifest,a.v16_court,a.v16_pixels))
    v17_pub=load(a.v17_public)
    v17_prop=load(a.v17_proposal)
    v17_court=load(a.v17_court)
    source_row=next((z for z in v17_pub.get("source_results",[])
                     if z.get("source")==a.source),None)
    require(v17_pub.get("schema")==V17_PUBLIC and source_row is not None
        and sha(a.v17_court)==source_row["court_sha256"]
        and v17_prop.get("schema")==V17_PROPOSAL
        and v17_court.get("schema")==V17_COURT
        and v17_court.get("source_pose_ray_proposal_sha256")==sha(a.v17_proposal)
        and v17_court.get("status")==
            "V17_REAL_POSED_SOURCE_GEOMETRIC_RAY_AND_RGBA_LINK_RECALCULATED__ARTIST_OWNER_UNKNOWN"
        and v17_prop.get("original_source_sha256")==identity["source_sha256"]
        and v17_prop.get("parents_sha256")==parents
        and v17_prop.get("training_authorized") is False
        and v17_court.get("graphics_owner_authorized") is False,
        "V18A_ORIGINAL_V17_RGBA_GEOMETRY_COURT_PARENT_SHA")
    require(job["v13b_array"]["sha256"]==sha(job["v13b_array"]["path"])
        and identity["source_sha256"]==job["source"]["sha256"]
        and job["selected_source_controls"]["sha256"]==
            sha(job["selected_source_controls"]["path"]),
        "V18A_ORIGINAL_V13B_AND_BLIND_CONTROL_PARENT_SHA")
    plan=load(job["selected_source_controls"]["path"])
    require(plan==manifest["selected_source_controls"]
        and plan["selected_control_count"]==2
        and plan["raster_pixels_seen_during_control_selection"] is False,
        "V18A_ORIGINAL_RASTER_BLIND_CONTROL_PLAN")
    global joint_selection_indices
    joint_selection_indices=tuple(
        (r["control_index_teacher_only"],r["axis_index_teacher_only"])
        for r in plan["controls"])
    with np.load(job["rest_geometry"]["path"],allow_pickle=False) as f:
        rest=np.asarray(f["rest_vertices_source_world"],np.float64)
        original_faces=np.asarray(f["faces"],np.int64)
    with np.load(job["v13b_array"]["path"],allow_pickle=False) as f:
        base=np.asarray(f["baseline_vertices"],np.float64)
        controls=np.asarray(f["native_control_perturbed_vertices"],np.float64)
    shape_check(rest,original_faces,base,controls,joint)
    require(np.array_equal(faces,original_faces),"V18A_V14_V16_SOURCE_FACE_IDENTITY")
    extent=max(1e-12,float(np.linalg.norm(np.ptp(rest,axis=0))))
    selected=select_blind_control_axes(base,controls,faces,extent)
    stained=face_stain_features(rest,faces,base,controls,joint,selected)
    edges=face_edges(faces)
    candidates={}
    results=[]
    # Group predictions computed without V17 observational target effects.
    for threshold in THRESHOLDS:
        labels=component_labels(stained,edges,threshold)
        n=int(labels.max(initial=-1)+1)
        counts=np.bincount(labels,minlength=n)
        heldout=heldout_response_error(faces,base,controls,selected,labels,extent)
        candidates[f"{threshold:g}"]=labels
        results.append({
            "frozen_stain_similarity_threshold":threshold,
            "predicted_connected_source_graphics_layer_candidate_count":n,
            "smallest_candidate_faces":int(counts.min()),
            "largest_candidate_faces":int(counts.max()),
            "single_face_candidate_count":int(np.count_nonzero(counts==1)),
            "heldout_authored_pose_2_motion_within_component_variance_fraction":
                heldout,
            "is_original_artist_part_count":False,
            "causal_edit_independence_proven":False
        })
    # Held-out V17 visibility/RGBA is a *separate evaluation*, not a feature
    # used for coloring or clustering. A first hit is merely geometric.
    heldout_observed=0;heldout_unknown=0;heldout_effect=0
    component_hit_counts={t:np.zeros(int(l.max(initial=-1)+1),np.int64)
                          for t,l in candidates.items()}
    for row in v17_prop["source_ray_rows_private_only"]:
        if row["source_pose_ordinal"]!=HELDOUT_POSE:continue
        hits=row["source_geometric_all_hits_ordered_near_to_far"]
        good=(row["geometric_stack_complete"] and
              not row["geometric_stack_has_depth_tie"] and bool(hits))
        if not good:heldout_unknown+=1;continue
        idx=hits[0]["face_index_teacher_only"]
        require(0<=idx<len(faces),"V18A_HELDOUT_RAY_FACE_ADDRESS")
        heldout_observed+=1
        heldout_effect+=int(row["real_original_source_joint_ab_visible_rgba_effect"])
        for t,labels in candidates.items():
            component_hit_counts[t][labels[idx]]+=1
    a.output_dir.mkdir(parents=True,exist_ok=True)
    npz=a.output_dir/"PRIVATE_REACTIVE_STAIN_CANDIDATE_V18A.npz"
    np.savez_compressed(npz,faces=faces,stain_channels=stained,
        graph_edges=edges,selected_source_control_axes_teacher_only=
        np.asarray(selected,dtype=np.int32),
        **{"labels_"+t.replace(".","p"):l for t,l in candidates.items()})
    cam={int(x["view_index"]):x for x in bundle["cameras"]}[0]
    debug=a.output_dir/"PRIVATE_V18A_REACTIVE_STAIN_POSE2_VIEW0.png"
    rasterize_debug_stain(faces,joint[HELDOUT_POSE,0,0],cam,
        candidates["0.1"],rgba[HELDOUT_POSE,0,newix["ORIGINAL_BASE"]],debug)
    report={
        "schema":SCHEMA,"status":STATUS,
        "source_name_teacher_only":a.source,
        "subject_id":identity["subject_id"],
        "source_sha256":identity["source_sha256"],
        "parent_source_immutable_sha256":{
            **parents,"v17_public":sha(a.v17_public),
            "v17_private_proposal":sha(a.v17_proposal),
            "v17_independent_court":sha(a.v17_court),
            "v13b_source_intervention":sha(job["v13b_array"]["path"]),
            "source_rest_geometry":sha(job["rest_geometry"]["path"]),
            "v14_source_control_plan":sha(job["selected_source_controls"]["path"])
        },
        "stain_control_selection_policy":
            "FIT_POSE0_GEOMETRY_ONLY_GREEDY_SOURCE_RESPONSE_DIVERSITY",
        "stain_channel_policy":
            "FIT_POSE0_POSE1_ANONYMOUS_SIGNED_LOCAL_RIGID_RESIDUAL_AND_REAL_JOINT_AB",
        "fit_source_authored_poses":[0,1],
        "heldout_source_authored_pose":2,
        "actual_native_source_axes_probed_teacher_only":[list(x) for x in selected],
        "source_control_count_for_staining":len(selected),
        "source_face_count":len(faces),
        "true_shared_edge_graph_edges":len(edges),
        "feature_channel_count":stained.shape[1],
        "candidate_policies":results,
        "independent_heldout_v17_geometric_ray_first_hit_cases":heldout_observed,
        "independent_heldout_v17_unknown_ray_cases":heldout_unknown,
        "independent_heldout_v17_visible_effect_pixel_cases":heldout_effect,
        "candidate_component_heldout_geometric_observation_counts_teacher_only":
            {k:v.tolist() for k,v in component_hit_counts.items()},
        "private_candidate_npz_sha256":sha(npz),
        "private_rgb_stain_diagnostic_sha256":sha(debug),
        "source_appearance_transformed_into_model_input":False,
        "source_3d_rig_bone_or_material_identity_used_as_artist_layer":False,
        "source_control_id_as_model_feature":False,
        "number_of_artist_parts_imposed":False,
        "teacher_control_response_is_product_input":False,
        "heldout_edit_replay_proven":False,
        "full_alpha_draw_order_and_CAA_proven":False,
        "original_artist_same_drawable_labels_minted":0,
        "original_artist_different_drawable_labels_minted":0,
        "source_graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False
    }
    outfile=a.output_dir/"PRIVATE_REACTIVE_STAIN_REPORT_V18A.json"
    outfile.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({
        "source":a.source,"subject_id":identity["subject_id"],
        "status":STATUS,
        "stain_channels":stained.shape[1],
        "selected_native_control_axes":len(selected),
        "predicted_layer_candidates":[
            [z["frozen_stain_similarity_threshold"],
             z["predicted_connected_source_graphics_layer_candidate_count"],
             z["heldout_authored_pose_2_motion_within_component_variance_fraction"]]
            for z in results],
        "private_candidate_sha":report["private_candidate_npz_sha256"],
        "model_training_authorized":False
    },sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True,choices=("CAT","TRICERATOPS","COWFISH"))
    for name in ("v14_public","v16_public","v17_public","v14_dir",
                 "v14_job","v16_manifest","v16_court","v16_pixels",
                 "v17_proposal","v17_court","output_dir"):
        ap.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    execute(ap.parse_args())


if __name__=="__main__":
    main()
