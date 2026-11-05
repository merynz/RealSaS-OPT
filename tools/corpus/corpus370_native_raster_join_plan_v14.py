"""V14 source-only and raster-blind preselection of original native controls.

This does NOT choose artist layers, source group count, or semantics. It
pre-registers (a) the strongest nonrigid locally responding source carrier,
(b) an actual global-rigid responding source carrier where available, otherwise
a second independent local carrier, BEFORE any new V14 raster is observed.
Source controls and original rig names stay private research instrumentation.
"""
from __future__ import annotations
import numpy as np

SCHEMA="RealSaS.Corpus370NativeActuatorRasterBlindSelection.v14"

def require(ok,msg):
    if not ok:raise ValueError(msg)

def rigid_rms(source,target):
    source=np.asarray(source,np.float64);target=np.asarray(target,np.float64)
    require(source.shape==target.shape and source.ndim==2 and source.shape[1]==3
            and np.isfinite(source).all() and np.isfinite(target).all()
            and len(source)>=3,"V14_INVALID_SOURCE_RIGID_RESPONSE")
    x=source-source.mean(axis=0);y=target-target.mean(axis=0)
    u,_,vh=np.linalg.svd(x.T@y)
    d=-1. if np.linalg.det(vh.T@u.T)<0 else 1.
    r=vh.T@np.diag([1.,1.,d])@u.T
    z=x@r.T+target.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((target-z)**2,axis=1))))

def select(original_rest, baseline, interventions, native_pose_indices,
           source_control_inventory_count, expected_poses):
    rest=np.asarray(original_rest,np.float64)
    base=np.asarray(baseline,np.float64)
    variant=np.asarray(interventions,np.float64)
    require(rest.ndim==2 and rest.shape[1]==3 and
            base.shape==(3,len(rest),3)
            and variant.shape==(3,source_control_inventory_count,3,2,len(rest),3)
            and type(source_control_inventory_count) is int
            and 1<=source_control_inventory_count<=128
            and type(native_pose_indices) is list
            and native_pose_indices==expected_poses
            and len(native_pose_indices)==3
            and np.isfinite(rest).all() and np.isfinite(base).all()
            and np.isfinite(variant).all(),"V14_ORIGINAL_AUTHORED_SOURCE_VERTEX_ARRAY_CONTRACT")
    extent=max(float(np.linalg.norm(np.ptp(rest,axis=0))),1e-12)
    threshold=2e-4*extent
    rows=[]
    for i in range(source_control_inventory_count):
        for axis in range(3):
            plus=variant[0,i,axis,0];minus=variant[0,i,axis,1]
            d=plus-base[0]
            movement=float(np.max(np.linalg.norm(d,axis=-1)))
            residue=rigid_rms(base[0],plus)
            reverse=rigid_rms(base[0],minus)
            if movement<=threshold: continue
            rows.append({
                "control_index_teacher_only":i,
                "axis_index_teacher_only":axis,
                "source_global_rigid_residual":residue,
                "source_reverse_global_rigid_residual":reverse,
                "source_max_displacement":movement,
                "native_rig_control_kind_inferred_from_raster":False,
                "training_authorized":False,
            })
    require(rows,"V14_SOURCE_HAS_NO_MEASURABLE_CONTROL_ABSTAIN")
    local=sorted((r for r in rows if r["source_global_rigid_residual"]>threshold),
                 key=lambda r:(-r["source_global_rigid_residual"],-r["source_max_displacement"],
                               r["control_index_teacher_only"],r["axis_index_teacher_only"]))
    require(local,"V14_SOURCE_HAS_NO_NATIVE_NONRIGID_RESPONSE_ABSTAIN")
    pick=[dict(local[0],source_response_class_teacher_only="LOCAL_NONRIGID")]
    global_like=sorted((r for r in rows
                        if r["source_global_rigid_residual"]<=threshold
                        and r["control_index_teacher_only"]!=pick[0]["control_index_teacher_only"]),
                        key=lambda r:(-r["source_max_displacement"],
                                      r["control_index_teacher_only"],r["axis_index_teacher_only"]))
    if global_like:
        pick.append(dict(global_like[0],source_response_class_teacher_only="GLOBAL_RIGID_LIKE"))
    else:
        second=next((r for r in local if r["control_index_teacher_only"]!=pick[0]["control_index_teacher_only"]),None)
        if second is not None:
            pick.append(dict(second,source_response_class_teacher_only="SECOND_LOCAL_NONRIGID"))
    require(len(pick) in (1,2),"V14_ORIGINAL_CARRIER_SELECTION_BUDGET")
    return {
        "schema":SCHEMA,
        "status":"ORIGINAL_NATIVE_GEOMETRY_ONLY_PRECOMMITTED_BEFORE_V14_RGBA",
        "native_pose_indices_teacher_only":native_pose_indices,
        "source_control_inventory_count_teacher_only":source_control_inventory_count,
        "controls":[pick[0],*pick[1:]],
        "selected_control_count":len(pick),
        "source_relative_activity_threshold":threshold/extent,
        "raster_pixels_seen_during_control_selection":False,
        "source_bone_or_object_identity_as_artist_target":False,
        "source_response_classes_are_not_artist_drawables":True,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }

def arms_for_plan(plan):
    require(plan.get("schema")==SCHEMA
            and plan.get("raster_pixels_seen_during_control_selection") is False
            and plan.get("graphics_owner_authorized") is False
            and plan.get("training_authorized") is False
            and isinstance(plan.get("controls"),list)
            and len(plan["controls"])==plan["selected_control_count"]
            and plan["selected_control_count"] in (1,2),"V14_RASTER_BLIND_PRECOMMIT_VALIDATION")
    arms=["ORIGINAL_BASE","SOUP_BASE","SOUP_NOOP",
          "ROOT_SHAM_ORIGINAL","ROOT_SHAM_SOUP"]
    pairs=[]
    seen=set()
    for ci,r in enumerate(plan["controls"]):
        c=r["control_index_teacher_only"];a=r["axis_index_teacher_only"]
        require(type(c) is int and c>=0 and type(a) is int and a in range(3)
                and (c,a) not in seen
                and r["training_authorized"] is False
                and r["native_rig_control_kind_inferred_from_raster"] is False,
                "V14_DUPLICATE_UNSAFE_NATIVE_ACTUATOR_CONTROL")
        seen.add((c,a))
        for sign in ("PLUS","MINUS"):
            real=f"SOURCE_C{ci:02d}_{sign}"
            soup=f"SOUP_C{ci:02d}_{sign}"
            arms.extend((real,soup))
            pairs.append((real,soup))
    return arms,pairs
