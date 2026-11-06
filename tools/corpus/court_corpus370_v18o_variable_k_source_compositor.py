#!/usr/bin/env python3
"""V18o generic K-layer source compositor PURE CORE (not teacher truth).

Inputs are *separately rendered* per-layer premultipliable source straight
RGBA and per-layer Cycles View Z diagnostic evidence for exactly the same
camera sample index. Supports K>=3 and real nonbinary material alpha without
using source FULL outcome, rig names, part semantics or manually fixed layer
order. Source layer correspondence and fidelity must still be proved by
actual independent physical source render/edit courts before any authority.
"""
from __future__ import annotations
import numpy as np

TIE_EPS=.0002
TOL=.025
MAX_SOURCE_GROUPS=64


def require(ok,reason):
    if not ok:raise ValueError(reason)


def k_compose(layer_rgba,depth_rgba,*,tie_epsilon=TIE_EPS):
    colors=np.asarray(layer_rgba,dtype=np.float64)
    depth=np.asarray(depth_rgba,dtype=np.float64)
    require(colors.ndim==5 and colors.shape==depth.shape
            and colors.shape[0]==8 and 3<=colors.shape[1]<=MAX_SOURCE_GROUPS
            and colors.shape[2]>0 and colors.shape[3]>0
            and colors.shape[-1]==4
            and np.isfinite(colors).all() and np.isfinite(depth).all()
            and 0<tie_epsilon<=.001
            and np.min(colors[...,3])>=-1e-5
            and np.max(colors[...,3])<=1.+1e-5
            and np.min(depth[...,3])>=-1e-5
            and np.max(depth[...,3])<=1.+1e-5,
            "V18O_K_SOURCE_LAYER_RENDER_SHAPE_ALPHA_OR_FINITE")
    s,k,h,w,_=colors.shape
    ca=np.clip(colors[...,3],0,1)
    da=np.clip(depth[...,3],0,1)
    require(np.all((da<=1e-6)|(da>=1-1e-6)),
            "V18O_K_SOURCE_GEOMETRY_PER_SAMPLE_ALPHA_BINARY_REQUIRED")
    hit=da>.5
    require(not np.any((ca>1e-6)&~hit),
            "V18O_K_SOURCE_COLOR_WITHOUT_SAME_SAMPLE_GEOMETRY")
    z=depth[...,0]
    valid_z=np.where(hit,(z>=.019)&(z<=.981),True)
    invalid=~np.all(valid_z,axis=1)
    # Missing surface = +inf. Group order in input is anonymous and must not
    # change the physical result; only numerical per-ray measured depth sorts.
    source_depth=np.where(hit,z,np.inf)
    order=np.argsort(source_depth,axis=1,kind="stable")
    sorted_z=np.take_along_axis(source_depth,order,axis=1)
    sorted_hit=np.take_along_axis(hit,order,axis=1)
    # Ties among consecutive actual geometry hits do not have an evidence
    # ordering; UNKNOWN, even if one layer's source paint alpha is zero.
    near=(np.abs(np.diff(sorted_z,axis=1))<=tie_epsilon)
    both=(sorted_hit[:,:-1]&sorted_hit[:,1:])
    tied=np.any(both&near,axis=1)
    valid=~(invalid|tied)
    # Original source alpha is not geometry alpha; use original material color
    # and alpha as rendered from each physically isolated source group.
    color_premul=np.concatenate(
        (colors[...,:3]*ca[...,None],ca[...,None]),axis=-1)
    sorted_color=np.take_along_axis(
        color_premul,order[...,None],axis=1)
    out=np.zeros((s,h,w,4),dtype=np.float64)
    for i in range(k-1,-1,-1):
        top=sorted_color[:,i]
        out=top+out*(1.-top[...,3:4])
    out[~valid]=0.
    require(np.isfinite(out).all()
        and np.min(out[...,3])>=-1e-6 and np.max(out[...,3])<=1+1e-6,
        "V18O_K_ORACLE_FREE_COMPOSITING_NUMERIC_OVERFLOW")
    return {
       "premultiplied_sample_rgba":out,
       "qualified_sample_mask":valid,
       "source_depth_tie_mask":tied,
       "source_invalid_depth_mask":invalid,
       "source_at_least_three_geometry_hits_mask":
           np.count_nonzero(hit,axis=1)>=3,
       "source_any_nonbinary_material_alpha_mask":
           np.any((ca>1e-6)&(ca<1-1e-6),axis=1),
       "source_K":k,
       "source_samples":s,
       "source_image_height":h,
       "source_image_width":w,
       "source_diagnostic_original_full_used_for_sorting":False,
       "source_group_ids_are_anonymous_only":True,
       "artist_graphics_owner_authorized":False,
       "iris_training_authorized":False,
       "tessa_training_authorized":False,
       "materialization_authorized":False,
       "training_authorized":False,
    }


def independently_score_against_full(prediction,full_rgba,tolerance=TOL):
    """Separate comparator: reference FULL is never passed to k_compose."""
    full=np.asarray(full_rgba,dtype=np.float64)
    pred=prediction["premultiplied_sample_rgba"]
    mask=prediction["qualified_sample_mask"]
    require(full.shape==pred.shape and np.isfinite(full).all()
        and np.min(full[...,3])>=-1e-5
        and np.max(full[...,3])<=1.00001,
        "V18O_K_ORIGINAL_SOURCE_FULL_REFERENCE_RGBA_SHAPE")
    alpha=np.clip(full[...,3],0.,1.)
    full_pre=np.concatenate((full[...,:3]*alpha[...,None],
                             alpha[...,None]),axis=-1)
    residual=np.max(np.abs(pred-full_pre),axis=-1)
    complete=np.all(mask,axis=0)
    pred_mean=np.mean(pred,axis=0)
    reference_mean=np.mean(full_pre,axis=0)
    pixel_diff=np.max(np.abs(pred_mean-reference_mean),axis=-1)
    return {
       "total_source_sample_pixel_instances":int(mask.size),
       "qualified_source_sample_pixel_instances":int(np.count_nonzero(mask)),
       "unknown_source_sample_pixel_instances":int(np.count_nonzero(~mask)),
       "qualified_source_sample_rgba_failures":int(np.count_nonzero(mask&(residual>tolerance))),
       "qualified_source_all_8_sample_pixels":int(np.count_nonzero(complete)),
       "unknown_source_all_8_sample_pixels":int(np.count_nonzero(~complete)),
       "qualified_source_8sample_pixel_rgba_failures":
           int(np.count_nonzero(complete&(pixel_diff>tolerance))),
       "full_source_used_only_in_independent_scoring":True,
       "source_semantic_part_labels_used":False,
       "artist_graphics_owner_authorized":False,
       "training_authorized":False,
    }
