#!/usr/bin/env python3
"""Variable-K Cycles *same-sample* source RGBA/depth composer (teacher only).

Generalizes V18m/V18n's two physically isolated parts to arbitrary K>=1.
Only isolated original source material RGBA and corresponding independently
measured original source View-Z depth are predictors. FULL is never an input.
No artist drawable/IRIS training authority is produced.
"""
from __future__ import annotations
import numpy as np

SCHEMA="RealSaS.Corpus370SampleIndexedVariableKSourceCompositor.v1"
DEPTH_TIE_EPS=0.0002


def compose_isolated_original_source_k(
        layer_rgba, layer_view_z_rgba, *, depth_tie_eps=DEPTH_TIE_EPS):
    """Inputs [S,K,H,W,4], 1-sample-per-Cycles-offset, same view/camera.

    Pixel sample may have K source layer hits in the original 3D scene.
    Original source face labels are source-private; this is ONLY a mechanics
    of rendered source appearances, not a segmentation/identity truth.
    """
    color=np.asarray(layer_rgba,dtype=np.float64)
    depth=np.asarray(layer_view_z_rgba,dtype=np.float64)
    if (color.ndim!=5 or color.shape!=depth.shape or color.shape[-1]!=4
        or color.shape[1]<1 or min(color.shape[2:4])<1
        or not np.isfinite(color).all() or not np.isfinite(depth).all()
        or not 0<depth_tie_eps<=.001
        or np.min(color[...,3]) < -1e-5 or np.max(color[...,3])>1.00001):
        raise ValueError("SAMPLE_K_SOURCE_COLOR_DEPTH_SHAPE_INVALID")
    ca=np.clip(color[...,3],0.,1.)
    da=np.clip(depth[...,3],0.,1.)
    if np.any((da>1e-6)&(da<1-1e-6)):
        raise ValueError("SAMPLE_K_SOURCE_DEPTH_COVERAGE_NOT_BINARY")
    geo=da>.5
    if np.any((ca>1e-6)&~geo):
        raise ValueError("SAMPLE_K_SOURCE_MATERIAL_VISIBLE_WITHOUT_GEOMETRY")
    z=depth[...,0]
    valid_z=np.where(geo,(z>=.019)&(z<=.981),True)
    invalid=np.any(~valid_z,axis=1)
    # Nearest first. Absent geometry must never take a depth-order slot.
    zz=np.where(geo,z,np.inf)
    indices=np.argsort(zz,axis=1,kind="stable")
    ordered_z=np.take_along_axis(zz,indices,axis=1)
    ordered_geo=np.take_along_axis(geo,indices,axis=1)
    diffs=np.diff(ordered_z,axis=1)
    close=(ordered_geo[:,1:] & ordered_geo[:,:-1]
           & np.isfinite(diffs) & (diffs<=depth_tie_eps))
    ties=np.any(close,axis=1)
    unknown=invalid|ties
    # RGBA in source material frames is STRAIGHT color; convert to premul.
    packed=np.concatenate([color[...,:3]*ca[...,None],ca[...,None]],axis=-1)
    ordered_rgba=np.take_along_axis(packed,indices[...,None],axis=1)
    # Porter-Duff OVER from back to front. This handles semitransparency
    # when an original independently established material alpha has it.
    out=np.zeros_like(ordered_rgba[:,0])
    for k in range(color.shape[1]-1,-1,-1):
        fg=ordered_rgba[:,k]
        out=fg+(1.-fg[...,3:4])*out
    out[unknown]=0.
    if not np.isfinite(out).all():
        raise ValueError("SAMPLE_K_SOURCE_COMPOSITOR_NONFINITE")
    return {
        "schema":SCHEMA,
        "predicted_source_sample_premultiplied_rgba":out,
        "qualified_sample_mask":~unknown,
        "unknown_depth_tie_mask":ties,
        "invalid_source_depth_mask":invalid,
        "physical_source_group_hit_count":np.sum(geo,axis=1),
        "frontmost_layer_index":np.where(np.any(geo,axis=1)&~unknown,indices[:,0],-1),
        "sample_decision_used_full_target":False,
        "source_artist_drawable_owner":"UNKNOWN",
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
