"""Blind, label-permutation-invariant ARTIST-drawable partition benchmark V1.

An original human-art authored layer/slot/attachment project is an INDEPENDENT
evaluation reference; it is NEVER the source of model inference labels. The
producer must flatten it into one anonymous element domain and prove identical
source render. Mechanical glTF/FBX object IDs are NOT authored 2D drawables.
All results are EVALUATION ONLY, never training/data/owner authority.
"""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA="RealSaS.BlindedIndependentArtistDrawablePartitionCourt.v1"
SOURCE_KIND="HUMAN_AUTHORED_EDITABLE_2D_DRAWABLE"
PREDICTION_KIND="GENERIC_ANONYMOUS_MERGED_SURFACE_PARTITION"

def require(value, reason):
    if not value:
        raise ValueError(reason)

def array_digest(values):
    a=np.ascontiguousarray(np.asarray(values, dtype="<i8"))
    return hashlib.sha256(a.tobytes()).hexdigest()

def evaluate(author_ids, prediction_ids, element_weights, observable_mask,
             artist_source, anonymous_prediction, *, min_overlap_share=.05):
    """Return court-only metrics. Label numbers are opaque & freely permutable.

    Note: each element must be a *common rendered/source-face address* with
    exactly the same identity in the independently flattened input and the
    hidden artist reference. For layered PSD and Spine mesh formats, construction
    of this paired domain is a separate mandatory source adapter.
    """
    require(artist_source.get("source_kind")==SOURCE_KIND
            and artist_source.get("reference_from_original_artist_project") is True
            and artist_source.get("human_authored_not_generated_proxy") is True
            and artist_source.get("artist_drawable_ids_hidden_from_algorithm") is True
            and artist_source.get("author_project_SHA256") is not None
            and artist_source.get("source_composite_parity_qualified") is True
            and artist_source.get("layer_semantics_authoritative_only_for_court") is True
            and artist_source.get("training_authorized") is False
            and anonymous_prediction.get("prediction_kind")==PREDICTION_KIND
            and anonymous_prediction.get("original_merged_surface_SHA256")==
                artist_source.get("original_merged_surface_SHA256")
            and anonymous_prediction.get("element_address_SHA256")==
                artist_source.get("element_address_SHA256")
            and anonymous_prediction.get("author_slot_mesh_material_names_exposed") is False
            and anonymous_prediction.get("author_layer_count_exposed") is False
            and anonymous_prediction.get("layer_reference_used_in_training") is False
            and anonymous_prediction.get("artist_label_access_before_prediction") is False
            and anonymous_prediction.get("precommitted_before_artist_reference_opened") is True
            and anonymous_prediction.get("training_authorized") is False,
            "ARTIST_COURT_INDEPENDENT_REAL_AUTHOR_ORIGINAL_MERGED_SOURCE_NO_LEAKAGE")
    ids=np.asarray(author_ids)
    pred=np.asarray(prediction_ids)
    w=np.asarray(element_weights,np.float64)
    observed=np.asarray(observable_mask)
    require(ids.ndim==pred.ndim==w.ndim==observed.ndim==1
            and len(ids)==len(pred)==len(w)==len(observed)
            and len(ids)>0
            and ids.dtype.kind in "iu" and pred.dtype.kind in "iu"
            and observed.dtype.kind=="b"
            and np.isfinite(w).all() and np.all(w>0)
            and np.isfinite(float(w.sum()))
            and isinstance(min_overlap_share,float) and 0.<min_overlap_share<.5,
            "ARTIST_COURT_UNMATCHED_TRIANGLE_OR_PIXEL_DOMAIN")
    require(artist_source.get("artist_label_array_SHA256")==array_digest(ids)
            and anonymous_prediction.get("anonymous_partition_array_SHA256")==array_digest(pred),
            "ARTIST_COURT_OPAQUE_HIDDEN_AUTHOR_LABEL_BYTES")
    # -1 is UNKNOWN. Do not call source invisible/ambiguous pixels known.
    valid_ref=observed&(ids>=0)
    valid_pred=valid_ref&(pred>=0)
    reference_area=float(w[valid_ref].sum())
    evaluated_area=float(w[valid_pred].sum())
    require(reference_area>0 and evaluated_area>0,
            "ARTIST_COURT_NO_OVERLAPPING_OBSERVABLE_ART_LAYER_AREA_ABSTAIN")
    # Score evaluated part but always report coverage so UNKNOWN is not rewarded.
    g,gi=np.unique(ids[valid_pred],return_inverse=True)
    p,pi=np.unique(pred[valid_pred],return_inverse=True)
    contingency=np.zeros((len(g),len(p)),np.float64)
    np.add.at(contingency,(gi,pi),w[valid_pred])
    ga=contingency.sum(axis=1)
    pa=contingency.sum(axis=0)
    total=float(contingency.sum())
    intersection=contingency
    union=ga[:,None]+pa[None,:]-intersection
    iou=np.divide(intersection,union,out=np.zeros_like(intersection),
                  where=union>0)
    # One-to-one Hungarian optimum maximizes *total IoU* (label invariant).
    row,col=linear_sum_assignment(-iou)
    matched_iou=float(iou[row,col].sum()/max(1,len(g)))
    matched_intersection=float(intersection[row,col].sum()/total)
    # Conditional, observed-subset partition agreement. A reference layer can
    # correctly be spanned by several independent mechanical factors.
    prob=contingency/total
    pg=ga/total;pp=pa/total
    hg=float(-np.sum(pg[pg>0]*np.log2(pg[pg>0])))
    hp=float(-np.sum(pp[pp>0]*np.log2(pp[pp>0])))
    with np.errstate(divide="ignore",invalid="ignore"):
        joint=np.divide(prob,pg[:,None]*pp[None,:],
                        out=np.ones_like(prob),where=prob>0)
        mi=float(np.sum(prob[prob>0]*np.log2(joint[prob>0])))
    vi=max(0.,hg+hp-2*mi)
    # Source gt layer split / predicted group merge counted using only supported
    # overlapping masses. This is diagnostic, not a required one-to-one truth.
    significant_gt=(intersection/ga[:,None])>=min_overlap_share
    significant_pred=(intersection/pa[None,:])>=min_overlap_share
    split_count=int(np.count_nonzero(significant_gt.sum(axis=1)>1))
    merge_count=int(np.count_nonzero(significant_pred.sum(axis=0)>1))
    # Weight unsupported source elements. Visibility/occlusion means *known*
    # layer truth is not equivalent to a physically unobserved 2D image label.
    result={
        "schema":SCHEMA,
        "status":"INDEPENDENT_AUTHORED_DRAWABLE_PARTITION_EVALUATION_ONLY",
        "source_kind":SOURCE_KIND,
        "reference_known_observable_element_count":int(valid_ref.sum()),
        "predicted_known_observable_element_count":int(valid_pred.sum()),
        "reference_observable_weight":reference_area,
        "prediction_observable_weight_coverage":evaluated_area/reference_area,
        "author_layer_count_in_evaluated_domain":len(g),
        "anonymous_prediction_region_count_in_evaluated_domain":len(p),
        "hungarian_mean_matched_iou":matched_iou,
        "hungarian_matched_intersection_weight_fraction":matched_intersection,
        "weighted_variation_of_information_bits":vi,
        "author_layers_with_significant_algorithm_splits":split_count,
        "predicted_regions_merging_significant_author_layers":merge_count,
        "overlap_share_threshold":min_overlap_share,
        "evaluated_artist_ground_truth_only_not_mechanical_source_owner":True,
        "author_mesh_bone_material_slot_names_used_as_student_prior":False,
        "artist_layer_count_leaked_to_algorithm":False,
        "unknown_or_unsupported_artist_layer_causality_can_be_masked":True,
        "source_native_2d_edit_behavior_qualified":False,
        "cross_camera_draw_order_qualified":False,
        "3d_ownership_source_authority_qualified":False,
        "graphics_owner_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }
    return result
