#!/usr/bin/env python3
"""Original artist-blind topology LEAKAGE audit, not an ownership model.

Two mandatory phases:
1. PREDICT: connected components using ONLY anonymous merged triangles.
2. AUDIT: compare blind prediction to untouched original author slot reference.

An un-welded concatenation of the original artist's meshes can encode the
answer in its disconnected components! High accuracy here is tautological,
and must NOT be mistaken for inference from a flattened image or homogeneous
geometry. Render/alpha parity is intentionally FALSE for this research gate.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from tools.corpus.author_spine_blind_flatten_v1 import (
    MERGED,HIDDEN,canonical_bytes,digest
)

SCHEMA="RealSaS.AuthorSourceGeometryComponentLeakageAudit.v1"
BASELINE="RealSaS.AnonymousMergedGeometryConnectedComponentsBaseline.v1"

def require(ok,reason):
    if not ok:raise ValueError(reason)

def connected_face_components(merged):
    require(merged.get("schema")==MERGED
            and merged.get("source_composite_parity_qualified") is False
            and merged.get("original_art_layer_count_visible") is False
            and merged.get("original_slot_attachment_bone_names_present") is False
            and merged.get("training_authorized") is False,
            "ARTIST_BASELINE_CANNOT_INSPECT_SOURCE_LAYER_NAMES_OR_REFERENCE")
    pts=np.asarray(merged["vertices_setup_xy"],np.float64)
    faces=np.asarray(merged["faces"],np.int64)
    require(pts.ndim==2 and pts.shape[1]==2
            and faces.ndim==2 and faces.shape[1]==3 and len(faces)>0
            and faces.min()>=0 and faces.max()<len(pts)
            and np.isfinite(pts).all(),
            "ARTIST_BASELINE_INVALID_ANONYMOUS_MERGED_FACE_ADDRESS")
    parent=np.arange(len(pts),dtype=np.int64)
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=int(parent[i])
        return i
    def unite(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for a,b,c in faces:unite(int(a),int(b));unite(int(a),int(c))
    groups={}
    labels=[]
    for a,b,c in faces:
        roots={find(int(i)) for i in (a,b,c)}
        require(len(roots)==1,"ARTIST_BASELINE_TRIANGLE_NOT_SAME_GRAPH_COMPONENT")
        root=next(iter(roots))
        if root not in groups:groups[root]=len(groups)
        labels.append(groups[root])
    return labels

def predict(merged):
    labels=connected_face_components(merged)
    result={
        "schema":BASELINE,
        "status":"ANONYMOUS_SOURCE_TOPOLOGY_DISCONNECTED_COMPONENT_BASELINE_ONLY",
        "anonymous_merged_surface_sha256":digest(canonical_bytes(merged)),
        "face_partition_id":[int(z) for z in labels],
        "face_partition_SHA256":digest(canonical_bytes(labels)),
        "predicted_topological_component_count":len(set(labels)),
        "readset_only_merged_vertices_and_triangle_incidence":True,
        "artist_reference_not_read":True,
        "source_bone_material_slot_names_as_labels":False,
        "original_appearance_image_input_used":False,
        "source_artist_topology_leakage_risk":True,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }
    return result

def audit(merged,prediction,hidden):
    require(merged["schema"]==MERGED and hidden["schema"]==HIDDEN
            and prediction["schema"]==BASELINE
            and hidden["anonymous_merged_surface_sha256"]==
                prediction["anonymous_merged_surface_sha256"]==
                digest(canonical_bytes(merged))
            and hidden["source_appearance_rgba_alpha_parity_qualified"] is False
            and hidden["original_spine_setup_pose_native_runtime_parity_qualified"] is False
            and prediction["source_artist_topology_leakage_risk"] is True
            and prediction["artist_reference_not_read"] is True
            and prediction["training_authorized"] is False,
            "ARTIST_SOURCE_COMPONENT_LEAKAGE_PREVIEW_PROVENANCE_NOT_GOLD_ART_ACCURACY")
    original=np.asarray(hidden["artist_slot_id_teacher_only_per_face"],np.int64)
    predicted=np.asarray(prediction["face_partition_id"],np.int64)
    require(original.shape==predicted.shape and original.ndim==1
            and len(original)==len(merged["faces"])
            and digest(canonical_bytes(predicted.tolist()))==
                prediction["face_partition_SHA256"]
            and (original>=0).all() and (predicted>=0).all(),
            "ARTIST_SOURCE_COMPONENT_PRIVATE_REFERENCE_COMMON_FACE_DOMAIN")
    a,ai=np.unique(original,return_inverse=True)
    b,bi=np.unique(predicted,return_inverse=True)
    table=np.zeros((len(a),len(b)),dtype=np.int64)
    np.add.at(table,(ai,bi),1)
    # Conditional agreement with the source author's visible setup attachments,
    # not artist image layer identity. Incidence graph often reveals same cuts.
    component_purity=float(table.max(axis=0).sum()/len(original))
    source_group_purity=float(table.max(axis=1).sum()/len(original))
    source_group_splits=int(np.count_nonzero((table>0).sum(axis=1)>1))
    prediction_component_merges=int(np.count_nonzero((table>0).sum(axis=0)>1))
    exact_matching=bool(component_purity==1 and source_group_purity==1
                        and len(a)==len(b))
    return {
        "schema":SCHEMA,
        "status":"REAL_SOURCE_TOPOLOGY_LEAKAGE_DIAGNOSTIC_ONLY__NOT_GOLD_DRAWABLE_BENCHMARK",
        "anonymous_input_sha256":prediction["anonymous_merged_surface_sha256"],
        "face_partition_sha256":prediction["face_partition_SHA256"],
        "original_artist_reference_sha256":digest(canonical_bytes(hidden)),
        "original_artist_setup_supported_slot_count_PRIVATE_COURT_ONLY":int(len(a)),
        "anonymous_topological_component_count":int(len(b)),
        "source_topology_component_face_purity":component_purity,
        "source_slot_reference_face_purity":source_group_purity,
        "original_source_setup_slots_split_by_topology":source_group_splits,
        "topological_components_mixing_multiple_original_author_slots":prediction_component_merges,
        "tautological_one_to_one_component_match":exact_matching,
        "source_merged_graph_retains_original_artist_component_seams":True,
        "this_is_true_artist_drawable_recovery":False,
        "source_original_RGBA_composite_qualified":False,
        "full_spine_transform_IK_qualified":False,
        "heldout_animated_artist_edit_equivalence_qualified":False,
        "original_unsupported_source_attachment_count":len(
            hidden["unsupported_original_attachment_cases_teacher_only"]),
        "original_author_slot_count_not_exposed_during_predict":True,
        "actual_product_image_only_input_benchmark":False,
        "graphics_owner_authorized":False,
        "training_authorized":False,
    }

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("predict")
    a.add_argument("--anonymous-input",type=Path,required=True)
    a.add_argument("--prediction",type=Path,required=True)
    b=sub.add_parser("audit")
    b.add_argument("--anonymous-input",type=Path,required=True)
    b.add_argument("--prediction",type=Path,required=True)
    b.add_argument("--private-reference",type=Path,required=True)
    b.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    merged=json.loads(args.anonymous_input.read_text())
    if args.command=="predict":
        result=predict(merged)
        args.prediction.parent.mkdir(parents=True,exist_ok=True)
        args.prediction.write_bytes(canonical_bytes(result))
        print(json.dumps({"status":result["status"],
                          "predicted_topological_component_count":
                              result["predicted_topological_component_count"],
                          "source_artist_topology_leakage_risk":True,
                          "training_authorized":False},sort_keys=True))
    else:
        prior=json.loads(args.prediction.read_text())
        hidden=json.loads(args.private_reference.read_text())
        result=audit(merged,prior,hidden)
        args.receipt.parent.mkdir(parents=True,exist_ok=True)
        args.receipt.write_bytes(canonical_bytes(result))
        print(json.dumps(result,sort_keys=True))
if __name__=="__main__":
    main()
