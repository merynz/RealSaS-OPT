#!/usr/bin/env python3
"""V18p: identify V18a source topology ceiling versus measured response cuts.

Read-only independent, SHA-pinned original source. No segmentation semantics,
no hidden artist labels, no student targets. A shared geometric edge is NOT
evidence of same drawable; separate source vertex islands are NOT proof of
different drawable. Both structures are diagnostic only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    SCHEMA as STAIN_SCHEMA, STATUS as STAIN_STATUS,
    THRESHOLDS, face_edges, component_labels
)
from tools.corpus.court_corpus370_v18o_anonymous_partition_admissibility import (
    SCHEMA as V18O_SCHEMA, STATUS as V18O_STATUS,
)

SCHEMA="RealSaS.Corpus370IndependentResponseTopologyIdentifiabilityCourt.v18p"
STATUS="SOURCE_TOPOLOGY_AND_AXIS_RESPONSE_ABLATION_MEASURED__DRAWABLE_OWNER_UNKNOWN"
FROZEN=(.045,.1,.18)
FLAGS=("graphics_owner_authorized","iris_training_authorized",
       "tessa_training_authorized","materialization_authorized",
       "training_authorized")


def require(ok,code):
    if not ok:raise ValueError(code)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_labels(n,edges,allow=None):
    parent=np.arange(n,dtype=np.int64)
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    if allow is None:
        allow=np.ones(len(edges),dtype=bool)
    for (a,b),valid in zip(edges,allow):
        if valid:
            a,b=root(int(a)),root(int(b))
            if a!=b:parent[max(a,b)]=min(a,b)
    raw=np.asarray([root(i) for i in range(n)],dtype=np.int64)
    return np.unique(raw,return_inverse=True)[1]


def counts_for_labels(labels):
    counts=np.bincount(labels)
    weights=counts/counts.sum()
    return {
        "component_count":int(len(counts)),
        "largest_faces":int(counts.max()),
        "singleton_count":int(np.count_nonzero(counts==1)),
        "effective_component_count":float(1./np.sum(weights*weights)),
    }


def source_audit(faces,features,edges,axis_count,original_labels):
    faces=np.asarray(faces,dtype=np.int64)
    feat=np.asarray(features,dtype=np.float64)
    edges=np.asarray(edges,dtype=np.int64)
    n=len(faces)
    require(faces.shape==(n,3) and n>2
            and feat.ndim==2 and feat.shape[0]==n
            and axis_count>=2 and feat.shape[1]==20*axis_count+8
            and edges.ndim==2 and edges.shape[1]==2
            and np.isfinite(feat).all() and (edges>=0).all()
            and (edges<n).all() and
            np.array_equal(edges,np.unique(edges,axis=0))
            and np.array_equal(edges,face_edges(faces)),
            "V18P_SOURCE_ORIGINAL_TOPOLOGY_AND_MULTICHANNEL_RESPONSE_INVALID")
    topo=canonical_labels(n,edges)
    topo_profile=counts_for_labels(topo)
    cut_dist=np.sqrt(np.mean(np.square(
        feat[edges[:,0]]-feat[edges[:,1]]),axis=1)) if len(edges) else np.zeros(0)
    per_axis=np.stack([
        np.sqrt(np.mean(np.square(
            feat[edges[:,0],i*20:(i+1)*20]-
            feat[edges[:,1],i*20:(i+1)*20]),axis=1))
        for i in range(axis_count)],axis=1) if len(edges) else np.zeros((0,axis_count))
    nonlinear=np.sqrt(np.mean(np.square(
        feat[edges[:,0],-8:]-feat[edges[:,1],-8:]),axis=1)) if len(edges) else np.zeros(0)
    thresholds=[]
    for t in FROZEN:
        expected=np.asarray(original_labels[t],np.int64)
        reproduced=component_labels(feat.astype(np.float32),edges,t)
        require(np.array_equal(expected,reproduced),
                "V18P_ORIGINAL_SOURCE_V18A_FACE_PARTITION_NOT_REPRODUCED")
        # Impose no *new* partitions as gold. Axis-only ablation is a
        # stress diagnostic of identifiability, not a trained student label.
        per_axis_partition=[
            counts_for_labels(canonical_labels(n,edges,per_axis[:,k]<=t))
            for k in range(axis_count)]
        observed=counts_for_labels(reproduced)
        cuts=cut_dist>t
        # Every V18a group is a connected subset of an existing topology
        # component; its part-count floor comes from artist source topology.
        is_consistent=bool(np.all(topo[edges[:,0]]==topo[edges[:,1]]))
        thresholds.append({
            "frozen_original_similarity_threshold":float(t),
            "original_V18a_candidates":observed,
            "topological_component_floor":topo_profile["component_count"],
            "original_candidate_count_above_topology_floor":
                observed["component_count"]-topo_profile["component_count"],
            "original_fraction_candidate_count_from_topology_floor":
                float(topo_profile["component_count"]/observed["component_count"]),
            "shared_source_mesh_edge_count":int(len(edges)),
            "shared_source_edges_cut_by_response":int(np.count_nonzero(cuts)),
            "fraction_original_shared_edges_cut_by_response":
                float(np.mean(cuts)) if len(cuts) else None,
            "source_edges_discriminated_by_at_least_one_axis":
                int(np.count_nonzero(np.any(per_axis>t,axis=1))),
            "source_edges_not_discriminated_by_any_axis":
                int(np.count_nonzero(~np.any(per_axis>t,axis=1))),
            "source_edges_where_joint_is_above_threshold_but_no_single_axis_is":
                int(np.count_nonzero(cuts & ~np.any(per_axis>t,axis=1))),
            "source_edges_only_one_axis_exceeds_threshold":
                int(np.count_nonzero(np.sum(per_axis>t,axis=1)==1)),
            "source_edges_all_axes_exceed_threshold":
                int(np.count_nonzero(np.all(per_axis>t,axis=1))),
            "source_edges_nonlinear_coupling_exceeds_threshold":
                int(np.count_nonzero(nonlinear>t)),
            "source_individual_axis_partition_counts":per_axis_partition,
            "original_source_graph_local_threshold_replay_verified":True,
            "topology_components_contain_all_source_edges":is_consistent,
            "source_semantic_labels_used":False,
            "treat_topology_island_boundary_as_art_drawable_edge":False,
        })
    # Response collapse across *disconnected* source vertex islands is an
    # identifiability witness: motion alone cannot give a guaranteed layer
    # link, but we never merge islands as assumed same graphics.
    sums=np.zeros((topo_profile["component_count"],feat.shape[1]))
    masses=np.bincount(topo).astype(np.int64)
    np.add.at(sums,topo,feat)
    means=sums/masses[:,None]
    # Avoid quadratic memory in the full source; 320 islands <= ~50k pairs.
    pairs=0
    close={str(t):0 for t in FROZEN}
    for i in range(len(means)):
        d=np.sqrt(np.mean(np.square(means[i+1:]-means[i]),axis=1))
        pairs+=len(d)
        for t in FROZEN:
            close[str(t)]+=int(np.count_nonzero(d<=t))
    return {
        "original_source_face_count":n,
        "original_source_topology_island_profile":topo_profile,
        "original_source_selected_control_axis_count":int(axis_count),
        "original_source_signed_response_feature_dimension":int(feat.shape[1]),
        "original_source_graph_shared_edges":int(len(edges)),
        "original_source_topology_island_pair_count":pairs,
        "original_source_different_island_mean_response_near_pairs":close,
        "original_source_topology_island_mean_response_proximity_is_not_artist_same_drawable":True,
        "original_source_threshold_diagnostics":thresholds,
        "original_source_teacher_axis_and_face_ids_never_model_inputs":True,
        "original_source_artist_same_drawable_pairs_minted":0,
        "original_source_artist_different_drawable_pairs_minted":0,
    }


def audit(stain_report,stain_npz,stain_court,v18o_court,
          expected_stain_court_sha,expected_v18o_court_sha):
    require(all(isinstance(z,str) and len(z)==64 for z in (
        expected_stain_court_sha,expected_v18o_court_sha))
        and sha(stain_court)==expected_stain_court_sha
        and sha(v18o_court)==expected_v18o_court_sha,
        "V18P_ORIGINAL_SOURCE_PARENT_COURT_SHA_INVALID")
    s=load(stain_report);sc=load(stain_court);v=load(v18o_court)
    require(s.get("schema")==STAIN_SCHEMA
        and s.get("status")==STAIN_STATUS
        and sc.get("schema")==
            "RealSaS.Corpus370IndependentReactiveStainProposalCourt.v18a"
        and sc.get("status")==
            "ORIGINAL_NATIVE_SOURCE_STAIN_GRAPH_REPRODUCED__ARTIST_EDIT_TRUTH_UNKNOWN"
        and sc.get("private_stain_report_sha256")==sha(stain_report)
        and sc.get("private_stain_arrays_sha256")==sha(stain_npz)
        and v.get("schema")==V18O_SCHEMA and v.get("status")==V18O_STATUS
        and v.get("exact_v18a_stain_report_sha256")==sha(stain_report)
        and v.get("exact_v18a_label_array_sha256")==sha(stain_npz)
        and v.get("exact_v18a_independent_court_sha256")==sha(stain_court)
        and s.get("source_sha256")==sc.get("original_source_sha256")==
            v.get("original_source_sha256")
        and s.get("subject_id")==sc.get("subject_id")==
            v.get("original_source_id_private")
        and s.get("fit_source_authored_poses")==[0,1]
        and s.get("heldout_source_authored_pose")==2
        and s.get("source_graphics_owner_authorized") is False
        and all(v.get(k) is False for k in FLAGS)
        and all(s.get(k) is False for k in FLAGS if k!="graphics_owner_authorized"),
        "V18P_PRIVATE_V18A_V18O_SOURCE_AND_AUTHORITY_BINDING")
    with np.load(stain_npz,allow_pickle=False) as f:
        needed={"faces","stain_channels","graph_edges",
                "selected_source_control_axes_teacher_only"}|{
                    "labels_"+str(t).replace(".","p") for t in FROZEN}
        require(set(f.files)==needed,
                "V18P_SOURCE_FROZEN_LABEL_ARRAY_KEYS_CHANGED")
        faces=np.asarray(f["faces"],np.int64)
        features=np.asarray(f["stain_channels"],np.float64)
        edges=np.asarray(f["graph_edges"],np.int64)
        axes=np.asarray(f["selected_source_control_axes_teacher_only"],np.int64)
        require(axes.ndim==2 and axes.shape[1]==2,
                "V18P_SOURCE_SELECTED_AGENT_AXIS_SHAPE")
        labels={t:np.asarray(f["labels_"+str(t).replace(".","p")],np.int32)
                for t in FROZEN}
    result=source_audit(faces,features,edges,len(axes),labels)
    require(result["original_source_face_count"]==s.get("source_face_count")
        and result["original_source_signed_response_feature_dimension"]==
            s.get("feature_channel_count")
        and result["original_source_graph_shared_edges"]==
            s.get("true_shared_edge_graph_edges"),
        "V18P_SOURCE_GRAPH_AND_SIGNED_RESPONSE_CENSUS_CHANGED")
    for row,orig in zip(result["original_source_threshold_diagnostics"],
                        v["all_original_source_frozen_thresholds"]):
        require(row["original_V18a_candidates"]["component_count"]==
            orig["candidate_component_count"]
            and row["original_V18a_candidates"]["largest_faces"]==
            orig["largest_component_faces"]
            and row["original_V18a_candidates"]["singleton_count"]==
            orig["singleton_component_count"],
            "V18P_SOURCE_INDEPENDENT_V18O_PARTITION_METRIC_MISMATCH")
    return {
        "schema":SCHEMA,"status":STATUS,
        "subject_id_private":s["subject_id"],
        "original_source_sha256":s["source_sha256"],
        "exact_v18a_manifest_sha256":sha(stain_report),
        "exact_v18a_array_sha256":sha(stain_npz),
        "exact_v18a_independent_court_sha256":sha(stain_court),
        "exact_v18o_independent_court_sha256":sha(v18o_court),
        "source_identifiability_audit":result,
        "new_drawable_partition_constructed":False,
        "source_response_shared_or_different_does_not_prove_graphic_owner":True,
        "original_source_artist_drawable_owner":"UNKNOWN",
        "face_topology_connectivity_is_not_graphic_owner":True,
        "source_geometry_face_ontology_not_promoted":True,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    parser=argparse.ArgumentParser()
    for k in ("stain_report","stain_npz","stain_court","v18o_court","output"):
        parser.add_argument("--"+k.replace("_","-"),required=True,type=Path)
    parser.add_argument("--expected-stain-court-sha",required=True)
    parser.add_argument("--expected-v18o-court-sha",required=True)
    a=parser.parse_args()
    v=audit(a.stain_report,a.stain_npz,a.stain_court,a.v18o_court,
            a.expected_stain_court_sha,a.expected_v18o_court_sha)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":v["status"],
        "subject":v["subject_id_private"],
        "audit":v["source_identifiability_audit"],
        "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
