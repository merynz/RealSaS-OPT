#!/usr/bin/env python3
"""V18q typed source-private graph evidence and NON-authoritative hypotheses.

V18q extends, not replaces, the previously courted per-ray Graphics Evidence
IR v1. It joins original V18a source mechanical axes and true source face
topology to a separately typed, UNKNOWN-first graph. No appearance, image,
UV, artist ownership or heldout source-edit outcome can affect proposal
generation because none is supplied to the graph predictor.

Original source V18o heldout EDIT measurements are carried *only as court
observations* and never used to choose graph edges, thresholds or candidates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.corpus.compile_corpus370_v18a_reactive_staining import (
    SCHEMA as V18A_SCHEMA, STATUS as V18A_STATUS, THRESHOLDS,
    component_labels, face_edges
)
from tools.corpus.court_corpus370_v18p_response_topology_identifiability import (
    SCHEMA as V18P_SCHEMA, STATUS as V18P_STATUS,
    canonical_labels, counts_for_labels, source_audit
)
from tools.corpus.court_corpus370_v18o_anonymous_partition_admissibility import (
    SCHEMA as V18O_SCHEMA, STATUS as V18O_STATUS
)

SCHEMA="RealSaS.Corpus370TypedGraphAffinityHypothesisIR.v18q"
STATUS="PRIVATE_SOURCE_GRAPH_EVIDENCE_AND_ALTERNATIVE_HYPOTHESES__ARTIST_OWNER_UNKNOWN"
DEFAULT_THRESHOLD=.1
NEIGHBORS_PER_ISLAND=6
MAX_LIFTED=4096
NONPROMOTION=("graphics_owner_authorized","iris_training_authorized",
              "tessa_training_authorized","materialization_authorized",
              "training_authorized")

def require(ok,reason):
    if not ok:raise ValueError(reason)

def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def metric_counts(labels):
    v=counts_for_labels(np.asarray(labels,dtype=np.int64))
    return {
       "K":v["component_count"],"largest_source_faces":v["largest_faces"],
       "single_face_components":v["singleton_count"],
       "effective_K_source_face_mass":v["effective_component_count"],
    }

def graph_proposals(faces,stain,edges,axes,original_labels):
    """Pure FIT-only original source V18a arrays; no source pixel outcomes."""
    faces=np.asarray(faces,np.int64)
    stain=np.asarray(stain,np.float64)
    edges=np.asarray(edges,np.int64)
    axes=np.asarray(axes,np.int64)
    require(faces.ndim==2 and faces.shape[1]==3 and len(faces)>=3
        and stain.shape==(len(faces),20*len(axes)+8)
        and axes.ndim==2 and axes.shape[1]==2 and len(axes)>=2
        and np.isfinite(stain).all()
        and np.array_equal(edges,face_edges(faces))
        and all(np.array_equal(
                component_labels(stain.astype(np.float32),edges,t),
                original_labels[t]) for t in THRESHOLDS),
        "V18Q_ORIGINAL_FIT_ONLY_SOURCE_TOPOLOGY_AXIS_AND_STAIN_CONTRACT")
    k=len(axes);n=len(faces)
    topo=canonical_labels(n,edges)
    islands=len(np.unique(topo))
    t=DEFAULT_THRESHOLD
    peraxis=np.stack([
        np.sqrt(np.mean(np.square(
           stain[edges[:,0],20*i:20*(i+1)]-
           stain[edges[:,1],20*i:20*(i+1)]),axis=1))
        for i in range(k)],axis=1) if len(edges) else np.empty((0,k))
    pooled=np.sqrt(np.mean(np.square(
        stain[edges[:,0]]-stain[edges[:,1]]),axis=1)) if len(edges) else np.empty(0)
    nl=np.sqrt(np.mean(np.square(stain[edges[:,0],-8:]-
                               stain[edges[:,1],-8:]),axis=1)) if len(edges) else np.empty(0)
    # Canonical local source edges: signed evidence carries *real* independent
    # responses, but every owner decision remains explicitly UNKNOWN.
    source_local=[]
    for ix,(a,b) in enumerate(edges):
        distances=[float(v) for v in peraxis[ix]]
        source_local.append({
          "face_a_source_teacher_only":int(a),
          "face_b_source_teacher_only":int(b),
          "support":"SOURCE_TRUE_SHARED_VERTEX_EDGE",
          "source_mechanical_pooled_distance":float(pooled[ix]),
          "source_mechanical_per_axis_distance":distances,
          "source_nonlinear_joint_distance":float(nl[ix]),
          "source_pooled_above_original_threshold":bool(pooled[ix]>t),
          "source_any_axis_above_original_threshold":bool(any(x>t for x in distances)),
          "source_all_axes_below_or_equal_original_threshold":
              bool(all(x<=t for x in distances)),
          "original_source_uv_material_boundary":"UNOBSERVED",
          "original_source_cross_view_raster_relation":"UNOBSERVED",
          "source_editor_independent_control_relation":"UNKNOWN",
          "can_merge_artist_drawable":"UNKNOWN",
          "must_separate_artist_drawable":"UNKNOWN",
          "evidence_restriction":"MECHANICAL_AND_SHARED_TOPOLOGY_ONLY",
        })
    # Lifted graph hypotheses across *disconnected original vertex-index
    # topology islands*. The response similarity is only a retrieval key:
    # WITHOUT source geometry UV/material/appearance relation there is no
    # admissible owner merge and no committed union across islands.
    totals=np.bincount(topo,minlength=islands)
    summed=np.zeros((islands,stain.shape[1]),dtype=np.float64)
    np.add.at(summed,topo,stain)
    center=summed/totals[:,None]
    wanted=[]
    for i in range(islands):
        d=np.sqrt(np.mean(np.square(center-center[i]),axis=1))
        d[:i+1]=np.inf
        hits=np.flatnonzero(d<=t)
        if len(hits):
            chosen=sorted(hits,key=lambda j:(float(d[j]),int(j)))[:NEIGHBORS_PER_ISLAND]
            for j in chosen:
                wanted.append((float(d[j]),int(i),int(j)))
    wanted.sort(key=lambda v:(v[0],v[1],v[2]))
    wanted=wanted[:MAX_LIFTED]
    lifted=[{
       "topology_island_a_source_teacher_only":i,
       "topology_island_b_source_teacher_only":j,
       "source_mean_mechanical_response_distance":d,
       "source_lifted_hypothesis_type":"NONLOCAL_RESPONSE_SIMILARITY_LOOKUP_ONLY",
       "original_source_spatial_surface_proximity":"UNOBSERVED",
       "original_source_uv_or_material_continuity":"UNOBSERVED",
       "original_source_occlusion_and_visibility_continuity":"UNOBSERVED",
       "source_edit_effect_separability":"UNKNOWN",
       "can_merge_artist_drawable":"UNKNOWN",
       "must_separate_artist_drawable":"UNKNOWN",
       "source_appearance_independent_confirmation_required":True,
    } for d,i,j in wanted]
    # Axis OR is a research STRESS TEST. If even one axis is distinct,
    # keep a split in this *hypothesis*, never mint a cannot-link edge.
    hypothesis_axis_all=canonical_labels(n,edges,
                                 np.all(peraxis<=t,axis=1))
    baseline=np.asarray(original_labels[t],dtype=np.int64)
    topo_labels=canonical_labels(n,edges)
    variants=[
       {"candidate_key":"V18A_POOLED_LOCAL_BASELINE","metrics":metric_counts(baseline),
        "source_only_hypothesis_not_artist_truth":True},
       {"candidate_key":"SOURCE_TOPOLOGY_ONLY_CONTROL","metrics":metric_counts(topo_labels),
        "source_only_hypothesis_not_artist_truth":True},
       {"candidate_key":"MULTI_AXIS_OR_CUT_STRESS_ONLY","metrics":metric_counts(hypothesis_axis_all),
        "source_only_hypothesis_not_artist_truth":True},
    ]
    return {
       "source_total_faces":n,
       "source_signed_response_dimension":int(stain.shape[1]),
       "source_selected_axes_count":k,
       "source_original_topology_island_count":islands,
       "source_original_local_edge_count":len(source_local),
       "source_original_lifted_candidate_count":len(lifted),
       "source_face_observation_ir_teacher_private":{
           "source_face_ref":"source_face_index_in_frozen_V18a_original_arrays",
           "source_topology_island_per_face_teacher_private":
               [int(x) for x in topo],
           "source_mechanical_signal_storage":
               "PRIVATE_V18A_ARRAY_SHA_BOUND__stain_channels[face,:]",
           "original_face_uv_material_addresses":"MISSING_IN_THIS_SOURCE_INPUT",
           "original_face_raster_visibility":"MISSING_IN_THIS_SOURCE_INPUT",
           "source_face_artist_owner":"UNKNOWN",
       },
       "source_local_graph_affinity_evidence_ir_teacher_private":source_local,
       "source_nonlocal_graph_affinity_hypotheses_teacher_private":lifted,
       "source_candidate_drawable_hypotheses_teacher_private":variants,
       "source_axis_cut_stress_not_a_calibrated_cannot_link":True,
       "source_lifted_response_proximity_not_a_calibrated_must_link":True,
       "source_nonlocal_merge_edges_admitted":0,
       "source_local_split_edges_admitted":0,
       "source_artist_same_drawable_pairs_minted":0,
       "source_artist_different_drawable_pairs_minted":0,
       "student_target_admitted":False,
    }


def compile_original(stain_manifest,stain_npz,stain_court,
                     v18p_court,v18o_court,
                     expected_stain_sha,expected_v18p_sha,expected_v18o_sha):
    required=(expected_stain_sha,expected_v18p_sha,expected_v18o_sha)
    require(all(isinstance(s,str) and len(s)==64 for s in required)
        and sha(stain_court)==expected_stain_sha
        and sha(v18p_court)==expected_v18p_sha
        and sha(v18o_court)==expected_v18o_sha,
        "V18Q_EXACT_ORIGINAL_PARENTS_SHA")
    m=read(stain_manifest);s=read(stain_court)
    p=read(v18p_court);o=read(v18o_court)
    require(m.get("schema")==V18A_SCHEMA and m.get("status")==V18A_STATUS
        and s.get("private_stain_report_sha256")==sha(stain_manifest)
        and s.get("private_stain_arrays_sha256")==sha(stain_npz)
        and s.get("original_source_sha256")==m.get("source_sha256")
        and p.get("schema")==V18P_SCHEMA and p.get("status")==V18P_STATUS
        and o.get("schema")==V18O_SCHEMA and o.get("status")==V18O_STATUS
        and p.get("exact_v18a_manifest_sha256")==sha(stain_manifest)
        and p.get("exact_v18a_array_sha256")==sha(stain_npz)
        and o.get("exact_v18a_label_array_sha256")==sha(stain_npz)
        and p.get("exact_v18o_independent_court_sha256")==sha(v18o_court)
        and p.get("original_source_sha256")==m.get("source_sha256")
        and o.get("original_source_sha256")==m.get("source_sha256")
        and p.get("subject_id_private")==m.get("subject_id")
        and o.get("original_source_id_private")==m.get("subject_id")
        and m.get("source_graphics_owner_authorized") is False
        and all(p.get(k) is False and o.get(k) is False for k in NONPROMOTION),
        "V18Q_ORIGINAL_SOURCE_MULTICOURT_IDENTITY_AND_NEGATIVE_AUTHORITY")
    with np.load(stain_npz,allow_pickle=False) as z:
        require(set(z.files)=={"faces","stain_channels","graph_edges",
                 "selected_source_control_axes_teacher_only"}|
                {"labels_"+str(t).replace(".","p") for t in THRESHOLDS},
                "V18Q_ORIGINAL_SOURCE_REQUIRED_ARRAYS")
        faces=np.asarray(z["faces"],np.int64)
        features=np.asarray(z["stain_channels"],np.float64)
        edges=np.asarray(z["graph_edges"],np.int64)
        axes=np.asarray(z["selected_source_control_axes_teacher_only"],np.int64)
        labels={t:np.asarray(z["labels_"+str(t).replace(".","p")],np.int32)
                for t in THRESHOLDS}
    graph=graph_proposals(faces,features,edges,axes,labels)
    require(graph["source_total_faces"]==m.get("source_face_count")
        and graph["source_original_local_edge_count"]==
            m.get("true_shared_edge_graph_edges")
        and graph["source_signed_response_dimension"]==
            m.get("feature_channel_count"),
        "V18Q_ORIGINAL_SOURCE_GRAPH_CENSUS")
    # This parent is only a SOURCE HOLDOUT evaluator, never the graph's
    # source features or candidate-selection mechanism.
    edit=o["original_heldout_real_source_material_edit_locus"]
    court={
       "source_observed_original_heldout_edit_count":1,
       "selected_source_group_fraction_all_faces":
           edit["source_selected_candidate_fraction_of_all_faces"],
       "selected_source_group_removal_observable_samples":
           edit["source_selected_group_removal_sample_effects_total"],
       "selected_source_group_isolation_observable_samples":
           edit["source_selected_group_isolation_sample_effects_total"],
       "source_edit_consequence_for_new_graph_hypotheses":"NOT_EVALUATED",
       "source_v18o_edit_outcome_used_to_rank_hypotheses":False,
       "original_source_cross_pose_graphics_owner_stability":"UNKNOWN",
       "original_source_apparent_source_material_alpha_distribution":
           "SEE_SEPARATE_V18N_INDEPENDENT_COURT",
    }
    return {
       "schema":SCHEMA,"status":STATUS,
       "subject_id_private":m["subject_id"],
       "original_source_sha256":m["source_sha256"],
       "parents_sha256":{
         "v18a_report":sha(stain_manifest),
         "v18a_arrays":sha(stain_npz),
         "v18a_independent_court":sha(stain_court),
         "v18p_independent_court":sha(v18p_court),
         "v18o_independent_court":sha(v18o_court),
       },
       "source_private_graph":graph,
       "independent_edit_court_observations_not_for_fitting":court,
       "graph_label_policy":"MECHANICAL_PROPOSAL_ONLY__ARTIST_DRAWABLE_UNKNOWN",
       "source_original_face_uv_material_relation":"NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS",
       "source_original_multiview_visibility_relation":"NOT_PRESENT_IN_SHA_BOUND_V18A_ARRAYS",
       "compiler_graphics_proof":"NOT_QUALIFIED",
       "artist_graphics_ownership":"UNKNOWN",
       "same_drawable_training_pairs_qualified":0,
       "different_drawable_training_pairs_qualified":0,
       "graphics_owner_authorized":False,
       "iris_training_authorized":False,
       "tessa_training_authorized":False,
       "materialization_authorized":False,
       "training_authorized":False,
    }


def main():
    a=argparse.ArgumentParser()
    for name in ("stain_manifest","stain_npz","stain_court",
                 "v18p_court","v18o_court","output"):
        a.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    for name in ("expected_stain_sha","expected_v18p_sha","expected_v18o_sha"):
        a.add_argument("--"+name.replace("_","-"),required=True)
    v=a.parse_args()
    result=compile_original(v.stain_manifest,v.stain_npz,v.stain_court,
                            v.v18p_court,v.v18o_court,
                            v.expected_stain_sha,v.expected_v18p_sha,
                            v.expected_v18o_sha)
    v.output.parent.mkdir(parents=True,exist_ok=True)
    v.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    g=result["source_private_graph"]
    print(json.dumps({"status":result["status"],
          "subject":result["subject_id_private"],
          "topology_islands":g["source_original_topology_island_count"],
          "lifted_response_only_hypotheses":g["source_original_lifted_candidate_count"],
          "source_candidate_complexity":g["source_candidate_drawable_hypotheses_teacher_private"],
          "graphics_owner_authorized":False,
          "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":main()
