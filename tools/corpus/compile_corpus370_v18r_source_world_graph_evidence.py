#!/usr/bin/env python3
"""V18r real original-source pose geometry evidence for V18q graph hypotheses.

Original source V13b Blender evaluated 3D vertices are independently bound
to exact V18a rest-geometry, source SHA and V13b intervention bytes. Source
FIT poses 0/1 compute candidate geometry; authored pose2 is heldout scoring
ONLY. No heldout response/render outcome drives proposals, no UV/material
continuity or graphic owner labels are invented.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA="RealSaS.Corpus370SourceWorldGraphGeometryEvidence.v18r"
STATUS="SOURCE_POSE_GEOMETRY_BOUND_TO_TYPED_GRAPH__NO_GRAPHIC_OWNER_PROOF"
GAPS=(.005,.02,.05)
CREASE_DEGREES=30.
DENY=("graphics_owner_authorized","iris_training_authorized",
      "tessa_training_authorized","materialization_authorized",
      "training_authorized")


def require(ok,why):
    if not ok: raise ValueError(why)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _rounded(value):
    return round(float(value),10)


def original_geometry_metrics(faces,baseline,topology_ids,local_graph,lifted):
    """Pure source geometry; candidate retrieval is received, never refitted."""
    faces=np.asarray(faces,dtype=np.int64)
    baseline=np.asarray(baseline,dtype=np.float64)
    topology_ids=np.asarray(topology_ids,dtype=np.int64)
    n=len(faces)
    require(faces.ndim==2 and faces.shape==(n,3) and n>2
        and baseline.ndim==3 and baseline.shape[0]==3
        and baseline.shape[2]==3
        and np.isfinite(baseline).all()
        and len(topology_ids)==n and np.all(topology_ids>=0)
        and len(local_graph)>=0 and len(lifted)>=0
        and faces.min()>=0 and faces.max()<baseline.shape[1],
        "V18R_SOURCE_3D_FACE_OR_AUTHOR_POSE_SHAPE_INVALID")
    extent=float(np.linalg.norm(np.ptp(baseline[0],axis=0)))
    require(extent>1e-9,"V18R_SOURCE_ZERO_EXTENT")
    tri=baseline[:,faces]  # P3,F,vertex3,xyz3
    normals=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0])
    magnitude=np.linalg.norm(normals,axis=-1)
    valid=magnitude>max(1e-12,extent*extent*1e-12)
    unit=normals/np.maximum(magnitude[...,None],1e-24)
    vids={}
    for island in np.unique(topology_ids):
        idx=np.flatnonzero(topology_ids==island)
        vids[int(island)]=np.unique(faces[idx].reshape(-1))
    face_by_island={int(k):np.flatnonzero(topology_ids==k)
                    for k in vids}
    # Physical source vertex-to-vertex gap: only a hypothesis screening
    # measure, NOT an exact triangle surface distance or editor ownership.
    def gap_from_vertices(a,b,pidx):
        x=baseline[pidx,vids[a]]
        y=baseline[pidx,vids[b]]
        return float(np.min(np.linalg.norm(x[:,None,:]-y[None,:,:],axis=2))/extent)
    local=[]
    degenerate=0
    for e in local_graph:
        a=e["face_a_source_teacher_only"];bb=e["face_b_source_teacher_only"]
        require(type(a) is int and type(bb) is int
            and 0<=a<n and 0<=bb<n and a<bb,
            "V18R_LOCAL_SOURCE_FACE_EDGE_ORDER")
        ang=[]
        for pose in range(3):
            if not (valid[pose,a] and valid[pose,bb]):
                ang.append(None)
            else:
                angle=float(np.degrees(np.arccos(np.clip(
                    np.dot(unit[pose,a],unit[pose,bb]),-1.,1.))))
                ang.append(_rounded(angle))
        if ang[0] is None or ang[1] is None: degenerate+=1
        fit_normal_crease=bool(
            ang[0] is not None and ang[1] is not None
            and ang[0]>=CREASE_DEGREES and ang[1]>=CREASE_DEGREES)
        local.append({
            "face_a_source_teacher_only":a,
            "face_b_source_teacher_only":bb,
            "source_vertex_shared_edge":True,
            "fit_pose_0_normal_angle_degrees":ang[0],
            "fit_pose_1_normal_angle_degrees":ang[1],
            "heldout_pose_2_normal_angle_degrees_court_only":ang[2],
            "fit_consistent_geometric_normal_crease_30deg":fit_normal_crease,
            "heldout_geometric_crease_30deg_court_only":
                (None if ang[2] is None else bool(ang[2]>=CREASE_DEGREES)),
            "source_mechanical_any_axis_divergent":
                e["source_any_axis_above_original_threshold"],
            "source_mechanical_pooled_divergent":
                e["source_pooled_above_original_threshold"],
            "source_geometric_crease_is_not_artist_boundary":True,
            "source_uv_material_relation":"UNOBSERVED",
            "source_occlusion_relationship":"UNOBSERVED",
            "source_graphic_owner_separation":"UNKNOWN",
        })
    candidate=[]
    for e in lifted:
        a=e["topology_island_a_source_teacher_only"]
        bb=e["topology_island_b_source_teacher_only"]
        require(type(a) is int and type(bb) is int
            and a in vids and bb in vids and a<bb,
            "V18R_LIFTED_SOURCE_ISLAND_ADDRESSES_INVALID")
        g0=gap_from_vertices(a,bb,0)
        g1=gap_from_vertices(a,bb,1)
        gh=gap_from_vertices(a,bb,2)
        fit_far=max(g0,g1)
        # Report world local proximity on fit and heldout separately; no
        # ratio or source-nearest threshold is used as SAME owner admission.
        candidate.append({
            "source_island_a_teacher_only":a,
            "source_island_b_teacher_only":bb,
            "fit_pose_0_min_VERTEX_to_VERTEX_gap_per_extent":_rounded(g0),
            "fit_pose_1_min_VERTEX_to_VERTEX_gap_per_extent":_rounded(g1),
            "heldout_pose_2_min_VERTEX_to_VERTEX_gap_per_extent_COURT_ONLY":
                _rounded(gh),
            "source_fit_both_pose_min_vertex_gap_le_0p005":
                bool(fit_far<=GAPS[0]),
            "source_fit_both_pose_min_vertex_gap_le_0p02":
                bool(fit_far<=GAPS[1]),
            "source_fit_both_pose_min_vertex_gap_le_0p05":
                bool(fit_far<=GAPS[2]),
            "source_heldout_min_vertex_gap_le_0p02_COURT_ONLY":
                bool(gh<=GAPS[1]),
            "fit_pose_gap_max_relative_extent":_rounded(fit_far),
            "heldout_pose_gap_relative_extent_COURT_ONLY":_rounded(gh),
            "source_surface_exact_gap":"NOT_MEASURED_VERTEX_PROXY_ONLY",
            "source_vertex_gap_does_not_establish_surface_or_uv_continuity":True,
            "source_image_uv_material_appearance":"UNOBSERVED",
            "original_artist_owner_merge":"UNKNOWN",
            "source_lifted_is_only_proximity_screen":True,
        })
    summary=[]
    for threshold in GAPS:
        fit=[z for z in candidate if z["fit_pose_gap_max_relative_extent"]<=threshold]
        held=[z for z in fit if z[
          "heldout_pose_gap_relative_extent_COURT_ONLY"]<=threshold]
        summary.append({
            "normalized_min_vertex_gap_threshold":threshold,
            "fit_both_pose_candidate_count":len(fit),
            "heldout_pose_remains_near_count_COURT_ONLY":len(held),
            "heldout_pose_separates_count_COURT_ONLY":len(fit)-len(held),
            "not_drawable_owner_merge_count":0,
        })
    local_summary={
        "source_adjacent_edge_count":len(local),
        "fit_source_normal_crease_30deg_edge_count":sum(
            e["fit_consistent_geometric_normal_crease_30deg"] for e in local),
        "fit_both_pose_valid_normals_count":len(local)-degenerate,
        "fit_normal_crease_and_single_axis_mechanical_divergence_count":sum(
            e["fit_consistent_geometric_normal_crease_30deg"] and
            e["source_mechanical_any_axis_divergent"] for e in local),
        "fit_normal_crease_and_pooled_mechanical_divergence_count":sum(
            e["fit_consistent_geometric_normal_crease_30deg"] and
            e["source_mechanical_pooled_divergent"] for e in local),
        "heldout_crease_remains_for_fit_crease_COURT_ONLY":sum(
            e["fit_consistent_geometric_normal_crease_30deg"] and
            e["heldout_geometric_crease_30deg_court_only"] is True for e in local),
        "normal_orientation_may_be_geometry_winding_not_editor_boundary":True,
    }
    return {
        "source_fit_pose_ordinals":[0,1],
        "source_original_heldout_pose_ordinal_COURT_ONLY":2,
        "source_total_faces":n,
        "source_total_vertices":int(baseline.shape[1]),
        "source_topology_islands":len(vids),
        "source_original_3d_extent_world_teacher_only":_rounded(extent),
        "source_local_geometry_edge_evidence_private":local,
        "source_lifted_world_proximity_evidence_private":candidate,
        "source_world_proximity_sensitivity_summary":summary,
        "source_local_crease_summary":local_summary,
        "source_CAA_or_UV_appearance_layer_evidence":"NOT_PRESENT_IN_THIS_SOURCE_GEOMETRY_AUDIT",
        "source_material_alpha_source":"NOT_PRESENT_IN_THIS_SOURCE_GEOMETRY_AUDIT",
        "original_source_artist_graphic_ownership":"UNKNOWN",
        "no_hard_merge_or_cannot_link_minted":True,
        "heldout_pose_not_used_to_generate_candidates":True,
        "source_teacher_world_vertices_not_student_model_input":True,
        "source_geometric_proximity_only_is_not_drawable_proof":True,
    }


def compile_source(v18q_ir,v18q_court,v18a_manifest,v18a_npz,
                   v13b_manifest,v13b_array,v13b_court,
                   expected_v18q_court_sha,expected_v13b_court_sha):
    require(len(expected_v18q_court_sha)==64
        and len(expected_v13b_court_sha)==64
        and sha(v18q_court)==expected_v18q_court_sha
        and sha(v13b_court)==expected_v13b_court_sha,
        "V18R_ORIGINAL_V18Q_V13B_PARENT_COURT_SHA_PIN")
    g=read(v18q_ir);gc=read(v18q_court)
    a=read(v18a_manifest);m=read(v13b_manifest);mc=read(v13b_court)
    require(g.get("schema")=="RealSaS.Corpus370TypedGraphAffinityHypothesisIR.v18q"
        and gc.get("schema")=="RealSaS.Corpus370IndependentTypedGraphAffinityHypothesisCourt.v18q"
        and gc.get("status")=="ORIGINAL_SOURCE_GRAPH_HYPOTHESES_REPLAYED__NO_OWNER_OR_STUDENT_GOLD"
        and gc.get("source_original_graph_hypothesis_ir_sha256")==sha(v18q_ir)
        and a.get("schema")=="RealSaS.Corpus370NativeReactiveStainCandidate.v18a"
        and a.get("private_candidate_npz_sha256")==sha(v18a_npz)
        and a.get("parent_source_immutable_sha256",{}).get(
            "v13b_source_intervention")==sha(v13b_array)
        and a.get("parent_source_immutable_sha256",{}).get(
            "source_rest_geometry")==m.get("rest_source_sha256")
        and m.get("source_pose_articulation_array_sha256")==sha(v13b_array)
        and m.get("schema")=="RealSaS.Corpus370NativeCarrierActuatorInterventionRender.v13b"
        and m.get("status")==
            "V13B_REAL_ORIGINAL_SOURCE_ANONYMOUS_NATIVE_CARRIER_INTERVENTION__INDEPENDENT_COURT_REQUIRED"
        and mc.get("schema")=="RealSaS.Corpus370AnonymousNativeMixedCarrierResponseCourt.v13b"
        and mc.get("status")=="REAL_ORIGINAL_NATIVE_RIG_OR_OBJECT_ACTUATOR_RESPONSE_OBSERVED_NO_OWNER"
        and mc.get("parents",{}).get(
            "native_anonymous_intervention_manifest_sha256")==sha(v13b_manifest)
        and mc.get("parents",{}).get(
            "native_anonymous_intervention_vertex_array_sha256")==sha(v13b_array)
        and g.get("original_source_sha256")==gc.get("original_source_sha256")==
            a.get("source_sha256")==m.get("source_sha256")==mc.get("source_sha256")
        and g.get("subject_id_private")==gc.get("original_subject_id_private")==
            a.get("subject_id")==m.get("subject_id")==mc.get("subject_id")
        and all(g.get(k) is False and gc.get(k) is False for k in DENY)
        and mc.get("graphics_owner_authorized") is False
        and mc.get("training_authorized") is False
        and m.get("training_authorized") is False
        and m.get("graphics_owner_authorized") is False
        and a.get("fit_source_authored_poses")==[0,1]
        and a.get("heldout_source_authored_pose")==2
        and m.get("original_source_teacher_pose_count")==3
        and len(m.get("original_selected_pose_indices_teacher_only",[]))==3,
        "V18R_SOURCE_ORIGINAL_POSE_AND_V18A_V13B_V18Q_CHAIN_NOT_BOUND")
    with np.load(v18a_npz,allow_pickle=False) as d:
        faces=np.asarray(d["faces"],np.int64)
    with np.load(v13b_array,allow_pickle=False) as d:
        require(set(d.files)=={
            "baseline_vertices","native_control_perturbed_vertices"},
            "V18R_ORIGINAL_V13B_VERTEX_ARRAY_MEMBERS_INVALID")
        baseline=np.asarray(d["baseline_vertices"],np.float64)
    graph=g["source_private_graph"]
    require(len(faces)==m["source_face_count"]==graph["source_total_faces"]
        and baseline.shape==(3,m["source_rest_vertex_count"],3),
        "V18R_SOURCE_TRIANGLE_POSE_VERTEX_CENSUS")
    geom=original_geometry_metrics(
        faces,baseline,
        graph["source_face_observation_ir_teacher_private"][
            "source_topology_island_per_face_teacher_private"],
        graph["source_local_graph_affinity_evidence_ir_teacher_private"],
        graph["source_nonlocal_graph_affinity_hypotheses_teacher_private"])
    require(geom["source_topology_islands"]==
        graph["source_original_topology_island_count"]
        and len(geom["source_local_geometry_edge_evidence_private"])==
            graph["source_original_local_edge_count"]
        and len(geom["source_lifted_world_proximity_evidence_private"])==
            graph["source_original_lifted_candidate_count"],
        "V18R_SOURCE_GRAPH_EVIDENCE_CENSUS_OR_IDENTITY")
    return {
        "schema":SCHEMA,"status":STATUS,
        "subject_id_private":g["subject_id_private"],
        "original_source_sha256":g["original_source_sha256"],
        "original_source_art_3d_geometry_teacher_only":True,
        "original_authored_poses_teacher_only":
            m["original_selected_pose_indices_teacher_only"],
        "parents_sha256":{
            "v18q_source_ir":sha(v18q_ir),
            "v18q_independent_court":sha(v18q_court),
            "v18a_source_manifest":sha(v18a_manifest),
            "v18a_original_arrays":sha(v18a_npz),
            "v13b_source_pose_manifest":sha(v13b_manifest),
            "v13b_actual_source_pose_vertices":sha(v13b_array),
            "v13b_independent_court":sha(v13b_court)},
        "original_source_world_evidence_teacher_private":geom,
        "visual_surface_appearance_and_uv_boundary":"UNOBSERVED_NOT_FABRICATED",
        "source_graphics_compiler_drawable_owner":"UNKNOWN",
        "source_geometry_proximity_hypotheses_are_not_graphics_owner_labels":True,
        "source_heldout_pose2_only_court_no_fit":True,
        "same_drawable_training_pairs_qualified":0,
        "different_drawable_training_pairs_qualified":0,
        "graphics_owner_authorized":False,
        "iris_training_authorized":False,
        "tessa_training_authorized":False,
        "materialization_authorized":False,
        "training_authorized":False,
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("v18q_ir","v18q_court","v18a_manifest","v18a_npz",
                 "v13b_manifest","v13b_array","v13b_court","output"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    parser.add_argument("--expected-v18q-court-sha",required=True)
    parser.add_argument("--expected-v13b-court-sha",required=True)
    args=parser.parse_args()
    result=compile_source(args.v18q_ir,args.v18q_court,args.v18a_manifest,
         args.v18a_npz,args.v13b_manifest,args.v13b_array,args.v13b_court,
         args.expected_v18q_court_sha,args.expected_v13b_court_sha)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                     allow_nan=False)+"\n")
    g=result["original_source_world_evidence_teacher_private"]
    print(json.dumps({"status":result["status"],
      "subject":result["subject_id_private"],
      "local_crease_summary":g["source_local_crease_summary"],
      "lifted_vertex_world_gap_sensitivity":
          g["source_world_proximity_sensitivity_summary"],
      "source_world_heldout_not_used_for_fit":True,
      "training_authorized":False},sort_keys=True),flush=True)


if __name__=="__main__":main()
