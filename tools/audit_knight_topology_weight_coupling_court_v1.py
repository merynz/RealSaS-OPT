from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    deformation_envelope_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    rigging_surface_from_dict,
)
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd, _solve
from tools.audit_knight_residual_topology_severity_court_v1 import _actual_face_edge_max
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch, _edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask, exact, face_indices, load, stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.audit_knight_weight_outlier_signal_court_v1 import teacher_matrix
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_TOPOLOGY_WEIGHT_COUPLING_COURT_PREREG_V1_20260929.json")


def _truth_region(bank,source,sk,cand):
    _,_,tri,sf=teacher_weights(bank,source,sk,cand)
    src_comp=source_components(int(sf.max())+1,sf)
    tri_comp=np.asarray([src_comp[int(row[0])] for row in sf],dtype=np.int64)
    return np.asarray([tri_comp[int(t)] for t in tri],dtype=np.int64)


def _validity_summary(mask,face_any_invalid,face_invalid_count):
    idx=np.where(mask)[0]
    if not len(idx):
        return {"face_count":0}
    invalid=face_any_invalid[idx]
    return {
      "face_count":int(len(idx)),
      "any_teacher_invalid_face_count":int(np.count_nonzero(invalid)),
      "all_teacher_valid_face_count":int(np.count_nonzero(~invalid)),
      "any_teacher_invalid_fraction":float(np.mean(invalid)),
      "invalid_vertex_count_p50":float(np.quantile(face_invalid_count[idx],.5)),
      "invalid_vertex_count_p95":float(np.quantile(face_invalid_count[idx],.95)),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True)
    p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_TOPOLOGY_WEIGHT_COUPLING_RESULT":
        raise RuntimeError("TOPOLOGY_WEIGHT_COUPLING_PREREG_DRIFT")

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    P=np.asarray([v.P for v in cand.vertices],np.float64)
    F=face_indices(cand)
    dense=dense_supported_face_mask(rr,cand,surface)

    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()))
        Wa=np.asarray(z["arachne"],np.float64)
    Wa=np.maximum(Wa,0.0);Wa/=np.maximum(Wa.sum(axis=1,keepdims=True),1e-15)

    base=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),bool)
    unsafe[np.asarray(base["unsafe_face_indices"],np.int64)]=True
    edges,edge_faces,fei=_edge_table(P,F)
    stretch=_edge_stretch(P,Wa,edges,jids,sk,cams,env)
    l1=np.sum(np.abs(Wa[edges[:,0]]-Wa[edges[:,1]]),axis=1)
    jsd=_edge_jsd(Wa,edges)
    dense_edge=np.zeros(len(edges),bool);scope=np.zeros(len(edges),bool)
    for ei,e in enumerate(map(tuple,edges.tolist())):
        fs=edge_faces[e]
        dense_edge[ei]=any(bool(dense[fi]) for fi in fs)
        scope[ei]=any(bool(dense[fi] and unsafe[fi]) for fi in fs)
    rep=scope&(stretch>4.0)&(l1>1.0)
    inferred,solver=_solve(len(P),edges,dense_edge,rep,jsd)
    pred_mixed=np.asarray([len(set(int(inferred[v]) for v in row))>1 for row in F],bool)

    # Teacher starts only here.
    truth=_truth_region(a.teacher_bank,a.teacher_source,sk,cand)
    truth_mixed=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)
    Wteach,valid=teacher_matrix(a.teacher_bank,sk,cand,jids)
    face_invalid_count=np.sum((~valid)[F],axis=1)
    face_any_invalid=face_invalid_count>0

    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1)
    Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())
    actual_e,_,_=_actual_face_edge_max(P,Wt,F,jids,sk,cams,rr,source_report)

    fp=(~truth_mixed)&pred_mixed
    fn=truth_mixed&(~pred_mixed)
    unsafe_fn=unsafe&fn
    catastrophic_fn=fn&(actual_e>10.0)
    unsafe_cat_fn=unsafe&catastrophic_fn

    groups={
      "FALSE_POSITIVE_SPLIT":_validity_summary(fp,face_any_invalid,face_invalid_count),
      "FALSE_NEGATIVE_CROSS_REGION":_validity_summary(fn,face_any_invalid,face_invalid_count),
      "UNSAFE_FALSE_NEGATIVE_CROSS_REGION":_validity_summary(unsafe_fn,face_any_invalid,face_invalid_count),
      "ACTUAL_GT10_FALSE_NEGATIVE_CROSS_REGION":_validity_summary(catastrophic_fn,face_any_invalid,face_invalid_count),
      "UNSAFE_ACTUAL_GT10_FALSE_NEGATIVE_CROSS_REGION":_validity_summary(unsafe_cat_fn,face_any_invalid,face_invalid_count),
      "TRUE_POSITIVE_CROSS_REGION":_validity_summary(truth_mixed&pred_mixed,face_any_invalid,face_invalid_count),
      "TRUE_SAME_REGION":_validity_summary((~truth_mixed)&(~pred_mixed),face_any_invalid,face_invalid_count),
    }

    report={
      "schema":"RealSaS.KnightTopologyWeightCouplingCourt.v1",
      "status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
      "preregistration":str(PREREG),
      "teacher_used_by_partition_solver":False,
      "solver":solver,
      "vertex_teacher_validity":{
        "vertex_count":int(len(valid)),
        "teacher_valid_vertex_count":int(np.count_nonzero(valid)),
        "teacher_invalid_vertex_count":int(np.count_nonzero(~valid)),
        "teacher_valid_fraction":float(np.mean(valid)),
      },
      "groups":groups,
      "finding":{
        "false_positive_split_invalid_fraction":groups["FALSE_POSITIVE_SPLIT"].get("any_teacher_invalid_fraction"),
        "unsafe_catastrophic_false_negative_invalid_fraction":groups["UNSAFE_ACTUAL_GT10_FALSE_NEGATIVE_CROSS_REGION"].get("any_teacher_invalid_fraction"),
      },
      "claim_boundary":"The partition and all Arachne-derived evidence are frozen before teacher_valid_mask, teacher topology or teacher weights are loaded."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_TOPOLOGY_WEIGHT_COUPLING_PASS",json.dumps(report,sort_keys=True))


if __name__=="__main__":
    main()
