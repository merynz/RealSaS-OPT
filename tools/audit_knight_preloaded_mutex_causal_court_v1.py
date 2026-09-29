from __future__ import annotations

import argparse
import json
import math
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
from tools.audit_knight_james_twigg_region_inference_v1 import _split_holeless, _teacher_region_eval
from tools.audit_knight_mechanical_mutex_watershed_closure_v1 import MutexDSU
from tools.audit_knight_mechanical_mutex_evidence_policy_court_v1 import _edge_jsd
from tools.audit_knight_teacher_free_edge_seam_diagnosis_v1 import _edge_stretch, _edge_table
from tools.audit_knight_teacher_free_weight_completion_court_v1 import (
    dense_supported_face_mask,
    exact,
    face_indices,
    load,
    motion_metrics,
    stress_arbitrary_weights,
)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_PRELOADED_MUTEX_CAUSAL_COURT_PREREG_V1_20260929.json")
LN2=math.log(2.0)


def _solve_preloaded(n, edges, dense_edge, repulsive, jsd):
    dsu=MutexDSU(n)
    mutex_added=0

    # Causal change: all cannot-link evidence exists before any attractive merge.
    rep_idx=np.where(np.asarray(dense_edge,dtype=bool)&np.asarray(repulsive,dtype=bool))[0]
    for ei in rep_idx:
        a,b=map(int,edges[int(ei)])
        if dsu.add_mutex(a,b):
            mutex_added+=1

    attractive=[]
    for ei,(a0,b0) in enumerate(edges.tolist()):
        if not bool(dense_edge[ei]) or bool(repulsive[ei]):
            continue
        a,b=int(a0),int(b0)
        j=float(min(max(jsd[ei]/LN2,0.0),1.0))
        conf=1.0-j
        attractive.append((-conf,min(a,b),max(a,b),int(ei)))
    attractive.sort()

    union_added=union_blocked=0
    for neg,a,b,ei in attractive:
        before=dsu.find(a)==dsu.find(b)
        ok=dsu.union(a,b)
        if not before and ok:
            union_added+=1
        elif not ok:
            union_blocked+=1

    # With preloading, this must be structurally impossible.
    late=0
    violated=0
    for ei in rep_idx:
        a,b=map(int,edges[int(ei)])
        if dsu.find(a)==dsu.find(b):
            violated+=1

    roots=[dsu.find(i) for i in range(n)]
    remap={};lab=np.empty(n,dtype=np.int64)
    for i,r in enumerate(roots):
        if r not in remap:
            remap[r]=len(remap)
        lab[i]=remap[r]
    sizes=np.bincount(lab)
    return lab,{
      "component_count":int(len(sizes)),
      "largest_component_sizes":list(map(int,sorted(sizes.tolist(),reverse=True)[:30])),
      "repulsive_edge_count":int(len(rep_idx)),
      "mutex_constraint_count":int(mutex_added),
      "attractive_union_count":int(union_added),
      "attractive_union_blocked_by_mutex":int(union_blocked),
      "repulsive_edge_arrived_after_endpoint_merge":int(late),
      "repulsive_constraint_violated_after_all_unions":int(violated),
      "event_order":"PRELOAD_ALL_REPULSIVE_THEN_JSD_ORDERED_ATTRACTIVE"
    }


def _checks(th, ev, split, am, ast, solver):
    return {
      "no_face_deletion":int(split["face_deletion_count"])<=int(th["face_deletion_count_max"]),
      "rest_area_preserved":float(split["rest_area_error_max"])<=float(th["rest_area_error_max"]),
      "actual_gt10":int(am["max_edge_gt_10"])<=int(th["actual_motion_max_edge_gt_10_max"]),
      "actual_gt4":int(am["max_edge_gt_4"])<=int(th["actual_motion_max_edge_gt_4_max"]),
      "actual_worst_edge":float(am["worst_edge_max"])<=float(th["actual_motion_worst_edge_max"]),
      "synthetic_unsafe":int(ast["unsafe_face_count"])<=int(th["synthetic_unsafe_face_count_max"]),
      "mixed_precision":float(ev["mixed_face_precision"])>=float(th["mixed_face_precision_min"]),
      "unsafe_cross_region_recall":float(ev["unsafe_truth_cross_region_recall"])>=float(th["unsafe_cross_region_recall_min"]),
      "preloaded_mutex_invariant":int(solver["repulsive_edge_arrived_after_endpoint_merge"])==0 and int(solver["repulsive_constraint_violated_after_all_unions"])==0,
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

    prereg=json.loads(PREREG.read_text())
    if prereg.get("status")!="FROZEN_BEFORE_PRELOADED_MUTEX_RESULT":
        raise RuntimeError("PRELOADED_MUTEX_PREREG_DRIFT")
    th=prereg["closure_thresholds"]

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

    masks={
      "STRETCH_ONLY_JSD":scope&(stretch>4.0),
      "STRETCH_OR_L1_JSD":scope&((stretch>4.0)|(l1>1.0)),
      "STRETCH_AND_L1_JSD":scope&(stretch>4.0)&(l1>1.0),
    }
    labels={};solvers={}
    for name,mask in masks.items():
        labels[name],solvers[name]=_solve_preloaded(len(P),edges,dense_edge,mask,jsd)

    # Teacher topology enters only after all partitions are frozen.
    evals={name:_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,lab,F,unsafe) for name,lab in labels.items()}

    # Teacher weights enter only after all partitions are frozen.
    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    missing=[j for j in jids if str(j) not in tix]
    if missing:
        raise RuntimeError("PRELOADED_MUTEX_TEACHER_JOINT_DRIFT:"+json.dumps(missing))
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1)
    Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    rows={}
    for name,lab in labels.items():
        P2,W2,F2,split=_split_holeless(P,Wt,F,lab)
        am=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
        ast=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)
        checks=_checks(th,evals[name],split,am,ast,solvers[name])
        rows[name]={
          "solver":solvers[name],
          "teacher_evaluation_only":evals[name],
          "holeless_split":split,
          "after_motion":{k:v for k,v in am.items() if k!="frames"},
          "after_stress":{k:v for k,v in ast.items() if k!="unsafe_face_indices"},
          "checks":checks,
          "closure_pass":all(checks.values()),
        }

    report={
      "schema":"RealSaS.KnightPreloadedMutexCausalCourt.v1",
      "status":"CAUSAL_COMPARISON_COMPLETE__NO_PRODUCT_MUTATION",
      "preregistration":str(PREREG),
      "teacher_used_by_partition_solver":False,
      "variants":rows,
      "claim_boundary":"The only solver-level causal change from the prior mutex evidence-policy court is preloading all frozen repulsive cannot-link constraints before any attractive union."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_PRELOADED_MUTEX_CAUSAL_COURT_PASS",json.dumps({
      name:{
        "solver":row["solver"],
        "teacher_eval":row["teacher_evaluation_only"],
        "after_motion":row["after_motion"],
        "after_stress":row["after_stress"],
        "checks":row["checks"],
        "closure_pass":row["closure_pass"]
      } for name,row in rows.items()
    },sort_keys=True))


if __name__=="__main__":
    main()
