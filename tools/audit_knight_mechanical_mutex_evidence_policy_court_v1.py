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

PREREG=Path("canonical/KNIGHT_MECHANICAL_MUTEX_EVIDENCE_POLICY_COURT_PREREG_V1_20260929.json")
LN2=math.log(2.0)


def _edge_jsd(W, edges):
    p=np.maximum(W[edges[:,0]],0.0)
    q=np.maximum(W[edges[:,1]],0.0)
    p=p/np.maximum(p.sum(axis=1,keepdims=True),1e-15)
    q=q/np.maximum(q.sum(axis=1,keepdims=True),1e-15)
    m=.5*(p+q)
    kp=np.where(p>0,p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)),0.0).sum(axis=1)
    kq=np.where(q>0,q*np.log(np.maximum(q,1e-300)/np.maximum(m,1e-300)),0.0).sum(axis=1)
    return .5*(kp+kq)


def _solve(n, edges, dense_edge, repulsive, jsd):
    events=[]
    for ei,(a0,b0) in enumerate(edges.tolist()):
        if not bool(dense_edge[ei]):
            continue
        a,b=int(a0),int(b0)
        j=float(min(max(jsd[ei]/LN2,0.0),1.0))
        if bool(repulsive[ei]):
            conf=j;kind=0
        else:
            conf=1.0-j;kind=1
        events.append((-conf,kind,min(a,b),max(a,b),int(ei)))
    events.sort()
    dsu=MutexDSU(n)
    mutex_added=union_added=union_blocked=rep_after_merge=0
    for neg,kind,a,b,ei in events:
        if kind==0:
            if dsu.find(a)==dsu.find(b):
                rep_after_merge+=1
            elif dsu.add_mutex(a,b):
                mutex_added+=1
        else:
            before=dsu.find(a)==dsu.find(b)
            ok=dsu.union(a,b)
            if not before and ok: union_added+=1
            elif not ok: union_blocked+=1
    roots=[dsu.find(i) for i in range(n)]
    remap={};label=np.empty(n,dtype=np.int64)
    for i,r in enumerate(roots):
        if r not in remap: remap[r]=len(remap)
        label[i]=remap[r]
    sizes=np.bincount(label)
    return label,{
      "component_count":int(len(sizes)),
      "largest_component_sizes":list(map(int,sorted(sizes.tolist(),reverse=True)[:30])),
      "repulsive_edge_count":int(np.count_nonzero(repulsive)),
      "mutex_constraint_count":int(mutex_added),
      "attractive_union_count":int(union_added),
      "attractive_union_blocked_by_mutex":int(union_blocked),
      "repulsive_edge_arrived_after_endpoint_merge":int(rep_after_merge),
      "event_count":int(len(events)),
    }


def _checks(th, ev, split, am, ast):
    return {
      "no_face_deletion":int(split["face_deletion_count"])<=int(th["face_deletion_count_max"]),
      "rest_area_preserved":float(split["rest_area_error_max"])<=float(th["rest_area_error_max"]),
      "actual_gt10":int(am["max_edge_gt_10"])<=int(th["actual_motion_max_edge_gt_10_max"]),
      "actual_gt4":int(am["max_edge_gt_4"])<=int(th["actual_motion_max_edge_gt_4_max"]),
      "actual_worst_edge":float(am["worst_edge_max"])<=float(th["actual_motion_worst_edge_max"]),
      "synthetic_unsafe":int(ast["unsafe_face_count"])<=int(th["synthetic_unsafe_face_count_max"]),
      "mixed_precision":float(ev["mixed_face_precision"])>=float(th["mixed_face_precision_min"]),
      "unsafe_cross_region_recall":float(ev["unsafe_truth_cross_region_recall"])>=float(th["unsafe_cross_region_recall_min"]),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
    p.add_argument("--weights-npz",type=Path,required=True);p.add_argument("--teacher-bank",type=Path,required=True)
    p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    prereg=json.loads(PREREG.read_text())
    if prereg.get("status")!="FROZEN_BEFORE_MUTEX_EVIDENCE_POLICY_RESULT":raise RuntimeError("MUTEX_POLICY_PREREG_DRIFT")
    th=prereg["closure_thresholds"]

    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict)
    sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
    cams=tuple(sorted(exact(rr,"cameras",qualified_camera_set_from_dict).cameras,key=lambda x:int(x.view_index)))
    env=load(rr/"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
    policy=load(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)
    surface=load(rr/"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
    P=np.asarray([v.P for v in cand.vertices],np.float64);F=face_indices(cand);dense=dense_supported_face_mask(rr,cand,surface)
    with np.load(a.weights_npz,allow_pickle=False) as z:
        jids=tuple(map(str,z["joint_ids"].tolist()));Wa=np.asarray(z["arachne"],np.float64)
    Wa=np.maximum(Wa,0.0);Wa/=np.maximum(Wa.sum(axis=1,keepdims=True),1e-15)

    base=stress_arbitrary_weights(P,Wa,F,jids,sk,cams,env,policy)
    unsafe=np.zeros(len(F),bool);unsafe[np.asarray(base["unsafe_face_indices"],np.int64)]=True
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
    labels={};solver={}
    for name,mask in masks.items():
        labels[name],solver[name]=_solve(len(P),edges,dense_edge,mask,jsd)

    # Teacher topology begins only after every variant partition is frozen.
    evals={name:_teacher_region_eval(a.teacher_bank,a.teacher_source,sk,cand,lab,F,unsafe) for name,lab in labels.items()}

    # Teacher weights begin only after every partition is frozen.
    Wt,tjids,_,_=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
    tix={str(j):i for i,j in enumerate(tjids)}
    missing=[j for j in jids if str(j) not in tix]
    if missing:raise RuntimeError("MUTEX_POLICY_TEACHER_JOINT_DRIFT:"+json.dumps(missing))
    Wt=np.stack([Wt[:,tix[str(j)]] for j in jids],axis=1);Wt=np.maximum(Wt,0.0);Wt/=np.maximum(Wt.sum(axis=1,keepdims=True),1e-15)
    source_report=json.loads(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json").read_text())

    rows={}
    for name,lab in labels.items():
        P2,W2,F2,split=_split_holeless(P,Wt,F,lab)
        am=motion_metrics(P2,W2,F2,jids,sk,cams,rr,source_report)
        ast=stress_arbitrary_weights(P2,W2,F2,jids,sk,cams,env,policy)
        checks=_checks(th,evals[name],split,am,ast)
        rows[name]={
          "solver":solver[name],"teacher_evaluation_only":evals[name],"holeless_split":split,
          "after_motion":{k:v for k,v in am.items() if k!="frames"},
          "after_stress":{k:v for k,v in ast.items() if k!="unsafe_face_indices"},
          "checks":checks,"closure_pass":all(checks.values())
        }
    report={
      "schema":"RealSaS.KnightMechanicalMutexEvidencePolicyCourt.v1",
      "status":"COMPARISON_COMPLETE__NO_PRODUCT_MUTATION",
      "preregistration":str(PREREG),
      "teacher_used_by_partition_solver":False,
      "variants":rows,
      "claim_boundary":"All candidate partitions are frozen before teacher topology or teacher weights are used."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_MECHANICAL_MUTEX_EVIDENCE_POLICY_COURT_PASS",json.dumps({
      name:{
        "solver":row["solver"],"teacher_eval":row["teacher_evaluation_only"],
        "after_motion":row["after_motion"],"after_stress":row["after_stress"],
        "checks":row["checks"],"closure_pass":row["closure_pass"]
      } for name,row in rows.items()
    },sort_keys=True))

if __name__=="__main__":main()
