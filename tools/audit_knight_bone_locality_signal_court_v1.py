from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
)
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights, _ctx

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
}

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()

def exact(rr,k,codec):
    rel,h=EXACT[k];p=rr/rel
    if sha(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
    return codec(json.loads(p.read_text()))

def surface_ids(candidate):
    out=[]
    for v in candidate.vertices:
        c=tuple(v.support_binding.coefficients)
        if len(c)!=1 or abs(float(c[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY_SUPPORT")
        out.append(str(c[0][0]))
    return tuple(out)

def teacher_matrix(bank_path,skeleton,candidate,joint_ids):
    with np.load(bank_path,allow_pickle=False) as z:
        sids=tuple(map(str,z["surface_ids"].tolist()))
        BW=np.asarray(z["weights"],float)
        valid=np.asarray(z["teacher_valid_mask"],np.uint8).astype(bool)
        bpos=np.asarray(z["target_positions_world"],float)
        bpar=np.asarray(z["target_parent_indices"],int)
    joints=tuple(skeleton.joints);ids=[str(j.canonical_joint_id) for j in joints];idx={x:i for i,x in enumerate(ids)}
    spos=np.asarray([j.position for j in joints],float)
    spar=np.asarray([-1 if j.parent_canonical_id is None else idx[str(j.parent_canonical_id)] for j in joints],int)
    C=np.linalg.norm(bpos[:,None]-spos[None],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),int);m[ri]=ci
    if sum(int((-1 if bpar[i]<0 else m[bpar[i]])==spar[m[i]]) for i in range(len(bpar)))!=len(bpar):raise RuntimeError("GRAPH_ALIGN_FAIL")
    jix={j:i for i,j in enumerate(joint_ids)}
    WT=np.zeros((len(sids),len(joint_ids)))
    for bc in range(len(m)):WT[:,jix[ids[m[bc]]]]=BW[:,bc]
    row={sid:i for i,sid in enumerate(sids)};cr=np.asarray([row[sid] for sid in surface_ids(candidate)],int)
    return WT[cr],valid[cr]

def point_segment_distance(P,A,B):
    AB=B-A
    den=np.sum(AB*AB,axis=1)
    AP=P[:,None,:]-A[None,:,:]
    t=np.sum(AP*AB[None,:,:],axis=2)/np.maximum(den[None,:],1e-15)
    t=np.clip(t,0.0,1.0)
    Q=A[None,:,:]+t[:,:,None]*AB[None,:,:]
    return np.linalg.norm(P[:,None,:]-Q,axis=2)

def q(x,p):
    x=np.asarray(x,float)
    return float(np.quantile(x,p)) if len(x) else None

def main():
    p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    rr=_ctx(a.authority_root,a.run_id)["run_root"]
    cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);skin=exact(rr,"skin",qualified_skin_from_dict)
    P=np.asarray([v.P for v in cand.vertices],float)
    joint_ids,W=_candidate_skin_weights(cand,skin,sk);Wteach,valid=teacher_matrix(a.teacher_bank,sk,cand,joint_ids)
    terr=np.sum(np.abs(W-Wteach),axis=1);bad1=terr>1.0;good=terr<=0.1

    joints=tuple(sk.joints)
    id_to_obj={str(j.canonical_joint_id):j for j in joints}
    pos={str(j.canonical_joint_id):np.asarray(j.position,float) for j in joints}
    A=[];B=[];meta=[]
    for jid in joint_ids:
        j=id_to_obj[jid]
        if j.parent_canonical_id is None:
            a0=pos[jid];b0=pos[jid]
        else:
            a0=pos[str(j.parent_canonical_id)];b0=pos[jid]
        A.append(a0);B.append(b0);meta.append({"joint_id":jid,"parent":None if j.parent_canonical_id is None else str(j.parent_canonical_id)})
    A=np.asarray(A);B=np.asarray(B)
    D=point_segment_distance(P,A,B)
    order=np.argsort(D,axis=1);rank=np.empty_like(order)
    rows=np.arange(len(P))[:,None];rank[rows,order]=np.arange(len(joint_ids))[None,:]
    pred=np.argmax(W,axis=1);teach=np.argmax(Wteach,axis=1);nearest=order[:,0]
    pd=D[np.arange(len(P)),pred];td=D[np.arange(len(P)),teach];nd=D[np.arange(len(P)),nearest]
    prank=rank[np.arange(len(P)),pred]+1;trank=rank[np.arange(len(P)),teach]+1
    ratio=pd/np.maximum(nd,1e-8)

    def group(mask):
        return {
          "count":int(mask.sum()),
          "pred_bone_distance_p50":q(pd[mask],.5),"pred_bone_distance_p95":q(pd[mask],.95),
          "teacher_bone_distance_p50":q(td[mask],.5),"teacher_bone_distance_p95":q(td[mask],.95),
          "pred_rank_p50":q(prank[mask],.5),"pred_rank_p95":q(prank[mask],.95),
          "teacher_rank_p50":q(trank[mask],.5),"teacher_rank_p95":q(trank[mask],.95),
          "pred_within_top2":int(np.count_nonzero(mask&(prank<=2))),
          "pred_within_top4":int(np.count_nonzero(mask&(prank<=4))),
          "teacher_within_top2":int(np.count_nonzero(mask&(trank<=2))),
          "teacher_within_top4":int(np.count_nonzero(mask&(trank<=4))),
          "pred_nearest_ratio_p50":q(ratio[mask],.5),"pred_nearest_ratio_p95":q(ratio[mask],.95),
        }

    courts=[]
    for k in (1,2,4,6,8,12):
        allowed=np.zeros_like(W,dtype=bool)
        allowed[rows,order[:,:k]]=True
        pred_dom_allowed=allowed[np.arange(len(P)),pred]
        teach_dom_allowed=allowed[np.arange(len(P)),teach]
        courts.append({
          "top_k":k,
          "bad_gt1_pred_dom_allowed":int(np.count_nonzero(bad1&pred_dom_allowed)),
          "bad_gt1_teacher_dom_allowed":int(np.count_nonzero(bad1&teach_dom_allowed)),
          "bad_gt1_count":int(bad1.sum()),
          "good_pred_dom_allowed":int(np.count_nonzero(good&pred_dom_allowed)),
          "good_count":int(good.sum()),
          "teacher_valid_teacher_dom_allowed":int(np.count_nonzero(valid&teach_dom_allowed)),
          "teacher_valid_count":int(valid.sum()),
        })

    top=[]
    for i in np.argsort(terr)[::-1][:256]:
        top.append({
          "vertex_index":int(i),"teacher_valid":bool(valid[i]),"teacher_l1_error":float(terr[i]),
          "position":P[i].tolist(),"nearest_joint":joint_ids[int(nearest[i])],"nearest_distance":float(nd[i]),
          "pred_joint":joint_ids[int(pred[i])],"pred_distance":float(pd[i]),"pred_rank":int(prank[i]),
          "teacher_joint":joint_ids[int(teach[i])],"teacher_distance":float(td[i]),"teacher_rank":int(trank[i]),
          "pred_to_nearest_distance_ratio":float(ratio[i]),
        })

    report={"schema":"RealSaS.KnightBoneLocalitySignalCourt.v1","status":"DIAGNOSTIC__NO_REPAIR",
      "groups":{"bad_l1_gt1":group(bad1),"good_l1_le0_1":group(good),"teacher_invalid_bad_l1_gt1":group((~valid)&bad1)},
      "top_k_court":courts,"top_error_rows":top,
      "finding":{
        "bad_rows_predicted_joint_is_generally_nonlocal":bool(q(prank[bad1],.5)>4),
        "teacher_joint_more_local_than_predicted_on_bad_rows":bool(q(trank[bad1],.5)<q(prank[bad1],.5)),
      },
      "claim_boundary":"Teacher is evaluation-only. Bone-segment distances and predicted weights are product-available."
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_BONE_LOCALITY_SIGNAL_COURT_PASS",json.dumps({"groups":report["groups"],"top_k":courts,"finding":report["finding"]},sort_keys=True))
if __name__=="__main__":main()
