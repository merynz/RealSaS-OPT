from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict,qualified_skeleton_from_dict,qualified_skin_from_dict
from tools.demo.render_knight_motion_preview_v1 import _candidate_skin_weights,_ctx
EX={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09")}
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(rr,k,codec):
 rel,h=EX[k];p=rr/rel
 if sh(p)!=h:raise RuntimeError("SHA_DRIFT:"+k)
 return codec(json.loads(p.read_text()))
def src_cat(a,b,sets):
 if a==b:return "SAME_FACE"
 n=len(sets[a]&sets[b]);return "SHARE_EDGE" if n==2 else "SHARE_VERTEX_ONLY" if n==1 else "NONINCIDENT"
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 rr=_ctx(a.authority_root,a.run_id)["run_root"];c=load(rr,"candidate",canonical_mesh_candidate_from_dict);s=load(rr,"skeleton",qualified_skeleton_from_dict);skin=load(rr,"skin",qualified_skin_from_dict)
 jids,Wpred=_candidate_skin_weights(c,skin,s);ji={j:i for i,j in enumerate(jids)}
 with np.load(a.bank,allow_pickle=False) as z:
  sids=tuple(map(str,z["surface_ids"].tolist()));BW=np.asarray(z["weights"],float);valid=np.asarray(z["teacher_valid_mask"],np.uint8).astype(bool);tri=np.asarray(z["source_triangle_index"],int);bpos=np.asarray(z["target_positions_world"],float);bpar=np.asarray(z["target_parent_indices"],int);dist=np.asarray(z["source_surface_distance"],float)
 with np.load(a.source,allow_pickle=False) as z:sf=np.asarray(z["faces"],int)
 joints=tuple(s.joints);ids=[str(x.canonical_joint_id) for x in joints];idx={x:i for i,x in enumerate(ids)};spos=np.asarray([x.position for x in joints],float);spar=np.asarray([-1 if x.parent_canonical_id is None else idx[str(x.parent_canonical_id)] for x in joints],int)
 C=np.linalg.norm(bpos[:,None]-spos[None],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),int);m[ri]=ci
 if sum(int((-1 if bpar[b]<0 else m[bpar[b]])==spar[m[b]]) for b in range(len(bpar)))!=len(bpar):raise RuntimeError("GRAPH_ALIGN_FAIL")
 WT=np.zeros((len(sids),len(jids)))
 for bc in range(len(m)):WT[:,ji[ids[m[bc]]]]=BW[:,bc]
 brow={sid:i for i,sid in enumerate(sids)};vid={str(v.candidate_vertex_id):i for i,v in enumerate(c.vertices)}
 crows=[]
 for v in c.vertices:
  co=tuple(v.support_binding.coefficients)
  if len(co)!=1 or abs(float(co[0][1])-1)>1e-12:raise RuntimeError("NONIDENTITY")
  crows.append(brow[str(co[0][0])])
 crows=np.asarray(crows,int);Wteach=WT[crows];vvalid=valid[crows];verr=np.sum(np.abs(Wpred-Wteach),axis=1);vdist=dist[crows]
 faces=np.asarray([[vid[str(x)] for x in f] for f in c.faces],int);sets=[set(map(int,x)) for x in sf.tolist()]
 strict=[]
 for f in faces:
  ts=[int(tri[crows[v]]) for v in f];cats=[src_cat(ts[0],ts[1],sets),src_cat(ts[1],ts[2],sets),src_cat(ts[2],ts[0],sets)];strict.append(all(x in ("SAME_FACE","SHARE_EDGE") for x in cats))
 strict=np.asarray(strict,bool)
 def fmax(W):
  x=W[faces];return np.maximum.reduce([np.sum(np.abs(x[:,0]-x[:,1]),1),np.sum(np.abs(x[:,1]-x[:,2]),1),np.sum(np.abs(x[:,2]-x[:,0]),1)])
 pl1=fmax(Wpred);tl1=fmax(Wteach);fall=np.all(vvalid[faces],axis=1);fanybad=~fall;fmaxerr=np.max(verr[faces],axis=1)
 report={"schema":"RealSaS.KnightStage32EvidenceSupportAudit.v1","status":"MEASURED__NO_REPAIR",
 "rows":{"total":len(verr),"valid":int(np.count_nonzero(vvalid)),"invalid":int(np.count_nonzero(~vvalid)),
  "gt_0_1_total":int(np.count_nonzero(verr>.1)),"gt_0_1_valid":int(np.count_nonzero((verr>.1)&vvalid)),"gt_0_1_invalid":int(np.count_nonzero((verr>.1)&~vvalid)),
  "gt_1_total":int(np.count_nonzero(verr>1)),"gt_1_valid":int(np.count_nonzero((verr>1)&vvalid)),"gt_1_invalid":int(np.count_nonzero((verr>1)&~vvalid)),
  "valid_error_p95":float(np.quantile(verr[vvalid],.95)),"valid_error_p99":float(np.quantile(verr[vvalid],.99)),"valid_error_max":float(np.max(verr[vvalid])),
  "invalid_error_p50":float(np.quantile(verr[~vvalid],.5)),"invalid_error_p95":float(np.quantile(verr[~vvalid],.95)),
  "valid_source_distance_p95":float(np.quantile(vdist[vvalid],.95)),"invalid_source_distance_p50":float(np.quantile(vdist[~vvalid],.5))},
 "strict_faces":{"count":int(np.count_nonzero(strict)),"pred_l1_gt_1":int(np.count_nonzero(strict&(pl1>1))),"teacher_l1_gt_1":int(np.count_nonzero(strict&(tl1>1))),
  "pred_l1_gt_1_all_vertices_teacher_valid":int(np.count_nonzero(strict&(pl1>1)&fall)),"pred_l1_gt_1_any_teacher_invalid":int(np.count_nonzero(strict&(pl1>1)&fanybad)),
  "pred_l1_gt_1_face_max_row_error_p50":float(np.quantile(fmaxerr[strict&(pl1>1)],.5)) if np.any(strict&(pl1>1)) else None},
 "finding":{}}
 report["finding"]={"large_stage32_errors_exist_on_teacher_valid_rows":bool(report["rows"]["gt_0_1_valid"]>0),"catastrophic_strict_faces_exist_with_all_teacher_rows_valid":bool(report["strict_faces"]["pred_l1_gt_1_all_vertices_teacher_valid"]>0)}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print("KNIGHT_STAGE32_EVIDENCE_SUPPORT_AUDIT_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
