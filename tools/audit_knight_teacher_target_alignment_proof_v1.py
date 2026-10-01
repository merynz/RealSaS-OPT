from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from compiler.realsas_compiler_core.artifact_codec_v2 import qualified_skeleton_from_dict
from tools.demo.render_knight_motion_preview_v1 import _ctx

REL="artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"
SHA="e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"

def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--bank",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 rr=_ctx(a.authority_root,a.run_id)["run_root"];sp=rr/REL
 if sh(sp)!=SHA:raise RuntimeError("SKELETON_SHA_DRIFT")
 sk=qualified_skeleton_from_dict(json.loads(sp.read_text()));joints=tuple(sk.joints)
 ids=[str(j.canonical_joint_id) for j in joints]; idx={x:i for i,x in enumerate(ids)}
 spos=np.asarray([j.position for j in joints],dtype=np.float64)
 spar=np.asarray([-1 if j.parent_canonical_id is None else idx[str(j.parent_canonical_id)] for j in joints],dtype=np.int64)
 with np.load(a.bank,allow_pickle=False) as z:
  bpos=np.asarray(z["target_positions_world"],dtype=np.float64);bpar=np.asarray(z["target_parent_indices"],dtype=np.int64)
 C=np.linalg.norm(bpos[:,None,:]-spos[None,:,:],axis=2);ri,ci=linear_sum_assignment(C);m=np.empty(len(bpos),dtype=np.int64);m[ri]=ci
 edge=[]
 ok=0
 for i in range(len(bpar)):
  bp=int(bpar[i]); sj=int(m[i]); actual=int(spar[sj]); expected=-1 if bp<0 else int(m[bp]); good=actual==expected;ok+=int(good)
  edge.append({"bank_col":i,"stage_joint_id":ids[sj],"bank_parent":bp,"expected_stage_parent":None if expected<0 else ids[expected],"actual_stage_parent":None if actual<0 else ids[actual],"graph_match":good,"position_error":float(C[i,sj])})
 # assignment ambiguity: nearest and second nearest raw position costs.
 margins=[]
 for i in range(len(bpos)):
  s=np.sort(C[i]);margins.append(float(s[1]-s[0]))
 report={"schema":"RealSaS.KnightTeacherTargetAlignmentProof.v1","status":"MEASURED__NO_REPAIR","hungarian_total_cost":float(C[ri,ci].sum()),"max_position_error":float(np.max(C[ri,ci])),"graph_edge_match_count":ok,"graph_edge_total":len(bpar),"graph_exact":bool(ok==len(bpar)),"minimum_raw_nearest_margin":float(min(margins)),"edges":edge}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_TEACHER_TARGET_ALIGNMENT_PROOF_PASS",json.dumps({k:report[k] for k in ("graph_exact","graph_edge_match_count","graph_edge_total","max_position_error","minimum_raw_nearest_margin")},sort_keys=True))
if __name__=="__main__":main()
