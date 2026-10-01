from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import canonical_mesh_candidate_from_dict,qualified_skeleton_from_dict
from tools.audit_knight_teacher_free_weight_completion_court_v1 import exact,face_indices
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.demo.render_knight_motion_preview_v1 import _ctx
PREREG=Path("canonical/KNIGHT_CANDIDATE_COMPONENT_AUTHORITY_AUDIT_PREREG_V1_20260929.json")
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_CANDIDATE_COMPONENT_AUDIT_RESULT":raise RuntimeError("CANDIDATE_COMPONENT_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"];cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict)
 F=face_indices(cand);comp=np.asarray([str(v.component_id) for v in cand.vertices],dtype=object)
 counts=Counter(map(str,comp.tolist()))
 pred=np.asarray([len(set(map(str,comp[row].tolist())))>1 for row in F],bool)
 # Teacher starts here.
 _,_,tri,sf=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand);src=source_components(int(sf.max())+1,sf)
 tri_comp=np.asarray([src[int(row[0])] for row in sf],np.int64);truth=np.asarray([tri_comp[int(t)] for t in tri],np.int64)
 tm=np.asarray([len(set(int(truth[v]) for v in row))>1 for row in F],bool)
 tp=int(np.count_nonzero(pred&tm));fp=int(np.count_nonzero(pred&(~tm)))
 report={"schema":"RealSaS.KnightCandidateComponentAuthorityAudit.v1","status":"DIAGNOSTIC_ONLY__NO_PRODUCT_MUTATION",
  "preregistration":str(PREREG),"teacher_used_by_operator":False,
  "candidate":{"vertex_count":len(comp),"component_count":len(counts),"largest_component_sizes":sorted(counts.values(),reverse=True)[:20],
   "component_ids_top":[[k,int(v)] for k,v in counts.most_common(20)],"predicted_mixed_face_count":int(pred.sum())},
  "teacher_eval_only":{"truth_mixed_face_count":int(tm.sum()),"tp":tp,"fp":fp,
   "mixed_face_precision":float(tp/max(1,tp+fp)),"mixed_face_recall":float(tp/max(1,tm.sum()))},
  "claim_boundary":"Candidate component_id is consumed exactly as sealed; teacher topology is evaluation-only."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_CANDIDATE_COMPONENT_AUTHORITY_AUDIT_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
