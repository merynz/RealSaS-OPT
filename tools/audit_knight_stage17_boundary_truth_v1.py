from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
 read_json,mechanical_partition_from_dict,canonical_mesh_candidate_from_dict,qualified_skeleton_from_dict)
from tools.audit_knight_teacher_weight_topology_only_court_v1 import teacher_weights
from tools.audit_knight_topology_region_pair_decomposition_court_v1 import source_components
from tools.demo.render_knight_motion_preview_v1 import _ctx

PREREG=Path("canonical/KNIGHT_STAGE17_BOUNDARY_TRUTH_AUDIT_PREREG_V1_20260929.json")
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True)
 p.add_argument("--teacher-bank",type=Path,required=True);p.add_argument("--teacher-source",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_STAGE17_BOUNDARY_TRUTH_AUDIT_RESULT":raise RuntimeError("STAGE17_BOUNDARY_TRUTH_PREREG_DRIFT")
 rr=_ctx(a.authority_root,a.run_id)["run_root"]
 part=mechanical_partition_from_dict(read_json(rr/"artifacts/17_MECHANICAL_PARTITION_QUALIFIED/mechanical_partition.json"))
 cand=canonical_mesh_candidate_from_dict(read_json(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json"))
 sk=qualified_skeleton_from_dict(read_json(rr/"artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json"))
 # Freeze exact candidate surface-id mapping and Stage17 decisions first.
 sid_to_vi={}
 for vi,v in enumerate(cand.vertices):
  terms=tuple(v.support_binding.terms)
  if str(v.support_binding.method)!="IDENTITY_SURFACE_NODE" or len(terms)!=1 or abs(float(terms[0][1])-1.0)>1e-12:
   continue
  sid_to_vi[str(terms[0][0])]=int(vi)
 rows=[]
 for bc in part.boundary_constraints:
  a_sid=str(bc.a_surface_id);b_sid=str(bc.b_surface_id)
  if a_sid in sid_to_vi and b_sid in sid_to_vi:
   rows.append((str(bc.decision),sid_to_vi[a_sid],sid_to_vi[b_sid]))
 # Teacher begins here.
 _,_,tri,sf=teacher_weights(a.teacher_bank,a.teacher_source,sk,cand)
 src=source_components(int(sf.max())+1,sf);tri_comp=np.asarray([src[int(r[0])] for r in sf],np.int64);truth=np.asarray([tri_comp[int(t)] for t in tri],np.int64)
 stats={}
 for decision in sorted(set(r[0] for r in rows)):
  sub=[r for r in rows if r[0]==decision];cross=sum(int(truth[u])!=int(truth[v]) for _,u,v in sub)
  stats[decision]={"evaluated_relation_count":len(sub),"teacher_cross_component_count":int(cross),
   "teacher_same_component_count":int(len(sub)-cross),"teacher_cross_fraction":float(cross/max(1,len(sub)))}
 preserve=stats.get("PRESERVE_CONTINUITY",{"evaluated_relation_count":0,"teacher_cross_component_count":0,"teacher_same_component_count":0,"teacher_cross_fraction":0.0})
 report={"schema":"RealSaS.KnightStage17BoundaryTruthAudit.v1","status":"READ_ONLY_EVALUATION",
  "preregistration":str(PREREG),"teacher_used_by_operator":False,
  "candidate_identity_surface_vertex_count":len(sid_to_vi),"evaluated_stage17_constraint_count":len(rows),
  "decision_teacher_eval_only":stats,
  "finding":{"preserve_constraints_cross_teacher_components":int(preserve["teacher_cross_component_count"]),
   "preserve_cross_fraction":float(preserve["teacher_cross_fraction"]),
   "stage17_separate_count":sum(str(x.decision)=="SEPARATE" for x in part.boundary_constraints)},
  "claim_boundary":"Stage17 decisions and candidate endpoint scope are frozen before teacher source components are loaded; teacher is evaluation-only."}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_STAGE17_BOUNDARY_TRUTH_AUDIT_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
