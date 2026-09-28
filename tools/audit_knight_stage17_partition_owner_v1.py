from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from compiler.realsas_compiler_core.artifact_codec_v2 import mechanical_partition_from_dict,rigging_surface_from_dict,canonical_mesh_candidate_from_dict
from tools.audit_knight_teacher_free_weight_completion_court_v1 import exact
from tools.demo.render_knight_motion_preview_v1 import _ctx
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload

PREREG=Path("canonical/KNIGHT_STAGE17_PARTITION_OWNER_AUDIT_PREREG_V1_20260929.json")

def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if json.loads(PREREG.read_text()).get("status")!="FROZEN_BEFORE_STAGE17_OWNER_AUDIT_RESULT":raise RuntimeError("STAGE17_OWNER_PREREG_DRIFT")
 ctx=_ctx(a.authority_root,a.run_id);rr=ctx["run_root"]
 # Use exact stage outputs from the sealed run.
 part=mechanical_partition_from_dict(stage_output_payload(ctx,"17_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1"))
 surf=rigging_surface_from_dict(stage_output_payload(ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
 cand=canonical_mesh_candidate_from_dict(stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
 decisions=Counter(str(x.decision) for x in part.boundary_constraints)
 comp_sizes=sorted([len(x.surface_ids) for x in part.components],reverse=True)
 cand_counts=Counter(str(v.component_id) for v in cand.vertices)
 report={
  "schema":"RealSaS.KnightStage17PartitionOwnerAudit.v1",
  "status":"READ_ONLY_OWNER_AUDIT",
  "preregistration":str(PREREG),
  "stage15":{"surface_node_count":len(surf.surface_nodes),"local_relation_count":len(surf.local_relations)},
  "stage17":{"component_count":len(part.components),"component_surface_sizes":comp_sizes[:20],
    "boundary_constraint_count":len(part.boundary_constraints),"boundary_decision_counts":dict(sorted(decisions.items())),
    "separate_boundary_count":int(decisions.get("SEPARATE",0)),"unknown_boundary_count":int(decisions.get("UNKNOWN",0)),
    "preserve_boundary_count":int(decisions.get("PRESERVE_CONTINUITY",0)),
    "partition_lineage_hash":part.partition_lineage_hash},
  "stage18":{"candidate_component_count":len(cand_counts),"candidate_component_vertex_sizes":sorted(cand_counts.values(),reverse=True)[:20],
    "partition_binding_hash":cand.partition_binding_hash,"candidate_lineage_hash":cand.candidate_lineage_hash},
  "owner_finding":{
    "collapse_present_by_stage17":bool(len(part.components)==1),
    "stage17_has_any_separate_boundary":bool(decisions.get("SEPARATE",0)>0),
    "stage18_exactly_inherits_stage17_partition":bool(cand.partition_binding_hash==part.partition_lineage_hash)
  },
  "claim_boundary":"This audit reads sealed Stage15/17/18 artifacts only; it does not infer source/teacher topology and does not mutate product state."
 }
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_STAGE17_PARTITION_OWNER_AUDIT_PASS",json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
