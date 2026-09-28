from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, component_carrier_policy_from_dict,
    deformation_envelope_from_dict, mechanical_partition_from_dict,
    mesh_policy_from_dict, qualified_camera_set_from_dict,
    qualified_observation_set_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import run_g3_local_frame_micro_stress_v2
from compiler.realsas_compiler_core.mesh.product_coverage_v1 import build_g5_coverage_matrix
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import run_skin_topology_compatibility_v1,seam_cut_candidate_v1
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_v2 import _component_observations
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    _evaluate_candidate_source_fidelity_v1, _source_foreground_masks_v1, _static_geometry_evidence,
)
from tools.demo.render_knight_motion_preview_v1 import _ctx

EXACT={
"candidate":("artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/canonical_mesh_candidate.json","0db35bbcdd3565cf42c74127fa33c66d39545e1ec84d1b1c74f1dac5bb6072d3"),
"skeleton":("artifacts/28_SKELETON_QUALIFIED/qualified_skeleton.json","e89d0b64cf23b2954f2b30fb37836ee2b3287749a79c90a76185437b9217e987"),
"skin":("artifacts/32_SKIN_QUALIFIED/qualified_skin.json","f1a937de488ec2a620ee292b3f865a3b2ca97b9466401555beb08b1d2f945a09"),
"cameras":("artifacts/05_CAMERA_CONTRACT_SOLVED/qualified_camera_set.json","312a9b1afe4ea1fcdc481232ac951fb6a6815fee532fbb93646e97e609b5b80a"),
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
def load(rr,rel,codec):
 p=rr/rel
 if not p.is_file():raise RuntimeError("MISSING:"+rel)
 return codec(json.loads(p.read_text()))
def source_fidelity(ctx,candidate):
 geometry,_demo=_static_geometry_evidence(ctx)
 cameras=exact(ctx["run_root"],"cameras",qualified_camera_set_from_dict)
 obs=qualified_observation_set_from_dict(json.loads((ctx["run_root"]/"artifacts/07_OBSERVATION_CONTRACT_QUALIFIED/qualified_observation_set.json").read_text()))
 masks=_source_foreground_masks_v1(ctx,obs)
 passed,rows=_evaluate_candidate_source_fidelity_v1(candidate=candidate,geometry=geometry,cameras=cameras,observation=obs,source_foreground=masks)
 return {"passed":bool(passed),"rows":rows}
def main():
 p=argparse.ArgumentParser();p.add_argument("--authority-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 ctx=_ctx(a.authority_root,a.run_id);rr=ctx["run_root"]
 cand=exact(rr,"candidate",canonical_mesh_candidate_from_dict);sk=exact(rr,"skeleton",qualified_skeleton_from_dict);skin=exact(rr,"skin",qualified_skin_from_dict);camera_set=exact(rr,"cameras",qualified_camera_set_from_dict);cams=camera_set.cameras
 surface=load(rr,"artifacts/15_RIGGING_SURFACE_QUALIFIED/qualified_rigging_surface.json",rigging_surface_from_dict)
 partition=load(rr,"artifacts/17_MECHANICAL_PARTITION_QUALIFIED/mechanical_partition.json",mechanical_partition_from_dict)
 carrier=load(rr,"artifacts/17_MECHANICAL_PARTITION_QUALIFIED/component_carrier_policy.json",component_carrier_policy_from_dict)
 envelope=load(rr,"artifacts/34_DEFORMATION_CAPABILITY_ENVELOPE/deformation_envelope.json",deformation_envelope_from_dict)
 policy=load(rr,"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json",mesh_policy_from_dict)

 comp=run_skin_topology_compatibility_v1(cand,surface=surface,skeleton=sk,skin=skin,envelope=envelope,cameras=cams,policy=policy,risk_l1_min=0.0)
 if comp["passed"]: repaired=cand; directive={"status":"NO_REPAIR_REQUIRED"}
 else: repaired,directive=seam_cut_candidate_v1(cand,comp["unsafe_face_indices"],report_hash=comp["report_hash"])

 # Re-run compatibility on repaired candidate with the same full-population risk semantics.
 comp2=run_skin_topology_compatibility_v1(repaired,surface=surface,skeleton=sk,skin=skin,envelope=envelope,cameras=cams,policy=policy,risk_l1_min=0.0)
 g3=run_g3_local_frame_micro_stress_v2(repaired,surface=surface,skeleton=sk,skin=skin,envelope=envelope,cameras=cams,policy=policy)
 observations,source_foreground,observation_set=_component_observations(ctx,surface=surface,partition=partition,carrier=carrier,cameras=cams)
 g5=build_g5_coverage_matrix(repaired,surface=surface,partition=partition,carrier_policy=carrier,mesh_policy=policy,observations=observations,cameras=cams,observation_set=observation_set,source_foreground_masks=source_foreground)
 failed_g5=[r for r in g5 if r["status"]!="PASS"]
 sf_before=source_fidelity(ctx,cand); sf_after=source_fidelity(ctx,repaired)

 def row_summary(rows):
  return [{
   "view_index":int(x["view_index"]),"passed":bool(x["passed"]),
   "silhouette_recall":float(x["silhouette_recall"]),"silhouette_precision":float(x["silhouette_precision"]),
   "largest_coherent_hole_fraction":float(x["largest_coherent_hole_fraction"]),
   "interior_uncovered_fraction":float(x["interior_uncovered_fraction"]),
   "component_recall":float(x["component_recall"]),
   "silhouette_edge_p95_px":float(x["silhouette_edge_p95_px"]),
  } for x in rows]
 report={
  "schema":"RealSaS.KnightRiskZeroRepairQualificationCourt.v1","status":"MEASURED__NO_AUTHORITY_MUTATION",
  "candidate":{"before_faces":len(cand.faces),"after_faces":len(repaired.faces),"removed":len(cand.faces)-len(repaired.faces),"lineage":repaired.candidate_lineage_hash},
  "compatibility_before":{"passed":comp["passed"],"unsafe_face_count":comp["unsafe_face_count"],"risky_face_count":comp["risky_face_count"],"report_hash":comp["report_hash"]},
  "compatibility_after":{"passed":comp2["passed"],"unsafe_face_count":comp2["unsafe_face_count"],"risky_face_count":comp2["risky_face_count"],"report_hash":comp2["report_hash"]},
  "g3":{"passed":bool(g3.passed),"failure_invariants":list(g3.failure_invariants),"report_hash":g3.report_hash},
  "g5":{"row_count":len(g5),"failed_count":len(failed_g5),"failed_rows":failed_g5},
  "source_fidelity_before":{"passed":sf_before["passed"],"rows":row_summary(sf_before["rows"])},
  "source_fidelity_after":{"passed":sf_after["passed"],"rows":row_summary(sf_after["rows"])},
  "directive":directive,
 }
 report["finding"]={
  "risk_zero_repair_passes_compatibility":bool(comp2["passed"]),
  "risk_zero_repair_passes_g3":bool(g3.passed),
  "risk_zero_repair_passes_g5":bool(len(failed_g5)==0),
  "risk_zero_repair_passes_source_fidelity":bool(sf_after["passed"]),
  "all_checked_mechanical_and_coverage_gates_pass":bool(comp2["passed"] and g3.passed and len(failed_g5)==0 and sf_after["passed"]),
 }
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print("KNIGHT_RISK_ZERO_REPAIR_QUALIFICATION_COURT_PASS",json.dumps({"candidate":report["candidate"],"g3":report["g3"],"g5_failed":len(failed_g5),"sf_before":sf_before["passed"],"sf_after":sf_after["passed"],**report["finding"]},sort_keys=True))
if __name__=="__main__":main()
