from __future__ import annotations
import argparse,json,time
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import (
    loadj, build_repaired_surface, report_quality, edge_topology,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import repair_candidate_fixed_vertex_flips_v1
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); rr=a.run_root.resolve()

    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_FLIP12_COURT",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    candidate=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({"court":"REPAIRED_FLIP12_V1","policy":policy.qualification_policy_lineage_hash}),
      explicit_face_provenance=explicit,
    )
    before=report_quality(candidate,policy)
    t=time.perf_counter()
    flipped,fr=repair_candidate_fixed_vertex_flips_v1(candidate,policy,max_passes=12)
    wall=time.perf_counter()-t
    after=report_quality(flipped,policy)
    topo=edge_topology(flipped.faces)

    report={
      "schema":"RealSaS.KnightRepairedFlip12Court.v1",
      "status":"MEASURED_AUDIT_ONLY__NO_PRODUCT_MUTATION",
      "before":before,
      "after":after,
      "accepted_flip_count":fr["accepted_flip_count"],
      "rejected_shape_deviation_count":fr["rejected_shape_deviation_count"],
      "rejected_topology_count":fr["rejected_topology_count"],
      "wall_seconds":wall,
      "topology":topo,
      "vertex_count":len(flipped.vertices),
      "face_count":len(flipped.faces),
      "support_binding_changed":False,
      "vertex_motion":False,
      "success":bool(topo["nonmanifold"]==0 and after["policy_violating_face_count"]<before["policy_violating_face_count"]),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_FLIP12="+json.dumps({
      "before_violations":before["policy_violating_face_count"],
      "after_violations":after["policy_violating_face_count"],
      "accepted_flip_count":fr["accepted_flip_count"],
      "before_min_angle":before["min_angle_deg"],
      "after_min_angle":after["min_angle_deg"],
      "before_max_aspect":before["max_aspect_longest_over_min_altitude"],
      "after_max_aspect":after["max_aspect_longest_over_min_altitude"],
      "nonmanifold_after":topo["nonmanifold"],
      "wall_seconds":wall,
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]: raise SystemExit(2)

if __name__=="__main__":main()
