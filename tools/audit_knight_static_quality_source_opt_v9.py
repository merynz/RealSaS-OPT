from __future__ import annotations
import argparse,json,time
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import loadj,build_repaired_surface,report_quality
from tools.audit_knight_static_quality_alternating_v3 import classify
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,mechanical_partition_from_dict,mesh_policy_from_dict,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_source_surface_optimize_v1 import (
    repair_candidate_source_surface_quality_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import _manifold_report
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import mechanical_quality_protected_surface_ids_v1
from compiler.realsas_compiler_core.hashing import content_sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--static-v8-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    current=canonical_mesh_candidate_from_dict(loadj(a.static_v8_dir/"final_candidate.json"))
    partition=mechanical_partition_from_dict(loadj(a.static_v8_dir/"partition.json"))
    protected=mechanical_quality_protected_surface_ids_v1(partition)

    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    refpart=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=refpart,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in refpart.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    reference=build_canonical_relation_candidate(
      surface,refpart,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_ALTERNATING_V3",
        "policy":policy.qualification_policy_lineage_hash,
      }),
      explicit_face_provenance=explicit,
    )

    before=report_quality(current,policy)
    topo0=_manifold_report(current.faces)
    if not topo0["passed"]:
        raise RuntimeError(f"STATIC_V9_INPUT_TOPOLOGY_FAIL:{topo0}")

    t0=time.perf_counter()
    repaired,rrpt=repair_candidate_source_surface_quality_v1(
        current,reference,policy,
        protected_surface_ids=protected,
        grid_schedule=(8,16,32,64),
        max_batches=8,max_moves=64,
    )
    elapsed=time.perf_counter()-t0
    after=report_quality(repaired,policy)
    topo=_manifold_report(repaired.faces)
    classification=classify(repaired,policy,partition)

    report={
      "schema":"RealSaS.KnightStaticQualitySourceSurfaceOptimizationCourt.v9",
      "status":"PASS" if topo["passed"] and int(after["policy_violating_face_count"])==0 else "FAIL",
      "input_candidate_lineage_hash":current.candidate_lineage_hash,
      "reference_candidate_lineage_hash":reference.candidate_lineage_hash,
      "before":before,
      "optimization_report":rrpt,
      "after":after,
      "topology":topo,
      "classification":classification,
      "elapsed_seconds":elapsed,
      "final_candidate_lineage_hash":repaired.candidate_lineage_hash,
      "final_vertex_count":len(repaired.vertices),
      "final_face_count":len(repaired.faces),
      "success":bool(topo["passed"] and int(after["policy_violating_face_count"])==0),
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"final_candidate.json").write_text(json.dumps(repaired.to_dict(),indent=2,sort_keys=True)+"\n")
    (out/"partition.json").write_text(json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STATIC_QUALITY_SOURCE_OPT_V9="+json.dumps({
      "before_violations":before["policy_violating_face_count"],
      "after_violations":after["policy_violating_face_count"],
      "accepted_moves":rrpt["accepted_move_count"],
      "before_min_angle":before["min_angle_deg"],
      "after_min_angle":after["min_angle_deg"],
      "before_max_aspect":before["max_aspect_longest_over_min_altitude"],
      "after_max_aspect":after["max_aspect_longest_over_min_altitude"],
      "topology_pass":topo["passed"],
      "boundary_faces":classification["boundary_face_count"],
      "protected_faces":classification["protected_support_face_count"],
      "interior_unprotected":classification["interior_unprotected_face_count"],
      "elapsed_seconds":elapsed,
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
