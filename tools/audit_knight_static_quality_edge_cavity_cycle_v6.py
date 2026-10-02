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
from compiler.realsas_compiler_core.canonical_mesh_quality_edge_cavity_remesh_v1 import (
    repair_candidate_edge_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_cavity_remesh_v1 import (
    repair_candidate_cavity_retriangulation_v1,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,_manifold_report,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_relaxation_v2 import (
    repair_candidate_projected_relaxation_batched_v2,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    mechanical_quality_protected_surface_ids_v1,
)
from compiler.realsas_compiler_core.hashing import content_sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--static-v5-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--max-cycles",type=int,default=6)
    a=ap.parse_args(); rr=a.run_root.resolve(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)

    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    current=canonical_mesh_candidate_from_dict(loadj(a.static_v5_dir/"final_candidate.json"))
    partition=mechanical_partition_from_dict(loadj(a.static_v5_dir/"partition.json"))
    protected=mechanical_quality_protected_surface_ids_v1(partition)

    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    reference_partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=reference_partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_ALTERNATING_V3",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in reference_partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    reference=build_canonical_relation_candidate(
      surface,reference_partition,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_ALTERNATING_V3",
        "policy":policy.qualification_policy_lineage_hash,
      }),
      explicit_face_provenance=explicit,
    )

    initial=report_quality(current,policy)
    topo0=_manifold_report(current.faces)
    if not topo0["passed"]:
        raise RuntimeError(f"EDGE_CAVITY_V6_INPUT_NOT_MANIFOLD:{topo0}")
    previous=int(initial["policy_violating_face_count"])
    cycles=[]
    t0=time.perf_counter()

    for ci in range(int(a.max_cycles)):
        row={"cycle":ci,"before_violations":previous}

        t=time.perf_counter()
        ec,ecr=repair_candidate_edge_cavity_retriangulation_v1(
            current,policy,protected_surface_ids=protected,max_batches=64,max_removed_edges=2048
        )
        qec=report_quality(ec,policy)
        row.update({
          "edge_cavity_count":int(ecr["accepted_edge_cavity_count"]),
          "edge_cavity_batches":int(ecr["batch_count"]),
          "after_edge_cavity_violations":int(qec["policy_violating_face_count"]),
          "edge_cavity_seconds":time.perf_counter()-t,
          "edge_cavity_rejected":ecr["rejected_counts"],
        })

        t=time.perf_counter()
        cav,cr=repair_candidate_cavity_retriangulation_v1(
            ec,policy,protected_surface_ids=protected,max_batches=64,max_removed_vertices=2048
        )
        qcav=report_quality(cav,policy)
        row.update({
          "vertex_cavity_count":int(cr["accepted_cavity_count"]),
          "after_vertex_cavity_violations":int(qcav["policy_violating_face_count"]),
          "vertex_cavity_seconds":time.perf_counter()-t,
        })

        t=time.perf_counter()
        fl,fr=repair_candidate_fixed_vertex_flips_topology_safe_v2(cav,policy,max_passes=12)
        qfl=report_quality(fl,policy)
        row.update({
          "flip_count":int(fr["accepted_flip_count"]),
          "after_flip_violations":int(qfl["policy_violating_face_count"]),
          "flip_seconds":time.perf_counter()-t,
        })

        t=time.perf_counter()
        co,col=repair_candidate_endpoint_collapses_batched_v2(
            fl,policy,max_batches=64,max_collapses=2048
        )
        qco=report_quality(co,policy)
        row.update({
          "collapse_count":int(col["accepted_collapse_count"]),
          "after_collapse_violations":int(qco["policy_violating_face_count"]),
          "collapse_seconds":time.perf_counter()-t,
        })

        t=time.perf_counter()
        rel,rrpt=repair_candidate_projected_relaxation_batched_v2(
            co,reference,policy,protected_surface_ids=protected,max_batches=64,max_moves=2048
        )
        qrel=report_quality(rel,policy)
        row.update({
          "relax_count":int(rrpt["accepted_move_count"]),
          "after_relax_violations":int(qrel["policy_violating_face_count"]),
          "relax_seconds":time.perf_counter()-t,
          "min_angle_deg":float(qrel["min_angle_deg"]),
          "max_aspect":float(qrel["max_aspect_longest_over_min_altitude"]),
          "vertices":len(rel.vertices),"faces":len(rel.faces),
          "topology":_manifold_report(rel.faces),
        })
        cycles.append(row)
        print("STATIC_EDGE_CAVITY_V6_CYCLE="+json.dumps(row,sort_keys=True),flush=True)

        current=rel
        now=int(qrel["policy_violating_face_count"])
        if now==0 or now>=previous:
            break
        previous=now

    final_quality=report_quality(current,policy)
    final_topology=_manifold_report(current.faces)
    final_class=classify(current,policy,partition)
    report={
      "schema":"RealSaS.KnightStaticQualityEdgeCavityCycleCourt.v6",
      "status":"PASS" if (
        final_topology["passed"] and final_quality["policy_violating_face_count"]==0
      ) else "FAIL",
      "input_candidate_lineage_hash":canonical_mesh_candidate_from_dict(
          loadj(a.static_v5_dir/"final_candidate.json")
      ).candidate_lineage_hash,
      "reference_candidate_lineage_hash":reference.candidate_lineage_hash,
      "initial_quality":initial,
      "cycles":cycles,
      "final_quality":final_quality,
      "final_topology":final_topology,
      "classification":final_class,
      "final_candidate_lineage_hash":current.candidate_lineage_hash,
      "final_vertex_count":len(current.vertices),
      "final_face_count":len(current.faces),
      "total_seconds":time.perf_counter()-t0,
      "audit_only_surface_remesh_used":any(
        r["edge_cavity_count"]>0 or r["vertex_cavity_count"]>0 for r in cycles
      ),
      "product_stage14_remesh_authority_required":True,
      "success":bool(final_topology["passed"] and final_quality["policy_violating_face_count"]==0),
    }
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    (out/"final_candidate.json").write_text(json.dumps(current.to_dict(),indent=2,sort_keys=True)+"\n")
    (out/"partition.json").write_text(json.dumps(partition.to_dict(),indent=2,sort_keys=True)+"\n")
    print("KNIGHT_STATIC_QUALITY_EDGE_CAVITY_V6="+json.dumps({
      "cycles":[
        [r["cycle"],r["before_violations"],r["after_edge_cavity_violations"],
         r["after_vertex_cavity_violations"],r["after_flip_violations"],
         r["after_collapse_violations"],r["after_relax_violations"],
         r["edge_cavity_count"],r["vertex_cavity_count"],r["flip_count"],
         r["collapse_count"],r["relax_count"]]
        for r in cycles
      ],
      "final_violations":final_quality["policy_violating_face_count"],
      "final_min_angle":final_quality["min_angle_deg"],
      "final_max_aspect":final_quality["max_aspect_longest_over_min_altitude"],
      "boundary_faces":final_class["boundary_face_count"],
      "protected_faces":final_class["protected_support_face_count"],
      "interior_unprotected":final_class["interior_unprotected_face_count"],
      "topology_pass":final_topology["passed"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:
        raise SystemExit(2)

if __name__=="__main__":main()
