from __future__ import annotations
import argparse,json,time
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import (
    loadj,build_repaired_surface,report_quality,edge_topology,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_topology_safe_flip_v2 import (
    repair_candidate_fixed_vertex_flips_topology_safe_v2,_manifold_report,
)
from compiler.realsas_compiler_core.canonical_mesh_quality_batched_collapse_v2 import (
    repair_candidate_endpoint_collapses_batched_v2,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();rr=a.run_root.resolve()

    t0=time.perf_counter()
    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(
        c.component_id,"MESH",("REPAIRED_STATIC_QUALITY_V2",),
        metadata={"automatic":True,"semantic_recognition_used":False}
      ) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    candidate=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({
        "court":"REPAIRED_STATIC_QUALITY_V2",
        "policy":policy.qualification_policy_lineage_hash
      }),
      explicit_face_provenance=explicit,
    )
    build_wall=time.perf_counter()-t0
    before=report_quality(candidate,policy)
    topo_before=_manifold_report(candidate.faces)
    if not topo_before["passed"]:
        raise RuntimeError(f"STATIC_QUALITY_V2_INPUT_NOT_MANIFOLD:{topo_before}")

    t1=time.perf_counter()
    flipped,fr=repair_candidate_fixed_vertex_flips_topology_safe_v2(
      candidate,policy,max_passes=12
    )
    flip_wall=time.perf_counter()-t1
    after_flip=report_quality(flipped,policy)

    t2=time.perf_counter()
    collapsed,cr=repair_candidate_endpoint_collapses_batched_v2(
      flipped,policy,max_batches=64,max_collapses=2048
    )
    collapse_wall=time.perf_counter()-t2
    after=report_quality(collapsed,policy)
    topo_after=_manifold_report(collapsed.faces)

    report={
      "schema":"RealSaS.KnightRepairedStaticQualityCourt.v2",
      "status":"PASS_STATIC_REPAIRED_QUALITY_V2" if (
        topo_after["passed"] and after["policy_violating_face_count"]==0
      ) else "FAIL_STATIC_REPAIRED_QUALITY_V2",
      "input":{
        "surface_nodes":len(surface.surface_nodes),
        "candidate_vertices":len(candidate.vertices),
        "candidate_faces":len(candidate.faces),
        "quality":before,
        "topology":topo_before,
      },
      "flip":{
        "accepted_flip_count":fr["accepted_flip_count"],
        "rejected_batch_conflict_count":fr["rejected_batch_conflict_count"],
        "quality":after_flip,
        "topology":fr["final_topology"],
        "wall_seconds":flip_wall,
      },
      "collapse":{
        "accepted_collapse_count":cr["accepted_collapse_count"],
        "batch_count":cr["batch_count"],
        "quality":cr["after"],
        "topology":cr["final_topology"],
        "rejected_boundary_vertex_count":cr["rejected_boundary_vertex_count"],
        "wall_seconds":collapse_wall,
      },
      "output":{
        "vertices":len(collapsed.vertices),
        "faces":len(collapsed.faces),
        "quality":after,
        "topology":topo_after,
      },
      "wall_seconds":{
        "build":build_wall,"flip":flip_wall,"collapse":collapse_wall,
        "total":build_wall+flip_wall+collapse_wall,
      },
      "success":bool(topo_after["passed"] and after["policy_violating_face_count"]==0),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_REPAIRED_STATIC_QUALITY_V2="+json.dumps({
      "input_nodes":len(surface.surface_nodes),
      "before_violations":before["policy_violating_face_count"],
      "after_flip_violations":after_flip["policy_violating_face_count"],
      "flip_count":fr["accepted_flip_count"],
      "after_collapse_violations":after["policy_violating_face_count"],
      "collapse_count":cr["accepted_collapse_count"],
      "collapse_batches":cr["batch_count"],
      "after_min_angle":after["min_angle_deg"],
      "after_max_aspect":after["max_aspect_longest_over_min_altitude"],
      "topology_pass":topo_after["passed"],
      "wall_seconds":report["wall_seconds"],
      "success":report["success"],
    },sort_keys=True),flush=True)
    if not report["success"]:raise SystemExit(2)

if __name__=="__main__":main()
