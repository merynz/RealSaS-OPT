from __future__ import annotations
import argparse,time,json
from pathlib import Path

from tools.audit_knight_repaired_quality_collapse_v1 import (
    loadj, build_repaired_surface, report_quality, edge_topology,
)
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR,build_component_carrier_policy,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.canonical_mesh_quality_repair_v1 import (
    repair_candidate_fixed_vertex_flips_v1,
    repair_candidate_endpoint_collapses_v1,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import mesh_policy_from_dict
from compiler.realsas_compiler_core.hashing import content_sha256

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-root",type=Path,required=True)
    ap.add_argument("--inverse-npz",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); rr=a.run_root.resolve()
    t=time.perf_counter()
    surface,explicit=build_repaired_surface(rr,a.inverse_npz)
    print("TIMING build_surface",time.perf_counter()-t,flush=True)
    partition=build_structural_partition(surface,boundary_overrides=())
    carrier=build_component_carrier_policy(
      partition=partition,
      decisions=tuple(ComponentCarrierDecisionIR(c.component_id,"MESH",("QUALITY_TIMING",),
        metadata={"automatic":True,"semantic_recognition_used":False}) for c in partition.components),
      metadata={"default_carrier":"MESH","automatic":True},
    )
    policy=mesh_policy_from_dict(loadj(rr/"artifacts/18_CANONICAL_MESH_ADDRESSING_BUILD/mesh_qualification_policy.json"))
    candidate=build_canonical_relation_candidate(
      surface,partition,carrier,
      producer_policy_hash=content_sha256({"court":"QUALITY_TIMING","policy":policy.qualification_policy_lineage_hash}),
      explicit_face_provenance=explicit,
    )
    print("TIMING candidate",time.perf_counter()-t,"faces",len(candidate.faces),flush=True)
    before=report_quality(candidate,policy)
    t1=time.perf_counter()
    flipped,fr=repair_candidate_fixed_vertex_flips_v1(candidate,policy,max_passes=2)
    print("TIMING flips2",time.perf_counter()-t1,"accepted",fr["accepted_flip_count"],"viol",fr["after"]["policy_violating_face_count"],flush=True)
    t2=time.perf_counter()
    collapsed,cr=repair_candidate_endpoint_collapses_v1(flipped,policy,max_collapses=16)
    print("TIMING collapse16",time.perf_counter()-t2,"accepted",cr["accepted_collapse_count"],"viol",cr["after"]["policy_violating_face_count"],flush=True)
    out={
      "schema":"RealSaS.KnightRepairedQualityTiming.v1",
      "before":before,"after_flip":fr["after"],"after_collapse":cr["after"],
      "flip_count":fr["accepted_flip_count"],"collapse_count":cr["accepted_collapse_count"],
      "topology":edge_topology(collapsed.faces),
    }
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
if __name__=="__main__":main()
